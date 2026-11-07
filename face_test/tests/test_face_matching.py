"""Headless tests for conservative per-frame face verification."""

import ast
import inspect
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import calibrate
import enroll
import face_core as core
import face_match
import verify


def _unit(index):
    value = np.zeros(core.EMBEDDING_SIZE, dtype=np.float32)
    value[index] = 1.0
    return value


def _nearby_samples(index):
    rows = []
    for offset in range(3):
        value = _unit(index)
        value[index + 1 + offset] = 0.02 * (offset + 1)
        rows.append(value / np.linalg.norm(value))
    return np.stack(rows)


def _scores(probe, enrollments):
    return face_match.score_embedding(probe, enrollments)


class FrameDecisionTests(unittest.TestCase):
    def test_genuine_frames_are_accepted(self):
        enrollments = {"Alice": _nearby_samples(0)}
        frame_results = [_scores(_unit(0), enrollments) for _ in range(8)]
        result = face_match.decide_votes(frame_results, threshold=0.5)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["name"], "Alice")
        self.assertEqual(result["votes"]["Alice"], 8)

    def test_impostor_frames_below_threshold_are_rejected(self):
        enrolled = _unit(0)
        probe = np.zeros(core.EMBEDDING_SIZE, dtype=np.float32)
        probe[0] = 0.49
        probe[1] = math.sqrt(1.0 - 0.49**2)
        frame_results = [_scores(probe, {"Alice": _nearby_samples(0)}) for _ in range(8)]
        self.assertFalse(face_match.decide_votes(frame_results, threshold=0.5)["accepted"])

    def test_four_votes_out_of_eight_are_rejected(self):
        enrolled = {"Alice": _nearby_samples(0)}
        genuine = _scores(_unit(0), enrolled)
        impostor_probe = np.zeros(core.EMBEDDING_SIZE, dtype=np.float32)
        impostor_probe[0] = 0.2
        impostor_probe[1] = math.sqrt(1.0 - 0.2**2)
        impostor = _scores(impostor_probe, enrolled)
        result = face_match.decide_votes([genuine] * 4 + [impostor] * 4, threshold=0.5)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["votes"]["Alice"], 4)

    def test_two_close_people_fail_the_margin_gate(self):
        scores = {"Alice": 0.81, "Bob": 0.78}
        name, _, margin, reason = face_match.frame_vote(scores, threshold=0.5)
        self.assertIsNone(name)
        self.assertAlmostEqual(margin, 0.03)
        self.assertEqual(reason, "low_margin")
        result = face_match.decide_votes([scores] * 8, threshold=0.5)
        self.assertFalse(result["accepted"])

    def test_higher_cosine_score_means_more_similar(self):
        enrolled = {"Alice": _nearby_samples(0)}
        closer = _unit(0)
        farther = np.zeros(core.EMBEDDING_SIZE, dtype=np.float32)
        farther[0] = 0.7
        farther[1] = math.sqrt(1.0 - 0.7**2)
        self.assertGreater(
            _scores(closer, enrolled)["Alice"],
            _scores(farther, enrolled)["Alice"],
        )

    def test_person_score_is_mean_of_top_three_not_single_best(self):
        rows = []
        for index, cosine in enumerate((0.9, 0.8, 0.7, 0.1), start=1):
            vector = np.zeros(core.EMBEDDING_SIZE, dtype=np.float32)
            vector[0] = cosine
            vector[index] = math.sqrt(1.0 - cosine**2)
            rows.append(vector)
        score = _scores(_unit(0), {"Alice": np.stack(rows)})["Alice"]
        self.assertAlmostEqual(score, (0.9 + 0.8 + 0.7) / 3, places=6)

    def test_calibration_recommends_a_fail_closed_separating_threshold(self):
        threshold, overlap = face_match.recommend_threshold(
            genuine_scores=[0.78, 0.83],
            impostor_scores=[0.42, 0.51],
        )
        self.assertAlmostEqual(threshold, 0.56)
        self.assertFalse(overlap)
        _, overlap = face_match.recommend_threshold([0.60], [0.61])
        self.assertTrue(overlap)

    def test_calibration_scores_save_without_images_and_drive_verification_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            saved = calibrate._save_calibration(
                path,
                {"Alice": [0.82, 0.80]},
                [0.41, 0.45],
            )
            self.assertAlmostEqual(saved["recommended_threshold"], 0.50)
            self.assertEqual(
                verify.active_threshold(None, data_dir=directory),
                (0.5, str(path)),
            )
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("image", text)


class SharedEmbeddingPipelineTests(unittest.TestCase):
    def test_enrollment_verification_and_calibration_call_shared_core_function(self):
        for module in (enroll, verify, calibrate):
            tree = ast.parse(inspect.getsource(module))
            calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_embedding"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "core"
            ]
            self.assertTrue(calls, f"{module.__name__} must use core.get_embedding")

    def test_core_embedding_aligns_original_frame_and_normalizes_feature(self):
        class FakeRecognizer:
            def __init__(self):
                self.frame = None
                self.row = None
                self.aligned = object()

            def alignCrop(self, frame, row):
                self.frame = frame
                self.row = row
                return self.aligned

            def feature(self, aligned):
                self.assert_aligned = aligned
                return np.full((1, core.EMBEDDING_SIZE), 2.0, dtype=np.float32)

        fake = FakeRecognizer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        row = np.asarray([20, 20, 60, 60, 35, 40, 65, 40, 50, 52, 40, 68, 60, 68, 0.99])
        with mock.patch.object(core, "_recognizer", fake):
            result = core.get_embedding(frame, row)
        self.assertIs(fake.frame, frame)
        np.testing.assert_array_equal(fake.row, row.astype(np.float32))
        self.assertIs(fake.assert_aligned, fake.aligned)
        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
