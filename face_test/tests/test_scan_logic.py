"""Headless tests for pose-derived enrollment coverage and validation."""

import math
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import enroll
import face_core as core


POSES = {
    "CENTER": (0.0, 0.0),
    "LEFT": (-22.0, 0.0),
    "RIGHT": (22.0, 0.0),
    "UP": (0.0, -12.0),
    "DOWN": (0.0, 12.0),
    "UP-LEFT": (-14.0, -9.0),
    "UP-RIGHT": (14.0, -9.0),
    "DOWN-LEFT": (-14.0, 9.0),
    "DOWN-RIGHT": (14.0, 9.0),
}


def _synthetic_embedding(pose_index, sample_index):
    vector = np.zeros(core.EMBEDDING_SIZE, dtype=np.float32)
    vector[0] = math.sqrt(0.55)
    vector[1 + pose_index] = math.sqrt(0.25)
    vector[10 + pose_index * 3 + sample_index] = math.sqrt(0.20)
    return vector


def _fill_pose(coverage, pose_name, start_time, pose_index):
    yaw, pitch = POSES[pose_name]
    now = float(start_time)
    sample_index = 0
    while coverage.count(pose_name) < 3:
        coverage.observe(
            pose_name,
            lambda: _synthetic_embedding(pose_index, sample_index),
            yaw,
            pitch,
            now,
            good=True,
        )
        if coverage.count(pose_name) > sample_index:
            sample_index = coverage.count(pose_name)
        now += 1.0 / 30.0
    return now


def _fill_all(coverage, omit=()):
    now = 0.0
    for index, name in enumerate(enroll.POSE_NAMES):
        if name not in omit:
            now = _fill_pose(coverage, name, now, index) + 0.3
    return now


def _rotation_matrix(yaw, pitch, roll):
    rx, _ = cv2.Rodrigues(np.asarray([math.radians(pitch), 0.0, 0.0]))
    ry, _ = cv2.Rodrigues(np.asarray([0.0, math.radians(-yaw), 0.0]))
    rz, _ = cv2.Rodrigues(np.asarray([0.0, 0.0, math.radians(roll)]))
    return rz @ ry @ rx


def _synthetic_face_row(yaw, pitch, roll=0.0, frame_size=(640, 480)):
    width, height = frame_size
    camera = np.asarray(
        [[width, 0.0, width / 2], [0.0, width, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation = _rotation_matrix(yaw, pitch, roll)
    rotation_vector, _ = cv2.Rodrigues(rotation)
    translation = np.asarray([[0.0], [0.0], [600.0]], dtype=np.float64)
    points, _ = cv2.projectPoints(
        core.YUNET_FACE_MODEL_3D,
        rotation_vector,
        translation,
        camera,
        np.zeros((4, 1), dtype=np.float64),
    )
    row = np.zeros(15, dtype=np.float32)
    points = points.reshape(5, 2).astype(np.float32)
    row[:4] = (points[:, 0].min(), points[:, 1].min(), np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    row[4:14] = points.reshape(-1)
    row[14] = 0.99
    return row


class PoseEstimationTests(unittest.TestCase):
    def test_solvepnp_pose_signs_match_mirrored_display_guidance(self):
        right_turn = core.estimate_head_pose(_synthetic_face_row(25, 0), (640, 480))
        left_turn = core.estimate_head_pose(_synthetic_face_row(-25, 0), (640, 480))
        looking_up = core.estimate_head_pose(_synthetic_face_row(0, -16), (640, 480))
        looking_down = core.estimate_head_pose(_synthetic_face_row(0, 16), (640, 480))
        self.assertGreater(right_turn[0], 15.0)
        self.assertLess(left_turn[0], -15.0)
        self.assertLess(looking_up[1], -10.0)
        self.assertGreater(looking_down[1], 10.0)
        self.assertAlmostEqual(right_turn[1], 0.0, delta=4.0)
        self.assertAlmostEqual(looking_up[0], 0.0, delta=4.0)

    def test_calibrated_yaw_wraparound_never_looks_like_a_full_turn(self):
        calibrated = enroll.calibrated_pose((179.0, 3.0, 0.0), (-179.0, 3.0))
        self.assertAlmostEqual(calibrated[0], -2.0)
        self.assertAlmostEqual(calibrated[1], 0.0)
        self.assertEqual(enroll.classify_pose(*calibrated), "CENTER")

    def test_neutral_median_unwraps_samples_crossing_180_degrees(self):
        neutral = enroll.median_pose_offset(
            [(179.0, 1.0), (-179.0, 2.0), (178.0, 0.0), (-178.0, 1.0), (180.0, 1.0)]
        )
        self.assertLess(abs(enroll.angle_delta_degrees(neutral[0], 180.0)), 2.0)
        self.assertAlmostEqual(neutral[1], 1.0)

    def test_pose_classifier_requires_all_angle_bounds(self):
        for name, pose in POSES.items():
            self.assertEqual(enroll.classify_pose(*pose), name)
        self.assertIsNone(enroll.classify_pose(0, 9))
        self.assertIsNone(enroll.classify_pose(10, 0))


class RequiredCoverageTests(unittest.TestCase):
    def test_sweep_of_all_nine_poses_completes_and_saves_verified_artifacts(self):
        coverage = enroll.PoseCoverage()
        _fill_all(coverage)
        self.assertTrue(coverage.complete)
        proof = enroll.coverage_proof(coverage)
        proof_ok = proof[0]
        self.assertTrue(proof_ok)
        cleaned, cohesion, reason = enroll.filter_and_validate(coverage)
        self.assertIsNone(reason)
        self.assertGreaterEqual(cohesion, 0.5)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(core, "DATA_DIR", Path(directory)):
                enroll.save_verified_enrollment("Synthetic", cleaned, proof)
                array = np.load(Path(directory) / "Synthetic.npy", allow_pickle=False)
                metadata = json.loads((Path(directory) / "Synthetic_meta.json").read_text())
                loaded = core.load_enrollments()
            self.assertEqual(array.shape, (27, 128))
            self.assertEqual(loaded["Synthetic"].shape, (27, 128))
            self.assertEqual(set(metadata["pose_counts"]), set(enroll.POSE_NAMES))
            self.assertTrue(all(value == 3 for value in metadata["pose_counts"].values()))

    def test_straight_ahead_for_forty_seconds_never_proves_full_coverage(self):
        coverage = enroll.PoseCoverage()
        for frame in range(40 * 30):
            now = frame / 30.0
            coverage.observe(
                "CENTER",
                lambda frame=frame: _synthetic_embedding(0, frame % 3),
                0.0,
                0.0,
                now,
                good=True,
            )
        proof_ok, _, _, _ = enroll.coverage_proof(coverage)
        self.assertFalse(coverage.complete)
        self.assertFalse(proof_ok)
        result = {
            "kind": "failure",
            "reason": "timeout; missing poses: " + ", ".join(coverage.missing()),
        }
        self.assertEqual(result["kind"], "failure")
        self.assertIn("UP", result["reason"])
        self.assertIn("DOWN-RIGHT", result["reason"])
        self.assertNotIn("Face scan complete", result["reason"])

    def test_retry_preserves_completed_poses_and_only_requests_missing_up(self):
        coverage = enroll.PoseCoverage()
        _fill_all(coverage, omit=("UP",))
        attempts = []
        retry_missing = []

        def attempt(number):
            attempts.append(number)
            if number == 1:
                return {"kind": "failure", "reason": "timeout missing UP"}
            self.assertEqual(coverage.missing(), ["UP"])
            retry_missing.append(coverage.missing())
            _fill_pose(coverage, "UP", 20.0, enroll.POSE_NAMES.index("UP"))
            return {"kind": "success", "reason": None} if coverage.complete else {
                "kind": "failure", "reason": "UP incomplete"
            }

        result, count = enroll.run_attempts(
            attempt,
            on_retry=lambda result, next_number, limit: True,
        )
        self.assertEqual(result["kind"], "success")
        self.assertEqual(count, 2)
        self.assertEqual(attempts, [1, 2])
        self.assertEqual(retry_missing, [["UP"]])

    def test_three_failed_attempts_do_not_save(self):
        calls = []
        saves = []

        def attempt(number):
            calls.append(number)
            return {"kind": "failure", "reason": "missing UP"}

        result, count = enroll.run_attempts(
            attempt,
            on_retry=lambda result, next_number, limit: True,
        )
        if result["kind"] == "success":
            saves.append(result)
        self.assertEqual(count, 3)
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(result["kind"], "failure")
        self.assertEqual(saves, [])

    def test_first_attempt_success_runs_exactly_once(self):
        calls = []
        result, count = enroll.run_attempts(
            lambda number: calls.append(number) or {"kind": "success", "reason": None}
        )
        self.assertEqual(result["kind"], "success")
        self.assertEqual(count, 1)
        self.assertEqual(calls, [1])

    def test_pose_requires_three_consecutive_good_frames_and_distinct_spaced_embeddings(self):
        coverage = enroll.PoseCoverage()
        capture = lambda index: _synthetic_embedding(1, index)
        self.assertFalse(coverage.observe("LEFT", capture(0), -22, 0, 0.0))
        self.assertFalse(coverage.observe(None, capture(0), -22, 0, 0.03, good=False))
        self.assertFalse(coverage.observe("LEFT", capture(0), -22, 0, 0.06))
        self.assertFalse(coverage.observe("LEFT", capture(0), -22, 0, 0.09))
        self.assertTrue(coverage.observe("LEFT", capture(0), -22, 0, 0.12))
        self.assertFalse(coverage.observe("LEFT", capture(0), -22, 0, 0.15))
        self.assertEqual(coverage.count("LEFT"), 1)

    def test_self_test_prompts_center_before_check_and_collects_five_spaced_frames(self):
        state = enroll.SelfTestCapture(0.0)
        vector = _synthetic_embedding(0, 0)
        self.assertEqual(
            state.observe(False, 0.1, lambda: vector),
            "Look straight at the camera",
        )
        self.assertEqual(
            state.observe(True, 0.2, lambda: vector),
            "Look straight at the camera",
        )
        state.observe(True, 0.23, lambda: vector)
        self.assertEqual(state.observe(True, 0.26, lambda: vector), "Checking...")
        for sample_time in (0.27, 0.48, 0.69, 0.90, 1.11, 1.32):
            state.observe(True, sample_time, lambda: vector)
        self.assertEqual(len(state.embeddings), 5)
        self.assertTrue(state.finished_at(1.76))

    def test_self_test_times_out_if_center_never_returns(self):
        state = enroll.SelfTestCapture(5.0)
        for index in range(100):
            state.observe(False, 5.0 + index / 30.0, lambda: _synthetic_embedding(0, 0))
        self.assertTrue(state.timed_out_waiting(8.5))
        self.assertEqual(state.embeddings, [])


class CircularUiTests(unittest.TestCase):
    def test_rendered_camera_circle_remains_square(self):
        canvas = core.render_ui(None, "Look straight", draw_ring=False)
        self.assertEqual(canvas.shape, (core.CANVAS_HEIGHT, core.CANVAS_WIDTH, 3))
        cx, cy = core.CAMERA_CENTER
        radius = core.CAMERA_RADIUS + 3
        region = canvas[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
        outline = (
            (region[:, :, 0] > 200)
            & (region[:, :, 1] > 200)
            & (region[:, :, 2] > 200)
            & (np.max(region, axis=2) - np.min(region, axis=2) < 3)
        )
        ys, xs = np.where(outline)
        self.assertGreater(len(xs), 500)
        self.assertEqual(int(xs.max() - xs.min()), int(ys.max() - ys.min()))
        distances = np.hypot(xs - radius, ys - radius)
        self.assertLess(float(distances.max() - distances.min()), 4.0)

    def test_enrollment_ring_ticks_fill_only_after_kept_embeddings(self):
        coverage = enroll.PoseCoverage()
        empty = np.zeros((core.CANVAS_HEIGHT, core.CANVAS_WIDTH, 3), dtype=np.uint8)
        enroll._draw_enrollment_ring(empty, coverage, 0.0, None)
        cx, cy = core.CAMERA_CENTER
        right_tick = empty[cy, cx + 213]
        self.assertLess(int(right_tick[1]), 100)

        _fill_pose(coverage, "RIGHT", 0.0, enroll.POSE_NAMES.index("RIGHT"))
        filled = np.zeros_like(empty)
        enroll._draw_enrollment_ring(filled, coverage, 0.0, None)
        right_tick = filled[cy, cx + 213]
        left_tick = filled[cy, cx - 213]
        self.assertGreater(int(right_tick[1]), int(right_tick[0]))
        self.assertLess(int(left_tick[1]), int(right_tick[1]))


if __name__ == "__main__":
    unittest.main()
