#!/usr/bin/env python3
"""Automatic YuNet/SFace verification using conservative eight-frame voting."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import face_core as core
import face_match


DEFAULT_THRESHOLD = face_match.DEFAULT_THRESHOLD
STABLE_HOLD_SECONDS = 0.7
SCAN_TIMEOUT_SECONDS = 5.0
SCAN_FRAME_COUNT = face_match.REQUIRED_FRAMES
RESULT_SECONDS = 2.0
WINDOW_NAME = "Face verification"


class StatusFade:
    def __init__(self):
        self.message = None
        self.changed_at = 0.0

    def alpha(self, message, now):
        if message != self.message:
            self.message = message
            self.changed_at = now
        return min(1.0, max(0.0, (now - self.changed_at) / 0.18))


def threshold_value(value):
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("threshold must be numeric") from error
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return threshold


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=threshold_value, help="Override calibrated/default threshold")
    parser.add_argument("--image", type=Path, help="Diagnostic score for one still image (not an 8-frame decision)")
    parser.add_argument("--attack-test", action="store_true", help="Print PASS only when the scanned subject is rejected")
    return parser.parse_args()


def active_threshold(override, data_dir=None):
    if override is not None:
        return float(override), "command line"
    path = (Path(data_dir) if data_dir else core.DATA_DIR) / "calibration.json"
    if path.exists():
        try:
            calibration = json.loads(path.read_text(encoding="utf-8"))
            raw_value = calibration.get("recommended_threshold")
            if raw_value is None:
                return DEFAULT_THRESHOLD, "default (calibration scores incomplete)"
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("recommended_threshold must be a finite non-negative number")
            return value, str(path)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not load calibration threshold from {path}: {error}") from error
    return DEFAULT_THRESHOLD, "default"


def compare_probe(probe, enrollments):
    """Compatibility helper returning the normalized top-three score per identity."""
    return face_match.score_embedding(probe, enrollments)


def print_frame_result(frame_number, scores, threshold):
    name, score, margin, reason = face_match.frame_vote(scores, threshold)
    print(f"Frame {frame_number}:")
    for person, value in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"  {person}: top-3 mean cosine={value:.6f}")
    gap = "n/a" if math.isinf(margin) else f"{margin:.6f}"
    print(
        f"  vote={name or 'NONE'} best={score:.6f} margin={gap} "
        f"reason={reason} threshold={threshold:.3f}"
    )
    return name, score, margin, reason


def report_decision(frame_results, threshold):
    decision = face_match.decide_votes(frame_results, threshold)
    all_names = sorted({name for scores in frame_results for name in scores})
    counts = {name: decision["votes"].get(name, 0) for name in all_names}
    print("Final votes: " + ", ".join(f"{name}={counts[name]}/8" for name in all_names))
    if decision["accepted"]:
        winner = decision["name"]
        winner_scores = [scores[winner] for scores in frame_results]
        score = float(np.mean(winner_scores))
        print(f"Final decision: SAME PERSON ({winner}, mean frame score {score:.6f})")
    else:
        score = max((max(values.values()) for values in frame_results), default=0.0)
        print(f"Final decision: NOT RECOGNIZED ({decision['reason']})")
    return decision, score


def verify_image(path, enrollments, threshold):
    """Print diagnostic scores; one still image cannot satisfy the eight-frame rule."""
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not read image: {path}")
    core.load_models()
    faces = core.detect_faces(frame)
    if len(faces) != 1:
        raise ValueError(f"Image must contain exactly one detected face; found {len(faces)}.")
    gate = core.QualityGate()
    status, face, _, _ = gate.evaluate(frame, faces)
    if status or face is None:
        raise ValueError(f"Image failed the enrollment quality gate: {status}.")
    probe = core.get_embedding(frame, face)
    scores = compare_probe(probe, enrollments)
    print("Single-image diagnostic scores (not a verification decision):")
    for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: top-3 mean cosine={score:.6f}")
    print(
        f"Result: NOT RECOGNIZED / INSUFFICIENT FRAMES "
        f"(one image; 8 good frames required, threshold {threshold:.3f})"
    )
    return None, max(scores.values()), False


def _show_result(camera, name, score, same, attack_test=False, message=None):
    start = time.monotonic()
    progress = [1.0] * core.TICK_COUNT
    color = (90, 220, 90) if same else (55, 55, 235)
    if message:
        result_text = message
    elif attack_test:
        result_text = "ATTACK TEST PASS" if not same else "ATTACK TEST FAIL"
    else:
        result_text = f"Recognized: {name}" if same else "Not recognized"
    status_text = f"Score: {score:.3f}" if score is not None else "8 clear frames required"
    fade = StatusFade()
    while time.monotonic() - start < RESULT_SECONDS:
        loop_started = time.monotonic()
        ok, frame = camera.read()
        if not ok or frame is None:
            frame = None
        now = time.monotonic()
        canvas = core.render_ui(
            frame,
            status_text,
            progress=progress,
            pulse=(math.sin((now - start) * math.pi) + 1) / 2,
            title="Face verification",
            result_color=color,
            result_text=result_text,
            status_alpha=fade.alpha(status_text, now),
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False
        time.sleep(max(0.0, 1 / 30 - (time.monotonic() - loop_started)))
    return True


def verify_webcam(enrollments, threshold, attack_test=False):
    core.load_models()
    enrollments = face_match.normalize_enrollments(enrollments)
    camera = core.open_camera()
    gate = core.QualityGate()
    fade = StatusFade()
    stable_since = None
    scan_started = None
    frame_scores = []
    last_frame = None
    next_sample_at = 0.0
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        print("Verification starts after holding one good face in the circle.")
        if len(enrollments) == 1:
            print(
                "WARNING: Only one identity is enrolled; false accepts are more likely. "
                "Enroll at least two people or collect impostor calibration data.",
                file=sys.stderr,
            )
        while True:
            loop_started = time.monotonic()
            now = loop_started
            ok, frame = camera.read()
            if not ok or frame is None:
                status = "Position your face in the circle"
                canvas = core.render_ui(
                    last_frame,
                    status,
                    title="Face verification",
                    status_alpha=fade.alpha(status, now),
                )
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                time.sleep(0.01)
                continue
            last_frame = frame
            faces = core.detect_faces(frame)
            status, face, _, pose = gate.evaluate(frame, faces)
            scan_phase = None

            if scan_started is None:
                if status:
                    stable_since = None
                elif stable_since is None:
                    stable_since = now
                    status = "Hold still"
                elif now - stable_since >= STABLE_HOLD_SECONDS:
                    scan_started = now
                    frame_scores = []
                    next_sample_at = now
                    status = "Scanning"
            else:
                elapsed = now - scan_started
                scan_phase = min(1.0, elapsed / SCAN_TIMEOUT_SECONDS)
                if not status and face is not None and now >= next_sample_at:
                    try:
                        # TODO(liveness): add a liveness check before accepting probe frames.
                        embedding = core.get_embedding(frame, face)
                        scores = face_match.score_embedding(embedding, enrollments)
                        frame_scores.append(scores)
                        print_frame_result(len(frame_scores), scores, threshold)
                    except (ValueError, cv2.error) as error:
                        print(f"Skipped one verification frame: {error}", file=sys.stderr)
                    next_sample_at = now + 0.08
                    status = "Scanning"

                if len(frame_scores) == SCAN_FRAME_COUNT:
                    decision, score = report_decision(frame_scores, threshold)
                    same = bool(decision["accepted"])
                    name = decision["name"]
                    if attack_test:
                        print("ATTACK TEST: " + ("FAIL (recognized)" if same else "PASS (rejected)"))
                    if not _show_result(
                        camera,
                        name,
                        score,
                        same,
                        attack_test=attack_test,
                    ):
                        break
                    scan_started = None
                    stable_since = None
                    frame_scores = []
                    status = "Position your face in the circle"
                elif elapsed >= SCAN_TIMEOUT_SECONDS:
                    print(
                        f"Scan quality failure: collected {len(frame_scores)}/{SCAN_FRAME_COUNT} "
                        "good frames within 5 seconds."
                    )
                    if attack_test:
                        print("ATTACK TEST: INCONCLUSIVE (clear scan unavailable)")
                    if not _show_result(
                        camera,
                        None,
                        None,
                        False,
                        attack_test=attack_test,
                        message="Can't get a clear scan, try again",
                    ):
                        break
                    scan_started = None
                    stable_since = None
                    frame_scores = []
                    status = "Position your face in the circle"

            display = core.render_ui(
                frame,
                status,
                scan_phase=scan_phase,
                direction=pose if face is not None else None,
                title="Face verification",
                status_alpha=fade.alpha(status, now),
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            time.sleep(max(0.0, 1 / 30 - (time.monotonic() - loop_started)))
    except KeyboardInterrupt:
        print("\nVerification stopped by Ctrl+C.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    try:
        enrollments = core.load_enrollments()
        threshold, source = active_threshold(args.threshold)
        print(f"Using cosine threshold {threshold:.3f} ({source}).")
        if args.image:
            if args.attack_test:
                raise ValueError("--attack-test requires a live webcam scan.")
            verify_image(args.image, enrollments, threshold)
        else:
            verify_webcam(enrollments, threshold, attack_test=args.attack_test)
    except KeyboardInterrupt:
        print("\nVerification stopped by Ctrl+C.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f"Verification error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
