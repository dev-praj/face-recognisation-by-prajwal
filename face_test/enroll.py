#!/usr/bin/env python3
"""Enroll a face only after measured coverage of all nine required head poses."""

import argparse
import json
import math
import os
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import face_core as core


# Tune angle bounds here. Angles are calibrated, smoothed solvePnP degrees.
POSE_CONFIG = {
    "calibration_seconds": 1.0,
    "attempt_timeout_seconds": 40.0,
    "stable_good_frames": 3,
    "embeddings_per_pose": 3,
    "minimum_interval_seconds": 0.2,
    "within_pose_max_cosine": 0.95,
    "consistency_min_cosine": 0.45,
    "minimum_pose_embeddings_after_filter": 2,
    "cohesion_min_cosine": 0.50,
    "self_test_min_cosine": 0.50,
    "self_test_frames": 5,
    "self_test_seconds": 1.5,
    "self_test_center_wait_seconds": 3.5,
    "smoothing_frames": 5,
    "success_animation_seconds": 1.2,
    "no_progress_guidance_seconds": 3.0,
    "minimum_yaw_span_degrees": 40.0,
    "minimum_pitch_span_degrees": 20.0,
    "direction_tick_count": 64,
    "ticks_per_direction_group": 8,
    "CENTER": {"yaw": (-8.0, 8.0), "pitch": (-8.0, 8.0)},
    "LEFT": {"yaw": (None, -18.0), "pitch": (-12.0, 12.0)},
    "RIGHT": {"yaw": (18.0, None), "pitch": (-12.0, 12.0)},
    "UP": {"yaw": (-12.0, 12.0), "pitch": (None, -10.0)},
    "DOWN": {"yaw": (-12.0, 12.0), "pitch": (10.0, None)},
    "UP-LEFT": {"yaw": (None, -12.0), "pitch": (None, -8.0)},
    "UP-RIGHT": {"yaw": (12.0, None), "pitch": (None, -8.0)},
    "DOWN-LEFT": {"yaw": (None, -12.0), "pitch": (8.0, None)},
    "DOWN-RIGHT": {"yaw": (12.0, None), "pitch": (8.0, None)},
}

FPS_TARGET = 30.0
WINDOW_NAME = "Face enrollment"
MAX_ATTEMPTS = 3
POSE_NAMES = (
    "CENTER",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "UP-LEFT",
    "UP-RIGHT",
    "DOWN-LEFT",
    "DOWN-RIGHT",
)
DIRECTION_GROUPS = (
    "RIGHT",
    "DOWN-RIGHT",
    "DOWN",
    "DOWN-LEFT",
    "LEFT",
    "UP-LEFT",
    "UP",
    "UP-RIGHT",
)
GROUP_ANGLES = {
    "RIGHT": 0.0,
    "DOWN-RIGHT": 45.0,
    "DOWN": 90.0,
    "DOWN-LEFT": 135.0,
    "LEFT": 180.0,
    "UP-LEFT": 225.0,
    "UP": 270.0,
    "UP-RIGHT": 315.0,
}
REJECTION_NAMES = (
    "no_face",
    "multi_face",
    "low_score",
    "too_small",
    "too_large",
    "off_center",
    "blurry",
    "moving",
    "duplicate",
    "bad_pose",
    "camera_hiccup",
)


def _within(value, bounds):
    low, high = bounds
    return (low is None or value >= low) and (high is None or value <= high)


def angle_delta_degrees(value, reference):
    """Return the signed shortest angular difference, avoiding +/-180 wrap jumps."""
    return (float(value) - float(reference) + 180.0) % 360.0 - 180.0


def median_pose_offset(samples):
    """Find a robust yaw/pitch neutral, unwrapping yaw around the first sample."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Neutral calibration needs finite yaw/pitch samples.")
    reference = float(values[0, 0])
    yaw_values = [reference + angle_delta_degrees(value, reference) for value in values[:, 0]]
    return (
        float(np.median(yaw_values)) % 360.0,
        float(np.median(values[:, 1])),
    )


def calibrated_pose(pose, neutral):
    return angle_delta_degrees(pose[0], neutral[0]), float(pose[1]) - float(neutral[1])


def classify_pose(yaw, pitch):
    """Return a required pose name only when both measured angles meet its bounds."""
    yaw, pitch = float(yaw), float(pitch)
    if not math.isfinite(yaw) or not math.isfinite(pitch):
        return None
    # Diagonals take precedence where their ranges overlap a cardinal range.
    for name in ("UP-LEFT", "UP-RIGHT", "DOWN-LEFT", "DOWN-RIGHT"):
        config = POSE_CONFIG[name]
        if _within(yaw, config["yaw"]) and _within(pitch, config["pitch"]):
            return name
    for name in ("CENTER", "LEFT", "RIGHT", "UP", "DOWN"):
        config = POSE_CONFIG[name]
        if _within(yaw, config["yaw"]) and _within(pitch, config["pitch"]):
            return name
    return None


def normalize_embedding(embedding):
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size != core.EMBEDDING_SIZE or not np.isfinite(vector).all():
        raise ValueError(f"Expected {core.EMBEDDING_SIZE} finite embedding values.")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Cannot normalize a zero-length embedding.")
    return vector / norm


class PoseSmoother:
    def __init__(self):
        self.samples = deque(maxlen=POSE_CONFIG["smoothing_frames"])

    def add(self, pose):
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.size != 3 or not np.isfinite(pose).all():
            raise ValueError("Pose must contain finite yaw, pitch, and roll.")
        self.samples.append(pose)
        return tuple(float(value) for value in np.median(np.asarray(self.samples), axis=0))


class PoseCoverage:
    """Keep up to three distinct embeddings for each measured pose."""

    def __init__(self):
        self.captures = {name: [] for name in POSE_NAMES}
        self.streak_pose = None
        self.streak_count = 0
        self.last_capture_at = {name: None for name in POSE_NAMES}
        self.last_rejection = None
        self.neutral_offset = None

    def count(self, pose):
        return len(self.captures[pose])

    def missing(self):
        return [
            name for name in POSE_NAMES
            if self.count(name) < POSE_CONFIG["embeddings_per_pose"]
        ]

    @property
    def complete(self):
        return not self.missing()

    def counts(self):
        return {name: self.count(name) for name in POSE_NAMES}

    def observe(self, pose_name, embedding, yaw, pitch, now, good=True):
        """Return whether an embedding was kept; pose must hold for 3 good frames."""
        self.last_rejection = None
        if not good or pose_name not in POSE_NAMES:
            self.streak_pose = None
            self.streak_count = 0
            return False
        if self.count(pose_name) >= POSE_CONFIG["embeddings_per_pose"]:
            self.streak_pose = pose_name
            self.streak_count = POSE_CONFIG["stable_good_frames"]
            return False
        if self.streak_pose != pose_name:
            self.streak_pose = pose_name
            self.streak_count = 1
            return False
        self.streak_count += 1
        if self.streak_count < POSE_CONFIG["stable_good_frames"]:
            return False
        last_at = self.last_capture_at[pose_name]
        if last_at is not None and now - last_at < POSE_CONFIG["minimum_interval_seconds"]:
            return False
        vector = normalize_embedding(embedding() if callable(embedding) else embedding)
        if any(
            core.cosine_sim(vector, item["embedding"])
            >= POSE_CONFIG["within_pose_max_cosine"]
            for item in self.captures[pose_name]
        ):
            self.last_rejection = "duplicate"
            return False
        self.captures[pose_name].append(
            {"embedding": vector, "yaw": float(yaw), "pitch": float(pitch), "time": float(now)}
        )
        self.last_capture_at[pose_name] = float(now)
        return True


class SelfTestCapture:
    """Wait for a centered pose, then collect spaced frames during the check window."""

    def __init__(self, started_at):
        self.started_at = float(started_at)
        self.centered_streak = 0
        self.checking_started_at = None
        self.last_capture_at = None
        self.embeddings = []

    @property
    def status(self):
        return "Checking..." if self.checking_started_at is not None else "Look straight at the camera"

    def finished_at(self, now):
        return (
            self.checking_started_at is not None
            and float(now) - self.checking_started_at >= POSE_CONFIG["self_test_seconds"]
        )

    def observe(self, centered, now, capture_embedding):
        now = float(now)
        if self.checking_started_at is None:
            self.centered_streak = self.centered_streak + 1 if centered else 0
            if self.centered_streak >= POSE_CONFIG["stable_good_frames"]:
                self.checking_started_at = now
            return self.status
        if (
            centered
            and len(self.embeddings) < POSE_CONFIG["self_test_frames"]
            and (
                self.last_capture_at is None
                or now - self.last_capture_at >= POSE_CONFIG["minimum_interval_seconds"]
            )
        ):
            self.embeddings.append(normalize_embedding(capture_embedding()))
            self.last_capture_at = now
        return self.status

    def timed_out_waiting(self, now):
        return (
            self.checking_started_at is None
            and float(now) - self.started_at >= POSE_CONFIG["self_test_center_wait_seconds"]
        )


def coverage_proof(coverage):
    """Return per-pose coverage and measured yaw/pitch spread validation."""
    rows = []
    all_yaw = []
    all_pitch = []
    counts_ok = True
    for name in POSE_NAMES:
        captures = coverage.captures[name]
        count = len(captures)
        counts_ok = counts_ok and count >= POSE_CONFIG["embeddings_per_pose"]
        if captures:
            yaws = [item["yaw"] for item in captures]
            pitches = [item["pitch"] for item in captures]
            yaw_range = (min(yaws), max(yaws))
            pitch_range = (min(pitches), max(pitches))
            all_yaw.extend(yaws)
            all_pitch.extend(pitches)
        else:
            yaw_range = (None, None)
            pitch_range = (None, None)
        rows.append((name, count, yaw_range, pitch_range))
    yaw_span = max(all_yaw) - min(all_yaw) if all_yaw else 0.0
    pitch_span = max(all_pitch) - min(all_pitch) if all_pitch else 0.0
    valid = (
        counts_ok
        and yaw_span >= POSE_CONFIG["minimum_yaw_span_degrees"]
        and pitch_span >= POSE_CONFIG["minimum_pitch_span_degrees"]
    )
    return valid, rows, yaw_span, pitch_span


def filter_and_validate(coverage):
    """Filter per-pose outliers and require global cohesion."""
    cleaned = {}
    for name in POSE_NAMES:
        captures = coverage.captures[name]
        vectors = [item["embedding"] for item in captures]
        keep = []
        for index, vector in enumerate(vectors):
            others = vectors[:index] + vectors[index + 1 :]
            if not others:
                continue
            mean = np.mean(others, axis=0)
            norm = float(np.linalg.norm(mean))
            if norm and core.cosine_sim(vector, mean / norm) >= POSE_CONFIG["consistency_min_cosine"]:
                keep.append(captures[index])
        if len(keep) < POSE_CONFIG["minimum_pose_embeddings_after_filter"]:
            return None, None, f"{name} has fewer than two consistent embeddings"
        cleaned[name] = keep

    vectors = [item["embedding"] for name in POSE_NAMES for item in cleaned[name]]
    scores = [
        core.cosine_sim(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    cohesion = float(np.mean(scores)) if scores else 0.0
    if cohesion < POSE_CONFIG["cohesion_min_cosine"]:
        return None, cohesion, "Face data is inconsistent, check lighting and make sure only you are in frame"
    return cleaned, cohesion, None


def self_test_score(probe, enrolled):
    normalized = normalize_embedding(probe)
    normalized_enrollments = [normalize_embedding(value) for value in enrolled]
    return max(float(np.dot(normalized, value)) for value in normalized_enrollments)


def run_attempts(attempt_runner, on_retry=None, max_attempts=MAX_ATTEMPTS):
    """Run until success/cancel or the maximum number of failed attempts."""
    for number in range(1, max_attempts + 1):
        result = attempt_runner(number)
        if result["kind"] in {"success", "cancel"}:
            return result, number
        if number < max_attempts and on_retry is not None:
            should_retry = on_retry(result, number + 1, max_attempts)
            if not should_retry:
                return {"kind": "cancel", "reason": "cancelled before retry"}, number
    return result, max_attempts


def _new_counters():
    return {key: 0 for key in ("frames_seen", "faces_detected", *REJECTION_NAMES)}


def _print_debug(counters, coverage, pose, current_pose):
    counts = ", ".join(
        f"{name} {coverage.count(name)}/{POSE_CONFIG['embeddings_per_pose']}"
        for name in POSE_NAMES
    )
    rejected = " ".join(f"{name}={counters[name]}" for name in REJECTION_NAMES)
    yaw_pitch = "n/a" if pose is None else f"{pose[0]:.1f}/{pose[1]:.1f}"
    print(
        f"[debug] yaw/pitch={yaw_pitch} pose={current_pose or 'none'} "
        f"frames={counters['frames_seen']} faces={counters['faces_detected']} "
        f"{counts} {rejected}"
    )


def _print_coverage(coverage, proof):
    _, rows, yaw_span, pitch_span = proof
    print("Pose coverage (kept / required; calibrated yaw and pitch ranges):")
    for name, count, yaw_range, pitch_range in rows:
        if count:
            ranges = (
                f"yaw={yaw_range[0]:.1f}..{yaw_range[1]:.1f}, "
                f"pitch={pitch_range[0]:.1f}..{pitch_range[1]:.1f}"
            )
        else:
            ranges = "yaw=n/a, pitch=n/a"
        print(
            f"  {name:10s} {count}/{POSE_CONFIG['embeddings_per_pose']}  {ranges}"
        )
    print(f"  overall   yaw span={yaw_span:.1f} deg, pitch span={pitch_span:.1f} deg")


def _pose_prompt(pose_name):
    return {
        "CENTER": "Look straight at the camera",
        "LEFT": "Slowly turn your head to the left",
        "RIGHT": "Slowly turn your head to the right",
        "UP": "Look up",
        "DOWN": "Look down",
        "UP-LEFT": "Look up and to the left",
        "UP-RIGHT": "Look up and to the right",
        "DOWN-LEFT": "Look down and to the left",
        "DOWN-RIGHT": "Look down and to the right",
    }[pose_name]


def _next_pose(coverage, measured_pose):
    if coverage.count("CENTER") < POSE_CONFIG["embeddings_per_pose"]:
        return "CENTER"
    if (
        measured_pose in POSE_NAMES
        and coverage.count(measured_pose) < POSE_CONFIG["embeddings_per_pose"]
    ):
        return measured_pose
    return next(
        (
            name for name in POSE_NAMES
            if coverage.count(name) < POSE_CONFIG["embeddings_per_pose"]
        ),
        "CENTER",
    )


def _nearest_group(coverage, yaw, pitch):
    remaining = [
        group for group in DIRECTION_GROUPS
        if coverage.count(group) < POSE_CONFIG["embeddings_per_pose"]
    ]
    if not remaining:
        return None
    angle = math.degrees(math.atan2(pitch, yaw)) % 360.0
    return min(
        remaining,
        key=lambda group: abs((angle - GROUP_ANGLES[group] + 180.0) % 360.0 - 180.0),
    )


def _draw_enrollment_ring(canvas, coverage, pulse, target_group, center_indicator=True):
    cx, cy = core.CAMERA_CENTER
    for tick in range(POSE_CONFIG["direction_tick_count"]):
        group_index = tick // POSE_CONFIG["ticks_per_direction_group"]
        group = DIRECTION_GROUPS[group_index]
        angle = math.radians(tick * 360.0 / POSE_CONFIG["direction_tick_count"])
        start = (cx + math.cos(angle) * 202, cy + math.sin(angle) * 202)
        end = (cx + math.cos(angle) * 224, cy + math.sin(angle) * 224)
        capture_count = coverage.count(group)
        within_group_tick = tick % POSE_CONFIG["ticks_per_direction_group"]
        filled = within_group_tick < round(
            capture_count
            * POSE_CONFIG["ticks_per_direction_group"]
            / POSE_CONFIG["embeddings_per_pose"]
        )
        color = (90, 220, 90) if filled else (68, 68, 68)
        if group == target_group and not filled:
            glow = 0.25 + 0.35 * (math.sin(pulse * math.tau) + 1.0) / 2.0
            color = tuple(int(base * (1 - glow) + 185 * glow) for base in color)
        cv2.line(
            canvas,
            tuple(int(round(value)) for value in start),
            tuple(int(round(value)) for value in end),
            color,
            7,
            cv2.LINE_AA,
        )
        for point in (start, end):
            cv2.circle(
                canvas,
                tuple(int(round(value)) for value in point),
                3,
                color,
                -1,
                cv2.LINE_AA,
            )

    if center_indicator:
        center_done = coverage.count("CENTER") >= POSE_CONFIG["embeddings_per_pose"]
        center_color = (90, 220, 90) if center_done else (125, 125, 125)
        cv2.circle(canvas, (cx, cy), 13, center_color, 2, cv2.LINE_AA)
        if center_done:
            cv2.circle(canvas, (cx, cy), 6, center_color, -1, cv2.LINE_AA)


def _draw_checklist(canvas, coverage):
    x_positions = (28, 374)
    y_start = 842
    for index, name in enumerate(POSE_NAMES):
        column = 0 if index < 5 else 1
        row = index if index < 5 else index - 5
        x = x_positions[column]
        y = y_start + row * 15
        done = coverage.count(name) >= POSE_CONFIG["embeddings_per_pose"]
        color = (90, 220, 90) if done else (130, 130, 130)
        mark = "[x]" if done else "[ ]"
        cv2.putText(
            canvas,
            f"{mark} {name} {coverage.count(name)}/{POSE_CONFIG['embeddings_per_pose']}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
        )


def _draw_frame(frame, status, coverage, pose, current_pose, now, status_changed_at):
    direction = None if pose is None else (-pose[0] / 45.0, pose[1] / 35.0)
    target_group = (
        _nearest_group(coverage, pose[0], pose[1]) if pose is not None else None
    )
    canvas = core.render_ui(
        frame,
        status,
        direction=direction,
        title="Face enrollment",
        status_alpha=min(1.0, max(0.0, (now - status_changed_at) / 0.18)),
        draw_ring=False,
    )
    _draw_enrollment_ring(canvas, coverage, now, target_group)
    _draw_checklist(canvas, coverage)
    cv2.imshow(WINDOW_NAME, canvas)


def _pose_spread_sufficient(coverage):
    proof = coverage_proof(coverage)
    return proof[0], proof


def _metadata(name, cleaned, proof):
    _, rows, yaw_span, pitch_span = proof
    pose_metadata = {}
    for pose in POSE_NAMES:
        items = cleaned[pose]
        yaws = [item["yaw"] for item in items]
        pitches = [item["pitch"] for item in items]
        pose_metadata[pose] = {
            "embedding_count": len(items),
            "yaw_range_degrees": [float(min(yaws)), float(max(yaws))],
            "pitch_range_degrees": [float(min(pitches)), float(max(pitches))],
        }
    return {
        "name": name,
        "pose_counts": {pose: len(cleaned[pose]) for pose in POSE_NAMES},
        "pose_ranges": pose_metadata,
        "overall_yaw_span_degrees": float(yaw_span),
        "overall_pitch_span_degrees": float(pitch_span),
        "pose_coverage": {row[0]: int(row[1]) for row in rows},
    }


def _success_animation(camera, coverage, deadline):
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= POSE_CONFIG["success_animation_seconds"]:
            return True
        if time.monotonic() >= deadline:
            return False
        ok, frame = camera.read()
        if not ok:
            frame = None
        phase = min(1.0, elapsed / POSE_CONFIG["success_animation_seconds"])
        canvas = core.render_ui(
            frame,
            "Face scan complete",
            title="Face enrollment",
            success_progress=phase,
            camera_dim=0.72,
            draw_ring=False,
        )
        _draw_enrollment_ring(canvas, coverage, phase, None, center_indicator=False)
        _draw_checklist(canvas, coverage)
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False


def save_verified_enrollment(name, cleaned, proof, before_commit=None):
    core.validate_person_name(name)
    embeddings = np.asarray(
        [item["embedding"] for pose in POSE_NAMES for item in cleaned[pose]],
        dtype=np.float32,
    )
    embeddings = np.asarray([normalize_embedding(item) for item in embeddings])
    core.DATA_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = core.DATA_DIR / f"{name}.npy"
    meta_path = core.DATA_DIR / f"{name}_meta.json"
    npy_temp = None
    meta_temp = None
    backups = {}
    installed = []
    committed = False
    try:
        with tempfile.NamedTemporaryFile(dir=core.DATA_DIR, suffix=".npy.tmp", delete=False) as handle:
            npy_temp = Path(handle.name)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=core.DATA_DIR, suffix=".json.tmp", delete=False
        ) as handle:
            meta_temp = Path(handle.name)
            json.dump(_metadata(name, cleaned, proof), handle, indent=2)
            handle.write("\n")
        with npy_temp.open("wb") as handle:
            np.save(handle, embeddings)
        if before_commit is not None and not before_commit():
            return False
        for destination in (npy_path, meta_path):
            if destination.exists():
                with tempfile.NamedTemporaryFile(
                    dir=core.DATA_DIR, suffix=".backup.tmp", delete=False
                ) as handle:
                    backup = Path(handle.name)
                backup.unlink()
                os.replace(destination, backup)
                backups[destination] = backup
        os.replace(npy_temp, npy_path)
        npy_temp = None
        installed.append(npy_path)
        os.replace(meta_temp, meta_path)
        meta_temp = None
        installed.append(meta_path)
        committed = True
    except OSError:
        for destination in installed:
            if destination.exists():
                destination.unlink()
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for temp_path in (npy_temp, meta_temp):
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        if committed:
            for backup in backups.values():
                if backup.exists():
                    backup.unlink()
    print(f"Saved {len(embeddings)} verified embeddings and pose metadata for {name!r}.")
    return True


def _collect_self_test(camera, gate, counters, neutral, cleaned, debug, coverage, deadline):
    capture = SelfTestCapture(time.monotonic())
    smoother = PoseSmoother()
    last_pose = None
    last_frame = None
    last_debug = time.monotonic()
    while time.monotonic() < deadline:
        now = time.monotonic()
        counters["frames_seen"] += 1
        ok, frame = camera.read()
        if not ok or frame is None:
            counters["camera_hiccup"] += 1
            capture.observe(False, now, lambda: None)
            cv2.waitKey(1)
            continue
        last_frame = frame
        current_pose = None
        current_measured_pose = None
        faces = core.detect_faces(frame)
        counters["faces_detected"] += len(faces)
        face, reason, _, raw_pose = gate.evaluate_detail(frame, faces)
        if reason:
            counters[reason] += 1
            capture.observe(False, now, lambda: None)
        else:
            try:
                last_pose = smoother.add(raw_pose)
                yaw, pitch = calibrated_pose(last_pose, neutral)
                capture.observe(
                    classify_pose(yaw, pitch) == "CENTER",
                    now,
                    lambda: core.get_embedding(frame, face),
                )
            except (ValueError, cv2.error):
                counters["bad_pose"] += 1
                capture.observe(False, now, lambda: None)

        if capture.timed_out_waiting(now):
            return "failure", None
        canvas = core.render_ui(
            last_frame, capture.status, title="Face enrollment", draw_ring=False
        )
        _draw_enrollment_ring(canvas, coverage, now, None)
        _draw_checklist(canvas, coverage)
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return "cancel", None
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return "cancel", None
        if debug and now - last_debug >= 1.0:
            relative = None if last_pose is None else (
                *calibrated_pose(last_pose, neutral), last_pose[2]
            )
            _print_debug(counters, coverage, relative, classify_pose(relative[0], relative[1]) if relative else None)
            last_debug = now
        if capture.finished_at(now):
            break

    if len(capture.embeddings) < POSE_CONFIG["self_test_frames"]:
        return "failure", None
    probe = normalize_embedding(
        np.mean(capture.embeddings[: POSE_CONFIG["self_test_frames"]], axis=0)
    )
    enrolled = [item["embedding"] for pose in POSE_NAMES for item in cleaned[pose]]
    score = self_test_score(probe, enrolled)
    return ("success", score) if score >= POSE_CONFIG["self_test_min_cosine"] else ("failure", score)


def _retry_screen(reason, coverage, number, max_attempts):
    missing = coverage.missing()
    summary = ", ".join(missing) if missing else "none"
    first = f"Missing: {summary} ({number} of {max_attempts})"
    second = reason[:75]
    start = time.monotonic()
    while time.monotonic() - start < 2.0:
        canvas = core.render_ui(None, first, title="Face enrollment", draw_ring=False)
        cv2.putText(
            canvas, second,
            (30, 790), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA,
        )
        _draw_enrollment_ring(canvas, coverage, time.monotonic(), None)
        _draw_checklist(canvas, coverage)
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return False
        time.sleep(0.01)
    return True


def _attempt_enrollment(attempt_number, name, coverage, debug):
    counters = _new_counters()
    camera = None
    reason = "unknown enrollment failure"
    final_verification = "not completed"
    coverage.streak_pose = None
    coverage.streak_count = 0
    try:
        camera = core.open_camera()
        gate = core.QualityGate()
        smoother = PoseSmoother()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        attempt_started = time.monotonic()
        last_debug = attempt_started
        last_frame = None
        neutral_samples = []
        calibration_started = None
        neutral = coverage.neutral_offset
        current_pose = None
        current_measured_pose = None
        status = "Look straight at the camera"
        status_changed_at = attempt_started
        previous_status = status
        last_capture_progress = attempt_started
        print(
            f"Attempt {attempt_number}/{MAX_ATTEMPTS}: capture 3 distinct frames at each "
            "of the 9 measured poses."
        )
        attempt_deadline = attempt_started + POSE_CONFIG["attempt_timeout_seconds"]
        capture_deadline = attempt_deadline - (
            POSE_CONFIG["self_test_center_wait_seconds"]
            + POSE_CONFIG["self_test_seconds"]
            + POSE_CONFIG["success_animation_seconds"]
            + 0.25
        )

        while (
            not coverage.complete
            and time.monotonic() < capture_deadline
        ):
            loop_time = time.monotonic()
            counters["frames_seen"] += 1
            ok, frame = camera.read()
            if not ok or frame is None:
                counters["camera_hiccup"] += 1
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return {"kind": "cancel", "reason": "cancelled", "counters": counters}
                if last_frame is not None:
                    _draw_frame(
                        last_frame, "Hold still", coverage, current_measured_pose,
                        current_pose, loop_time, status_changed_at,
                    )
                time.sleep(0.005)
                continue
            last_frame = frame
            faces = core.detect_faces(frame)
            counters["faces_detected"] += len(faces)
            face, rejection, _, raw_pose = gate.evaluate_detail(frame, faces)
            pose_for_display = None
            if rejection:
                counters[rejection] += 1
                status = core.QualityGate.STATUS[rejection]
                coverage.observe(None, None, 0, 0, loop_time, good=False)
                if neutral is None:
                    calibration_started = None
                    neutral_samples.clear()
                    smoother.samples.clear()
            else:
                try:
                    pose_for_display = smoother.add(raw_pose)
                    if calibration_started is None:
                        calibration_started = loop_time
                    calibration_elapsed = loop_time - calibration_started
                    if calibration_elapsed < POSE_CONFIG["calibration_seconds"]:
                        neutral_samples.append(pose_for_display[:2])
                        status = "Look straight at the camera"
                    else:
                        if neutral is None:
                            if not neutral_samples:
                                raise ValueError("No valid pose samples for neutral calibration.")
                            neutral = median_pose_offset(neutral_samples)
                            coverage.neutral_offset = neutral
                        yaw, pitch = calibrated_pose(pose_for_display, neutral)
                        current_measured_pose = (yaw, pitch, pose_for_display[2])
                        current_pose = classify_pose(yaw, pitch)
                        if (
                            current_pose is None
                            or coverage.count(current_pose) >= POSE_CONFIG["embeddings_per_pose"]
                        ):
                            coverage.observe(None, None, yaw, pitch, loop_time, good=False)
                            counters["bad_pose"] += 1 if current_pose is None else 0
                        else:
                            kept = coverage.observe(
                                current_pose,
                                lambda: core.get_embedding(frame, face),
                                yaw,
                                pitch,
                                loop_time,
                                good=True,
                            )
                            if kept:
                                last_capture_progress = loop_time
                            elif coverage.last_rejection == "duplicate":
                                counters["duplicate"] += 1

                        target = _next_pose(coverage, current_pose)
                        if (
                            loop_time - last_capture_progress
                            >= POSE_CONFIG["no_progress_guidance_seconds"]
                        ):
                            status = "Turn a little further"
                        elif current_pose == target and target is not None:
                            status = _pose_prompt(target)
                        else:
                            status = _pose_prompt(target)
                except (ValueError, cv2.error):
                    counters["bad_pose"] += 1
                    coverage.observe(None, None, 0, 0, loop_time, good=False)
                    status = "Position your face in the circle"

            if status != previous_status:
                previous_status = status
                status_changed_at = loop_time
            _draw_frame(
                frame, status, coverage, current_measured_pose, current_pose,
                loop_time, status_changed_at,
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return {"kind": "cancel", "reason": "cancelled", "counters": counters}
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return {"kind": "cancel", "reason": "window closed", "counters": counters}
            if debug and loop_time - last_debug >= 1.0:
                _print_debug(counters, coverage, current_measured_pose, current_pose)
                last_debug = loop_time
            elapsed = time.monotonic() - loop_time
            if elapsed < 1.0 / FPS_TARGET:
                time.sleep(1.0 / FPS_TARGET - elapsed)
            if coverage.complete:
                break

        proof_ok, proof = _pose_spread_sufficient(coverage)
        missing = coverage.missing()
        if missing:
            reason = "timeout; missing poses: " + ", ".join(missing)
            final_verification = f"FAILED: {reason}"
            return {
                "kind": "failure", "reason": reason, "counters": counters,
                "missing": missing, "coverage_proof": proof,
            }
        if not proof_ok:
            reason = (
                "measured pose spread is insufficient "
                f"(yaw {proof[2]:.1f}/{POSE_CONFIG['minimum_yaw_span_degrees']:.0f} deg, "
                f"pitch {proof[3]:.1f}/{POSE_CONFIG['minimum_pitch_span_degrees']:.0f} deg)"
            )
            final_verification = f"FAILED: {reason}"
            return {
                "kind": "failure", "reason": reason, "counters": counters,
                "missing": [], "coverage_proof": proof,
            }

        cleaned, cohesion, validation_error = filter_and_validate(coverage)
        if validation_error:
            reason = validation_error
            print(f"Consistency/cohesion: FAILED - {reason}")
            final_verification = f"FAILED: {reason}"
            return {
                "kind": "failure", "reason": reason, "counters": counters,
                "missing": [], "coverage_proof": proof,
            }
        print(
            f"Consistency: PASS (>=2 per pose); cohesion: PASS "
            f"(mean pairwise cosine {cohesion:.3f})."
        )
        final_verification = f"consistency PASS; cohesion PASS ({cohesion:.3f})"
        neutral_gate = core.QualityGate()
        self_test, score = _collect_self_test(
            camera,
            neutral_gate,
            counters,
            neutral,
            cleaned,
            debug,
            coverage,
            attempt_deadline - POSE_CONFIG["success_animation_seconds"],
        )
        if self_test == "cancel":
            return {"kind": "cancel", "reason": "cancelled during self-test", "counters": counters}
        if self_test != "success":
            reason = (
                "live self-test did not collect five good centered frames"
                if score is None
                else f"live self-test cosine {score:.3f} is below 0.500"
            )
            print(f"Live self-test: FAILED - {reason}")
            final_verification += f"; live self-test FAILED: {reason}"
            return {
                "kind": "failure", "reason": reason, "counters": counters,
                "missing": [], "coverage_proof": proof,
            }
        print(f"Live self-test: PASS (best cosine {score:.3f}).")
        final_verification += f"; live self-test PASS ({score:.3f})"
        if (
            time.monotonic() + POSE_CONFIG["success_animation_seconds"]
            > attempt_deadline
        ):
            reason = "40-second attempt cap reached before the success animation"
            final_verification += "; success animation FAILED: attempt timeout"
            return {
                "kind": "failure", "reason": reason, "counters": counters,
                "missing": [], "coverage_proof": proof,
            }
        # Success animation is reachable only after coverage and all checks pass.
        if not save_verified_enrollment(
            name,
            cleaned,
            proof,
            before_commit=lambda: _success_animation(
                camera, coverage, attempt_deadline
            ),
        ):
            if time.monotonic() >= attempt_deadline:
                reason = "40-second attempt cap reached before completion"
                final_verification += "; save FAILED: attempt timeout"
                return {
                    "kind": "failure", "reason": reason, "counters": counters,
                    "missing": [], "coverage_proof": proof,
                }
            return {"kind": "cancel", "reason": "cancelled during success animation", "counters": counters}
        final_verification += "; metadata and embeddings saved"
        return {
            "kind": "success", "reason": None, "counters": counters,
            "missing": [], "coverage_proof": proof,
        }
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        reason = str(error)
        final_verification = f"FAILED: {reason}"
        counters["camera_hiccup"] += 1
        return {
            "kind": "failure", "reason": reason, "counters": counters,
            "missing": coverage.missing(),
        }
    finally:
        _print_coverage(coverage, coverage_proof(coverage))
        print(f"Final verification: {final_verification}")
        counts = ", ".join(
            f"{pose}={coverage.count(pose)}/{POSE_CONFIG['embeddings_per_pose']}"
            for pose in POSE_NAMES
        )
        rejected = " ".join(f"{key}={value}" for key, value in counters.items())
        print(
            f"Attempt {attempt_number}/{MAX_ATTEMPTS} summary: {counts}; "
            f"frames_seen={counters['frames_seen']} faces_detected={counters['faces_detected']} "
            f"{rejected}"
        )
        if camera is not None:
            camera.release()


def confirm_overwrite(name):
    if not (core.DATA_DIR / f"{name}.npy").exists():
        return True
    answer = input(f"Enrollment for {name!r} exists. Overwrite it? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Name for this enrollment")
    parser.add_argument(
        "--image", type=Path,
        help="Not supported for complete enrollment: all nine poses must be measured live",
    )
    parser.add_argument("--debug", action="store_true", help="Print pose and rejection diagnostics")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        core.validate_person_name(args.name)
        if not confirm_overwrite(args.name):
            print("Enrollment cancelled; existing enrollment was not changed.")
            return 0
        if args.image:
            raise ValueError(
                "A still image cannot prove coverage of all nine poses. Use webcam enrollment."
            )
        core.load_models()
        coverage = PoseCoverage()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        def runner(number):
            return _attempt_enrollment(number, args.name, coverage, args.debug)

        def retry(result, next_number, limit):
            reason = result["reason"]
            print(f"Attempt failed: {reason}. Missing poses: {', '.join(coverage.missing()) or 'none'}.")
            return _retry_screen(reason, coverage, next_number, limit)

        result, attempts = run_attempts(runner, retry)
        if result["kind"] == "success":
            print(f"Enrollment complete after {attempts} attempt(s); all nine poses verified.")
            return 0
        if result["kind"] == "cancel":
            print(f"Enrollment cancelled: {result['reason']}. No new enrollment was saved.")
            return 1
        missing = coverage.missing()
        causes = {
            "no_face": "move into the camera view",
            "multi_face": "make sure only you are in frame",
            "low_score": "improve lighting",
            "too_small": "move closer",
            "too_large": "move farther away",
            "off_center": "center your face",
            "blurry": "improve lighting and hold steady",
            "moving": "turn more slowly",
            "bad_pose": "hold each requested angle steady so solvePnP can measure it",
        }
        counters = result.get("counters", {})
        if missing:
            likely_cause = "turn to the listed missing poses and hold each angle steady"
        elif any(counters.get(key, 0) for key in causes):
            likely = max(causes, key=lambda key: counters.get(key, 0))
            likely_cause = causes[likely]
        else:
            likely_cause = result["reason"]
        print(
            f"Enrollment failed after {attempts} attempts. "
            f"Missing poses: {', '.join(missing) or 'none'}. "
            f"Likely cause: {likely_cause}. No new enrollment was saved.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nEnrollment cancelled by Ctrl+C. No new enrollment was saved.")
        return 1
    except (EOFError, FileNotFoundError, RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f"Enrollment error: {error}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
