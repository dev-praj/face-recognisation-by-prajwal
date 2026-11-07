#!/usr/bin/env python3
"""Collect genuine/impostor SFace scores and recommend a verification threshold."""

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


SCAN_FRAME_COUNT = face_match.REQUIRED_FRAMES
SCAN_TIMEOUT_SECONDS = 5.0
SAMPLE_INTERVAL_SECONDS = 0.08
SCANS_PER_PERSON = 10
WINDOW_NAME = "Face calibration"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("genuine", "impostor"), required=True)
    parser.add_argument("--name", help="Enrolled identity for genuine calibration")
    parser.add_argument(
        "--people",
        type=int,
        default=2,
        help="Number of distinct impostor volunteers to scan (default: 2)",
    )
    parser.add_argument(
        "--impostor-images",
        type=Path,
        help="Use images in this folder instead of an impostor webcam session",
    )
    return parser.parse_args()


def _read_calibration(path):
    if not path.exists():
        return {"version": 1, "genuine_scores": {}, "impostor_scores": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read calibration file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Calibration file must contain a JSON object: {path}")
    genuine = value.get("genuine_scores", {})
    impostor = value.get("impostor_scores", [])
    if not isinstance(genuine, dict) or not isinstance(impostor, list):
        raise ValueError(f"Calibration scores have an invalid format: {path}")
    return {"version": 1, "genuine_scores": genuine, "impostor_scores": impostor}


def _save_calibration(path, genuine_scores, impostor_scores):
    genuine = [float(score) for scores in genuine_scores.values() for score in scores]
    impostor = [float(score) for score in impostor_scores]
    threshold, overlaps = face_match.recommend_threshold(genuine, impostor)
    result = {
        "version": 1,
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "recommended_threshold": threshold,
        "distributions_overlap": overlaps if threshold is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _print_distribution(label, values):
    if not values:
        print(f"{label}: no scores collected")
        return
    array = np.asarray(values, dtype=np.float64)
    print(
        f"{label}: n={len(array)} min={array.min():.4f} "
        f"mean={array.mean():.4f} max={array.max():.4f}"
    )
    edges = np.linspace(-1.0, 1.0, 11)
    counts, _ = np.histogram(array, bins=edges)
    print("  Histogram (cosine bins from -1 to 1):")
    for index, count in enumerate(counts):
        print(f"    {edges[index]: .1f} to {edges[index + 1]: .1f}: {'#' * int(count)} ({count})")


def _print_report(calibration):
    genuine = [
        float(score)
        for scores in calibration["genuine_scores"].values()
        for score in scores
    ]
    impostor = [float(score) for score in calibration["impostor_scores"]]
    _print_distribution("Genuine", genuine)
    _print_distribution("Impostor", impostor)
    threshold = calibration["recommended_threshold"]
    if threshold is None:
        print("Recommended threshold unavailable: collect both genuine and impostor scores.")
    elif calibration["distributions_overlap"]:
        print(
            "This model cannot reliably separate these people on this camera; "
            "improve lighting/camera placement or use a stronger model."
        )
        print(
            f"Fail-closed threshold recommendation: {threshold:.4f} "
            "(it may reject genuine scans)."
        )
    else:
        print(f"Recommended threshold: {threshold:.4f}")


def _press_enter(prompt):
    input(prompt + " Press Enter to start, or Ctrl+C to cancel. ")


def _collect_webcam_scan(camera, enrollments):
    gate = core.QualityGate()
    scores = []
    start = time.monotonic()
    next_sample_at = start
    last_frame = None
    while time.monotonic() - start < SCAN_TIMEOUT_SECONDS and len(scores) < SCAN_FRAME_COUNT:
        loop_started = time.monotonic()
        ok, frame = camera.read()
        if not ok or frame is None:
            status, face, pose = "Position your face in the circle", None, None
        else:
            last_frame = frame
            faces = core.detect_faces(frame)
            status, face, _, pose = gate.evaluate(frame, faces)
            if not status and face is not None and loop_started >= next_sample_at:
                # The exact enrollment/verification embedding path is shared in face_core.
                embedding = core.get_embedding(frame, face)
                scores.append(face_match.score_embedding(embedding, enrollments))
                next_sample_at = loop_started + SAMPLE_INTERVAL_SECONDS
                status = f"Scanning {len(scores)}/{SCAN_FRAME_COUNT}"
        canvas = core.render_ui(
            last_frame,
            status or f"Scanning {len(scores)}/{SCAN_FRAME_COUNT}",
            direction=pose if face is not None else None,
            title="Face calibration",
        )
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            raise KeyboardInterrupt
        time.sleep(max(0.0, 1 / 30 - (time.monotonic() - loop_started)))
    if len(scores) != SCAN_FRAME_COUNT:
        raise RuntimeError(
            f"Could not collect {SCAN_FRAME_COUNT} good frames within "
            f"{SCAN_TIMEOUT_SECONDS:.0f} seconds ({len(scores)} collected)."
        )
    return scores


def _scan_images(folder, enrollments):
    if not folder.is_dir():
        raise ValueError(f"Impostor image folder does not exist: {folder}")
    images = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No supported images found in: {folder}")
    scores = []
    core.load_models()
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Skipping unreadable image: {path}", file=sys.stderr)
            continue
        faces = core.detect_faces(frame)
        if len(faces) != 1:
            print(f"Skipping {path.name}: expected one face, found {len(faces)}.")
            continue
        gate = core.QualityGate()
        status, face, _, _ = gate.evaluate(frame, faces)
        if status or face is None:
            print(f"Skipping {path.name}: failed quality gate ({status}).")
            continue
        embedding = core.get_embedding(frame, face)
        values = face_match.score_embedding(embedding, enrollments)
        score = max(values.values())
        scores.append(score)
        print(f"Impostor image {path.name}: max enrolled-person score={score:.6f}")
    if len(scores) < SCANS_PER_PERSON:
        raise RuntimeError(
            f"Need at least {SCANS_PER_PERSON} usable impostor images; got {len(scores)}."
        )
    return scores


def main():
    args = parse_args()
    camera = None
    try:
        enrollments = face_match.normalize_enrollments(core.load_enrollments())
        if args.mode == "genuine" and args.name not in enrollments:
            raise ValueError("--name must match an enrolled identity for genuine calibration.")
        if args.mode == "impostor" and args.people < 2 and args.impostor_images is None:
            raise ValueError("Impostor webcam calibration needs at least two people.")
        if args.mode == "genuine" and args.impostor_images is not None:
            raise ValueError("--impostor-images is only valid with --mode impostor.")

        data_dir = Path(core.DATA_DIR)
        calibration_path = data_dir / "calibration.json"
        calibration = _read_calibration(calibration_path)
        core.load_models()

        if args.mode == "impostor" and args.impostor_images is not None:
            new_impostor_scores = _scan_images(args.impostor_images, enrollments)
            calibration["impostor_scores"].extend(new_impostor_scores)
        else:
            camera = core.open_camera()
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            if args.mode == "genuine":
                person_names = [args.name]
            else:
                person_names = [f"impostor-{index + 1}" for index in range(args.people)]
            for person_name in person_names:
                collected = []
                for scan_number in range(1, SCANS_PER_PERSON + 1):
                    if args.mode == "genuine":
                        prompt = (
                            f"Genuine scan {scan_number}/{SCANS_PER_PERSON}; "
                            "vary lighting and small head angles."
                        )
                    else:
                        prompt = (
                            f"{person_name}: impostor scan {scan_number}/{SCANS_PER_PERSON}; "
                            "use a person who is not enrolled."
                        )
                    _press_enter(prompt)
                    frame_scores = _collect_webcam_scan(camera, enrollments)
                    target = args.name if args.mode == "genuine" else None
                    score = face_match.scan_score(frame_scores, target_name=target)
                    collected.append(score)
                    print(f"Scan score: {score:.6f}")
                if args.mode == "genuine":
                    calibration["genuine_scores"].setdefault(args.name, []).extend(collected)
                else:
                    calibration["impostor_scores"].extend(collected)

        saved = _save_calibration(
            calibration_path,
            calibration["genuine_scores"],
            calibration["impostor_scores"],
        )
        _print_report(saved)
        print(f"Scores only saved to {calibration_path}")
    except KeyboardInterrupt:
        print("\nCalibration cancelled; incomplete current batch was not saved.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f"Calibration error: {error}", file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
