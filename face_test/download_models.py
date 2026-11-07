#!/usr/bin/env python3
"""Download the official OpenCV Zoo YuNet and SFace ONNX models."""

import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
        "face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/"
        "face_recognition_sface_2021dec.onnx"
    ),
}


def describe_file(path: Path) -> None:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    print(f"{path.name}: SHA256={digest.hexdigest()}  size={size} bytes")


def download_model(filename: str, url: str, destination: Path) -> bool:
    if destination.is_file() and destination.stat().st_size:
        print(f"Already present: {destination}")
        describe_file(destination)
        return True

    partial_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "face-test-model-downloader"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with partial_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        if not partial_path.stat().st_size:
            raise OSError("the server returned an empty file")
        partial_path.replace(destination)
    except (OSError, urllib.error.URLError) as error:
        partial_path.unlink(missing_ok=True)
        print(f"Download failed for {filename}: {error}", file=sys.stderr)
        print(f"Exact URL: {url}", file=sys.stderr)
        print(f"Download it manually and place it at: {destination}", file=sys.stderr)
        return False

    describe_file(destination)
    return True


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for filename, url in MODELS.items():
        if not download_model(filename, url, MODEL_DIR / filename):
            failed.append(filename)
    if failed:
        print("One or more models could not be downloaded.", file=sys.stderr)
        return 1
    print(f"Models are ready in: {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
