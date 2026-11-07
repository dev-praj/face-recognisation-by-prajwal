"""YuNet/SFace helpers, timed enrollment state, camera handling, and OpenCV UI."""

import math
import platform
import re
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
EMBEDDING_SIZE = 128
MAX_ENROLLMENT_EMBEDDINGS = 27
TICK_COUNT = 60
SECTOR_COUNT = 12
CANVAS_WIDTH = 720
CANVAS_HEIGHT = 960
CAMERA_CENTER = (CANVAS_WIDTH // 2, 432)
CAMERA_RADIUS = 175
RING_INNER_RADIUS = 202
RING_OUTER_RADIUS = 224
DETECTION_MAX_DIMENSION = 960
YUNET_FACE_MODEL_3D = np.asarray(
    [
        (-32.0, 30.0, 0.0),
        (32.0, 30.0, 0.0),
        (0.0, 0.0, -30.0),
        (-25.0, -25.0, -5.0),
        (25.0, -25.0, -5.0),
    ],
    dtype=np.float64,
)
_detector = None
_recognizer = None


def load_models():
    """Load YuNet and SFace once, reporting missing model files clearly."""
    global _detector, _recognizer
    missing = [path for path in (YUNET_PATH, SFACE_PATH) if not path.is_file()]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing model file(s):\n{paths}\nRun: python download_models.py")
    _detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.5, 0.3, 5000)
    _recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    return _detector, _recognizer


def detect_faces(frame):
    """Detect on a uniformly downscaled copy and return rows in frame coordinates."""
    if _detector is None:
        raise RuntimeError("Models are not loaded. Call load_models() first.")
    if frame is None or frame.ndim < 2 or frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError("Cannot detect faces in an empty frame.")
    height, width = frame.shape[:2]
    scale = min(1.0, DETECTION_MAX_DIMENSION / max(height, width))
    if scale < 1:
        detection_frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_frame = frame
    detection_height, detection_width = detection_frame.shape[:2]
    _detector.setInputSize((detection_width, detection_height))
    _, rows = _detector.detect(detection_frame)
    if rows is None:
        return []
    return [row for row in map_face_rows(rows, (detection_width, detection_height), (width, height))]


def map_face_rows(rows, source_size, target_size):
    """Map YuNet bboxes and all five landmarks into target-frame coordinates."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    mapped = np.asarray(rows, dtype=np.float32).copy()
    if mapped.ndim == 1:
        mapped = mapped[np.newaxis, :]
    if mapped.ndim != 2 or mapped.shape[1] < 15 or not np.isfinite(mapped[:, :15]).all():
        raise ValueError("YuNet rows must contain finite bbox, landmarks, and score values.")
    mapped[:, [0, 2, 4, 6, 8, 10, 12]] *= target_width / source_width
    mapped[:, [1, 3, 5, 7, 9, 11, 13]] *= target_height / source_height
    return mapped


def face_box_bounds(frame, face_row):
    """Validate a face row and return its clipped box before cropping or alignment."""
    height, width = frame.shape[:2]
    row = np.asarray(face_row, dtype=np.float32).reshape(-1)
    if row.size < 15 or not np.isfinite(row[:15]).all():
        raise ValueError("Invalid YuNet face row.")
    x, y, box_width, box_height = row[:4]
    x0 = max(0, int(math.floor(x)))
    y0 = max(0, int(math.floor(y)))
    x1 = min(width, int(math.ceil(x + box_width)))
    y1 = min(height, int(math.ceil(y + box_height)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Face box does not intersect the frame.")
    return x0, y0, x1, y1


def get_embedding(frame, face_row):
    """Align the original unmirrored frame and return an L2-normalized SFace vector."""
    if _recognizer is None:
        raise RuntimeError("Models are not loaded. Call load_models() first.")
    face_box_bounds(frame, face_row)
    row = np.asarray(face_row, dtype=np.float32).reshape(-1).copy()
    height, width = frame.shape[:2]
    points = row[4:14].reshape(5, 2)
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width):
        raise ValueError("Face landmarks extend outside the original frame.")
    if np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
        raise ValueError("Face landmarks extend outside the original frame.")
    row[0] = max(0.0, row[0])
    row[1] = max(0.0, row[1])
    row[2] = min(width - row[0], row[2])
    row[3] = min(height - row[1], row[3])
    aligned = _recognizer.alignCrop(frame, row)
    feature = np.asarray(_recognizer.feature(aligned), dtype=np.float32).reshape(-1)
    if feature.size != EMBEDDING_SIZE or not np.isfinite(feature).all():
        raise ValueError(f"SFace must return {EMBEDDING_SIZE} finite values.")
    norm = float(np.linalg.norm(feature))
    if norm <= 0:
        raise ValueError("SFace returned a zero-length embedding.")
    return feature / norm


def cosine_sim(a, b):
    first = np.asarray(a, dtype=np.float32).reshape(-1)
    second = np.asarray(b, dtype=np.float32).reshape(-1)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Cosine similarity requires same-size finite vectors.")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0:
        raise ValueError("Cosine similarity is undefined for a zero-length vector.")
    return float(np.dot(first, second) / denominator)


def estimate_head_pose(face_row, frame_size):
    """Estimate (yaw, pitch, roll) in degrees from original-frame YuNet landmarks."""
    row = np.asarray(face_row, dtype=np.float32).reshape(-1)
    if row.size < 14 or not np.isfinite(row[4:14]).all():
        raise ValueError("Face row does not contain five valid landmarks.")
    width, height = (int(value) for value in frame_size)
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    image_points = row[4:14].reshape(5, 2).astype(np.float64)
    if (
        np.any(image_points[:, 0] < 0)
        or np.any(image_points[:, 0] >= width)
        or np.any(image_points[:, 1] < 0)
        or np.any(image_points[:, 1] >= height)
    ):
        raise ValueError("Face landmarks extend outside the original frame.")
    eye_distance = float(np.linalg.norm(image_points[0] - image_points[1]))
    if not np.isfinite(eye_distance) or eye_distance <= 1e-6:
        raise ValueError("Inter-eye distance is zero.")
    camera_matrix = np.asarray(
        [
            [float(width), 0.0, width / 2.0],
            [0.0, float(width), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)
    try:
        solved, rotation_vector, translation = cv2.solvePnP(
            YUNET_FACE_MODEL_3D,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as error:
        raise ValueError(f"Could not estimate head pose with solvePnP: {error}") from error
    if not solved or not np.isfinite(rotation_vector).all():
        raise ValueError("Could not estimate head pose with solvePnP.")
    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rotation_vector, translation = cv2.solvePnPRefineLM(
                YUNET_FACE_MODEL_3D,
                image_points,
                camera_matrix,
                distortion,
                rotation_vector,
                translation,
            )
        except cv2.error as error:
            raise ValueError(f"Could not refine head pose: {error}") from error
    rotation, _ = cv2.Rodrigues(rotation_vector)
    yaw = math.degrees(math.atan2(rotation[0, 2], rotation[2, 2]))
    pitch = math.degrees(
        math.atan2(
            -rotation[1, 2],
            math.sqrt(rotation[0, 2] ** 2 + rotation[2, 2] ** 2),
        )
    )
    roll = math.degrees(math.atan2(rotation[1, 0], rotation[1, 1]))
    if not np.isfinite((yaw, pitch, roll)).all():
        raise ValueError("Head pose estimate is not finite.")
    # Display is mirrored; negate camera-space yaw so a right turn moves right on screen.
    return -yaw, pitch, roll


def pose_to_sector(pose, offset=(0.0, 0.0)):
    """Map calibrated pose to one of 12 sectors, mirroring yaw for display."""
    yaw = float(pose[0]) - float(offset[0])
    pitch = float(pose[1]) - float(offset[1])
    if not math.isfinite(yaw) or not math.isfinite(pitch):
        raise ValueError("Pose contains a non-finite value.")
    angle = math.atan2(pitch, -yaw) % (2 * math.pi)
    return int(math.floor(angle * SECTOR_COUNT / (2 * math.pi) + 0.5)) % SECTOR_COUNT


def nearest_unfilled_sector(pose, offset, filled):
    remaining = set(range(SECTOR_COUNT)) - set(filled)
    if not remaining:
        return None
    requested = pose_to_sector(pose, offset)
    return min(
        remaining,
        key=lambda sector: min(
            (sector - requested) % SECTOR_COUNT,
            (requested - sector) % SECTOR_COUNT,
        ),
    )


class QualityGate:
    """Evaluate detection and image quality in the original frame's coordinates."""

    STATUS = {
        "no_face": "Position your face in the circle",
        "multi_face": "Only one face please",
        "low_score": "Position your face in the circle",
        "too_small": "Move closer",
        "too_large": "Move back",
        "off_center": "Position your face in the circle",
        "blurry": "Hold still",
        "moving": "Hold still",
        "bad_pose": "Position your face in the circle",
    }

    def __init__(self, sharpness_ratio=0.4):
        self.sharpness_ratio = float(sharpness_ratio)
        self.sharpness_samples = []
        self.previous_center = None
        self.last_reason = None
        self.last_sharpness = None

    def reset_motion(self):
        self.previous_center = None

    def evaluate_detail(self, frame, faces, score_threshold=0.7):
        """Return (face, reason, center, pose); accepted frames have reason=None."""
        self.last_reason = None
        self.last_sharpness = None
        height, width = frame.shape[:2]
        if len(faces) > 1:
            self.reset_motion()
            return None, "multi_face", None, None
        if not faces:
            self.reset_motion()
            return None, "no_face", None, None

        face = np.asarray(faces[0], dtype=np.float32).reshape(-1)
        if face.size < 15 or not np.isfinite(face[:15]).all():
            return None, "off_center", None, None
        if float(face[14]) < score_threshold:
            return None, "low_score", None, None

        x, y, face_width, face_height = (float(value) for value in face[:4])
        center = (x + face_width / 2, y + face_height / 2)
        if face_width < width * 0.12:
            return None, "too_small", center, None
        if face_width > width * 0.70:
            return None, "too_large", center, None
        if not (width * 0.20 <= center[0] <= width * 0.80):
            return None, "off_center", center, None
        if not (height * 0.20 <= center[1] <= height * 0.80):
            return None, "off_center", center, None
        try:
            x0, y0, x1, y1 = face_box_bounds(frame, face)
        except ValueError:
            return None, "off_center", center, None

        if self.previous_center is not None:
            if math.dist(center, self.previous_center) > face_width * 0.15:
                self.previous_center = center
                return None, "moving", center, None
        self.previous_center = center

        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.last_sharpness = sharpness
        if len(self.sharpness_samples) < 10:
            self.sharpness_samples.append(sharpness)
        elif sharpness < float(np.median(self.sharpness_samples)) * self.sharpness_ratio:
            return None, "blurry", center, None

        try:
            pose = estimate_head_pose(face, (width, height))
        except ValueError:
            return None, "bad_pose", center, None
        return face, None, center, pose

    def evaluate(self, frame, faces, score_threshold=0.7):
        """Compatibility wrapper returning a UI message and accepted face details."""
        face, reason, center, pose = self.evaluate_detail(frame, faces, score_threshold)
        self.last_reason = reason
        return self.STATUS.get(reason, ""), face, center, pose[:2] if pose is not None else None


def center_square_crop(frame, with_origin=False):
    height, width = frame.shape[:2]
    side = min(height, width)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    square = frame[y0 : y0 + side, x0 : x0 + side]
    return (square, (x0, y0)) if with_origin else square


def validate_person_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}", name):
        raise ValueError("Name must be 1-64 characters; use letters, numbers, spaces, _ or -.")
    return name


def save_enrollment(name, embeddings):
    validate_person_name(name)
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if (
        array.ndim != 2
        or array.shape[1] != EMBEDDING_SIZE
        or not 1 <= array.shape[0] <= MAX_ENROLLMENT_EMBEDDINGS
    ):
        raise ValueError("Enrollment must contain 1-27 embeddings of 128 values each.")
    if not np.isfinite(array).all():
        raise ValueError("Enrollment contains non-finite values.")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Enrollment contains a zero-length embedding.")
    array = array / norms[:, np.newaxis]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / f"{name}.npy"
    np.save(destination, array)
    print(f"Saved {array.shape[0]} embeddings for {name!r} to {destination}")


def load_enrollments():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"No enrollment folder found: {DATA_DIR}")
    files = sorted(DATA_DIR.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No enrollments found in {DATA_DIR}. Enroll a person first.")
    enrollments = {}
    for path in files:
        try:
            values = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError) as error:
            raise ValueError(f"Corrupted enrollment file {path}: {error}") from error
        if (
            values.ndim != 2
            or values.shape[1] != EMBEDDING_SIZE
            or not 1 <= values.shape[0] <= MAX_ENROLLMENT_EMBEDDINGS
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Corrupted enrollment file {path}: expected a finite (N, 128) array.")
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= 0):
            raise ValueError(f"Corrupted enrollment file {path}: zero-length embedding.")
        enrollments[path.stem] = values / norms[:, np.newaxis]
    return enrollments


def open_camera():
    if platform.system() == "Linux":
        backend = cv2.CAP_V4L2
    elif platform.system() == "Windows":
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_ANY
    camera = None
    for index in range(4):
        candidate = cv2.VideoCapture(index, backend)
        if candidate.isOpened():
            camera = candidate
            break
        candidate.release()
    if camera is None:
        raise RuntimeError(
            "Could not open camera indexes 0-3. On Linux check `ls /dev/video*` and camera "
            "permissions; on Windows check camera privacy permissions."
        )
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        discarded = 0
        failures = 0
        while discarded < 10:
            ok, _ = camera.read()
            if ok:
                discarded += 1
                failures = 0
            else:
                failures += 1
                if failures >= 30:
                    raise RuntimeError("Camera opened but did not produce warm-up frames.")
        return camera
    except (cv2.error, RuntimeError):
        camera.release()
        raise


def _clamp_point(x, y):
    return (
        min(CANVAS_WIDTH - 1, max(0, int(round(x)))),
        min(CANVAS_HEIGHT - 1, max(0, int(round(y)))),
    )


def _draw_rounded_line(image, start, end, color, thickness):
    start = _clamp_point(*start)
    end = _clamp_point(*end)
    cv2.line(image, start, end, color, thickness, cv2.LINE_AA)
    radius = max(1, thickness // 2)
    cv2.circle(image, start, radius, color, -1, cv2.LINE_AA)
    cv2.circle(image, end, radius, color, -1, cv2.LINE_AA)


def _fit_text_scale(text, initial_scale, max_width, thickness):
    scale = initial_scale
    while scale > 0.15:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if width <= max_width:
            return scale
        scale *= 0.9
    return scale


def _sector_for_tick(tick):
    return tick // 5


def _draw_ring(
    canvas, progress=None, pulse=0.0, scan_phase=None, target_sector=None, ring_color=None
):
    cx, cy = CAMERA_CENTER
    alphas = progress if progress is not None else [0.0] * TICK_COUNT
    for tick in range(TICK_COUNT):
        angle = 2 * math.pi * tick / TICK_COUNT
        start = (cx + math.cos(angle) * RING_INNER_RADIUS, cy + math.sin(angle) * RING_INNER_RADIUS)
        end = (cx + math.cos(angle) * RING_OUTER_RADIUS, cy + math.sin(angle) * RING_OUTER_RADIUS)
        alpha = min(1.0, max(0.0, float(alphas[tick])))
        eased = alpha * alpha * (3 - 2 * alpha)
        base = np.array((68, 68, 68), dtype=np.float32)
        green = np.array((90, 220, 90), dtype=np.float32)
        color = tuple(int(value) for value in base * (1 - eased) + green * eased)
        sector = _sector_for_tick(tick)
        if target_sector == sector:
            glow = 0.35 + 0.25 * (math.sin(pulse * 2 * math.pi) + 1) / 2
            color = tuple(int(a * (1 - glow) + 150 * glow) for a in color)
        if scan_phase is not None:
            light_tick = int((scan_phase % 1.0) * TICK_COUNT)
            distance = (tick - light_tick) % TICK_COUNT
            if distance < 10:
                blue = (255, 190, 90)
                amount = (1 - distance / 10) * 0.9
                color = tuple(int(a * (1 - amount) + b * amount) for a, b in zip(color, blue))
        if progress is not None and all(value >= 0.99 for value in progress):
            color = tuple(int(channel * (0.72 + 0.28 * pulse)) for channel in green)
        if ring_color is not None:
            color = ring_color
        thickness = 7
        _draw_rounded_line(canvas, start, end, color, thickness)

    if scan_phase is not None:
        start_angle = int((scan_phase % 1.0) * 360)
        cv2.ellipse(
            canvas,
            (cx, cy),
            (RING_OUTER_RADIUS + 7, RING_OUTER_RADIUS + 7),
            0,
            start_angle,
            start_angle + 48,
            (105, 120, 135),
            2,
            cv2.LINE_AA,
        )


def _draw_camera_circle(canvas, frame):
    cx, cy = CAMERA_CENTER
    if frame is not None and frame.size:
        square = center_square_crop(frame)
        mirrored = cv2.flip(square, 1)
        diameter = CAMERA_RADIUS * 2
        camera_square = cv2.resize(mirrored, (diameter, diameter), interpolation=cv2.INTER_AREA)
        mask = np.zeros((diameter, diameter), dtype=np.uint8)
        cv2.circle(mask, (CAMERA_RADIUS, CAMERA_RADIUS), CAMERA_RADIUS - 1, 255, -1, cv2.LINE_AA)
        roi = canvas[cy - CAMERA_RADIUS : cy + CAMERA_RADIUS, cx - CAMERA_RADIUS : cx + CAMERA_RADIUS]
        np.copyto(roi, camera_square, where=mask[:, :, np.newaxis] > 0)
    cv2.circle(canvas, (cx, cy), CAMERA_RADIUS, (215, 215, 215), 2, cv2.LINE_AA)


def render_ui(
    frame,
    status,
    progress=None,
    pulse=0.0,
    scan_phase=None,
    direction=None,
    target_sector=None,
    title="Face enrollment",
    result_color=None,
    result_text=None,
    success_progress=None,
    status_alpha=1.0,
    camera_dim=1.0,
    draw_ring=True,
):
    """Render a fixed 720x960, circular-camera Face ID-style interface."""
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    _draw_camera_circle(canvas, frame)
    if camera_dim < 1.0:
        x0 = CAMERA_CENTER[0] - CAMERA_RADIUS
        x1 = CAMERA_CENTER[0] + CAMERA_RADIUS
        y0 = CAMERA_CENTER[1] - CAMERA_RADIUS
        y1 = CAMERA_CENTER[1] + CAMERA_RADIUS
        roi = canvas[y0:y1, x0:x1]
        cv2.convertScaleAbs(roi, dst=roi, alpha=max(0.0, camera_dim))
    if draw_ring:
        _draw_ring(canvas, progress, pulse, scan_phase, target_sector, result_color)
    if direction is not None:
        yaw, pitch = direction
        dx = -float(yaw)
        dy = float(pitch)
        magnitude = math.hypot(dx, dy)
        if magnitude > 1e-6 and math.isfinite(magnitude):
            radius = min(CAMERA_RADIUS - 30, magnitude * 260)
            dot = _clamp_point(
                CAMERA_CENTER[0] + dx / magnitude * radius,
                CAMERA_CENTER[1] + dy / magnitude * radius,
            )
            cv2.circle(canvas, dot, 13, (70, 125, 190), -1, cv2.LINE_AA)
            cv2.circle(canvas, dot, 6, (230, 245, 255), -1, cv2.LINE_AA)

    if success_progress is not None:
        cv2.circle(canvas, CAMERA_CENTER, 67, (70, 190, 80), -1, cv2.LINE_AA)
        check = (
            (CAMERA_CENTER[0] - 33, CAMERA_CENTER[1] + 1),
            (CAMERA_CENTER[0] - 8, CAMERA_CENTER[1] + 25),
            (CAMERA_CENTER[0] + 38, CAMERA_CENTER[1] - 28),
        )
        progress_value = min(1.0, max(0.0, success_progress))
        if progress_value > 0:
            if progress_value < 0.5:
                end = tuple(
                    int(check[0][i] + (check[1][i] - check[0][i]) * progress_value * 2)
                    for i in range(2)
                )
                _draw_rounded_line(canvas, check[0], end, (245, 245, 245), 8)
            else:
                _draw_rounded_line(canvas, check[0], check[1], (245, 245, 245), 8)
                fraction = (progress_value - 0.5) * 2
                end = tuple(
                    int(check[1][i] + (check[2][i] - check[1][i]) * fraction)
                    for i in range(2)
                )
                _draw_rounded_line(canvas, check[1], end, (245, 245, 245), 8)

    cv2.putText(canvas, title, (28, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (215, 215, 215), 1, cv2.LINE_AA)
    if result_text:
        text = result_text
        scale = _fit_text_scale(text, 0.62 if len(text) < 28 else 0.48, CANVAS_WIDTH - 32, 1)
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
        cv2.putText(
            canvas,
            text,
            (max(12, (CANVAS_WIDTH - size[0]) // 2), CAMERA_CENTER[1] + CAMERA_RADIUS + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    status_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0]
    status_scale = min(0.62, (CANVAS_WIDTH - 36) / max(1, status_size[0]) * 0.62)
    status_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, status_scale, 1)[0]
    text_canvas = canvas.copy()
    cv2.putText(
        text_canvas,
        status,
        (max(18, (CANVAS_WIDTH - status_size[0]) // 2), 820),
        cv2.FONT_HERSHEY_SIMPLEX,
        status_scale,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    alpha = min(1.0, max(0.0, status_alpha))
    if alpha < 1:
        cv2.addWeighted(text_canvas, alpha, canvas, 1 - alpha, 0, canvas)
    else:
        canvas = text_canvas
    hint = "Press Q or ESC to cancel"
    hint_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(
        canvas,
        hint,
        ((CANVAS_WIDTH - hint_size[0]) // 2, 925),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (105, 105, 105),
        1,
        cv2.LINE_AA,
    )
    return canvas


def letterbox_canvas(canvas, output_size):
    """Fit the fixed portrait canvas into an output size without distorting it."""
    output_width, output_height = output_size
    if output_width <= 0 or output_height <= 0:
        raise ValueError("Output dimensions must be positive.")
    scale = min(output_width / canvas.shape[1], output_height / canvas.shape[0])
    fitted = cv2.resize(
        canvas,
        (max(1, round(canvas.shape[1] * scale)), max(1, round(canvas.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    x0 = (output_width - fitted.shape[1]) // 2
    y0 = (output_height - fitted.shape[0]) // 2
    output[y0 : y0 + fitted.shape[0], x0 : x0 + fitted.shape[1]] = fitted
    return output

