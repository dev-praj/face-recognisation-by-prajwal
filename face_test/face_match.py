"""Normalized SFace cosine scoring and conservative multi-frame decisions."""

import numpy as np

import face_core as core


DEFAULT_THRESHOLD = 0.50
MINIMUM_MARGIN = 0.05
REQUIRED_VOTES = 6
REQUIRED_FRAMES = 8
TOP_EMBEDDINGS = 3


def normalize_embedding(embedding):
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size != core.EMBEDDING_SIZE or not np.isfinite(vector).all():
        raise ValueError(f"Expected {core.EMBEDDING_SIZE} finite embedding values.")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Embedding must have non-zero length.")
    return vector / norm


def normalize_enrollments(enrollments):
    if not enrollments:
        raise ValueError("No enrolled identities are available.")
    normalized = {}
    for name, embeddings in enrollments.items():
        values = np.asarray(embeddings, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[1] != core.EMBEDDING_SIZE
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Invalid enrollment embeddings for {name!r}.")
        rows = np.stack([normalize_embedding(row) for row in values])
        if len(np.unique(rows, axis=0)) != len(rows):
            raise ValueError(f"Enrollment for {name!r} contains duplicate embeddings.")
        normalized[name] = rows
    return normalized


def score_embedding(probe, enrollments):
    """Score each person by mean of the top three normalized cosine matches."""
    vector = normalize_embedding(probe)
    people = normalize_enrollments(enrollments)
    scores = {}
    for name, rows in people.items():
        similarities = rows @ vector
        top = np.sort(similarities)[-min(TOP_EMBEDDINGS, len(similarities)) :]
        scores[name] = float(np.mean(top))
    return scores


def frame_vote(scores, threshold, margin=MINIMUM_MARGIN):
    """Return winning name, score, and margin only when both gates pass."""
    if not scores:
        raise ValueError("Cannot vote without enrolled identities.")
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    name, score = ranked[0]
    gap = float("inf") if len(ranked) == 1 else float(score - ranked[1][1])
    if score < threshold:
        return None, float(score), gap, "below_threshold"
    if len(ranked) >= 2 and gap < margin:
        return None, float(score), gap, "low_margin"
    return name, float(score), gap, "vote"


def decide_votes(frame_results, threshold, required_frames=REQUIRED_FRAMES):
    """Accept only a person with >=6 threshold/margin-qualified frame votes."""
    if len(frame_results) != required_frames:
        return {
            "accepted": False,
            "name": None,
            "votes": {},
            "reason": f"need {required_frames} good frames",
        }
    counts = {}
    for result in frame_results:
        name, _, _, _ = frame_vote(result, threshold)
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    winner = max(counts, key=counts.get) if counts else None
    if winner is not None and counts[winner] >= REQUIRED_VOTES:
        return {"accepted": True, "name": winner, "votes": counts, "reason": "accepted"}
    return {"accepted": False, "name": None, "votes": counts, "reason": "insufficient_votes"}


def scan_score(frame_results, target_name=None):
    """Conservative scan statistic: lowest genuine frame score or highest impostor score."""
    if not frame_results:
        raise ValueError("A calibration scan must contain at least one good frame.")
    if target_name is not None:
        try:
            return float(min(scores[target_name] for scores in frame_results))
        except KeyError as error:
            raise ValueError(f"Target identity {target_name!r} is not enrolled.") from error
    return float(max(max(scores.values()) for scores in frame_results))


def recommend_threshold(genuine_scores, impostor_scores, margin=MINIMUM_MARGIN):
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)
    if (
        genuine.size == 0
        or impostor.size == 0
        or not np.isfinite(genuine).all()
        or not np.isfinite(impostor).all()
    ):
        return None, False
    threshold = max(0.45, float(np.max(impostor)) + float(margin))
    overlaps = float(np.max(impostor)) >= float(np.min(genuine))
    return threshold, overlaps
