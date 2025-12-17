"""Scoring heuristics for candidates."""

from __future__ import annotations

from ecc.moments.features import extract_features


def score_candidate(candidate: dict) -> float:
    """Compute a simple heuristic score (higher is better)."""
    features = extract_features(candidate)
    duration_ms = features["duration_ms"]
    word_count = features["word_count"]
    words_per_second = features["words_per_second"]

    score = 0.0

    # Prefer mid-length clips (3–8s).
    if 3000 <= duration_ms <= 8000:
        score += 1.0
    elif duration_ms < 2000:
        score -= 0.5
    elif duration_ms > 12000:
        score -= 0.5

    # Favor higher speaking pace as a rough energy proxy.
    score += min(words_per_second / 3.0, 1.0)

    # Slight bonus for more content.
    score += min(word_count / 40.0, 0.5)

    return score
