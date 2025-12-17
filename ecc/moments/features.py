"""Feature extraction for scoring engagement."""

from __future__ import annotations


def extract_features(candidate: dict) -> dict:
    """Compute simple text/duration features for heuristic scoring."""
    text = str(candidate.get("text", "")).strip()
    duration_ms = int(candidate.get("duration_ms", 0)) or max(
        1, int(candidate.get("end_ms", 0)) - int(candidate.get("start_ms", 0))
    )
    word_count = int(candidate.get("word_count", 0)) or len(text.split())
    duration_s = max(duration_ms / 1000.0, 0.1)
    words_per_second = word_count / duration_s

    return {
        "duration_ms": duration_ms,
        "word_count": word_count,
        "words_per_second": words_per_second,
    }
