"""Select final set of candidates under pacing/diversity constraints."""

from __future__ import annotations

from typing import List

from ecc.moments.features import extract_features
from ecc.moments.score import score_candidate


def _overlaps(a: dict, b: dict) -> bool:
    """Check if two candidates overlap in time within the same scene."""
    if a.get("scene_id") != b.get("scene_id"):
        return False
    return not (
        int(a.get("end_ms", 0)) <= int(b.get("start_ms", 0))
        or int(a.get("start_ms", 0)) >= int(b.get("end_ms", 0))
    )


def _overlaps_existing(candidate: dict, selected: List[dict]) -> bool:
    return any(_overlaps(candidate, s) for s in selected)


def select_clips(
    candidates: list[dict],
    target_min_sec: int = 30,
    target_max_sec: int = 45,
    max_candidates: int = 12,
) -> list[dict]:
    """
    Greedy selection to hit a total duration window.

    Returns selected candidates ordered by start time.
    """
    scored: List[dict] = []
    for cand in candidates:
        data = dict(cand)
        features = extract_features(data)
        data.setdefault("duration_ms", features["duration_ms"])
        data.setdefault("word_count", features["word_count"])
        data["score"] = float(data.get("score", score_candidate(data)))
        scored.append(data)

    scored.sort(key=lambda c: c["score"], reverse=True)

    selected: List[dict] = []
    total_ms = 0
    max_total_ms = target_max_sec * 1000
    min_total_ms = target_min_sec * 1000

    for cand in scored:
        if len(selected) >= max_candidates:
            break
        duration_ms = int(cand.get("duration_ms", 0))
        if duration_ms <= 0:
            continue
        if _overlaps_existing(cand, selected):
            continue
        if total_ms + duration_ms <= max_total_ms:
            selected.append(cand)
            total_ms += duration_ms

    if total_ms < min_total_ms:
        for cand in scored:
            if cand in selected:
                continue
            if len(selected) >= max_candidates:
                break
            duration_ms = int(cand.get("duration_ms", 0))
            if duration_ms <= 0:
                continue
            if _overlaps_existing(cand, selected):
                continue
            selected.append(cand)
            total_ms += duration_ms
            if total_ms >= min_total_ms:
                break

    selected.sort(key=lambda c: (int(c.get("start_ms", 0)), int(c.get("end_ms", 0))))
    return selected
