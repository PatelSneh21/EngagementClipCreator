"""Utilities to normalize and merge transcript segments."""

from __future__ import annotations

import re
from typing import Iterable, List

from .transcript_schema import TranscriptSegment


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _split_by_punctuation(
    segment: TranscriptSegment, max_segment_ms: int
) -> List[TranscriptSegment]:
    """Split a long segment into smaller ones using sentence punctuation."""
    duration_ms = segment.end_ms - segment.start_ms
    if duration_ms <= max_segment_ms:
        return [segment]

    parts = [p for p in re.split(r"(?<=[.!?])\s+", segment.text.strip()) if p]
    if len(parts) <= 1:
        return [segment]

    total_chars = sum(len(p) for p in parts) or 1
    start_ms = segment.start_ms
    new_segments: List[TranscriptSegment] = []

    for idx, part in enumerate(parts):
        if idx == len(parts) - 1:
            end_ms = segment.end_ms
        else:
            portion = len(part) / total_chars
            end_ms = start_ms + max(1, int(duration_ms * portion))
        new_segments.append(
            TranscriptSegment(start_ms=start_ms, end_ms=end_ms, text=part.strip())
        )
        start_ms = end_ms

    return new_segments


def cleanup_transcript(
    segments: Iterable[TranscriptSegment],
    min_segment_ms: int = 600,
    max_gap_ms: int = 200,
    max_segment_ms: int = 8000,
) -> List[TranscriptSegment]:
    """
    Normalize, merge tiny segments, and split very long segments.

    Args:
        segments: Iterable of TranscriptSegment.
        min_segment_ms: Merge segments shorter than this threshold.
        max_gap_ms: Merge segments separated by gaps smaller than this threshold.
        max_segment_ms: Split segments longer than this threshold by punctuation.
    """
    merged: List[TranscriptSegment] = []
    current: TranscriptSegment | None = None

    for seg in segments:
        text = _normalize_text(seg.text)
        if not text:
            continue
        seg = TranscriptSegment(start_ms=seg.start_ms, end_ms=seg.end_ms, text=text)

        seg_duration = seg.end_ms - seg.start_ms
        if current is None:
            current = seg
            continue

        gap_ms = seg.start_ms - current.end_ms
        if seg_duration < min_segment_ms or gap_ms <= max_gap_ms:
            current = TranscriptSegment(
                start_ms=current.start_ms,
                end_ms=seg.end_ms,
                text=f"{current.text} {seg.text}".strip(),
            )
        else:
            merged.extend(_split_by_punctuation(current, max_segment_ms))
            current = seg

    if current is not None:
        merged.extend(_split_by_punctuation(current, max_segment_ms))

    return merged
