"""Build candidate segments by aligning transcript text to scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

from ecc.asr.transcript_cleanup import cleanup_transcript
from ecc.asr.transcript_schema import Transcript, TranscriptSegment
from ecc.segmentation.candidate_schema import CandidateSegment


def _load_transcript(transcript_path: Path) -> Transcript:
    """Load a transcript JSON file into a Transcript model."""
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    return Transcript.model_validate(data)


def _load_scenes(scene_path: Path) -> List[dict]:
    """Load scenes JSON as a list of dicts with start_ms/end_ms/scene_id."""
    if not scene_path.exists():
        raise FileNotFoundError(f"Scenes file not found: {scene_path}")
    scenes = json.loads(scene_path.read_text(encoding="utf-8"))
    if not isinstance(scenes, list):
        raise ValueError("Scenes JSON must be a list.")
    return scenes


def _overlaps(scene_start: int, scene_end: int, seg_start: int, seg_end: int) -> bool:
    """Return True if transcript segment overlaps the scene window."""
    return seg_end > scene_start and seg_start < scene_end


def _segments_for_scene(
    segments: Sequence[TranscriptSegment],
    scene_start: int,
    scene_end: int,
) -> List[TranscriptSegment]:
    """Filter transcript segments overlapping a scene and clamp to scene boundaries."""
    overlapping: List[TranscriptSegment] = []
    for seg in segments:
        if _overlaps(scene_start, scene_end, seg.start_ms, seg.end_ms):
            start_ms = max(scene_start, seg.start_ms)
            end_ms = min(scene_end, seg.end_ms)
            if end_ms <= start_ms:
                continue
            overlapping.append(
                TranscriptSegment(start_ms=start_ms, end_ms=end_ms, text=seg.text.strip())
            )
    return overlapping


def _chunk_segments(
    segments: Sequence[TranscriptSegment],
    scene_id: int,
    scene_start: int,
    scene_end: int,
    min_window_ms: int,
    max_window_ms: int,
) -> List[CandidateSegment]:
    """Group transcript segments into candidate windows within a scene."""
    candidates: List[CandidateSegment] = []
    current_text: List[str] = []
    current_start: int | None = None
    current_end: int | None = None
    segment_count = 0

    def flush() -> None:
        nonlocal current_text, current_start, current_end, segment_count
        if current_start is None or current_end is None:
            return
        duration_ms = current_end - current_start
        if duration_ms < min_window_ms:
            return
        text = " ".join(t for t in current_text if t).strip()
        if not text:
            return
        candidates.append(
            CandidateSegment(
                candidate_id=f"scene-{scene_id}-{len(candidates)}",
                scene_id=scene_id,
                start_ms=current_start,
                end_ms=current_end,
                text=text,
                duration_ms=duration_ms,
                word_count=len(text.split()),
                segment_count=segment_count,
            )
        )

    for seg in segments:
        if current_start is None:
            current_start = seg.start_ms
            current_end = seg.end_ms
            current_text = [seg.text]
            segment_count = 1
            continue

        proposed_end = seg.end_ms
        if proposed_end - current_start > max_window_ms:
            flush()
            current_start = seg.start_ms
            current_end = seg.end_ms
            current_text = [seg.text]
            segment_count = 1
            continue

        current_text.append(seg.text)
        current_end = proposed_end
        segment_count += 1

    flush()

    # If nothing met min_window_ms but we have content, emit a single candidate.
    if not candidates and segments:
        text = " ".join(seg.text for seg in segments).strip()
        if text:
            duration_ms = max(1, scene_end - scene_start)
            candidates.append(
                CandidateSegment(
                    candidate_id=f"scene-{scene_id}-0",
                    scene_id=scene_id,
                    start_ms=scene_start,
                    end_ms=scene_end,
                    text=text,
                    duration_ms=duration_ms,
                    word_count=len(text.split()),
                    segment_count=len(segments),
                )
            )

    return candidates


def build_candidate_segments(
    transcript: Transcript | Path | dict,
    scenes: Sequence[dict] | Path,
    min_window_ms: int = 3000,
    max_window_ms: int = 8000,
) -> List[CandidateSegment]:
    """
    Build clip candidates by aligning transcript segments to scene windows.

    Args:
        transcript: Transcript object or path/JSON to load.
        scenes: Sequence of scene dicts or a path to scenes.json.

    Returns:
        List of CandidateSegment, one per scene with overlapping transcript text.
    """
    # Normalize transcript input
    if isinstance(transcript, Transcript):
        transcript_obj = transcript
    elif isinstance(transcript, Path):
        transcript_obj = _load_transcript(transcript)
    elif isinstance(transcript, dict):
        transcript_obj = Transcript.model_validate(transcript)
    else:
        raise TypeError("transcript must be Transcript, Path, or dict.")

    # Normalize scenes input
    if isinstance(scenes, Path):
        scenes_list = _load_scenes(scenes)
    else:
        scenes_list = list(scenes)

    candidates: List[CandidateSegment] = []

    cleaned_segments = cleanup_transcript(transcript_obj.segments)

    for scene in scenes_list:
        scene_id = scene.get("scene_id")
        start_ms = scene.get("start_ms")
        end_ms = scene.get("end_ms")
        if scene_id is None or start_ms is None or end_ms is None:
            raise ValueError("Scene entries must include scene_id, start_ms, end_ms.")

        overlapping_segments = _segments_for_scene(
            cleaned_segments, int(start_ms), int(end_ms)
        )
        if not overlapping_segments:
            continue

        candidates.extend(
            _chunk_segments(
                overlapping_segments,
                scene_id=int(scene_id),
                scene_start=int(start_ms),
                scene_end=int(end_ms),
                min_window_ms=min_window_ms,
                max_window_ms=max_window_ms,
            )
        )

    return candidates


def write_candidates(candidates: Iterable[CandidateSegment], output_path: Path) -> None:
    """Write candidate list to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump() for c in candidates]
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
