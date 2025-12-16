"""Build candidate segments by aligning transcript text to scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

from ecc.asr.transcript_schema import Transcript
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


def build_candidate_segments(
    transcript: Transcript | Path | dict,
    scenes: Sequence[dict] | Path,
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

    for scene in scenes_list:
        scene_id = scene.get("scene_id")
        start_ms = scene.get("start_ms")
        end_ms = scene.get("end_ms")
        if scene_id is None or start_ms is None or end_ms is None:
            raise ValueError("Scene entries must include scene_id, start_ms, end_ms.")

        overlapping_text: List[str] = []
        for seg in transcript_obj.segments:
            if _overlaps(start_ms, end_ms, seg.start_ms, seg.end_ms):
                overlapping_text.append(seg.text.strip())

        if not overlapping_text:
            # If no transcript overlaps, skip this scene.
            continue

        candidate_text = " ".join(t for t in overlapping_text if t)
        if not candidate_text:
            continue

        candidates.append(
            CandidateSegment(
                candidate_id=f"scene-{scene_id}",
                scene_id=int(scene_id),
                start_ms=int(start_ms),
                end_ms=int(end_ms),
                text=candidate_text,
            )
        )

    return candidates


def write_candidates(candidates: Iterable[CandidateSegment], output_path: Path) -> None:
    """Write candidate list to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump() for c in candidates]
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
