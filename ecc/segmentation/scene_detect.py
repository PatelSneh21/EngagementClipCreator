"""Scene/shot detection using PySceneDetect."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def _load_pyscenedetect() -> Any:
    """Import PySceneDetect components with a clear error if missing."""
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError as exc:
        raise RuntimeError(
            "PySceneDetect is not installed. Install with `pip install 'scenedetect>=0.6'` "
            "or add the seg extra: `pip install -e .[seg]`."
        ) from exc
    return ContentDetector, SceneManager, open_video


def detect_scenes(input_video: Path, threshold: float = 27.0) -> List[Dict[str, int]]:
    """
    Detect scenes in a video and return a list of boundaries in milliseconds.

    Args:
        input_video: Path to video file.
        threshold: ContentDetector threshold (higher = fewer scenes).

    Returns:
        List of dicts: {"scene_id": int, "start_ms": int, "end_ms": int}

    Raises:
        FileNotFoundError: if input_video does not exist.
        RuntimeError: if PySceneDetect/ffmpeg decoding fails.
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    ContentDetector, SceneManager, open_video = _load_pyscenedetect()

    video = open_video(str(input_video))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))

    manager.detect_scenes(video)
    scene_list = manager.get_scene_list()

    scenes: List[Dict[str, int]] = []
    for idx, (start, end) in enumerate(scene_list):
        # PySceneDetect uses FrameTimecodes; convert to milliseconds.
        scenes.append(
            {
                "scene_id": idx,
                "start_ms": int(start.get_seconds() * 1000),
                "end_ms": int(end.get_seconds() * 1000),
            }
        )
    return scenes
