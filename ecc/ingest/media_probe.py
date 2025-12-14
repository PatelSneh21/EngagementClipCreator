"""Inspect media metadata (duration, fps, audio tracks) using ffprobe."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict


def probe_media(input_video: Path) -> Dict[str, Any]:
    """
    Run ffprobe and return its JSON output as a dict.

    Raises:
        FileNotFoundError: if the input file does not exist.
        RuntimeError: if ffprobe fails or is not installed.
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(input_video),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe not found. Please install FFmpeg (ffprobe is part of it)."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to parse ffprobe output as JSON.") from exc
