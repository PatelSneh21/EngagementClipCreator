"""Inspect media metadata (duration, fps, audio tracks)."""

from __future__ import annotations

from pathlib import Path


def probe_media(input_video: Path) -> dict:
    """TODO: wrap ffprobe output."""
    raise NotImplementedError("Media probing not implemented yet.")
