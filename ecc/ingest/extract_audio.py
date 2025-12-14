"""Extract mono 16k audio for ASR."""

from __future__ import annotations

import subprocess
from pathlib import Path


def extract_audio(input_video: Path, output_wav: Path) -> Path:
    """
    Use ffmpeg to extract mono 16 kHz WAV from the input video.
    Args:
        input_video: Source video file.
        output_wav: Destination WAV file path.

    Returns:
        The output_wav path.

    Raises:
        FileNotFoundError: if the input video does not exist.
        RuntimeError: if ffmpeg is not installed or extraction fails.
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i",
        str(input_video),
        "-vn",  # no video
        "-ac",
        "1",  # mono
        "-ar",
        "16000",  # 16 kHz
        "-y",  # overwrite
        str(output_wav),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg not found. Please install FFmpeg."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    return output_wav
