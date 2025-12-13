"""ASR stage placeholder."""

from __future__ import annotations

from pathlib import Path

from .transcript_schema import Transcript


def transcribe_audio(audio_path: Path) -> Transcript:
    """
    TODO: Implement whisper or faster-whisper transcription.
    Right now this just documents the expected interface.
    """
    raise NotImplementedError("ASR transcription not implemented yet.")
