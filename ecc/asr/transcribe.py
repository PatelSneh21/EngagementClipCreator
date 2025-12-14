"""ASR transcription using faster-whisper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .transcript_schema import Transcript, TranscriptSegment


def _load_model(model_size: str = "small") -> Any:
    """Load faster-whisper model, raising a clear error if missing."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install with `pip install 'faster-whisper>=0.10'` "
            "or add the asr extra: `pip install -e .[asr]`."
        ) from exc

    # Let faster-whisper pick device/compute_type; users can tune later.
    return WhisperModel(model_size, device="auto", compute_type="auto")


def transcribe_audio(
    audio_path: Path,
    model_size: str = "small",
    output_json: Path | None = None,
) -> Transcript:
    """
    Transcribe audio into a Transcript with millisecond timings.

    Args:
        audio_path: Path to mono 16 kHz WAV file.
        model_size: faster-whisper model size (tiny/base/small/medium/large-v3).
        output_json: Optional path to write transcript JSON.

    Returns:
        Transcript object with segments.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _load_model(model_size)

    segments_iter, _info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
    )

    segments: list[TranscriptSegment] = []
    for seg in segments_iter:
        segments.append(
            TranscriptSegment(
                start_ms=int(seg.start * 1000),
                end_ms=int(seg.end * 1000),
                text=seg.text.strip(),
            )
        )

    transcript = Transcript(segments=segments)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(transcript.model_dump(), f, ensure_ascii=False, indent=2)

    return transcript
