from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TranscriptSegment(BaseModel):
    """Single ASR segment with millisecond timing."""

    model_config = ConfigDict(extra="forbid")

    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=0)
    text: str = Field(..., min_length=1)


class Transcript(BaseModel):
    """Full transcript returned by the ASR stage."""

    model_config = ConfigDict(extra="forbid")

    segments: list[TranscriptSegment] = Field(default_factory=list)
