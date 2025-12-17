from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CandidateSegment(BaseModel):
    """A clip candidate aligned to a scene with its transcript text."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., description="Unique id for this candidate.")
    scene_id: int = Field(..., ge=0, description="Scene identifier this candidate comes from.")
    start_ms: int = Field(..., ge=0, description="Start time of the candidate in ms.")
    end_ms: int = Field(..., ge=0, description="End time of the candidate in ms.")
    text: str = Field(..., min_length=1, description="Concatenated transcript text for the scene.")
    duration_ms: int = Field(..., ge=1, description="Duration of the candidate in ms.")
    word_count: int = Field(..., ge=0, description="Approximate word count for text.")
    segment_count: int = Field(
        ..., ge=1, description="Number of transcript segments used."
    )
