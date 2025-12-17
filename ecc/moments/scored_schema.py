from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, ConfigDict, Field

from ecc.segmentation.candidate_schema import CandidateSegment


class ScoredCandidate(CandidateSegment):
    """Candidate with heuristic score and extracted features."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(..., description="Heuristic score (higher is better).")
    features: Dict[str, float] = Field(
        default_factory=dict, description="Computed features used for scoring."
    )
