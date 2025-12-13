from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ECCConfig(BaseModel):
    """Top-level configuration for a single ECC run."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["mode_a", "mode_b"] = Field(
        "mode_a", description="Mode A: article/plot driven. Mode B: auto-highlight."
    )
    input_video: Path = Field(..., description="Path to the source video (mp4/mov).")
    article_path: Path | None = Field(
        None,
        description="Plot/article text for Mode A (required when mode is mode_a).",
    )
    output_dir: Path = Field(
        Path("runs"), description="Where to write artifacts and final outputs."
    )
    clip_length_seconds: tuple[int, int] = Field(
        (30, 45), description="Min/max duration targets for the final clip."
    )
