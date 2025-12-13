from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .config import ECCConfig


class ECCJob(BaseModel):
    """Metadata for a single ECC run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., description="Unique id for this run.")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation time (UTC).",
    )
    config: ECCConfig
    working_dir: Path = Field(
        Path("runs"), description="Folder where this job's artifacts live."
    )
