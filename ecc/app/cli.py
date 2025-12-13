from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Engagement Clip Creator CLI skeleton")
console = Console()


def _ensure_runs_dir(runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


@app.command()
def info() -> None:
    """Print a quick reminder of what ECC is for."""
    console.print(
        "[bold]ECC[/] converts long-form videos (up to ~3h) into 30–45s shorts.\n"
        "- Mode A: plot-driven recap teaser using provided article/plot.\n"
        "- Mode B: auto-highlight trailer that scores and stitches engaging moments.\n"
        "Pipeline: ingest → ASR → segmentation → selection → narration/TTS → render."
    )


@app.command()
def init_run(
    run_id: str | None = typer.Option(
        None, help="Optional run id; defaults to timestamp-based id."
    ),
    runs_dir: Path = typer.Option(
        Path("runs"),
        help="Directory where run artifacts are stored.",
    ),
) -> None:
    """Create a run folder under runs/ so you can store artifacts."""
    run_name = run_id or time.strftime("run-%Y%m%d-%H%M%S")
    run_path = _ensure_runs_dir(runs_dir) / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Initialized[/] run directory at {run_path}")


if __name__ == "__main__":
    app()
