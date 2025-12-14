from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console

from ecc.asr.transcribe import transcribe_audio
from ecc.ingest.extract_audio import extract_audio
from ecc.ingest.media_probe import probe_media

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


@app.command(name="probe-media")
def probe_media_cmd(
    input_video: Path = typer.Argument(..., exists=True, help="Path to input video")
) -> None:
    """Inspect media metadata using ffprobe and print JSON output."""
    try:
        info = probe_media(input_video)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Probe failed:[/] {exc}")
        raise typer.Exit(code=1)

    console.print_json(data=info)


@app.command(name="ingest-audio")
def ingest_audio_cmd(
    input_video: Path = typer.Argument(..., exists=True, help="Path to input video"),
    run_id: str | None = typer.Option(
        None, help="Optional run id; defaults to timestamp-based id."
    ),
    runs_dir: Path = typer.Option(
        Path("runs"), help="Directory where run artifacts are stored."
    ),
    output_wav: Path | None = typer.Option(
        None,
        help="Optional output wav path. Defaults to runs/<run_id>/audio.wav",
    ),
) -> None:
    """
    Probe media and extract mono 16 kHz audio for ASR into a run folder.
    """
    run_name = run_id or time.strftime("run-%Y%m%d-%H%M%S")
    run_path = _ensure_runs_dir(runs_dir) / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    wav_path = output_wav or (run_path / "audio.wav")

    try:
        probe_info = probe_media(input_video)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Probe failed:[/] {exc}")
        raise typer.Exit(code=1)

    duration = None
    try:
        duration = float(probe_info.get("format", {}).get("duration", 0))
    except (TypeError, ValueError):
        duration = None

    try:
        extract_audio(input_video, wav_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Audio extraction failed:[/] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[green]Run:[/] {run_path}")
    if duration:
        console.print(f"Duration: {duration:.2f}s")
    console.print(f"Audio written to: {wav_path}")


@app.command(name="asr")
def asr_cmd(
    audio_path: Path = typer.Argument(
        ..., exists=True, help="Path to mono 16 kHz WAV audio"
    ),
    model_size: str = typer.Option(
        "small",
        help="faster-whisper model size (tiny/base/small/medium/large-v3)",
    ),
    run_id: str | None = typer.Option(
        None, help="Optional run id; defaults to timestamp-based id."
    ),
    runs_dir: Path = typer.Option(
        Path("runs"), help="Directory where run artifacts are stored."
    ),
    output_json: Path | None = typer.Option(
        None,
        help="Optional transcript JSON output path. Defaults to runs/<run_id>/transcript.json",
    ),
) -> None:
    """Transcribe audio with faster-whisper and write a transcript JSON."""
    run_name = run_id or time.strftime("run-%Y%m%d-%H%M%S")
    run_path = _ensure_runs_dir(runs_dir) / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    transcript_path = output_json or (run_path / "transcript.json")

    try:
        transcript = transcribe_audio(
            audio_path=audio_path,
            model_size=model_size,
            output_json=transcript_path,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]ASR failed:[/] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[green]Run:[/] {run_path}")
    console.print(f"Segments: {len(transcript.segments)}")
    console.print(f"Transcript written to: {transcript_path}")


if __name__ == "__main__":
    app()
