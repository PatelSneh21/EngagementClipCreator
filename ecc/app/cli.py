from __future__ import annotations

import time
import json
from pathlib import Path

import typer
from rich.console import Console

from ecc.asr.transcribe import transcribe_audio
from ecc.ingest.extract_audio import extract_audio
from ecc.ingest.media_probe import probe_media
from ecc.segmentation.build_candidates import (
    build_candidate_segments,
    write_candidates,
)
from ecc.segmentation.scene_detect import detect_scenes

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


@app.command(name="scene-detect")
def scene_detect_cmd(
    input_video: Path = typer.Argument(..., exists=True, help="Path to input video"),
    threshold: float = typer.Option(
        27.0,
        help="Content detector threshold (higher -> fewer scenes).",
    ),
    run_id: str | None = typer.Option(
        None, help="Optional run id; defaults to timestamp-based id."
    ),
    runs_dir: Path = typer.Option(
        Path("runs"), help="Directory where run artifacts are stored."
    ),
    output_json: Path | None = typer.Option(
        None,
        help="Optional output path. Defaults to runs/<run_id>/scenes.json",
    ),
) -> None:
    """Detect scene boundaries and write them as JSON."""
    run_name = run_id or time.strftime("run-%Y%m%d-%H%M%S")
    run_path = _ensure_runs_dir(runs_dir) / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    scenes_path = output_json or (run_path / "scenes.json")

    try:
        scenes = detect_scenes(input_video=input_video, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Scene detection failed:[/] {exc}")
        raise typer.Exit(code=1)

    scenes_path.parent.mkdir(parents=True, exist_ok=True)
    scenes_path.write_text(
        json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    console.print(f"[green]Run:[/] {run_path}")
    console.print(f"Scenes detected: {len(scenes)}")
    console.print(f"Scenes written to: {scenes_path}")


@app.command(name="build-candidates")
def build_candidates_cmd(
    run_id: str = typer.Option(
        ..., help="Run id whose transcript/scenes to use (expects runs/<run_id>/ files)."
    ),
    runs_dir: Path = typer.Option(
        Path("runs"), help="Directory where run artifacts are stored."
    ),
    transcript_path: Path | None = typer.Option(
        None, help="Optional transcript.json path. Defaults to runs/<run_id>/transcript.json"
    ),
    scenes_path: Path | None = typer.Option(
        None, help="Optional scenes.json path. Defaults to runs/<run_id>/scenes.json"
    ),
    output_json: Path | None = typer.Option(
        None,
        help="Optional output path. Defaults to runs/<run_id>/candidates.json",
    ),
) -> None:
    """Build candidate segments by aligning transcript to scenes."""
    run_path = _ensure_runs_dir(runs_dir) / run_id
    transcript_file = transcript_path or (run_path / "transcript.json")
    scenes_file = scenes_path or (run_path / "scenes.json")
    output_path = output_json or (run_path / "candidates.json")

    try:
        candidates = build_candidate_segments(
            transcript=transcript_file,
            scenes=scenes_file,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Candidate build failed:[/] {exc}")
        raise typer.Exit(code=1)

    try:
        write_candidates(candidates, output_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to write candidates:[/] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[green]Run:[/] {run_path}")
    console.print(f"Candidates built: {len(candidates)}")
    console.print(f"Candidates written to: {output_path}")


# Run entire workflow using `ecc run-ecc-workflow path/to/video.mp4 --run-id demo --model-size small`
@app.command(name="run-ecc-workflow")
def run_ecc_workflow(
    input_video: Path = typer.Argument(..., exists=True, help="Path to input video"),
    model_size: str = typer.Option(
        "small",
        help="faster-whisper model size (tiny/base/small/medium/large-v3)",
    ),
    threshold: float = typer.Option(
        27.0,
        help="Scene detection content detector threshold (higher -> fewer scenes).",
    ),
    run_id: str | None = typer.Option(
        None, help="Optional run id; defaults to timestamp-based id."
    ),
    runs_dir: Path = typer.Option(
        Path("runs"), help="Directory where run artifacts are stored."
    ),
    skip_scenes: bool = typer.Option(
        False, help="Skip scene detection step."
    ),
    skip_candidates: bool = typer.Option(
        False, help="Skip candidate building step."
    ),
) -> None:
    """
    Run the ECC pipeline steps: probe -> extract audio -> ASR -> scene detect -> candidates.
    Saves probe.json, audio.wav, transcript.json, scenes.json, candidates.json under runs/<run_id>/.
    """
    run_name = run_id or time.strftime("run-%Y%m%d-%H%M%S")
    run_path = _ensure_runs_dir(runs_dir) / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    probe_path = run_path / "probe.json"
    audio_path = run_path / "audio.wav"
    transcript_path = run_path / "transcript.json"
    scenes_path = run_path / "scenes.json"
    candidates_path = run_path / "candidates.json"

    try:
        probe_info = probe_media(input_video)
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        probe_path.write_text(
            json.dumps(probe_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Probe failed:[/] {exc}")
        raise typer.Exit(code=1)

    duration = None
    try:
        duration = float(probe_info.get("format", {}).get("duration", 0))
    except (TypeError, ValueError):
        duration = None

    try:
        extract_audio(input_video, audio_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Audio extraction failed:[/] {exc}")
        raise typer.Exit(code=1)

    try:
        transcript = transcribe_audio(
            audio_path=audio_path,
            model_size=model_size,
            output_json=transcript_path,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]ASR failed:[/] {exc}")
        raise typer.Exit(code=1)

    scenes = []
    if not skip_scenes:
        try:
            scenes = detect_scenes(input_video=input_video, threshold=threshold)
            scenes_path.write_text(
                json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Scene detection failed:[/] {exc}")
            raise typer.Exit(code=1)

    if not skip_candidates:
        if not scenes:
            console.print("[yellow]No scenes detected; skipping candidate build.[/]")
        else:
            try:
                candidates = build_candidate_segments(
                    transcript=transcript,
                    scenes=scenes,
                )
                write_candidates(candidates, candidates_path)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Candidate build failed:[/] {exc}")
                raise typer.Exit(code=1)

    console.print(f"[green]Run:[/] {run_path}")
    if duration:
        console.print(f"Duration: {duration:.2f}s")
    console.print(f"Probe written to: {probe_path}")
    console.print(f"Audio written to: {audio_path}")
    console.print(f"Transcript written to: {transcript_path}")
    if not skip_scenes:
        console.print(f"Scenes written to: {scenes_path} (count={len(scenes)})")
    if not skip_candidates and scenes:
        console.print(f"Candidates written to: {candidates_path}")
    console.print(f"Segments: {len(transcript.segments)}")

if __name__ == "__main__":
    app()
