# Engagement Clip Creator (ECC) / Brainrot Clip Creator (BCC)

## Overview
ECC converts long-form videos (≤3 hours) into 30–45s short-form clips for Shorts/Reels. Two modes:
- Mode A — Plot-driven recap teaser: given a plot/article, build a narrated montage.
- Mode B — Auto-highlight trailer: find engaging moments automatically and render a trailer-style clip.

Primary goals: end-to-end automation (ingest → analyze → select → narrate → render) with repeatable outputs and configurable style knobs. Start with CLI MVP, optional web wrapper later.

## Inputs / Outputs
- Inputs: long-form video (mp4/mov, up to ~3h); for Mode A also plot/article text (md/txt).
- Outputs: final 30–45s mp4; optional SRT/VTT captions; intermediate artifacts (metadata, transcript, scenes, candidates, EDL).

## Requirements
- Mode A: creator supplies plot/article; ECC selects matching moments and narrates a teaser.
- Mode B: creator lets ECC auto-find engaging moments and generate trailer-style narration.

## Architecture (MVP)
Python pipeline orchestrating media tools:
- Ingest: probe metadata; extract audio (16k mono wav); optional proxy video.
- ASR: transcript with timestamps.
- Segmentation: scene/shot detection; build candidate segments from transcript + scenes.
- Selection:
  - Mode A: article → beat sheet → retrieve matching candidates.
  - Mode B: score candidates for engagement and diversity.
- Planning: convert selections into an EDL with pacing constraints.
- Narration & TTS: generate narration script; synthesize voiceover.
- Rendering (FFmpeg): cut, concat, mix audio (duck + normalize), captions, export 9:16.
- Storage: per-run artifacts under `runs/<run_id>/...` using JSON for metadata.

## Tech Stack
- Python 3.11+
- FFmpeg/ffprobe; faster-whisper or openai-whisper; PySceneDetect; sentence-transformers + faiss-cpu; pysubs2 or srt; pydantic; typer.
- Pluggable providers: LLM for beat sheet/narration; TTS provider for voiceover.

## Data Models (Artifacts)
- Transcript segment: `start_ms`, `end_ms`, `text`
- Scene segment: `start_ms`, `end_ms`, `scene_id`
- Candidate segment: `candidate_id`, `start_ms`, `end_ms`, `scene_id`, `text`, `features?`
- Beat (Mode A): `beat_id`, `title`, `summary`, `keywords?`
- EDL clip: `clip_id`, `candidate_id`, `in_ms`, `out_ms`, `order`, `caption?`

## Clip Selection Logic
- Mode A: article → beats → retrieve top-k candidates; rerank for punchy dialogue, emotion markers, diversity, spoiler cutoff; pick 8–15 micro-clips totaling 30–45s.
- Mode B: score candidates via audio energy, excitement heuristics, diversity, spoiler cutoff; pick top under pacing/diversity constraints with hook-first ordering.

## Narration & Captions
- Mode A: narrate beats (spoiler-safe).
- Mode B: trailer-style setup + stakes + tease.
- Captions: at least narration; optional dialogue captions burned in.

## Rendering Pipeline
- Cut and concat with consistent encoding.
- Audio mix: duck original under narration; normalize loudness.
- Export vertical 9:16 with safe margins; burn-in captions or attach sidecar SRT.

## Goals
-  Local prototype, Mode A end-to-end (basic retrieval).
-  Quality jump (scene detection, better candidates, pacing, audio ducking, captions).
- : Mode B scoring and trailer narration.
-  Optional web wrapper (upload → background job → download).

## Repo Layout 
```
ecc/
  app/cli.py
  ingest/media_probe.py, extract_audio.py, make_proxy.py
  asr/transcribe.py, transcript_schema.py
  segmentation/scene_detect.py, build_candidates.py
  retrieval/embed_store.py, faiss_index.py, match_beats.py
  moments/features.py, score.py, select.py
  planning/edl.py, pacing.py
  narration/beat_sheet.py, script_writer.py, tts.py
  render/cut.py, concat.py, audio_mix.py, captions.py, export.py
  schemas/config.py, job.py
runs/<run_id>/...
```

## Task Breakdown
-  scaffolding/tooling, config/schemas.
-  ingest (probe, extract audio, proxy).
-  ASR + cleanup.
-  scene detection + candidates.
-  Mode A beats + retrieval.
-  Mode B scoring/selection.
- : EDL + pacing.
-  narration + TTS.
-  rendering.
-  CLI orchestration.
-  optional web wrapper.

## End product capability
Given a 30–180m video and a plot/article, produce a 30–45s mp4 with stitched clips matching beats, audible narration, sane audio levels, optional captions. Mode B produces a 30–45s mp4 of energetic moments with narration.

## Stretch capabilities
 CLI-driven Mode A end-to-end; improve segmentation/pacing; add Mode B; wrap with web UI only after outputs are strong.
