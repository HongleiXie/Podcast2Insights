# Podcast2Insights (Demo)

Turn English/Chinese (including mixed) podcast audio into a plain UTF-8 `.txt` transcript.

## What this demo does

- Accepts audio from either:
  - file upload (max 100MB)
  - podcast URL
- Runs async transcription jobs so long audio does not timeout.
- Supports two engines:
  - `faster_whisper`
  - `qwen3_asr_1_7b`
- Normalizes audio to mono 16k WAV, chunks it, transcribes, and merges chunk text.
- Produces `{job_id}.txt` output.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe`
- `yt-dlp` (needed for non-direct MP3 URLs)

## Quick start (uv)

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Open: [http://localhost:8000](http://localhost:8000)

## Use from the web page

1. Choose an engine.
2. Provide either a file or a podcast URL (not both).
3. Submit.
4. Wait for status to become `completed`.
5. Download the `.txt` transcript.

## API

### `POST /transcriptions`

Create a transcription job.

- File mode (`multipart/form-data`):
  - `file`: audio/video file
  - `engine`: `faster_whisper` or `qwen3_asr_1_7b`
- URL mode (`application/json`):

```json
{
  "podcast_url": "https://example.com/episode.mp3",
  "engine": "faster_whisper"
}
```

Response:

```json
{
  "job_id": "<id>",
  "status": "queued"
}
```

### `GET /transcriptions/{job_id}`

Check job status (`queued | running | completed | failed`) and metadata.

### `GET /transcriptions/{job_id}/text`

Download transcript as UTF-8 plain text once completed.

## Development

Run tests:

```bash
uv run pytest -q
```

## Notes

- This is a single-machine, single-worker FIFO demo.
- Output lines include timestamps and speaker labels (`[HH:MM:SS] Speaker X: ...`).
- Transcription is `transcribe` mode (no translation), so mixed language is preserved.
- `faster_whisper` defaults are tuned for CPU latency: model `small`, `compute_type=int8`, and `beam_size=1`.
- You can tune these with env vars: `FASTER_WHISPER_MODEL_NAME`, `FASTER_WHISPER_DEVICE`, `FASTER_WHISPER_COMPUTE_TYPE`, `FASTER_WHISPER_BEAM_SIZE`.
- For `qwen3_asr_1_7b`, the app uses the official `qwen-asr` package runtime.
- You can tune Qwen with env vars:
  - `QWEN3_ASR_MODEL_NAME` (default `Qwen/Qwen3-ASR-1.7B`)
  - `QWEN3_ASR_DEVICE` (`auto|cuda|mps|cpu`, default `auto`, where `auto` prefers CUDA then CPU)
  - `QWEN3_ASR_DTYPE` (`auto|float16|bfloat16|float32`, default `auto`)
  - `QWEN3_ASR_MAX_NEW_TOKENS` (default `128`)
  - `QWEN3_ASR_TOKENS_PER_SECOND` (default `4.0`, used to adapt decode length per chunk)
- On CPU, `qwen3_asr_1_7b` can still be slow even for short audio; use a GPU when possible.
