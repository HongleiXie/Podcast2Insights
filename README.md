# Podcast2Insights
Upload a podcast → get a transcript → ask questions about it. A self-hosted, fully local toy version of NotebookLM. English, Chinese, and mixed-language audio supported. Please checkout the [technical report](https://github.com/HongleiXie/Podcast2Insights/blob/main/TECHNICAL_REPORT.md) for details.

## What it does

1. **Transcribe** — upload an audio file or paste a podcast URL. The app transcribes it and produces a timestamped `.txt` file.
2. **Index** — transcript is automatically chunked and embedded into a local FAISS vector index.
3. **Ask** — type any question. The app retrieves the most relevant passages and streams a grounded answer with timestamp citations.

Everything runs locally. No data leaves your machine.

## System requirements

| | Minimum | Recommended |
|---|---|---|
| **OS** | macOS (Apple Silicon) | macOS M3 / M4 Pro |
| **RAM** | 16 GB | 48 GB |
| **Python** | 3.11-3.13 | 3.11-3.13 |
| **Disk** | ~10 GB free | ~15 GB free |

> The recommended spec is for running the full stack (mlx-whisper large-v3 + bge-m3 + Qwen3-8B) comfortably without swapping. The minimum spec can run smaller model variants.

## Dependencies

Install these before running:

- [uv](https://docs.astral.sh/uv/) — Python package manager
- [ffmpeg](https://ffmpeg.org/) — audio processing (`brew install ffmpeg`)
- [Ollama](https://ollama.com/) — local LLM server (`brew install ollama`)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — for non-direct podcast URLs (optional)

## First-time setup

```bash
# 1. Pull the LLM (once, ~5 GB download)
ollama pull qwen3:8b

# 2. Install Python dependencies
uv sync
```

## Running locally

```bash
# Terminal 1 — LLM server
ollama serve

# Terminal 2 — app
uv run uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

## Usage

1. Select an ASR engine (`mlx_whisper` recommended on Apple Silicon).
2. Upload an audio file or paste a podcast URL.
3. Click **Transcribe** and wait for the job to complete.
4. Once the index is ready (shown by a status pill), type a question and click **Ask**.
5. The answer streams in with `[HH:MM:SS]` timestamp citations.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/transcriptions` | Create a transcription job (file upload or JSON URL) |
| `GET` | `/transcriptions/{job_id}` | Poll job status and metadata |
| `GET` | `/transcriptions/{job_id}/text` | Download completed transcript |
| `GET` | `/transcriptions/{job_id}/index-status` | Poll RAG index status (`building` / `ready` / `failed`) |
| `POST` | `/transcriptions/{job_id}/query` | Stream a grounded answer (SSE) |

## Configuration

Key environment variables (all optional):

| Variable | Default | Description |
|---|---|---|
| `MLX_WHISPER_MODEL` | `mlx-community/whisper-large-v3-mlx` | ASR model for mlx_whisper |
| `EMBED_MODEL` | `BAAI/bge-m3` | Sentence embedding model |
| `EMBED_DEVICE` | `mps` | Embedding device (`mps` / `cpu`) |
| `OLLAMA_MODEL` | `qwen3:8b` | LLM served by Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API base URL |
| `OLLAMA_THINKING` | `false` | Enable Qwen3 chain-of-thought thinking mode |
| `TOP_K` | `5` | Number of chunks retrieved per query |

## Development

```bash
uv run pytest -q
```
