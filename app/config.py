import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root before any os.getenv() call.
# This makes env vars available regardless of how the app is launched
# (VSCode, terminal, uvicorn directly, etc.).
# Shell-level exports take precedence over .env values (override=False default).
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
AUDIO_DIR = DATA_DIR / "audio"
OUTPUT_DIR = DATA_DIR / "output"
TMP_DIR = DATA_DIR / "tmp"

for path in (DATA_DIR, JOBS_DIR, AUDIO_DIR, OUTPUT_DIR, TMP_DIR):
    path.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
CHUNK_SECONDS = 90
OVERLAP_SECONDS = 2

FASTER_WHISPER_MODEL_NAME = os.getenv("FASTER_WHISPER_MODEL_NAME", "small")
FASTER_WHISPER_DEVICE = os.getenv("FASTER_WHISPER_DEVICE", "auto")
FASTER_WHISPER_COMPUTE_TYPE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8")
FASTER_WHISPER_BEAM_SIZE = int(os.getenv("FASTER_WHISPER_BEAM_SIZE", "1"))

QWEN3_ASR_MODEL_NAME = os.getenv("QWEN3_ASR_MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
QWEN3_ASR_DEVICE = os.getenv("QWEN3_ASR_DEVICE", "auto")
QWEN3_ASR_DTYPE = os.getenv("QWEN3_ASR_DTYPE", "auto")
QWEN3_ASR_MAX_NEW_TOKENS = int(os.getenv("QWEN3_ASR_MAX_NEW_TOKENS", "1024"))

# MLX Whisper (Apple Silicon, Metal-accelerated)
MLX_WHISPER_MODEL = os.getenv("MLX_WHISPER_MODEL", "mlx-community/whisper-large-v3-mlx")

# Speaker diarisation via pyannote.audio
# Requires a HuggingFace token — see app/diarizer.py for setup instructions.
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
# Set DIARIZE=false to disable even when HF_TOKEN is present
DIARIZE: bool = os.getenv("DIARIZE", "true").lower() == "true"

# Embedding model (bge-m3 — multilingual, 1024-dim)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "mps")  # mps | cpu

# Ollama / LLM
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
# Set True to enable Qwen3 chain-of-thought thinking mode (slower but more thorough)
OLLAMA_THINKING = os.getenv("OLLAMA_THINKING", "false").lower() == "true"

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
