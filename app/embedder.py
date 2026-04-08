"""Embedding layer using BAAI/bge-m3 via sentence-transformers.

bge-m3 characteristics
-----------------------
* 1024-dimensional dense vectors
* 8192-token context window — comfortably fits large speaker-turn chunks
* Native multilingual support: handles English, Chinese, and mixed text equally
* Must use normalized_embeddings=True for cosine similarity via dot product
  (required by FAISS IndexFlatIP)

Device selection
----------------
Default is "mps" (Metal on Apple Silicon).  Falls back to "cpu" if MPS is
unavailable — useful when developing on non-Apple hardware.
"""

from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBED_DEVICE, EMBED_MODEL
from .models import Chunk

# bge-m3 output dimension — used by indexer to initialise FAISS
EMBED_DIM = 1024


def _resolve_device(configured: str) -> str:
    """Return a concrete device string, falling back to cpu if MPS unavailable."""
    if configured.lower() != "mps":
        return configured
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load and cache the bge-m3 model (downloaded once to ~/.cache/huggingface)."""
    device = _resolve_device(EMBED_DEVICE)
    return SentenceTransformer(EMBED_MODEL, device=device)


def embed_chunks(chunks: list[Chunk]) -> np.ndarray:
    """Embed a list of Chunk objects.

    Returns a float32 array of shape (len(chunks), EMBED_DIM) with
    L2-normalised rows ready for dot-product similarity search.
    """
    model = get_embedder()
    texts = [f"{chunk.speaker}: {chunk.text}" for chunk in chunks]
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return np.array(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string.

    Returns a float32 array of shape (1, EMBED_DIM), normalised.
    bge-m3 recommends prepending 'Represent this sentence: ' for queries,
    but performs well without it for conversational queries.
    """
    model = get_embedder()
    vector = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(vector, dtype=np.float32)
