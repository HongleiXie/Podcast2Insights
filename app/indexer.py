"""FAISS vector index: build and search.

Index choice: IndexFlatIP (exact inner-product search)
------------------------------------------------------
* "Flat" = no quantisation, no approximation — exact nearest-neighbour.
* Inner-product on L2-normalised vectors is equivalent to cosine similarity.
* At podcast scale (hundreds to low-thousands of chunks) this is fast enough
  and avoids the tuning overhead of IVF/HNSW approximate indexes.
"""

from __future__ import annotations

import faiss
import numpy as np

from .embedder import EMBED_DIM
from .models import Chunk


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Create and populate a FAISS IndexFlatIP from a (N, EMBED_DIM) array."""
    if vectors.ndim != 2 or vectors.shape[1] != EMBED_DIM:
        raise ValueError(f"Expected vectors of shape (N, {EMBED_DIM}), got {vectors.shape}")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors.astype(np.float32))
    return index


def search(
    index: faiss.IndexFlatIP,
    query_vec: np.ndarray,
    chunks: list[Chunk],
    k: int = 5,
) -> list[Chunk]:
    """Return the top-k most relevant Chunk objects for *query_vec*.

    *query_vec* must be L2-normalised (shape (1, EMBED_DIM)).
    Results are ordered by descending cosine similarity score.
    """
    k = min(k, index.ntotal)
    if k == 0:
        return []

    _scores, indices = index.search(query_vec.astype(np.float32), k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
