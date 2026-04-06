"""In-memory store that holds the FAISS index and chunks for each job.

Lifecycle per job_id
---------------------
1. pipeline sets status="building"  (transcription just completed)
2. pipeline embeds chunks, builds FAISS index
3. pipeline sets status="ready"     (Q&A endpoint now usable)
4. On any error: status="failed"

Thread-safety
-------------
A threading.Lock guards all reads and writes because the worker thread
(pipeline) writes state while the FastAPI event loop reads it.

Persistence
-----------
Intentionally in-memory only.  FAISS indexes are fast to rebuild (~seconds
for a typical podcast) and storing them on disk adds complexity with no real
benefit for a single-user demo.  On server restart users simply re-upload.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import faiss

from .models import Chunk


@dataclass
class SessionState:
    status: str = "building"  # "building" | "ready" | "failed"
    chunks: list[Chunk] = field(default_factory=list)
    index: faiss.IndexFlatIP | None = None
    error: str | None = None


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def set(self, job_id: str, state: SessionState) -> None:
        with self._lock:
            self._sessions[job_id] = state

    def get(self, job_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(job_id)


# Module-level singleton — imported by pipeline.py and main.py
session_store = SessionStore()
