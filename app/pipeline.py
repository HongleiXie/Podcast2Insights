from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .asr import create_engine
from .audio_utils import chunk_audio, dedupe_overlap, normalize_to_wav, probe_duration_seconds
from .chunker import parse_transcript
from .config import AUDIO_DIR, MAX_UPLOAD_BYTES, OUTPUT_DIR, TMP_DIR
from .embedder import embed_chunks
from .indexer import build_index
from .models import JobStatus
from .session_store import SessionState, session_store
from .store import JobStore


class TranscriptionWorker:
    def __init__(self, store: JobStore) -> None:
        self.store = store
        self.q: queue.Queue[str] = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def enqueue(self, job_id: str) -> None:
        self.q.put(job_id)

    def _run(self) -> None:
        while True:
            job_id = self.q.get()
            try:
                self._process(job_id)
            except Exception as exc:
                self.store.set_status(job_id, JobStatus.failed, failure_reason=str(exc))
            finally:
                self.q.task_done()

    def _process(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return

        started = time.time()
        self.store.set_status(job_id, JobStatus.running)

        source_audio = AUDIO_DIR / f"{job_id}_source"
        if job.input_source.value == "podcast_url":
            downloaded = download_from_url(job.source_ref or "", job_id)
            source_audio = downloaded
        else:
            possible = list(AUDIO_DIR.glob(f"{job_id}_source*"))
            if possible:
                source_audio = possible[0]
            if not source_audio.exists():
                raise FileNotFoundError("Uploaded audio not found")

        normalized_wav = normalize_to_wav(source_audio, job_id)
        duration = probe_duration_seconds(normalized_wav)
        chunks = chunk_audio(normalized_wav, job_id)

        engine = create_engine(job.engine)
        temp_txt = TMP_DIR / f"{job_id}.partial.txt"
        temp_txt.write_text("", encoding="utf-8")

        transcript_text = ""
        for chunk_path, start_sec, end_sec in chunks:
            chunk_text = engine.transcribe_chunk(
                chunk_path,
                {
                    "task": "transcribe",
                    "chunk_duration_seconds": max(0.0, end_sec - start_sec),
                    "chunk_start_seconds": start_sec,
                },
            )
            merged = dedupe_overlap(transcript_text, chunk_text)
            if merged:
                transcript_text = (transcript_text + "\n" + merged).strip()
                temp_txt.write_text(transcript_text + "\n", encoding="utf-8")

        output_txt = OUTPUT_DIR / f"{job_id}.txt"
        output_txt.write_text(transcript_text + "\n", encoding="utf-8")

        job = self.store.get(job_id)
        if job is None:
            return

        job.status = JobStatus.completed
        job.transcript_path = str(output_txt)
        job.metadata.duration_seconds = round(duration, 3)
        job.metadata.elapsed_seconds = round(time.time() - started, 3)
        job.metadata.chunk_count = len(chunks)
        self.store.save(job)

        # ── Auto-index for Q&A ────────────────────────────────────────────
        # Runs in the same worker thread immediately after transcription.
        # Errors here are isolated — they don't affect the transcript status.
        self._build_rag_index(job_id, output_txt)

    def _build_rag_index(self, job_id: str, transcript_path: Path) -> None:
        session_store.set(job_id, SessionState(status="building"))
        try:
            rag_chunks = parse_transcript(transcript_path, job_id)
            if not rag_chunks:
                # Empty transcript — mark ready with no chunks (queries will
                # return a graceful "no content" response)
                session_store.set(job_id, SessionState(status="ready"))
                return
            vectors = embed_chunks(rag_chunks)
            index = build_index(vectors)
            session_store.set(
                job_id,
                SessionState(status="ready", chunks=rag_chunks, index=index),
            )
        except Exception as exc:
            session_store.set(
                job_id,
                SessionState(status="failed", error=str(exc)),
            )


def download_from_url(url: str, job_id: str) -> Path:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL")

    if parsed.path.lower().endswith(".mp3"):
        out = AUDIO_DIR / f"{job_id}_source.mp3"
        try:
            r = requests.get(url, timeout=30, stream=True)
            r.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Could not download audio URL: {exc}") from exc
        total = 0
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise RuntimeError("URL audio exceeds 100MB max size")
                f.write(chunk)
        return out

    if not shutil.which("yt-dlp"):
        raise RuntimeError("yt-dlp is required for non-direct mp3 URLs")

    out_tmpl = AUDIO_DIR / f"{job_id}_source.%(ext)s"
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "mp3",
        "-o",
        str(out_tmpl),
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "yt-dlp failed")

    candidates = sorted(AUDIO_DIR.glob(f"{job_id}_source.*"))
    if not candidates:
        raise RuntimeError("No audio produced by yt-dlp")
    if candidates[0].stat().st_size > MAX_UPLOAD_BYTES:
        raise RuntimeError("URL audio exceeds 100MB max size")
    return candidates[0]
