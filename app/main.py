from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import AUDIO_DIR, DIARIZE, HF_TOKEN, MAX_UPLOAD_BYTES, OLLAMA_MODEL
from .indexer import search
from .models import (
    CreateJobResponse,
    Engine,
    GetJobResponse,
    IndexStatus,
    InputSource,
    JobRecord,
    JobStatus,
    QueryRequest,
    TranscriptMetadata,
    UrlSubmission,
)
from .pipeline import TranscriptionWorker
from .qa import stream_answer
from .session_store import session_store
from .store import JobStore

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".webm", ".ogg"}
ALLOWED_MIME_PREFIX = ("audio/", "video/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log the effective config once at startup so issues are immediately visible.
    token_status = f"set ({HF_TOKEN[:4]}…)" if HF_TOKEN else "NOT SET"
    diarize_status = (
        "enabled"
        if (DIARIZE and HF_TOKEN)
        else ("disabled (HF_TOKEN missing)" if DIARIZE else "disabled (DIARIZE=false)")
    )
    logger.info("── Podcast2Insights startup ──────────────────────")
    logger.info("  HF_TOKEN     : %s", token_status)
    logger.info("  Diarisation  : %s", diarize_status)
    logger.info("  LLM model    : %s", OLLAMA_MODEL)
    logger.info("──────────────────────────────────────────────────")
    yield


app = FastAPI(title="Podcast2Insights Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

store = JobStore()
worker = TranscriptionWorker(store)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/transcriptions", response_model=CreateJobResponse)
async def create_transcription(
    request: Request,
    file: UploadFile | None = File(default=None),
    engine_form: str | None = Form(default=None, alias="engine"),
) -> CreateJobResponse:
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        if not file:
            raise HTTPException(status_code=400, detail="Missing file upload")
        if not engine_form:
            raise HTTPException(status_code=400, detail="Missing engine")

        try:
            engine = Engine(engine_form)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid engine") from exc

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported file extension")
        if file.content_type and not file.content_type.startswith(ALLOWED_MIME_PREFIX):
            raise HTTPException(status_code=400, detail="Unsupported mime type")

        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds 100MB max size")

        job_id = uuid.uuid4().hex
        audio_path = AUDIO_DIR / f"{job_id}_source{suffix}"
        audio_path.write_bytes(data)

        job = JobRecord(
            job_id=job_id,
            status=JobStatus.queued,
            engine=engine,
            input_source=InputSource.file_upload,
            source_ref=str(audio_path),
            metadata=TranscriptMetadata(engine=engine),
        )
        store.save(job)
        worker.enqueue(job_id)
        return CreateJobResponse(job_id=job_id, status=job.status)

    if "application/json" in content_type:
        try:
            payload = UrlSubmission.model_validate(await request.json())
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc
        try:
            probe = requests.get(str(payload.podcast_url), timeout=10, stream=True)
            probe.raise_for_status()
            probe.close()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unreachable URL: {exc}") from exc

        job_id = uuid.uuid4().hex
        job = JobRecord(
            job_id=job_id,
            status=JobStatus.queued,
            engine=payload.engine,
            input_source=InputSource.podcast_url,
            source_ref=str(payload.podcast_url),
            metadata=TranscriptMetadata(engine=payload.engine),
        )
        store.save(job)
        worker.enqueue(job_id)
        return CreateJobResponse(job_id=job_id, status=job.status)

    raise HTTPException(
        status_code=415,
        detail="Use multipart/form-data (file) or application/json (podcast_url)",
    )


@app.get("/transcriptions/{job_id}", response_model=GetJobResponse)
def get_transcription(job_id: str) -> GetJobResponse:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    text_url = None
    if job.status == JobStatus.completed:
        text_url = f"/transcriptions/{job_id}/text"

    return GetJobResponse(
        job_id=job.job_id,
        status=job.status,
        engine=job.engine,
        input_source=job.input_source,
        created_at=job.created_at,
        metadata=job.metadata,
        text_url=text_url,
    )


@app.get("/transcriptions/{job_id}/index-status", response_model=IndexStatus)
def get_index_status(job_id: str) -> IndexStatus:
    """Poll this after a job completes to know when Q&A is ready."""
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    state = session_store.get(job_id)
    if state is None:
        # Transcription may still be running (index hasn't been kicked off yet)
        status = "building" if job.status == JobStatus.completed else "building"
        return IndexStatus(job_id=job_id, status=status)

    return IndexStatus(job_id=job_id, status=state.status)


@app.post("/transcriptions/{job_id}/query")
async def query_podcast(job_id: str, body: QueryRequest) -> StreamingResponse:
    """Stream a grounded answer to *body.question* over SSE.

    The frontend reads the stream with the Fetch Streams API and appends
    tokens to the answer box in real time.  Each event is:

        data: {"token": "..."}\n\n

    A final sentinel ``data: [DONE]\\n\\n`` signals end-of-stream.
    """
    state = session_store.get(job_id)
    if state is None or state.status != "ready":
        detail = "Index not ready" if state else "Job not found or not indexed"
        raise HTTPException(status_code=409, detail=detail)

    if not state.chunks:
        raise HTTPException(status_code=409, detail="No transcript content to search")

    from .embedder import embed_query

    query_vec = embed_query(body.question)
    retrieved = search(state.index, query_vec, state.chunks, k=body.top_k)

    return StreamingResponse(
        stream_answer(body.question, retrieved),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )


@app.get("/transcriptions/{job_id}/text")
def get_transcription_text(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.completed or not job.transcript_path:
        raise HTTPException(status_code=409, detail="Transcript not ready")

    path = Path(job.transcript_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript file missing")

    return FileResponse(path, media_type="text/plain; charset=utf-8", filename=f"{job_id}.txt")
