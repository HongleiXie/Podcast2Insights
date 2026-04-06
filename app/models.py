from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Engine(str, Enum):
    faster_whisper = "faster_whisper"
    qwen3_asr_1_7b = "qwen3_asr_1_7b"
    mlx_whisper = "mlx_whisper"


class InputSource(str, Enum):
    file_upload = "file_upload"
    podcast_url = "podcast_url"


class UrlSubmission(BaseModel):
    podcast_url: HttpUrl
    engine: Engine


class TranscriptMetadata(BaseModel):
    engine: Engine
    duration_seconds: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    chunk_count: Optional[int] = None
    failure_reason: Optional[str] = None


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    engine: Engine
    input_source: InputSource
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_ref: Optional[str] = None
    transcript_path: Optional[str] = None
    metadata: TranscriptMetadata


class Chunk(BaseModel):
    """A single speaker-turn chunk parsed from a transcript."""

    job_id: str
    index: int
    speaker: str
    text: str
    start_ts: str  # "HH:MM:SS" — first line timestamp in this turn
    end_ts: str    # "HH:MM:SS" — last line timestamp in this turn


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class IndexStatus(BaseModel):
    job_id: str
    # "building" while embedding+indexing, "ready" when searchable,
    # "failed" on error, "not_found" if job unknown
    status: str


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus


class GetJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    engine: Engine
    input_source: InputSource
    created_at: datetime
    metadata: TranscriptMetadata
    text_url: Optional[str] = None
