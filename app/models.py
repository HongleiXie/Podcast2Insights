from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Engine(StrEnum):
    faster_whisper = "faster_whisper"
    qwen3_asr_1_7b = "qwen3_asr_1_7b"
    mlx_whisper = "mlx_whisper"


class InputSource(StrEnum):
    file_upload = "file_upload"
    podcast_url = "podcast_url"


class UrlSubmission(BaseModel):
    podcast_url: HttpUrl
    engine: Engine


class TranscriptMetadata(BaseModel):
    engine: Engine
    duration_seconds: float | None = None
    elapsed_seconds: float | None = None
    chunk_count: int | None = None
    failure_reason: str | None = None


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    engine: Engine
    input_source: InputSource
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_ref: str | None = None
    transcript_path: str | None = None
    metadata: TranscriptMetadata


class Chunk(BaseModel):
    """A single speaker-turn chunk parsed from a transcript."""

    job_id: str
    index: int
    speaker: str
    text: str
    start_ts: str  # "HH:MM:SS" — first line timestamp in this turn
    end_ts: str  # "HH:MM:SS" — last line timestamp in this turn


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
    text_url: str | None = None
