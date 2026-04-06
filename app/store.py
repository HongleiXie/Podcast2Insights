from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import JOBS_DIR
from .models import JobRecord, JobStatus


class JobStore:
    def __init__(self, jobs_dir: Path = JOBS_DIR) -> None:
        self.jobs_dir = jobs_dir

    def _path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def save(self, job: JobRecord) -> None:
        job.updated_at = datetime.now(timezone.utc)
        self._path(job.job_id).write_text(job.model_dump_json(indent=2), encoding="utf-8")

    def get(self, job_id: str) -> Optional[JobRecord]:
        path = self._path(job_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return JobRecord.model_validate(data)

    def set_status(self, job_id: str, status: JobStatus, *, failure_reason: Optional[str] = None) -> JobRecord:
        job = self.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job_id: {job_id}")
        job.status = status
        if failure_reason:
            job.metadata.failure_reason = failure_reason
        self.save(job)
        return job
