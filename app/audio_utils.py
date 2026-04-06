from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

from .config import CHUNK_SECONDS, OVERLAP_SECONDS, TMP_DIR


class AudioError(RuntimeError):
    pass


def _run(cmd: list[str]) -> None:
    if not shutil.which(cmd[0]):
        raise AudioError(f"Required command '{cmd[0]}' is not installed")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(proc.stderr.strip() or f"Command failed: {' '.join(cmd)}")


def probe_duration_seconds(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    if not shutil.which("ffprobe"):
        raise AudioError("ffprobe is required but not installed")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(proc.stderr.strip() or "Could not probe duration")
    return float(proc.stdout.strip())


def normalize_to_wav(input_path: Path, job_id: str) -> Path:
    out = TMP_DIR / f"{job_id}_normalized.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(out),
    ]
    _run(cmd)
    return out


def chunk_audio(input_wav: Path, job_id: str, chunk_seconds: int = CHUNK_SECONDS, overlap_seconds: int = OVERLAP_SECONDS) -> List[Tuple[Path, float, float]]:
    duration = probe_duration_seconds(input_wav)
    chunks: List[Tuple[Path, float, float]] = []

    if duration <= float(chunk_seconds):
        return [(input_wav, 0.0, duration)]

    start = 0.0
    idx = 0
    step = max(1, chunk_seconds - overlap_seconds)

    while start < duration:
        end = min(duration, start + chunk_seconds)
        output = TMP_DIR / f"{job_id}_chunk_{idx:05d}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output),
        ]
        _run(cmd)
        chunks.append((output, start, end))
        start += step
        idx += 1

    return chunks


def dedupe_overlap(previous_text: str, new_text: str, max_overlap_chars: int = 500) -> str:
    prev = previous_text.strip()
    nxt = new_text.strip()
    if not prev:
        return nxt
    if not nxt:
        return ""

    tail = prev[-max_overlap_chars:]
    max_len = min(len(tail), len(nxt))

    for k in range(max_len, 20, -1):
        if tail[-k:] == nxt[:k]:
            return nxt[k:].lstrip()
    return nxt
