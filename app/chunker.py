"""Parse a transcript .txt file into speaker-turn chunks for RAG indexing.

Expected transcript line format (produced by all ASR engines in this project):
    [HH:MM:SS] Speaker X: transcribed text here

Strategy
--------
* Consecutive lines from the **same speaker** are merged into one chunk.
* If a single speaker run would exceed MAX_CHARS_PER_CHUNK, it is split so
  retrieval stays focused (avoids embedding huge walls of text).
* start_ts / end_ts carry the HH:MM:SS timestamps of the first and last line
  in each chunk — used for citation in Q&A answers.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .models import Chunk

# Matches: [HH:MM:SS] Any Speaker Label: transcript text
_LINE_RE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s+(.+?):\s+(.+)$")

# ~300 tokens for bge-m3; 1 token ≈ 4 chars (conservative for CJK mixed text)
MAX_CHARS_PER_CHUNK = 1200


def _flush(
    job_id: str,
    index: int,
    speaker: str,
    start_ts: str,
    end_ts: str,
    texts: list[str],
) -> Optional[Chunk]:
    text = " ".join(texts).strip()
    if not text:
        return None
    return Chunk(
        job_id=job_id,
        index=index,
        speaker=speaker,
        text=text,
        start_ts=start_ts,
        end_ts=end_ts,
    )


def parse_transcript(txt_path: Path, job_id: str) -> list[Chunk]:
    """Return a list of Chunk objects parsed from *txt_path*.

    Returns an empty list if the file is empty or contains no parseable lines.
    """
    raw = txt_path.read_text(encoding="utf-8").splitlines()

    # Parse every recognised line into (timestamp, speaker, text)
    parsed: list[tuple[str, str, str]] = []
    for line in raw:
        m = _LINE_RE.match(line.strip())
        if m:
            parsed.append((m.group(1), m.group(2).strip(), m.group(3).strip()))

    if not parsed:
        return []

    chunks: list[Chunk] = []
    cur_speaker = parsed[0][1]
    cur_start = parsed[0][0]
    cur_end = parsed[0][0]
    cur_texts: list[str] = []

    for ts, speaker, text in parsed:
        same_speaker = speaker == cur_speaker
        would_overflow = sum(len(t) for t in cur_texts) + len(text) > MAX_CHARS_PER_CHUNK

        if not same_speaker or would_overflow:
            chunk = _flush(job_id, len(chunks), cur_speaker, cur_start, cur_end, cur_texts)
            if chunk:
                chunks.append(chunk)
            cur_speaker = speaker
            cur_start = ts
            cur_texts = [text]
        else:
            cur_texts.append(text)

        cur_end = ts

    # Flush the final accumulated group
    chunk = _flush(job_id, len(chunks), cur_speaker, cur_start, cur_end, cur_texts)
    if chunk:
        chunks.append(chunk)

    return chunks
