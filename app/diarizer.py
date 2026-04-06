"""Speaker diarisation using pyannote.audio.

What this does
--------------
Runs pyannote's speaker-diarization-3.1 pipeline on a full normalised WAV
file and returns a timeline of (start_sec, end_sec, speaker_label) tuples.
speaker_at() then maps any transcript segment timestamp to the right label.

Setup required (one-time)
--------------------------
1. Create a free account at https://huggingface.co
2. Accept the model conditions at:
   https://huggingface.co/pyannote/speaker-diarization-3.1
3. Generate an access token at https://huggingface.co/settings/tokens
4. Set the env variable:  HF_TOKEN=hf_...

If HF_TOKEN is not set, diarisation is silently skipped and all segments
fall back to "Speaker A".

Device
------
The pipeline is moved to MPS (Apple Silicon GPU) if available, otherwise CPU.
Diarisation of a 1-hour podcast takes roughly 2-5 min on an M-series chip.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from .config import HF_TOKEN

logger = logging.getLogger(__name__)

# Ordered list of labels assigned as new speakers appear
_LABEL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@lru_cache(maxsize=1)
def _get_pipeline():
    """Load and cache the pyannote diarisation pipeline."""
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Run: uv add pyannote.audio"
        ) from exc

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )

    # Move to MPS if available, else stay on CPU
    device = "cpu"
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass

    return pipeline.to(__import__("torch").device(device))


def diarize(audio_path: Path) -> list[tuple[float, float, str]]:
    """Return a speaker timeline for *audio_path*.

    Each entry is (start_seconds, end_seconds, "Speaker X") where X is a
    letter assigned in order of first appearance (A, B, C, …).

    Returns an empty list if diarisation is unavailable or fails.
    """
    if not HF_TOKEN:
        logger.debug("HF_TOKEN not set — skipping diarisation")
        return []

    try:
        pipeline = _get_pipeline()
        annotation = pipeline(str(audio_path))
    except Exception as exc:
        logger.warning("Diarisation failed, falling back to single speaker: %s", exc)
        return []

    # Normalise pyannote's raw labels (e.g. "SPEAKER_00") to "Speaker A", etc.
    speaker_map: dict[str, str] = {}

    def normalise(raw: str) -> str:
        if raw not in speaker_map:
            idx = len(speaker_map)
            letter = _LABEL_CHARS[idx] if idx < len(_LABEL_CHARS) else str(idx)
            speaker_map[raw] = f"Speaker {letter}"
        return speaker_map[raw]

    timeline: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        timeline.append((turn.start, turn.end, normalise(speaker)))

    return timeline


def speaker_at(
    timeline: list[tuple[float, float, str]],
    timestamp: float,
    default: str = "Speaker A",
) -> str:
    """Return the speaker label active at *timestamp*, or *default*."""
    for start, end, label in timeline:
        if start <= timestamp <= end:
            return label
    return default
