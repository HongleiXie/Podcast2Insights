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
4. Add to your .env file:  HF_TOKEN=hf_...

If HF_TOKEN is not set, diarisation is skipped and all segments fall back
to "Speaker A".  A WARNING is logged so the skip is always visible in the
uvicorn console.

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
    """Load and cache the pyannote diarisation pipeline.

    Raises RuntimeError with a clear message on any failure so the caller
    can surface it in the job log rather than silently falling back.
    """
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed. Run: uv add pyannote.audio"
        ) from exc

    try:
        # huggingface_hub ≥ 0.17 prefers `token` over `use_auth_token`.
        # Pass via login() so both old and new hub versions are happy.
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load pyannote pipeline: {exc}\n"
            "Common causes:\n"
            "  • HF_TOKEN is invalid or expired\n"
            "  • You haven't accepted the model conditions at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  • No internet connection on first load (model not cached yet)"
        ) from exc

    # Move to MPS if available, else stay on CPU
    device = "cpu"
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass

    logger.info("Loaded pyannote diarisation pipeline on device=%s", device)
    return pipeline.to(__import__("torch").device(device))


def diarize(audio_path: Path) -> list[tuple[float, float, str]]:
    """Return a speaker timeline for *audio_path*.

    Each entry is (start_seconds, end_seconds, "Speaker X") where X is a
    letter assigned in order of first appearance (A, B, C, …).

    Returns an empty list if diarisation is unavailable or fails, and always
    logs a WARNING so the reason is visible in the uvicorn console.
    """
    if not HF_TOKEN:
        logger.warning(
            "Speaker diarisation SKIPPED — HF_TOKEN is not set. "
            "All speakers will be labelled 'Speaker A'. "
            "See app/diarizer.py for setup instructions."
        )
        return []

    logger.info("Starting speaker diarisation for %s …", audio_path.name)

    try:
        pipeline = _get_pipeline()
        annotation = pipeline(str(audio_path))
    except Exception as exc:
        logger.warning(
            "Speaker diarisation FAILED — falling back to single speaker. "
            "Reason: %s",
            exc,
        )
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

    logger.info(
        "Diarisation complete: %d speaker segments, %d unique speakers",
        len(timeline),
        len(speaker_map),
    )
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
