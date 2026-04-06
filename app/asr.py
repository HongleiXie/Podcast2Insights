from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Any

from .config import (
    FASTER_WHISPER_BEAM_SIZE,
    FASTER_WHISPER_COMPUTE_TYPE,
    FASTER_WHISPER_DEVICE,
    FASTER_WHISPER_MODEL_NAME,
    MLX_WHISPER_MODEL,
    QWEN3_ASR_DEVICE,
    QWEN3_ASR_DTYPE,
    QWEN3_ASR_MAX_NEW_TOKENS,
    QWEN3_ASR_MODEL_NAME,
)
from .models import Engine


class ASRError(RuntimeError):
    pass


class ASREngine(ABC):
    @abstractmethod
    def transcribe_chunk(self, audio_path: Path, options: dict[str, Any]) -> str:
        raise NotImplementedError


def _format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _speaker_label(raw_speaker: Any) -> str:
    text = str(raw_speaker or "").strip()
    if not text:
        return "Speaker A"
    if text.lower().startswith("speaker "):
        return text
    return f"Speaker {text}"


def _format_line(timestamp_seconds: float, speaker: Any, text: str) -> str:
    clean = text.strip()
    if not clean:
        return ""
    return f"[{_format_timestamp(timestamp_seconds)}] {_speaker_label(speaker)}: {clean}"


class FasterWhisperEngine(ASREngine):
    def __init__(
        self,
        model_name: str = FASTER_WHISPER_MODEL_NAME,
        *,
        device: str = FASTER_WHISPER_DEVICE,
        compute_type: str = FASTER_WHISPER_COMPUTE_TYPE,
        beam_size: int = FASTER_WHISPER_BEAM_SIZE,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise ASRError(f"faster-whisper import failed: {exc}") from exc

        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.beam_size = max(1, int(beam_size))

    def transcribe_chunk(self, audio_path: Path, options: dict[str, Any]) -> str:
        task = options.get("task", "transcribe")
        chunk_start = float(options.get("chunk_start_seconds", 0.0))
        segments, _ = self.model.transcribe(
            str(audio_path),
            task=task,
            vad_filter=True,
            beam_size=self.beam_size,
            language=None,
        )
        timeline = options.get("diarization_timeline", [])
        lines: list[str] = []
        for seg in segments:
            text = str(getattr(seg, "text", "")).strip()
            if not text:
                continue
            start = float(getattr(seg, "start", 0.0) or 0.0)
            abs_ts = chunk_start + start
            # Prefer pyannote timeline; fall back to faster-whisper's own label
            speaker = getattr(seg, "speaker", None)
            if not speaker and timeline:
                from .diarizer import speaker_at
                speaker = speaker_at(timeline, abs_ts)
            line = _format_line(abs_ts, speaker, text)
            if line:
                lines.append(line)
        return "\n".join(lines).strip()


class Qwen3ASREngine(ASREngine):
    def __init__(
        self,
        model_name: str = QWEN3_ASR_MODEL_NAME,
        *,
        device: str = QWEN3_ASR_DEVICE,
        dtype: str = QWEN3_ASR_DTYPE,
        max_new_tokens: int = QWEN3_ASR_MAX_NEW_TOKENS,
    ) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except Exception as exc:
            raise ASRError(f"qwen-asr/torch import failed: {exc}") from exc

        self._torch = torch
        self.device_map = self._resolve_device_map(torch, device)
        self.torch_dtype = self._resolve_torch_dtype(torch, dtype, self.device_map)
        self.max_new_tokens = max(16, int(max_new_tokens))
        # Heuristic for short chunks: enough headroom for CJK/mixed transcripts without over-decoding.
        self.tokens_per_second = max(2.0, float(os.getenv("QWEN3_ASR_TOKENS_PER_SECOND", "4.0")))

        torch.set_grad_enabled(False)

        try:
            self.model = Qwen3ASRModel.from_pretrained(
                model_name,
                dtype=self.torch_dtype,
                device_map=self.device_map,
                max_inference_batch_size=1,
                max_new_tokens=self.max_new_tokens,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            hint = (
                "Failed to load Qwen3 ASR model via `qwen-asr`. "
                "Ensure the app is started with the project environment (`uv run ...`) "
                "and dependencies are synced (`uv sync`)."
            )
            raise ASRError(
                f"{hint} Python={sys.executable}. "
                f"Original error: {exc}"
            ) from exc

    def _resolve_device_map(self, torch: Any, configured: str) -> str:
        cfg = configured.strip().lower()
        if cfg and cfg != "auto":
            if cfg == "cuda" and torch.cuda.is_available():
                return "cuda:0"
            if cfg == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            if cfg == "cpu":
                return "cpu"
            return cfg

        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _resolve_torch_dtype(self, torch: Any, configured: str, device_map: str) -> Any:
        cfg = configured.strip().lower()
        if cfg and cfg != "auto":
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            if cfg in mapping:
                return mapping[cfg]

        if device_map.startswith("cuda"):
            return torch.bfloat16
        if device_map == "mps":
            return torch.float16
        return torch.float32

    def _chunk_token_limit(self, options: dict[str, Any]) -> int:
        duration = options.get("chunk_duration_seconds")
        if isinstance(duration, (int, float)) and duration > 0:
            estimated = int(24 + float(duration) * self.tokens_per_second)
            return min(self.max_new_tokens, max(24, estimated))
        return self.max_new_tokens

    def _extract_text(self, results: Any) -> str:
        if not results:
            return ""

        texts: list[str] = []
        for item in results:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
            else:
                text = str(getattr(item, "text", "")).strip()
            if text:
                texts.append(text)
        return " ".join(texts).strip()

    def _extract_structured_lines(self, results: Any, chunk_start: float) -> list[str]:
        lines: list[str] = []
        for idx, item in enumerate(results):
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                start = item.get("start", item.get("start_time", item.get("begin", 0.0)))
                speaker = item.get("speaker", item.get("speaker_id", item.get("spk")))
            else:
                text = str(getattr(item, "text", "")).strip()
                start = getattr(item, "start", getattr(item, "start_time", getattr(item, "begin", 0.0)))
                speaker = getattr(item, "speaker", getattr(item, "speaker_id", getattr(item, "spk", None)))

            if not text:
                continue

            try:
                rel_start = float(start or 0.0)
            except Exception:
                rel_start = float(idx)

            line = _format_line(chunk_start + rel_start, speaker, text)
            if line:
                lines.append(line)
        return lines

    def transcribe_chunk(self, audio_path: Path, options: dict[str, Any]) -> str:
        chunk_start = float(options.get("chunk_start_seconds", 0.0))
        prior_max_new_tokens = self.model.max_new_tokens
        self.model.max_new_tokens = self._chunk_token_limit(options)
        try:
            results = self.model.transcribe(
                audio=str(audio_path),
                language=None,
            )
        except Exception as exc:
            raise ASRError(f"Qwen3 ASR inference failed: {exc}") from exc
        finally:
            self.model.max_new_tokens = prior_max_new_tokens

        lines = self._extract_structured_lines(results, chunk_start)
        if lines:
            return "\n".join(lines).strip()

        fallback = self._extract_text(results)
        return _format_line(chunk_start, "Speaker A", fallback) if fallback else ""


class MLXWhisperEngine(ASREngine):
    """Apple Silicon Metal-accelerated Whisper via the mlx-whisper library.

    Uses the same [HH:MM:SS] Speaker A: text output format as the other
    engines so downstream chunking and deduplication work unchanged.

    Speaker diarisation is not available in mlx-whisper, so all segments
    are labelled "Speaker A".  The chunker will group them by time windows
    (MAX_CHARS_PER_CHUNK) rather than speaker boundaries — still works well
    for retrieval.
    """

    def __init__(self, model_name: str = MLX_WHISPER_MODEL) -> None:
        try:
            import mlx_whisper  # noqa: F401 — validate import at init time
        except Exception as exc:
            raise ASRError(f"mlx-whisper import failed: {exc}") from exc
        self.model_name = model_name

    def transcribe_chunk(self, audio_path: Path, options: dict[str, Any]) -> str:
        try:
            import mlx_whisper
        except Exception as exc:
            raise ASRError(f"mlx-whisper import failed: {exc}") from exc

        chunk_start = float(options.get("chunk_start_seconds", 0.0))

        try:
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self.model_name,
                language=None,       # auto-detect English / Chinese / mixed
                word_timestamps=False,
                verbose=False,
            )
        except Exception as exc:
            raise ASRError(f"mlx-whisper inference failed: {exc}") from exc

        timeline = options.get("diarization_timeline", [])
        segments = result.get("segments", [])
        lines: list[str] = []
        for seg in segments:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            start = float(seg.get("start", 0.0))
            abs_ts = chunk_start + start
            if timeline:
                from .diarizer import speaker_at
                speaker = speaker_at(timeline, abs_ts)
            else:
                speaker = "Speaker A"
            line = _format_line(abs_ts, speaker, text)
            if line:
                lines.append(line)

        return "\n".join(lines).strip()


def create_engine(engine_name: Engine) -> ASREngine:
    if engine_name == Engine.faster_whisper:
        return _get_faster_whisper_engine()
    if engine_name == Engine.qwen3_asr_1_7b:
        return _get_qwen3_engine()
    if engine_name == Engine.mlx_whisper:
        return _get_mlx_whisper_engine()
    raise ASRError(f"Unsupported engine: {engine_name}")


@lru_cache(maxsize=1)
def _get_faster_whisper_engine() -> FasterWhisperEngine:
    return FasterWhisperEngine()


@lru_cache(maxsize=1)
def _get_qwen3_engine() -> Qwen3ASREngine:
    return Qwen3ASREngine()


@lru_cache(maxsize=1)
def _get_mlx_whisper_engine() -> MLXWhisperEngine:
    return MLXWhisperEngine()
