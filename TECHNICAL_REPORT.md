# Podcast2Insights: Technical Report

## Executive Summary

Podcast2Insights is a fully local, end-to-end RAG system built on Apple Silicon that transcribes podcast audio and answers natural-language questions about the content with grounded, timestamped responses. It was built solo over a weekend as a learning project and portfolio artifact — not a production service. The system works end-to-end on real audio, including bilingual English/Chinese podcasts. The most significant open problems are LLM hallucination beyond the retrieved context and inconsistent output language when querying in Chinese.

---

## Problem Framing

**What problem:** Given a podcast episode, let a user ask free-form questions and get answers grounded in what was actually said — a toy version of NotebookLM.

**Why I built it:** A personal pain point. I regularly listen to long, technically dense podcasts across English and Chinese sources (Spotify, YouTube, 小宇宙) also sometimes I recored technical talks — and wanted a way to extract key takeaways without re-listening. Existing tools (Podwise, Snipd, NotebookLM) solve this but sit behind paywalls or require uploading audio to third-party servers. Building it myself meant zero ongoing cost, full data privacy, and a concrete excuse to implement a RAG pipeline from scratch.

**Success definition:** This is an educational project. No business metrics or quantitative performance thresholds were defined — the goal was to learn by implementing every layer of a RAG pipeline from scratch rather than through orchestration frameworks. Formal offline evaluation (retrieval recall, answer faithfulness, BLEU/ROUGE) was deliberately out of scope for a weekend build. This is an honest gap; a production system would require a held-out evaluation set with measurable targets before any quality claims.

**Constraints that shaped the design:**

| Constraint | Impact |
|---|---|
| Zero ongoing cost | Ruled out all cloud APIs; everything runs locally |
| Apple Silicon only (M4 Pro, 48 GB) | Determined model selection and acceleration backend |
| Solo, one weekend | Forced scope cuts; no fine-tuning, no persistence, no eval harness |
| Portfolio legibility | Favoured explicit implementation over framework abstractions |

---

## System Design Overview

```
Audio File / URL
  → ffmpeg normalisation (mono, 16 kHz WAV)
  → 90s chunks with 2s overlap
  → mlx-whisper large-v3 (Metal, per chunk)
  → overlap deduplication
  → Transcript [HH:MM:SS] Speaker A: text

Transcript (auto-triggered after transcription)
  → Speaker-turn chunker (≤1,200 chars)
  → bge-m3 embedder (1024-dim, MPS, normalised)
  → FAISS IndexFlatIP (exact cosine, in-memory)

Query
  → bge-m3 query embedding
  → FAISS top-5 retrieval
  → Grounding prompt + Qwen3-8B via Ollama (stream=True)
  → SSE token stream → marked.js markdown render
```

The backend is FastAPI with a single-worker FIFO queue. Both pipeline phases run sequentially in the same thread. The FAISS index and chunks are held in-memory per session — lost on restart by design.

---

## Key Design Decisions

### ASR: mlx-whisper over faster-whisper

**Decided:** mlx-whisper large-v3 via Apple's MLX framework.

**Why:** faster-whisper uses CTranslate2, which runs CPU-only on Apple Silicon. mlx-whisper targets Metal, reducing transcription time for a 1-hour podcast from roughly 40 minutes (faster-whisper/CPU) to ~10 minutes. No quality difference between the two for large-v3.

**Alternatives rejected:** Whisper API (cost, privacy — audio would leave the machine); AssemblyAI/Deepgram (same reasons, plus vendor lock-in).

**Trade-off accepted:** mlx-whisper is macOS/arm64-only, requiring a `sys_platform == 'darwin'` marker in `pyproject.toml` and CPU fallback in CI.

### Embedding: bge-m3 over multilingual alternatives

**Decided:** `BAAI/bge-m3` via sentence-transformers on MPS.

**Why:** The two hard requirements were strong Chinese/English bilingual quality and a context window large enough for speaker-turn chunks. bge-m3 is the only freely available model that meets both: it supports 100+ languages with production-grade quality and has an 8,192-token context window. The next best alternatives (`multilingual-e5-large`, `paraphrase-multilingual-MiniLM`) cap at 512 tokens, which would force overly aggressive chunking and risk truncating semantically coherent turns.

**Trade-off accepted:** bge-m3 is 570 MB. On first run it downloads from HuggingFace. Subsequent runs use the local cache.

### Vector Index: FAISS IndexFlatIP over managed stores

**Decided:** Exact inner-product search, in-memory, no persistence.

**Why:** A podcast transcript produces hundreds to low-thousands of chunks. At that scale, approximate indexes (IVF, HNSW) offer no latency benefit and add tuning complexity (nlist, efSearch, minimum training vectors). ChromaDB and Pinecone local mode were considered but both embed a persistence layer that is unnecessary for a session-scoped demo.

**Trade-off accepted:** The index is lost on server restart. This is acceptable for a single-user tool; a persistent index would require serialising FAISS state and storing chunk metadata in SQLite.

### LLM: Qwen3-8B via Ollama over API models

**Decided:** Qwen3-8B served locally through Ollama (OpenAI-compatible endpoint).

**Why:** Fits comfortably in the memory budget (~5 GB quantised, leaving headroom for ASR and embedding), has native Chinese/English bilingual capability, and costs nothing per query. The Ollama API is OpenAI-compatible, so the `base_url` can be swapped to any provider without code changes.

**Thinking mode disabled:** Qwen3 supports chain-of-thought "thinking" mode. For grounded Q&A over retrieved excerpts, this adds 5–15 seconds of latency with minimal benefit — the answer space is already constrained by the prompt. Disabled via `/no_think` prefix; configurable via `OLLAMA_THINKING` env var.

**Trade-off accepted:** 8B parameters is the threshold where hallucination becomes noticeable (see Failure Modes). A 14B or 32B model would reduce this at the cost of inference speed.

### Speaker Diarisation: Removed

**Decided:** All segments labelled `Speaker A`. Diarisation was implemented (pyannote/speaker-diarization-3.1) and then removed.

**Why removed:** pyannote introduced a hard dependency on `torchaudio` with an exact version constraint (`torchaudio==2.6.0` to match `torch==2.6.0` — a binary ABI requirement). The dependency resolver silently installed `torchaudio==2.11.0`, causing a runtime symbol error that was non-obvious to diagnose. Beyond the dependency problem, running pyannote on top of mlx-whisper and bge-m3 simultaneously exhausted memory on the MacBook Pro and caused system crashes. The accuracy improvement did not justify the engineering and operational cost for a single-user demo.

**Trade-off accepted:** All speakers are `Speaker A`. Retrieval and citation still work correctly; only multi-speaker attribution is lost.

---

## Evaluation Strategy

No formal evaluation was conducted. This is an intentional scope cut given the one-weekend timeline and educational goal.

**What was validated manually:**
- End-to-end transcription on real podcast audio (English-only, Chinese-only, mixed)
- Retrieval returning topically relevant chunks on spot-check questions
- Streaming delivery and markdown rendering in the browser

**What was not evaluated:**
- Retrieval recall or precision on a held-out question set
- Answer faithfulness (does the answer stay within the retrieved context?)
- Transcription WER against a reference transcript

A minimal evaluation harness — even 20–30 manually labeled question/answer pairs — would make quality claims defensible and expose retrieval failures that spot-checking misses.

---

## Known Failure Modes

**LLM hallucination.** Qwen3-8B occasionally generates claims not present in the retrieved excerpts, despite the system prompt instructing it to answer only from the provided text. This is a known behaviour of instruction-tuned models at this parameter scale. Mitigation options: stricter prompt wording, a faithfulness classifier as a post-processing filter, or a larger model (14B+). Not addressed in the current build.

**Language inconsistency.** Asking a question in Chinese returns an answer in English. The system prompt instructs the model to "respond in the same language as the question," but Qwen3-8B does not consistently follow this for the grounded RAG format. Root cause is likely the prompt structure — the large English excerpt block biases the model toward English continuation. Targeted fix: language-specific instruction placement or explicit language detection and enforcement. Flagged for future work.

**Single speaker label.** As noted above, all transcript segments carry `Speaker A`. For interview-format podcasts this reduces the value of speaker-specific queries ("What did the guest say about X?").

---

## Scalability & Production Readiness

This is a prototype. It is explicitly not production-ready.

- **Throughput:** One job at a time (FIFO single-worker queue). A second upload blocks until the first completes.
- **State:** In-memory; lost on restart.
- **Latency (measured on M4 Pro, 48 GB):** ~10 min transcription for a 1-hour podcast; index build negligible; time-to-first-token ~2–3 seconds.
- **No monitoring or alerting.** Errors surface in the uvicorn console and the job `failure_reason` field.

---

## What Was Left Out and Why

| Item | Reason |
|---|---|
| Speaker diarisation | Crashed the machine; ABI dependency hell; ROI not justified |
| Formal evaluation harness | Out of scope for a weekend; flagged as the most important next step |
| Persistent session state | Adds serialisation complexity; re-upload is acceptable for one user |
| Multi-turn conversation | Each query is independent; stateless design kept the API simple |
| Cloud deployment | Local-only was a hard constraint; no data leaves the machine |
| Fine-tuning | No labelled data; no training infrastructure; off-the-shelf models sufficient |

---

## Lessons Learned & Open Questions

**What worked better than expected:** The full stack fits comfortably in 48 GB unified memory, and mlx-whisper on Metal is fast enough that transcription is not a user-experience bottleneck for typical podcast lengths. bge-m3 retrieval quality on mixed-language queries was strong with no tuning.

**What was harder than expected:** Dependency management for ML packages on Apple Silicon is fragile. The `torch`/`torchaudio` ABI constraint is invisible until runtime and the failure mode (missing C++ symbol) is opaque. Lesson: pin transitive ML dependencies explicitly and verify with an integration smoke test.

**What I'd do differently:** Define two or three measurable success criteria before building — even informal ones like "retrieval returns at least one relevant chunk for 90% of my test questions." It takes 30 minutes to label 20 examples and gives the entire project a much clearer definition of done.

**Open questions:**
- Does a larger LLM (14B, 32B) reduce hallucination to an acceptable rate on grounded RAG prompts?
- Can language consistency be fixed with prompt engineering alone, or does it require explicit language detection?
- What is the retrieval recall on a real evaluation set? Is speaker-turn chunking actually better than fixed-token chunking for this domain?
