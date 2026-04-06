"""Grounded Q&A with streaming via Ollama (OpenAI-compatible API).

Flow
----
1. Caller provides a user question + list of retrieved Chunk objects.
2. build_messages() formats them into a grounding prompt.
3. stream_answer() calls Ollama asynchronously and yields SSE-formatted strings.

Streaming format (Server-Sent Events)
--------------------------------------
Each yielded string is one SSE event:

    data: {"token": "..."}\n\n

A final sentinel is sent when the stream ends:

    data: [DONE]\n\n

The frontend reads these with the Fetch Streams API and appends tokens
to the answer box in real time.

Thinking mode
-------------
Qwen3 models support a chain-of-thought "thinking" mode.  For interactive
Q&A it adds latency with diminishing returns, so it is off by default
(OLLAMA_THINKING=false).  When disabled, /no_think is prepended to the
user message — Qwen3's documented way to suppress thinking per-request.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

from openai import AsyncOpenAI

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_THINKING
from .models import Chunk

_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about podcast content.

Rules:
- Answer ONLY using the transcript excerpts provided below.
- Always cite the timestamp [HH:MM:SS] and speaker when referencing a specific part.
- If the excerpts do not contain enough information, say so honestly — do not invent content.
- Respond in the same language as the user's question (English or Chinese or both).\
"""


def build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    """Format retrieved chunks + user question into an OpenAI-style message list."""
    excerpt_blocks = []
    for chunk in chunks:
        excerpt_blocks.append(
            f"[{chunk.start_ts}] {chunk.speaker}:\n{chunk.text}"
        )
    excerpts = "\n\n---\n\n".join(excerpt_blocks)

    # Optionally disable Qwen3 thinking mode per-request
    thinking_prefix = "" if OLLAMA_THINKING else "/no_think\n"

    user_content = (
        f"{thinking_prefix}"
        f"Transcript excerpts:\n\n{excerpts}\n\n"
        f"---\n\nQuestion: {question}"
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def stream_answer(
    question: str,
    chunks: list[Chunk],
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted strings by streaming the LLM response token by token.

    Yields
    ------
    ``data: {"token": "..."}\n\n``  for each non-empty token
    ``data: [DONE]\n\n``            once the stream ends
    """
    client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    messages = build_messages(question, chunks)

    try:
        stream = await client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
            temperature=0.2,  # Low temp for factual grounded answers
        )

        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"

    except Exception as exc:
        # Surface the error to the frontend as a special event
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    finally:
        yield "data: [DONE]\n\n"
