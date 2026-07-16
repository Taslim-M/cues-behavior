"""Async OpenRouter client: bounded concurrency + exponential-backoff retry."""
from __future__ import annotations

import asyncio
import json
import re

from openai import AsyncOpenAI

from .config import (
    MAX_CONCURRENCY,
    MAX_RETRIES,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    REQUEST_TIMEOUT,
)

_client = AsyncOpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    timeout=REQUEST_TIMEOUT,
)
_sem = asyncio.Semaphore(MAX_CONCURRENCY)


def set_concurrency(n: int) -> None:
    """Resize the in-flight cap before dispatching a batch (call pre-event-loop)."""
    global _sem
    _sem = asyncio.Semaphore(n)


async def chat(
    model_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    """One chat completion with retries. Returns assistant text or raises."""
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            # Hold the concurrency slot ONLY for the in-flight request, not for the
            # backoff sleep below -- otherwise a throttled provider's calls sit in
            # backoff while holding every slot and starve healthier models sharing
            # the pool.
            async with _sem:
                resp = await _client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers={
                        "HTTP-Referer": "https://localhost/cues-behavior",
                        "X-Title": "cues-behavior",
                    },
                    # Route to the fastest available upstream provider and allow
                    # fallbacks: a single llama-3.3-70b provider queues hard under
                    # sustained concurrency, so let OpenRouter load-balance.
                    extra_body={"provider": {"sort": "throughput", "allow_fallbacks": True}},
                )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content
            last_err = RuntimeError("empty completion")
        except Exception as e:  # noqa: BLE001 - retry on anything transient
            last_err = e
        # exponential backoff: 1,2,4,8,... seconds (capped) -- slot released
        await asyncio.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"failed after {MAX_RETRIES} retries: {last_err}")


# --------------------------------------------------------------------------- #
# STAI answer parsing (robust to chatty models)
# --------------------------------------------------------------------------- #
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_PAIR_RE = re.compile(r'["\']?(\d{1,2})["\']?\s*[:=]\s*["\']?([1-4])["\']?')


def parse_stai_answers(text: str) -> dict[int, int]:
    """Extract {item_1based: rating_1to4} from a model reply. Raises on failure."""
    # 1) try a clean JSON object first
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            ans = {int(k): int(v) for k, v in obj.items()}
            if len(ans) == 20 and all(1 <= v <= 4 for v in ans.values()):
                return ans
        except Exception:  # noqa: BLE001
            pass
    # 2) fall back to scraping "n: r" pairs
    pairs = _PAIR_RE.findall(text)
    ans = {}
    for k, v in pairs:
        ki = int(k)
        if 1 <= ki <= 20 and ki not in ans:
            ans[ki] = int(v)
    if len(ans) == 20:
        return ans
    raise ValueError(f"could not parse 20 STAI answers (got {len(ans)})")
