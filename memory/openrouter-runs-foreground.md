---
name: openrouter-runs-foreground
description: In this harness, long async OpenRouter batch runs must be run foreground, not backgrounded; TaskStop leaves zombie python on Windows
metadata:
  type: feedback
---

When running the async OpenRouter batch scripts in this repo (`src/run_exp3_persona*`, `run_exp2_cues`, etc.), **run them in the FOREGROUND** with a generous `timeout` (e.g. 600000). Backgrounded asyncio runs (`run_in_background: true`, or commands the harness auto-backgrounds) reliably **stall at 0 completions** even though the API itself is fast (a single call returns <1s) — the event loop does not make network progress when detached. A full 432-datapoint model run completes foreground in ~3–5 min at concurrency 12.

**Why:** empirically confirmed across many attempts on 2026-06-25 — every backgrounded run produced 0 files for 10+ min; the identical command foreground finished 420/420 in 3m23s.

**How to apply:**
- Run each model foreground, one at a time, to stay under the auto-background threshold (~one model per call). Runs are resumable (existing datapoints are skipped), so chunk freely.
- `TaskStop` kills the bash wrapper but **leaves the python child alive as a zombie** on Windows. After stopping a run, `taskkill //F //IM python3.11.exe` to clear zombies — a leftover stuck process contends with new runs.
- The llama providers on OpenRouter throttle hard above ~8 concurrent (bursts pass, sustained load 429s into backoff); concurrency 8–12 is the safe range. qwen routes to a different provider and is faster.
- `client.py` holds the concurrency semaphore only during the request, not during retry backoff (fixed so a throttled model can't starve healthier ones sharing the pool).

Related: [[exp3-persona-pipeline]]
