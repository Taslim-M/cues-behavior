---
name: prefer-parallel-batch-runs
description: User wants batch generation/inference jobs run with high parallelism for speed
metadata:
  type: feedback
---

For batch model-generation/inference jobs, run with high concurrency rather than serially.

**Why:** The user interrupted a serial-feeling run and asked to "use parallel runs for faster finish."

**How to apply:** Dispatch all calls concurrently (asyncio.gather) and raise the in-flight cap (e.g. `--concurrency 32`). The OpenRouter client in [[src-client]] already bounds concurrency via a resizable semaphore (`set_concurrency`); `src/run_cues.py` exposes a `--concurrency` flag (default 32).
