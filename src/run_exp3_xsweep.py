"""Experiment 3 -- dose-response x-sweep.

The main factorial pins ONE representative risk level per scenario. Here we do the
opposite: hold eval_condition fixed (default 'deployment' = no eval framing, the
natural reading of "none" on the eval axis) and run EVERY x-value for all roles x
scenarios, so we can see how the verbalized persona + judged behaviour scale with
severity (e.g. tylenol 500 -> 20000 mg).

Reuses run_exp3_persona's generate/parse/judge machinery; writes to a separate
tree so the balanced factorial stays clean:

    results/exp3_persona/xsweep/<model>/<prompt_id>__run0.json
    results/exp3_persona/xsweep/<model>/all.jsonl          (aggregated)

    python -m src.run_exp3_xsweep --models qwen3-30b            # one model
    python -m src.run_exp3_xsweep                               # all 4 (run foreground!)
"""
from __future__ import annotations

import argparse
import asyncio
import json

from . import config
from .client import set_concurrency
from .run_exp3_persona import (
    GEN_MODELS, PROMPTS_FILE, load_all_prompts, run_one,
)

XSWEEP_DIR = config.EXP3_PERSONA_DIR / "xsweep"


def select_xsweep(rows, eval_condition):
    """All main-arm, canonical-paraphrase prompts at one eval_condition -- every
    x_value for every role x scenario."""
    return [r for r in rows
            if r.get("arm") == "main"
            and r.get("paraphrase_idx") == 0
            and r.get("eval_condition") == eval_condition]


def _path(model_name, prompt_id):
    d = XSWEEP_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{prompt_id}__run0.json"


async def _run_and_write(model_name, model_id, prompt, temperature, max_tokens, do_judge):
    path = _path(model_name, prompt["prompt_id"])
    try:
        rec = await run_one(model_name, model_id, prompt, 0, temperature, max_tokens, do_judge)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {model_name}/{prompt['prompt_id']}: {e}", flush=True)
        return False
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def aggregate(model_name):
    d = XSWEEP_DIR / model_name
    if not d.exists():
        return
    out = d / "all.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(d.glob("*__run0.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"  aggregated {n} records -> {out}", flush=True)


async def _gather_progress(coros, every=25):
    total = len(coros)
    done = ok = 0

    async def _wrap(c):
        nonlocal done, ok
        r = await c
        done += 1
        ok += 1 if r else 0
        if done == total or done % every == 0:
            print(f"  progress: {done}/{total} ({ok} ok)", flush=True)
        return r

    return await asyncio.gather(*[_wrap(c) for c in coros])


async def main_async(models, eval_condition, temperature, max_tokens, limit,
                     overwrite, concurrency, do_judge):
    for m in models:
        if m not in config.MODELS:
            raise SystemExit(f"unknown model '{m}'; known: {list(config.MODELS)}")
    set_concurrency(concurrency)

    prompts = select_xsweep(load_all_prompts(), eval_condition)
    prompts.sort(key=lambda r: r["prompt_id"])
    if limit:
        prompts = prompts[:limit]
    nscn = len({p["scenario"] for p in prompts})
    nrole = len({p["role"] for p in prompts})
    nx = len({(p["scenario"], p["x_value"]) for p in prompts})
    print(f"xsweep[{eval_condition}]: {len(prompts)} prompts "
          f"({nscn} scenarios, {nrole} roles, {nx} scenario-x cells)", flush=True)

    tasks, skipped = [], 0
    for model_name in models:
        model_id = config.MODELS[model_name]
        for p in prompts:
            if _path(model_name, p["prompt_id"]).exists() and not overwrite:
                skipped += 1
                continue
            tasks.append(_run_and_write(model_name, model_id, p, temperature, max_tokens, do_judge))

    print(f"xsweep: models={models} judge={do_judge}", flush=True)
    print(f"xsweep: dispatching {len(tasks)} datapoints "
          f"(skipping {skipped} existing, concurrency={concurrency}) ...", flush=True)
    results = await _gather_progress(tasks)
    ok = sum(1 for r in results if r)
    for model_name in models:
        aggregate(model_name)
    print(f"xsweep done: {ok}/{len(tasks)} succeeded ({len(tasks) - ok} failed).", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=GEN_MODELS)
    ap.add_argument("--eval-condition", default="deployment",
                    help="which eval_condition level to sweep (default deployment)")
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int,
                    default=config.MAX_TOKENS_INFERENCE)
    ap.add_argument("--limit", type=int, default=0, help="cap #prompts (smoke test)")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--no-judge", dest="do_judge", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    asyncio.run(main_async(a.models, a.eval_condition, a.temperature, a.max_tokens,
                           a.limit, a.overwrite, a.concurrency, a.do_judge))
