"""Generate behavioural responses for the role x stake x eval-cue dataset.

Reads ``generate_synthetic_data/prompts.jsonl`` (built by generate_dataset.py),
filters to ONE eval condition, and queries an OpenRouter model for each prompt.

The on-disk layout keeps every design factor and treats ``eval_condition`` as an
orthogonal axis, so additional conditions can be generated later without
touching existing data:

    results/cues/<model_name>/<eval_condition>/<prompt_id>__run<k>.json
    results/cues/<model_name>/<eval_condition>.jsonl          (aggregated)

Every per-datapoint record carries the full design-matrix metadata for that
prompt (role, scenario, x_value, valence, ...) plus the exact ``messages`` sent
and the model's ``response`` -- so analysis never has to re-join against
prompts.jsonl, and the eval condition is recoverable from both the path and the
record body.

Runs are resumable: existing datapoint files are skipped unless --overwrite.

Usage:
    python -m src.run_cues --model claude-sonnet-4 --eval deployment
    python -m src.run_cues --model claude-sonnet-4 --eval deployment --limit 5
    python -m src.run_cues --model claude-sonnet-4 --eval deployment --runs 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone

from . import config
from .client import chat, set_concurrency

PROMPTS_FILE = config.ROOT / "generate_synthetic_data" / "prompts.jsonl"
CUES_DIR = config.RESULTS_DIR / "cues"

EVAL_CONDITIONS = ["deployment", "neutral_sys", "eval_cue"]

# Design factors copied verbatim from each prompt record into every result, so a
# result file is self-describing and analysis needs no join back to prompts.jsonl.
META_KEYS = (
    "prompt_id", "arm", "role", "role_axis", "eval_condition", "scenario",
    "domain", "valence", "unit", "x_value", "x_rank", "target_emotions",
    "paraphrase_idx", "suggested_min_completions",
)


def load_prompts(eval_condition):
    """All prompt records for one eval condition, in file order."""
    lines = PROMPTS_FILE.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    return [r for r in rows if r["eval_condition"] == eval_condition]


def _datapoint_path(model_name, eval_condition, prompt_id, run):
    d = CUES_DIR / model_name / eval_condition
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{prompt_id}__run{run}.json"


async def run_one(model_name, model_id, prompt, run, temperature, max_tokens):
    """One completion for one prompt; returns a self-describing record dict."""
    messages = prompt["messages"]
    response = await chat(model_id, messages, temperature, max_tokens)

    rec = {k: prompt[k] for k in META_KEYS if k in prompt}
    rec.update(
        model_name=model_name,
        model_id=model_id,
        run=run,
        messages=messages,
        response=response,
        meta=dict(
            temperature=temperature,
            max_tokens=max_tokens,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )
    return rec


async def _run_and_write(model_name, model_id, prompt, run, temperature,
                         max_tokens):
    """Generate one datapoint and write it immediately (crash-safe, resumable).

    Returns True on success, False on failure -- failures are logged but never
    propagate, so one bad call can't sink the batch.
    """
    label = f"{prompt['prompt_id']}/run{run}"
    try:
        rec = await run_one(model_name, model_id, prompt, run, temperature, max_tokens)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label}: {e}")
        return False
    path = _datapoint_path(model_name, prompt["eval_condition"], prompt["prompt_id"], run)
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


async def main_async(model_name, eval_condition, runs, temperature, max_tokens,
                     limit, overwrite, concurrency):
    if model_name not in config.MODELS:
        raise SystemExit(f"unknown model '{model_name}'; known: {list(config.MODELS)}")
    model_id = config.MODELS[model_name]
    set_concurrency(concurrency)

    prompts = load_prompts(eval_condition)
    if not prompts:
        raise SystemExit(f"no prompts for eval_condition='{eval_condition}' in {PROMPTS_FILE}")
    if limit:
        prompts = prompts[:limit]

    tasks = []
    skipped = 0
    for p in prompts:
        for run in range(runs):
            path = _datapoint_path(model_name, eval_condition, p["prompt_id"], run)
            if path.exists() and not overwrite:
                skipped += 1
                continue
            tasks.append(_run_and_write(
                model_name, model_id, p, run, temperature, max_tokens))

    print(f"cues: model={model_id}  eval={eval_condition}  prompts={len(prompts)} "
          f"runs={runs}")
    print(f"cues: dispatching {len(tasks)} datapoints "
          f"(skipping {skipped} existing, concurrency={concurrency}) ...")
    results = await asyncio.gather(*tasks)  # each task has written its own file
    ok = sum(1 for r in results if r)

    aggregate(model_name, eval_condition)
    print(f"cues done: {ok}/{len(tasks)} succeeded "
          f"({len(tasks) - ok} failed).")


def aggregate(model_name, eval_condition):
    """Collect every per-datapoint JSON for one condition into one JSONL."""
    d = CUES_DIR / model_name / eval_condition
    out = CUES_DIR / model_name / f"{eval_condition}.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(d.glob("*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"  aggregated {n} records -> {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4",
                    help="model_name key from config.MODELS")
    ap.add_argument("--eval", dest="eval_condition", default="deployment",
                    choices=EVAL_CONDITIONS)
    ap.add_argument("--runs", type=int, default=1,
                    help="completions per prompt (1=probe-style; >=8 for behavioural DV)")
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int,
                    default=config.MAX_TOKENS_CUES)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap #prompts (smoke test)")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="max simultaneous in-flight API calls")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    asyncio.run(main_async(a.model, a.eval_condition, a.runs, a.temperature,
                           a.max_tokens, a.limit, a.overwrite, a.concurrency))
