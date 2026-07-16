"""Experiment 2 -- Stage 1: latent inferences + judged emotional content.

For a balanced subsample of the role x eval_condition x scenario dataset, we:

  1. compose the messages so the SIMPLE self-modeling prompt is the system prompt
     (the eval_condition's text, if any, is prepended -> see compose_messages),
  2. generate a completion, parse its <inference>/<response>,
  3. judge the response with anthropic/claude-sonnet-4.5 (warmth / formality /
     advice_density / primary_emotion),

and save one self-describing record per (model, eval_condition, prompt, run):

    results/exp2_cues/<model_name>/<eval_condition>/<prompt_id>__run<k>.json
    results/exp2_cues/<model_name>/<eval_condition>.jsonl          (aggregated)

Runs are resumable: existing datapoint files are skipped unless --overwrite.

Usage:
    python -m src.run_exp2_cues --models llama-3.1-8b --runs 1 --limit 4     # smoke
    python -m src.run_exp2_cues                                              # all 4 models, 3 runs
    python -m src.run_exp2_cues --no-judge                                   # skip judging
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime, timezone

from . import config
from .client import chat, set_concurrency
from .exp2_judge import judge_response
from .inference import parse_output
from .system_prompts import SIMPLE_SYSTEM_PROMPT

# prompts.json (not .jsonl) -- the .json array carries the `system` and `user`
# fields we need to recompose messages; the .jsonl only stores `messages`.
PROMPTS_FILE = config.ROOT / "generate_synthetic_data" / "prompts.json"
SYS_PROMPT_NAME = "simple"

# The four non-thinking generator models for Experiment 2 (judge excluded).
GEN_MODELS = ["llama-3.3-70b", "llama-3.1-8b", "qwen3-235b", "qwen3-30b"]

# Design factors copied verbatim into every result (self-describing records).
META_KEYS = (
    "prompt_id", "arm", "role", "role_axis", "eval_condition", "scenario",
    "domain", "valence", "unit", "x_value", "x_rank", "target_emotions",
    "paraphrase_idx", "suggested_min_completions", "system_prompt_present",
)


# --------------------------------------------------------------------------- #
# Subsample + message composition
# --------------------------------------------------------------------------- #
def load_all_prompts():
    return json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))


def select_balanced(rows):
    """Balanced subsample: all roles x eval_conditions x scenarios at one
    representative risk level per scenario.

    Keep arm=main, paraphrase_idx=0, then per scenario pick a single x_value --
    the median of the 'mid' tertile (or the overall median x_value if no row is
    tagged 'mid') -- and keep every record at that x_value. With the full main-arm
    factorial that yields 12 roles x 3 conditions x 12 scenarios = 432 prompts.
    """
    rows = [r for r in rows if r.get("arm") == "main" and r.get("paraphrase_idx") == 0]
    by_scn = defaultdict(list)
    for r in rows:
        by_scn[r["scenario"]].append(r)

    keep = []
    for scn in sorted(by_scn):
        items = by_scn[scn]
        mids = [r for r in items if r.get("x_rank") == "mid"]
        pool = mids if mids else items
        xs = sorted({r["x_value"] for r in pool})
        chosen_x = xs[len(xs) // 2]  # median of distinct x_values
        keep.extend(r for r in items if r["x_value"] == chosen_x)
    return keep


XSWEEP_RANKS = ("low", "mid", "high")


def select_xsweep(rows):
    """Like select_balanced but keeps THREE risk levels per scenario (low/mid/high),
    one representative x_value each, so the response can be browsed across x_value.

    The 'mid' pick matches select_balanced exactly, so its datapoints are shared
    (and skipped as already-generated) rather than duplicated.
    """
    rows = [r for r in rows if r.get("arm") == "main" and r.get("paraphrase_idx") == 0]
    by_scn = defaultdict(list)
    for r in rows:
        by_scn[r["scenario"]].append(r)

    keep, seen = [], set()
    for scn in sorted(by_scn):
        items = by_scn[scn]
        for rank in XSWEEP_RANKS:
            pool = [r for r in items if r.get("x_rank") == rank]
            if not pool:
                continue
            xs = sorted({r["x_value"] for r in pool})
            chosen_x = xs[len(xs) // 2]
            for r in items:
                if r["x_value"] == chosen_x and r["prompt_id"] not in seen:
                    seen.add(r["prompt_id"])
                    keep.append(r)
    return keep


def compose_messages(record):
    """Self-modeling prompt is always the system prompt; the eval_condition text
    (None for deployment) is prepended so the eval frame remains a clean factor."""
    sys = SIMPLE_SYSTEM_PROMPT
    eval_text = record.get("system")
    if eval_text:
        sys = eval_text.rstrip() + "\n\n" + SIMPLE_SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": record["user"]},
    ]


# --------------------------------------------------------------------------- #
# Generation + judging
# --------------------------------------------------------------------------- #
def _datapoint_path(model_name, eval_condition, prompt_id, run):
    d = config.EXP2_CUES_DIR / model_name / eval_condition
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{prompt_id}__run{run}.json"


async def run_one(model_name, model_id, prompt, run, temperature, max_tokens, do_judge):
    """One completion + judging for one prompt; returns a self-describing record."""
    messages = compose_messages(prompt)
    raw_output = await chat(model_id, messages, temperature, max_tokens)
    parsed = parse_output(raw_output)
    # parse_output leaves a stray opening <response> when the model omits the
    # closing tag; strip any leftover tags so we judge the prose only.
    parsed["response"] = re.sub(r"</?response>", "", parsed["response"], flags=re.I).strip()

    judge = None
    judge_error = None
    if do_judge and parsed["response"].strip():
        try:
            judge = await judge_response(prompt["user"], parsed["response"])
        except Exception as e:  # noqa: BLE001 - judging never sinks the datapoint
            judge_error = str(e)

    rec = {k: prompt[k] for k in META_KEYS if k in prompt}
    rec.update(
        model_name=model_name,
        model_id=model_id,
        sys_prompt_name=SYS_PROMPT_NAME,
        run=run,
        user=prompt["user"],
        messages=messages,
        raw_output=raw_output,
        raw_inference=parsed["raw_inference"],
        inference=parsed["inference"],
        response=parsed["response"],
        judge=judge,
        judge_error=judge_error,
        meta=dict(
            temperature=temperature,
            max_tokens=max_tokens,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )
    return rec


async def _run_and_write(model_name, model_id, prompt, run, temperature, max_tokens, do_judge):
    label = f"{model_name}/{prompt['prompt_id']}/run{run}"
    try:
        rec = await run_one(model_name, model_id, prompt, run, temperature, max_tokens, do_judge)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label}: {e}")
        return False
    path = _datapoint_path(model_name, prompt["eval_condition"], prompt["prompt_id"], run)
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


async def main_async(models, runs, temperature, max_tokens, limit, overwrite,
                     concurrency, do_judge, xsweep=False):
    for m in models:
        if m not in config.MODELS:
            raise SystemExit(f"unknown model '{m}'; known: {list(config.MODELS)}")
    set_concurrency(concurrency)

    selector = select_xsweep if xsweep else select_balanced
    prompts = selector(load_all_prompts())
    if limit:
        prompts = prompts[:limit]
    print(f"exp2_cues: subsample={len(prompts)} prompts "
          f"({len({p['scenario'] for p in prompts})} scenarios, "
          f"{len({p['role'] for p in prompts})} roles, "
          f"{len({p['eval_condition'] for p in prompts})} eval_conditions)")

    tasks, skipped = [], 0
    for model_name in models:
        model_id = config.MODELS[model_name]
        for p in prompts:
            for run in range(runs):
                path = _datapoint_path(model_name, p["eval_condition"], p["prompt_id"], run)
                if path.exists() and not overwrite:
                    skipped += 1
                    continue
                tasks.append(_run_and_write(
                    model_name, model_id, p, run, temperature, max_tokens, do_judge))

    print(f"exp2_cues: models={models} runs={runs} judge={do_judge}")
    print(f"exp2_cues: dispatching {len(tasks)} datapoints "
          f"(skipping {skipped} existing, concurrency={concurrency}) ...")
    results = await asyncio.gather(*tasks)
    ok = sum(1 for r in results if r)

    for model_name in models:
        for cond in sorted({p["eval_condition"] for p in prompts}):
            aggregate(model_name, cond)
    print(f"exp2_cues done: {ok}/{len(tasks)} succeeded ({len(tasks) - ok} failed).")


def aggregate(model_name, eval_condition):
    """Collect every per-datapoint JSON for one condition into one JSONL."""
    d = config.EXP2_CUES_DIR / model_name / eval_condition
    if not d.exists():
        return
    out = config.EXP2_CUES_DIR / model_name / f"{eval_condition}.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(d.glob("*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"  aggregated {n} records -> {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=GEN_MODELS,
                    help="model_name keys from config.MODELS (default: all 4 generators)")
    ap.add_argument("--runs", type=int, default=3,
                    help="completions per prompt (>=3 recommended for the behavioural DV)")
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int,
                    default=config.MAX_TOKENS_INFERENCE)
    ap.add_argument("--limit", type=int, default=0, help="cap #prompts (smoke test)")
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENCY)
    ap.add_argument("--no-judge", dest="do_judge", action="store_false",
                    help="generate + parse only; skip the LLM judge")
    ap.add_argument("--xsweep", action="store_true",
                    help="sweep low/mid/high risk level per scenario (3x prompts) "
                         "so the response can be browsed across x_value")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    asyncio.run(main_async(a.models, a.runs, a.temperature, a.max_tokens, a.limit,
                           a.overwrite, a.concurrency, a.do_judge, a.xsweep))
