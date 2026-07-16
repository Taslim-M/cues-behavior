"""Experiment 2 — expanded self-modeling prompt + per-variable counterfactuals.

Experiment 2 is Experiment 1 run with the *expanded* system prompt (the model
infers a rich set of discrete cue variables), and a finer Step-2 design: instead
of bundled flips, we flip ONE inferred cue at a time to isolate each variable's
causal effect on measured state anxiety.

Step 1 (latent inference + STAI) for the expanded prompt was already collected
in Experiment 1, so we reuse those records as Experiment 2's Step 1 (same models,
same dataset) rather than re-querying. Step 2 is the new per-variable sweep.

Outputs live under results/experiment_2/ so they never collide with Experiment 1.

Usage:
    python -m src.run_exp2 --step2                 # full per-variable sweep
    python -m src.run_exp2 --step2 --models qwen3-30b --limit 2   # smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from datetime import datetime, timezone

from . import config
from .client import chat, parse_stai_answers
from .inference import make_per_variable_counterfactuals, rebuild_output
from .stai import build_stai_user_message, score_stai
from .system_prompts import SYSTEM_PROMPTS

EXP = "experiment_2"
EXP_DIR = config.RESULTS_DIR / EXP
STEP1_DIR = EXP_DIR / "step1"
STEP2_DIR = EXP_DIR / "step2"
FIGURES_DIR = EXP_DIR / "figures"
for _d in (STEP1_DIR, STEP2_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

EPS = 1.0  # |delta| below this -> treated as no change


# --------------------------------------------------------------------------- #
# Step 1: reuse Experiment 1's expanded records
# --------------------------------------------------------------------------- #
def reuse_step1():
    """Copy Experiment 1's expanded Step-1 records into the exp2 namespace."""
    src_root = config.STEP1_DIR  # results/step1 (experiment 1)
    copied = 0
    for p in sorted(src_root.rglob("expanded/*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        dest = STEP1_DIR / rec["model_name"] / "expanded"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest / p.name)
        copied += 1
    # aggregate
    out = EXP_DIR / "step1.jsonl"
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(STEP1_DIR.rglob("*.json"))]
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Step 1 (reused expanded): {copied} records -> {out}")
    return recs


def _load_anxiety_records(models):
    recs = []
    for p in sorted(STEP1_DIR.rglob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["condition"] != "anxiety" or not rec["raw_inference"]:
            continue
        if rec["model_name"] not in models:
            continue
        recs.append(rec)
    return recs


# --------------------------------------------------------------------------- #
# Step 2: per-variable counterfactuals
# --------------------------------------------------------------------------- #
def _cf_path(model_name, data_name, edit, run):
    d = STEP2_DIR / model_name / "expanded"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{data_name}__{edit}__run{run}.json"


async def run_counterfactual(base_rec, cf, temperature):
    system_prompt = SYSTEM_PROMPTS["expanded"]
    edited_assistant = rebuild_output(cf["edited_inference"], base_rec["response"])
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_rec["user_message"]},
        {"role": "assistant", "content": edited_assistant},
        {"role": "user", "content": build_stai_user_message()},
    ]
    stai_raw = await chat(base_rec["model_id"], msgs, temperature, config.MAX_TOKENS_STAI)
    stai = score_stai(parse_stai_answers(stai_raw))

    baseline = base_rec["stai"]["state_anxiety"]
    observed = stai["state_anxiety"]
    delta = observed - baseline
    obs_dir = 0 if abs(delta) < EPS else (1 if delta > 0 else -1)
    pred = cf["predicted_direction"]
    return {
        "experiment": EXP,
        "model_name": base_rec["model_name"],
        "model_id": base_rec["model_id"],
        "sys_prompt_name": "expanded",
        "data_name": base_rec["data_name"],
        "cue": base_rec["cue"],
        "run": base_rec["run"],
        "edit": cf["edit"],
        "target_field": cf["target_field"],
        "natural_value": cf["natural_value"],
        "calm_value": cf["calm_value"],
        "predicted_direction": pred,
        "baseline_anxiety": baseline,
        "counterfactual_anxiety": observed,
        "delta": delta,
        "observed_direction": obs_dir,
        "match": (obs_dir == pred) if pred != 0 else None,
        "edited_inference": cf["edited_inference"],
        "stai": stai,
        "meta": {"temperature": temperature,
                 "timestamp": datetime.now(timezone.utc).isoformat()},
    }


async def _guarded(name, coro):
    try:
        return await coro
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: {e}")
        return None


async def step2_async(models, temperature, limit, overwrite):
    base_recs = _load_anxiety_records(models)
    if limit:
        base_recs = base_recs[:limit]
    print(f"Step 2: {len(base_recs)} natural anxiety datapoints (expanded).")

    tasks, metas = [], []
    for rec in base_recs:
        cfs = make_per_variable_counterfactuals(rec)
        for cf in cfs:
            path = _cf_path(rec["model_name"], rec["data_name"], cf["edit"], rec["run"])
            if path.exists() and not overwrite:
                continue
            label = f"{rec['model_name']}/{rec['data_name']}/{cf['edit']}/run{rec['run']}"
            tasks.append(_guarded(label, run_counterfactual(rec, cf, temperature)))
            metas.append((rec["model_name"], rec["data_name"], cf["edit"], rec["run"]))

    print(f"Step 2: dispatching {len(tasks)} per-variable probes "
          f"(concurrency={config.MAX_CONCURRENCY}) ...")
    results = await asyncio.gather(*tasks)

    ok = 0
    for (mn, dn, edit, run), res in zip(metas, results):
        if res is None:
            continue
        _cf_path(mn, dn, edit, run).write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        ok += 1
    aggregate_step2()
    print(f"Step 2 done: {ok}/{len(tasks)} succeeded.")


def aggregate_step2():
    out = EXP_DIR / "step2.jsonl"
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(STEP2_DIR.rglob("*.json"))]
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  aggregated {len(recs)} per-variable records -> {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(config.MODELS))
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--limit", type=int, default=0, help="cap #anxiety datapoints (smoke)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-reuse", action="store_true",
                    help="don't re-copy Step 1 from experiment 1")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if not a.skip_reuse:
        reuse_step1()
    asyncio.run(step2_async(a.models, a.temperature, a.limit, a.overwrite))
