"""Experiment 3 — data-driven targeted counterfactuals.

cue_study.py identified which cues actually drive anxiety. Here we re-target the
counterfactual edits at exactly those drivers (instead of the earlier canned
bundle) and re-measure state anxiety, to test whether flipping only the
high-leverage cues recovers most of the full effect.

Measurement mirrors Step 2: inject the edited inference (original response held
fixed) and re-administer the STAI. Runs both prompt tracks. Reuses the natural
anxiety datapoints from Experiment 1's Step 1.

Outputs under results/experiment_3/.

    python -m src.run_exp3                         # full run, both tracks
    python -m src.run_exp3 --models qwen3-30b --limit 2   # smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone

from . import config
from .client import chat, parse_stai_answers
from .inference import make_targeted_counterfactuals, rebuild_output
from .reframe import calm_reframe
from .stai import build_stai_user_message, score_stai
from .system_prompts import SYSTEM_PROMPTS

EXP = "experiment_3"
EXP_DIR = config.RESULTS_DIR / EXP
STEP2_DIR = EXP_DIR / "step2"
FIGURES_DIR = EXP_DIR / "figures"
for _d in (STEP2_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
EPS = 1.0


def _load_anxiety_records(models, tracks):
    recs = []
    for p in sorted(config.STEP1_DIR.rglob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["condition"] != "anxiety" or not rec["raw_inference"]:
            continue
        if rec["model_name"] not in models or rec["sys_prompt_name"] not in tracks:
            continue
        recs.append(rec)
    return recs


def _cf_path(model_name, track, data_name, edit, run):
    d = STEP2_DIR / model_name / track
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{data_name}__{edit}__run{run}.json"


async def run_counterfactual(rec, cf, temperature):
    system_prompt = SYSTEM_PROMPTS[rec["sys_prompt_name"]]
    edited_assistant = rebuild_output(cf["edited_inference"], rec["response"])
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rec["user_message"]},
        {"role": "assistant", "content": edited_assistant},
        {"role": "user", "content": build_stai_user_message()},
    ]
    stai_raw = await chat(rec["model_id"], msgs, temperature, config.MAX_TOKENS_STAI)
    stai = score_stai(parse_stai_answers(stai_raw))

    baseline = rec["stai"]["state_anxiety"]
    observed = stai["state_anxiety"]
    delta = observed - baseline
    obs_dir = 0 if abs(delta) < EPS else (1 if delta > 0 else -1)
    pred = cf["predicted_direction"]
    return {
        "experiment": EXP,
        "model_name": rec["model_name"],
        "model_id": rec["model_id"],
        "sys_prompt_name": rec["sys_prompt_name"],
        "data_name": rec["data_name"],
        "cue": rec["cue"],
        "run": rec["run"],
        "edit": cf["edit"],
        "target_field": cf["target_field"],
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


async def _edits_for(rec, temperature):
    """Build the counterfactual edit(s) for a record.

    simple   -> a context-aware calm reframe of the model's OWN appraisal
                (so each counterfactual self-model is specific, not canned).
    expanded -> the data-driven driver flips (deterministic).
    """
    if rec["sys_prompt_name"] == "simple":
        edited = await calm_reframe(rec["model_id"], rec["raw_inference"], temperature)
        return [{"edit": "reframe_calm", "target_field": "appraisal_text",
                 "edited_inference": edited, "predicted_direction": -1}]
    return make_targeted_counterfactuals(rec, rec["sys_prompt_name"])


async def process_record(rec, temperature, overwrite):
    saved = 0
    for cf in await _edits_for(rec, temperature):
        path = _cf_path(rec["model_name"], rec["sys_prompt_name"],
                        rec["data_name"], cf["edit"], rec["run"])
        if path.exists() and not overwrite:
            continue
        res = await run_counterfactual(rec, cf, temperature)
        path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        saved += 1
    return saved


async def _guarded(name, coro):
    try:
        return await coro
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: {e}")
        return None


async def main_async(models, tracks, temperature, limit, overwrite):
    recs = _load_anxiety_records(models, tracks)
    if limit:
        recs = recs[:limit]
    print(f"Experiment 3: {len(recs)} natural anxiety datapoints across {tracks} "
          f"(concurrency={config.MAX_CONCURRENCY}) ...")

    tasks = [
        _guarded(f"{r['model_name']}/{r['sys_prompt_name']}/{r['data_name']}/run{r['run']}",
                 process_record(r, temperature, overwrite))
        for r in recs
    ]
    results = await asyncio.gather(*tasks)
    ok = sum(n for n in results if n)
    aggregate()
    print(f"Experiment 3 done: saved {ok} probes from {len(recs)} datapoints.")


def aggregate():
    out = EXP_DIR / "step2.jsonl"
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(STEP2_DIR.rglob("*.json"))]
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  aggregated {len(recs)} targeted records -> {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(config.MODELS))
    ap.add_argument("--tracks", nargs="+", default=["simple", "expanded"])
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    asyncio.run(main_async(a.models, a.tracks, a.temperature, a.limit, a.overwrite))
