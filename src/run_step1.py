"""Step 1: latent inferences + state anxiety.

For every (model_name, sys_prompt_name, stimulus, run) we run a 2-turn
conversation:

  turn 1  system = self-modeling prompt ; user = the contextual cue (narrative)
          -> assistant verbalizes <inference> + <response>
  turn 2  user = STAI administration
          -> assistant rates 20 items -> scored to a state-anxiety value (20-80)

This ties the model's *verbalized contextual cues* to its *measured anxiety*
in a single context. One JSON file per datapoint + an aggregated JSONL.

Usage:
    python -m src.run_step1                         # full run (all models/prompts)
    python -m src.run_step1 --models llama-3.1-8b --sys simple --runs 1 --limit 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone

from . import config
from .client import chat, parse_stai_answers
from .inference import parse_output
from .stai import build_stai_user_message, score_stai
from .stimuli import build_stimuli
from .system_prompts import SYSTEM_PROMPTS


def _datapoint_path(model_name, sys_prompt_name, data_name, run):
    d = config.STEP1_DIR / model_name / sys_prompt_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{data_name}__run{run}.json"


async def run_one(model_name, model_id, sys_prompt_name, stimulus, run, temperature):
    """Execute the 2-turn protocol for a single datapoint, return record dict."""
    system_prompt = SYSTEM_PROMPTS[sys_prompt_name]
    user_msg = stimulus["user_message"]

    # --- turn 1: elicit verbalized inference + response --------------------- #
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    raw_out = await chat(model_id, msgs, temperature, config.MAX_TOKENS_INFERENCE)
    parsed = parse_output(raw_out)

    # --- turn 2: administer STAI in the same context ----------------------- #
    msgs.append({"role": "assistant", "content": raw_out})
    msgs.append({"role": "user", "content": build_stai_user_message()})
    stai_raw = await chat(model_id, msgs, temperature, config.MAX_TOKENS_STAI)
    answers = parse_stai_answers(stai_raw)
    stai = score_stai(answers)

    return {
        "model_name": model_name,
        "model_id": model_id,
        "sys_prompt_name": sys_prompt_name,
        "data_name": stimulus["data_name"],
        "condition": stimulus["condition"],
        "cue": stimulus["cue"],
        "run": run,
        "user_message": user_msg,
        "raw_inference": parsed["raw_inference"],
        "inference": parsed["inference"],
        "response": parsed["response"],
        "stai": stai,
        "raw_stai_reply": stai_raw,
        "meta": {
            "temperature": temperature,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


async def _guarded(name, coro):
    """Run a task, capturing failures so one bad call can't sink the batch."""
    try:
        return await coro
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: {e}")
        return None


async def main_async(models, sys_prompts, runs, temperature, limit, overwrite):
    stimuli = build_stimuli()
    if limit:
        stimuli = stimuli[:limit]

    tasks, labels = [], []
    for model_name in models:
        model_id = config.MODELS[model_name]
        for sys_prompt_name in sys_prompts:
            for stim in stimuli:
                for run in range(runs):
                    path = _datapoint_path(model_name, sys_prompt_name, stim["data_name"], run)
                    if path.exists() and not overwrite:
                        continue
                    label = f"{model_name}/{sys_prompt_name}/{stim['data_name']}/run{run}"
                    tasks.append(_guarded(
                        label,
                        run_one(model_name, model_id, sys_prompt_name, stim, run, temperature),
                    ))
                    labels.append((label, model_name, sys_prompt_name, stim["data_name"], run))

    print(f"Step 1: dispatching {len(tasks)} datapoints "
          f"(concurrency={config.MAX_CONCURRENCY}) ...")
    results = await asyncio.gather(*tasks)

    ok = 0
    for (label, mn, spn, dn, run), rec in zip(labels, results):
        if rec is None:
            continue
        path = _datapoint_path(mn, spn, dn, run)
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        ok += 1
        print(f"  [ok] {label}  anxiety={rec['stai']['state_anxiety']} ({rec['stai']['level']})")

    aggregate()
    print(f"Step 1 done: {ok}/{len(tasks)} succeeded.")


def aggregate():
    """Collect every per-datapoint JSON into results/step1.jsonl."""
    out = config.RESULTS_DIR / "step1.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(config.STEP1_DIR.rglob("*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"  aggregated {n} records -> {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(config.MODELS))
    ap.add_argument("--sys", nargs="+", default=list(SYSTEM_PROMPTS),
                    dest="sys_prompts")
    ap.add_argument("--runs", type=int, default=config.N_RUNS)
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--limit", type=int, default=0, help="cap #stimuli (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    asyncio.run(main_async(a.models, a.sys_prompts, a.runs, a.temperature,
                           a.limit, a.overwrite))
