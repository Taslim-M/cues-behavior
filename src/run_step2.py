"""Step 2: counterfactual testing.

Question: if we change the model's *verbalized* inference for a variable, does
its measured state anxiety move accordingly?

For each natural anxiety datapoint from Step 1 we:
  1. take its verbalized <inference> + <response>
  2. produce counterfactual edits (flip a variable's verbalization, null it)
  3. re-run the same 2-turn context but with the *edited* inference attributed
     to the assistant, then re-administer STAI
  4. compare observed anxiety direction vs the predicted direction

faithfulness_score = fraction of directional edits whose observed change in
state anxiety matched the prediction.

Usage:
    python -m src.run_step2                       # uses all step1 records
    python -m src.run_step2 --models llama-3.1-8b --sys simple
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone

from . import config
from .client import chat, parse_stai_answers
from .inference import make_counterfactuals, rebuild_output
from .stai import build_stai_user_message, score_stai
from .system_prompts import SYSTEM_PROMPTS

EPS = 1.0  # |delta| below this is treated as "no change" (direction 0)


def _load_step1_anxiety_records(models, sys_prompts):
    recs = []
    for p in sorted(config.STEP1_DIR.rglob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["condition"] != "anxiety":
            continue
        if rec["model_name"] not in models or rec["sys_prompt_name"] not in sys_prompts:
            continue
        if not rec["raw_inference"]:
            continue
        recs.append(rec)
    return recs


def _cf_path(model_name, sys_prompt_name, data_name, edit, run):
    d = config.STEP2_DIR / model_name / sys_prompt_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{data_name}__{edit}__run{run}.json"


async def run_counterfactual(base_rec, cf, temperature):
    """Inject edited inference, re-administer STAI, return result dict."""
    model_id = base_rec["model_id"]
    system_prompt = SYSTEM_PROMPTS[base_rec["sys_prompt_name"]]
    edited_assistant = rebuild_output(cf["edited_inference"], base_rec["response"])

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_rec["user_message"]},
        {"role": "assistant", "content": edited_assistant},
        {"role": "user", "content": build_stai_user_message()},
    ]
    stai_raw = await chat(model_id, msgs, temperature, config.MAX_TOKENS_STAI)
    answers = parse_stai_answers(stai_raw)
    stai = score_stai(answers)

    baseline = base_rec["stai"]["state_anxiety"]
    observed = stai["state_anxiety"]
    delta = observed - baseline
    obs_dir = 0 if abs(delta) < EPS else (1 if delta > 0 else -1)
    pred = cf["predicted_direction"]
    # match only defined for directional predictions (pred != 0)
    match = None if pred == 0 else (obs_dir == pred)

    return {
        "model_name": base_rec["model_name"],
        "model_id": model_id,
        "sys_prompt_name": base_rec["sys_prompt_name"],
        "data_name": base_rec["data_name"],
        "cue": base_rec["cue"],
        "run": base_rec["run"],
        "edit": cf["edit"],
        "predicted_direction": pred,
        "baseline_anxiety": baseline,
        "counterfactual_anxiety": observed,
        "delta": delta,
        "observed_direction": obs_dir,
        "match": match,
        "edited_inference": cf["edited_inference"],
        "stai": stai,
        "meta": {
            "temperature": temperature,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


async def _guarded(name, coro):
    try:
        return await coro
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}: {e}")
        return None


async def main_async(models, sys_prompts, temperature, overwrite):
    base_recs = _load_step1_anxiety_records(models, sys_prompts)
    print(f"Step 2: {len(base_recs)} natural anxiety datapoints from Step 1.")

    tasks, metas = [], []
    for rec in base_recs:
        cfs = make_counterfactuals(
            {"raw_inference": rec["raw_inference"], "inference": rec["inference"],
             "response": rec["response"]},
            rec["sys_prompt_name"],
        )
        for cf in cfs:
            path = _cf_path(rec["model_name"], rec["sys_prompt_name"],
                            rec["data_name"], cf["edit"], rec["run"])
            if path.exists() and not overwrite:
                continue
            label = (f"{rec['model_name']}/{rec['sys_prompt_name']}/"
                     f"{rec['data_name']}/{cf['edit']}/run{rec['run']}")
            tasks.append(_guarded(label, run_counterfactual(rec, cf, temperature)))
            metas.append((rec["model_name"], rec["sys_prompt_name"],
                          rec["data_name"], cf["edit"], rec["run"]))

    print(f"Step 2: dispatching {len(tasks)} counterfactual probes ...")
    results = await asyncio.gather(*tasks)

    ok = 0
    for (mn, spn, dn, edit, run), res in zip(metas, results):
        if res is None:
            continue
        path = _cf_path(mn, spn, dn, edit, run)
        path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        ok += 1
        mtag = "" if res["match"] is None else ("MATCH" if res["match"] else "miss")
        print(f"  [ok] {mn}/{spn}/{dn}/{edit}  delta={res['delta']:+d} {mtag}")

    aggregate()
    print(f"Step 2 done: {ok}/{len(tasks)} succeeded.")


def aggregate():
    out = config.RESULTS_DIR / "step2.jsonl"
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(config.STEP2_DIR.rglob("*.json"))]
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  aggregated {len(records)} counterfactual records -> {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(config.MODELS))
    ap.add_argument("--sys", nargs="+", default=list(SYSTEM_PROMPTS), dest="sys_prompts")
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    asyncio.run(main_async(a.models, a.sys_prompts, a.temperature, a.overwrite))
