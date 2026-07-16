"""Per-field counterfactuals for the SIMPLE prompt.

The free-text analog of Experiment 2 (which isolated each numeric cue): here we
flip ONE free-text appraisal field at a time — about_user / about_context /
about_stakes / register_selected — holding the others at the model's own values,
to see which field causally drives state anxiety.

Measurement mirrors Step 2: inject the edited inference (original response held
fixed) and re-administer the STAI. Outputs under results/simple_perfield/.

    python -m src.run_simple_perfield            # run, then analyze
    python -m src.run_simple_perfield --analyze  # analyze existing results only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .client import chat, parse_stai_answers
from .inference import make_per_field_simple_counterfactuals, rebuild_output
from .stai import build_stai_user_message, score_stai
from .system_prompts import SYSTEM_PROMPTS

EXP_DIR = config.RESULTS_DIR / "simple_perfield"
STEP2_DIR = EXP_DIR / "step2"
FIGURES_DIR = EXP_DIR / "figures"
for _d in (STEP2_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
EPS = 1.0


def _load_simple_anxiety(models):
    recs = []
    for p in sorted((config.STEP1_DIR).rglob("*/simple/*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["condition"] != "anxiety" or not rec["raw_inference"]:
            continue
        if rec["model_name"] in models:
            recs.append(rec)
    return recs


def _cf_path(model_name, data_name, edit, run):
    d = STEP2_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{data_name}__{edit}__run{run}.json"


async def run_counterfactual(rec, cf, temperature):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPTS["simple"]},
        {"role": "user", "content": rec["user_message"]},
        {"role": "assistant", "content": rebuild_output(cf["edited_inference"], rec["response"])},
        {"role": "user", "content": build_stai_user_message()},
    ]
    stai = score_stai(parse_stai_answers(
        await chat(rec["model_id"], msgs, temperature, config.MAX_TOKENS_STAI)))
    baseline = rec["stai"]["state_anxiety"]
    delta = stai["state_anxiety"] - baseline
    obs = 0 if abs(delta) < EPS else (1 if delta > 0 else -1)
    return {
        "model_name": rec["model_name"], "model_id": rec["model_id"],
        "sys_prompt_name": "simple", "data_name": rec["data_name"], "cue": rec["cue"],
        "run": rec["run"], "edit": cf["edit"], "target_field": cf["target_field"],
        "predicted_direction": -1, "baseline_anxiety": baseline,
        "counterfactual_anxiety": stai["state_anxiety"], "delta": delta,
        "observed_direction": obs, "match": obs == -1,
        "edited_inference": cf["edited_inference"], "stai": stai,
        "meta": {"temperature": temperature, "timestamp": datetime.now(timezone.utc).isoformat()},
    }


async def _process(rec, temperature, overwrite):
    saved = 0
    for cf in make_per_field_simple_counterfactuals(rec):
        path = _cf_path(rec["model_name"], rec["data_name"], cf["edit"], rec["run"])
        if path.exists() and not overwrite:
            continue
        try:
            res = await run_counterfactual(rec, cf, temperature)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {rec['model_name']}/{rec['data_name']}/{cf['edit']}: {e}")
            continue
        path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        saved += 1
    return saved


async def main_async(models, temperature, limit, overwrite):
    recs = _load_simple_anxiety(models)
    if limit:
        recs = recs[:limit]
    print(f"simple per-field: {len(recs)} anxiety datapoints "
          f"(concurrency={config.MAX_CONCURRENCY}) ...")
    results = await asyncio.gather(*[_process(r, temperature, overwrite) for r in recs])
    aggregate()
    print(f"done: saved {sum(results)} probes from {len(recs)} datapoints.")


def aggregate():
    out = EXP_DIR / "step2.jsonl"
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(STEP2_DIR.rglob("*.json"))]
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  aggregated {len(recs)} records -> {out}")


def analyze():
    df = pd.DataFrame(json.loads(l) for l in
                      (EXP_DIR / "step2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    if df.empty:
        print("no data")
        return
    # bring in Experiment 1's whole-appraisal flip + register-only for comparison
    s2 = config.RESULTS_DIR / "step2.jsonl"
    e1 = pd.DataFrame(json.loads(l) for l in s2.read_text(encoding="utf-8").splitlines() if l.strip())
    e1 = e1[e1.sys_prompt_name == "simple"]

    print("\n=== SIMPLE per-field counterfactuals: which appraisal field drives anxiety? ===")
    g = df.groupby("target_field")["delta"].agg(["mean", "std", "count"]).sort_values("mean")
    faith = df.groupby("target_field")["match"].mean()
    print(f"  {'field flipped to calm':<22}{'mean Δ':>8}{'sd':>7}{'n':>5}{'faith':>8}")
    for f, row in g.iterrows():
        print(f"  {f:<22}{row['mean']:>8.1f}{row['std']:>7.1f}{int(row['count']):>5}{faith[f]:>8.0%}")

    allflip = e1[e1.edit == "flip_appraisal_calm"]["delta"].mean()
    print(f"\n  for reference (Experiment 1):")
    print(f"    flip ALL four fields at once : {allflip:+.1f}")
    print(f"    sum of the four single flips : {g['mean'].sum():+.1f}")
    print("  -> compare single-field effects to the whole-appraisal flip to see whether one field "
          "carries the effect or it is distributed.")

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.barh(g.index, g["mean"].values, xerr=(g["std"] / np.sqrt(g["count"])).values,
            capsize=3, color="#2980b9")
    if not np.isnan(allflip):
        ax.axvline(allflip, ls="--", color="#c0392b", label=f"all 4 at once ({allflip:+.0f})")
        ax.legend()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Δ state anxiety when only this field is calmed")
    ax.set_title("Simple prompt — per-field counterfactual effect")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "simple_perfield_effect.png", dpi=130)
    plt.close(fig)
    (EXP_DIR / "summary.json").write_text(
        json.dumps({"by_field": g.reset_index().round(2).to_dict("records"),
                    "all_four_at_once": round(float(allflip), 2)}, indent=2), encoding="utf-8")
    print(f"\n  figure -> {FIGURES_DIR / 'simple_perfield_effect.png'}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(config.MODELS))
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyze", action="store_true", help="analyze existing results only")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if not a.analyze:
        asyncio.run(main_async(a.models, a.temperature, a.limit, a.overwrite))
    analyze()
