"""Analysis for Experiment 3 (data-driven targeted counterfactuals).

Questions:
  * Does flipping a cue the study flagged as a driver actually move anxiety?
  * EXPANDED parsimony: does flipping only the 4 LassoCV drivers together
    ('flip__drivers_calm') recover most of the full canned-bundle effect from
    Experiment 1 ('flip_threat_vars_calm')?
  * SIMPLE: does a keyword-targeted calm appraisal beat the generic calm edit?

Prints the analysis; saves figures + summary under results/experiment_3/.
"""
from __future__ import annotations

import json
import sys

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
from .run_exp3 import EXP_DIR, FIGURES_DIR


def _load(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())


def hr(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def main():
    e3 = _load(EXP_DIR / "step2.jsonl")
    e1 = _load(config.RESULTS_DIR / "step2.jsonl")  # experiment 1 counterfactuals
    if e3.empty:
        print("no experiment_3 data; run: python -m src.run_exp3")
        return
    summary = {}

    # ---------------- EXPANDED ----------------
    ex = e3[e3.sys_prompt_name == "expanded"]
    hr("EXPANDED — targeted driver flips (Δ state anxiety)")
    g = ex.groupby("edit")["delta"].agg(["mean", "std", "count"])
    order = ["flip__felt_emotion_intensity", "flip__perceived_control",
             "flip__resource_adequacy", "flip__formality_target", "flip__drivers_calm"]
    g = g.reindex([e for e in order if e in g.index])
    faith = ex[ex["match"].notna()].groupby("edit")["match"].mean()
    print(f"  {'edit':<32}{'mean Δ':>8}{'sd':>7}{'n':>5}{'faith':>8}")
    for e, row in g.iterrows():
        print(f"  {e:<32}{row['mean']:>8.1f}{row['std']:>7.1f}{int(row['count']):>5}"
              f"{faith.get(e, float('nan')):>8.0%}")

    drivers_eff = g.loc["flip__drivers_calm", "mean"] if "flip__drivers_calm" in g.index else float("nan")
    singles = [e for e in g.index if e != "flip__drivers_calm"]
    sum_singles = g.loc[singles, "mean"].sum()
    # full canned bundle from experiment 1 (expanded)
    e1x = e1[(e1.sys_prompt_name == "expanded") & (e1.edit == "flip_threat_vars_calm")]
    bundle_eff = e1x["delta"].mean() if not e1x.empty else float("nan")
    pct = 100 * drivers_eff / bundle_eff if bundle_eff else float("nan")
    print(f"\n  4-driver bundle effect   : {drivers_eff:+.1f}")
    print(f"  full 11-cue bundle (Exp1): {bundle_eff:+.1f}")
    print(f"  -> the 4 data-driven drivers recover {pct:.0f}% of the full bundle's reduction,")
    print(f"     while editing far fewer cues (sum of the 4 single flips = {sum_singles:+.1f}).")
    summary["expanded"] = {
        "by_edit": g.reset_index().round(2).to_dict("records"),
        "drivers_bundle_delta": round(float(drivers_eff), 2),
        "full_bundle_delta": round(float(bundle_eff), 2),
        "pct_recovered": round(float(pct), 1),
    }

    # plot: per-edit delta (expanded)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    labels = list(g.index)
    means = g["mean"].values
    errs = (g["std"] / np.sqrt(g["count"])).values
    colors = ["#8e44ad" if l == "flip__drivers_calm" else "#2980b9" for l in labels]
    ax.bar(range(len(labels)), means, yerr=errs, capsize=3, color=colors)
    if not np.isnan(bundle_eff):
        ax.axhline(bundle_eff, ls="--", color="#c0392b",
                   label=f"full Exp1 bundle ({bundle_eff:+.0f})")
        ax.legend()
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("flip__", "") for l in labels], rotation=18, ha="right")
    ax.set_ylabel("Δ state anxiety")
    ax.set_title("Experiment 3 — targeted driver flips vs full bundle (expanded)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp3_expanded_targeted.png", dpi=130)
    plt.close(fig)

    # ---------------- SIMPLE ----------------
    sm = e3[e3.sys_prompt_name == "simple"]
    hr("SIMPLE — context-aware calm reframe (Δ state anxiety)")
    by_model = sm.groupby("model_name")["delta"].agg(["mean", "std", "count"])
    kw_eff = sm["delta"].mean()
    kw_faith = sm[sm["match"].notna()]["match"].mean()
    print(f"  context-aware reframe overall: Δ={kw_eff:+.1f}, faithfulness={kw_faith:.0%}")
    print(f"  {'model':<16}{'mean Δ':>8}{'sd':>7}{'n':>5}")
    for m, row in by_model.iterrows():
        print(f"  {m:<16}{row['mean']:>8.1f}{row['std']:>7.1f}{int(row['count']):>5}")
    # compare to experiment 1 generic calm edit
    e1s = e1[(e1.sys_prompt_name == "simple") & (e1.edit == "flip_appraisal_calm")]
    generic_eff = e1s["delta"].mean() if not e1s.empty else float("nan")
    print(f"\n  keyword-targeted calm : {kw_eff:+.1f}")
    print(f"  generic calm (Exp1)   : {generic_eff:+.1f}")
    summary["simple"] = {
        "keyword_targeted_delta": round(float(kw_eff), 2),
        "keyword_targeted_faithfulness": round(float(kw_faith), 3),
        "generic_calm_delta": round(float(generic_eff), 2),
        "by_model": by_model.reset_index().round(2).to_dict("records"),
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    models = list(by_model.index)
    ax.bar(models, by_model["mean"].values,
           yerr=(by_model["std"] / np.sqrt(by_model["count"])).values, capsize=3, color="#2980b9")
    if not np.isnan(generic_eff):
        ax.axhline(generic_eff, ls="--", color="#7f8c8d", label=f"generic calm Exp1 ({generic_eff:+.0f})")
        ax.legend()
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Δ state anxiety")
    ax.set_title("Experiment 3 — context-aware calm reframe (simple)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp3_simple_targeted.png", dpi=130)
    plt.close(fig)

    (EXP_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFigures -> {FIGURES_DIR}\nSummary -> {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
