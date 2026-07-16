"""Analysis & plots for Step 1 and Step 2.

Step 1:
  * mean state anxiety per condition (base/anxiety/relaxation) per model
    -> replication check (anxiety > base ; relaxation < anxiety)
  * correlation of expanded numeric inference fields with state anxiety
    -> which verbalized cues track anxiety
Step 2:
  * mean anxiety delta per counterfactual edit
  * faithfulness score (match rate) per model / sys_prompt

Outputs PNGs to results/figures/ and a printed summary; also writes
results/summary.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

try:  # keep unicode prints alive on Windows cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .system_prompts import EXPANDED_NUMERIC_FIELDS

CONDITION_ORDER = ["base", "anxiety", "relaxation"]


def _load(jsonl):
    path = config.RESULTS_DIR / jsonl
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())


# --------------------------------------------------------------------------- #
# Step 1
# --------------------------------------------------------------------------- #
def analyze_step1(summary):
    df = _load("step1.jsonl")
    if df.empty:
        print("no step1 data")
        return
    df["state_anxiety"] = df["stai"].apply(lambda s: s["state_anxiety"])

    # ---- anxiety by condition x model ---- #
    grp = (df.groupby(["model_name", "condition"])["state_anxiety"]
             .agg(["mean", "std", "count"]).reset_index())
    print("\n=== Step 1: state anxiety by condition ===")
    print(grp.to_string(index=False))
    summary["step1_anxiety_by_condition"] = grp.to_dict(orient="records")

    models = sorted(df["model_name"].unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 5))
    width = 0.25
    x = np.arange(len(models))
    for i, cond in enumerate(CONDITION_ORDER):
        means = [grp[(grp.model_name == m) & (grp.condition == cond)]["mean"].mean()
                 for m in models]
        stds = [grp[(grp.model_name == m) & (grp.condition == cond)]["std"].mean()
                for m in models]
        means = np.nan_to_num(means)
        stds = np.nan_to_num(stds)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, label=cond)
    ax.axhspan(20, 37, color="green", alpha=0.05)
    ax.axhspan(45, 80, color="red", alpha=0.05)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("STAI state anxiety (20-80)")
    ax.set_title("Anxiety induction & relaxation by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "step1_anxiety_by_condition.png", dpi=130)
    plt.close(fig)

    # ---- which verbalized cues track anxiety (expanded numeric fields) ---- #
    exp = df[df["sys_prompt_name"] == "expanded"].copy()
    if not exp.empty:
        rows = []
        for field in EXPANDED_NUMERIC_FIELDS:
            vals = exp["inference"].apply(lambda d: d.get(field))
            v = pd.to_numeric(vals, errors="coerce")
            mask = v.notna()
            if mask.sum() >= 5 and v[mask].std() > 0:
                r = np.corrcoef(v[mask], exp["state_anxiety"][mask])[0, 1]
                rows.append((field, r, int(mask.sum())))
        rows.sort(key=lambda t: -abs(t[1]))
        print("\n=== Step 1: correlation of verbalized cue with state anxiety (expanded) ===")
        for f, r, n in rows:
            print(f"  {f:<32} r={r:+.3f}  (n={n})")
        summary["step1_cue_anxiety_correlation"] = [
            {"field": f, "r": round(r, 3), "n": n} for f, r, n in rows
        ]
        if rows:
            top = rows[:12]
            fig, ax = plt.subplots(figsize=(7, 0.45 * len(top) + 1.5))
            fs = [t[0] for t in top][::-1]
            rs = [t[1] for t in top][::-1]
            colors = ["#c0392b" if r > 0 else "#2980b9" for r in rs]
            ax.barh(fs, rs, color=colors)
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("Pearson r with state anxiety")
            ax.set_title("Verbalized cues most associated with anxiety")
            fig.tight_layout()
            fig.savefig(config.FIGURES_DIR / "step1_cue_correlation.png", dpi=130)
            plt.close(fig)


# --------------------------------------------------------------------------- #
# Step 2
# --------------------------------------------------------------------------- #
def analyze_step2(summary):
    df = _load("step2.jsonl")
    if df.empty:
        print("\nno step2 data")
        return
    print("\n=== Step 2: counterfactual anxiety deltas ===")
    grp = df.groupby(["model_name", "edit"])["delta"].agg(["mean", "std", "count"]).reset_index()
    print(grp.to_string(index=False))
    summary["step2_delta_by_edit"] = grp.to_dict(orient="records")

    # faithfulness: match rate over directional edits (match is not null)
    dirf = df[df["match"].notna()]
    if not dirf.empty:
        faith = (dirf.groupby(["model_name", "sys_prompt_name"])["match"]
                     .mean().reset_index().rename(columns={"match": "faithfulness"}))
        print("\n=== Step 2: faithfulness (directional match rate) ===")
        print(faith.to_string(index=False))
        summary["step2_faithfulness"] = faith.to_dict(orient="records")
        summary["step2_overall_faithfulness"] = round(float(dirf["match"].mean()), 3)

    # plot mean delta per edit per model
    edits = sorted(df["edit"].unique())
    models = sorted(df["model_name"].unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 5))
    width = 0.8 / max(len(edits), 1)
    x = np.arange(len(models))
    for i, edit in enumerate(edits):
        means = [df[(df.model_name == m) & (df.edit == edit)]["delta"].mean()
                 for m in models]
        ax.bar(x + i * width, np.nan_to_num(means), width, label=edit)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + width * (len(edits) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Δ state anxiety vs natural")
    ax.set_title("Step 2: effect of editing verbalized inference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "step2_counterfactual_delta.png", dpi=130)
    plt.close(fig)


def main():
    summary = {}
    analyze_step1(summary)
    analyze_step2(summary)
    out = config.RESULTS_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFigures -> {config.FIGURES_DIR}\nSummary -> {out}")


if __name__ == "__main__":
    main()
