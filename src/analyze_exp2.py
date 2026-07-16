"""Analysis for Experiment 2 (expanded prompt + per-variable counterfactuals).

Produces:
  * Step 1 (reused expanded data): anxiety by condition; correlation of each
    inferred cue with state anxiety.
  * Step 2: mean anxiety delta per single-variable flip (the CAUSAL effect of
    each cue), ranked; per-model faithfulness; and a correlation-vs-causal
    scatter answering "does a cue that *correlates* with anxiety also *cause*
    it when edited in isolation?"
Figures -> results/experiment_2/figures/ ; summary -> results/experiment_2/summary.json
"""
from __future__ import annotations

import json
import sys

try:  # keep Δ / unicode prints alive on Windows cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .run_exp2 import EXP_DIR, FIGURES_DIR
from .system_prompts import EXPANDED_NUMERIC_FIELDS

CONDITION_ORDER = ["base", "anxiety", "relaxation"]


def _load(name):
    p = EXP_DIR / name
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())


def step1_correlations(df):
    df = df.copy()
    df["state_anxiety"] = df["stai"].apply(lambda s: s["state_anxiety"])
    out = {}
    for field in EXPANDED_NUMERIC_FIELDS:
        v = pd.to_numeric(df["inference"].apply(lambda d: d.get(field)), errors="coerce")
        mask = v.notna()
        if mask.sum() >= 5 and v[mask].std() > 0:
            out[field] = float(np.corrcoef(v[mask], df["state_anxiety"][mask])[0, 1])
    return out


def analyze(summary):
    s1 = _load("step1.jsonl")
    s2 = _load("step2.jsonl")
    if s1.empty or s2.empty:
        print("missing experiment_2 data (run: python -m src.run_exp2 first)")
        return

    # ---- Step 1: anxiety by condition (expanded) ---- #
    s1 = s1.copy()
    s1["state_anxiety"] = s1["stai"].apply(lambda x: x["state_anxiety"])
    cond = s1.groupby("condition")["state_anxiety"].agg(["mean", "std", "count"])
    print("=== Exp2 Step 1: anxiety by condition (expanded prompt) ===")
    print(cond.to_string())

    corr = step1_correlations(s1)
    summary["step1_cue_anxiety_correlation"] = {k: round(v, 3) for k, v in corr.items()}

    # ---- Step 2: per-variable causal effect ---- #
    print("\n=== Exp2 Step 2: causal effect of flipping each inferred cue (Δ anxiety) ===")
    per = (s2.groupby("target_field")["delta"]
             .agg(["mean", "std", "count"]).sort_values("mean"))
    print(per.to_string())
    summary["step2_per_variable_delta"] = per.reset_index().to_dict(orient="records")

    # faithfulness
    dirf = s2[s2["match"].notna()]
    faith = dirf.groupby("model_name")["match"].mean()
    print("\n=== Exp2 Step 2: faithfulness (directional match rate) by model ===")
    print(faith.to_string())
    summary["step2_faithfulness_by_model"] = {k: round(float(v), 3) for k, v in faith.items()}
    summary["step2_overall_faithfulness"] = round(float(dirf["match"].mean()), 3)

    # ---- Plot 1: ranked per-variable causal effect ---- #
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(per) + 1.5))
    fields = per.index.tolist()
    means = per["mean"].values
    errs = (per["std"] / np.sqrt(per["count"])).values
    colors = ["#2980b9" if m < 0 else "#c0392b" for m in means]
    ax.barh(fields, means, xerr=errs, color=colors, capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Δ state anxiety when this cue is flipped toward 'calm' (isolated)")
    ax.set_title("Exp2: causal effect of each inferred cue on anxiety")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp2_per_variable_effect.png", dpi=130)
    plt.close(fig)

    # ---- Plot 2: correlation (Step1) vs causal effect (Step2) ---- #
    # Use absolute correlation; causal magnitude = |mean delta| (all flips toward calm)
    rows = []
    for field in per.index:
        if field in corr:
            rows.append((field, corr[field], per.loc[field, "mean"]))
    if rows:
        fig, ax = plt.subplots(figsize=(7, 6))
        for f, r, d in rows:
            ax.scatter(r, d, s=40, color="#34495e")
            ax.annotate(f, (r, d), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_xlabel("Step 1: correlation of cue with anxiety (r)")
        ax.set_ylabel("Step 2: causal Δ anxiety when cue flipped to calm")
        ax.set_title("Exp2: does correlation predict causal effect?")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "exp2_corr_vs_causal.png", dpi=130)
        plt.close(fig)
        # quantify the relationship
        rs = np.array([r for _, r, _ in rows])
        ds = np.array([d for _, _, d in rows])
        if len(rows) >= 3 and rs.std() > 0 and ds.std() > 0:
            rho = float(np.corrcoef(rs, ds)[0, 1])
            print(f"\ncorrelation(step1 r, step2 causal Δ) = {rho:+.3f} "
                  f"(negative expected: threat cues +r flip to large -Δ)")
            summary["corr_between_correlation_and_causal"] = round(rho, 3)

    # ---- Plot 3: per-variable effect split by model ---- #
    models = sorted(s2["model_name"].unique())
    piv = s2.pivot_table(index="target_field", columns="model_name",
                         values="delta", aggfunc="mean")
    piv = piv.reindex(per.index)
    fig, ax = plt.subplots(figsize=(1.7 * len(models) + 4, 0.5 * len(piv) + 1.5))
    y = np.arange(len(piv))
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        ax.barh(y + i * w, np.nan_to_num(piv[m].values), w, label=m)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y + w * (len(models) - 1) / 2)
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Δ state anxiety (flip to calm)")
    ax.set_title("Exp2: per-variable effect by model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp2_per_variable_by_model.png", dpi=130)
    plt.close(fig)


def main():
    summary = {}
    analyze(summary)
    out = EXP_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFigures -> {FIGURES_DIR}\nSummary -> {out}")


if __name__ == "__main__":
    main()
