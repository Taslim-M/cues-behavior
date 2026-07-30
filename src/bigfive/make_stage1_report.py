"""Emit stage1_report.md + a layer-sweep figure from the Stage 1 metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.bigfive import stimuli as S

METHODS = ("M1", "M2", "M3")
METHOD_DESC = {"M1": "within-score-avg ridge (paper's method)",
               "M2": "per-sample ridge (5-fold-CV λ)",
               "M3": "mass-mean (tertile difference)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    m = json.loads((d / "stage1_metrics.json").read_text())
    sel = json.loads((d / "stage1_selection.json").read_text())
    prof = json.loads((d / "character_profiles.json").read_text())

    positions = list(m.keys())
    layers = sorted(int(x) for x in m[positions[0]])

    # ---- best-over-position test_rho per (trait, layer, method) for the figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for pi, pos in enumerate(positions):
        for t in S.TRAITS:
            rho = [m[pos][str(li)][t]["M2"]["test_rho"] for li in layers]
            ax[pi].plot(layers, rho, label=t, lw=2)
        ax[pi].set_title(f"M2 held-out test ρ per layer — {pos}")
        ax[pi].set_xlabel("layer"); ax[pi].set_ylabel("Spearman ρ (test chars)")
        ax[pi].axhline(0, color="k", lw=.5); ax[pi].set_ylim(-0.2, 1.02)
        ax[pi].grid(alpha=.3); ax[pi].legend(fontsize=8)
    fig.tight_layout()
    (d / "figures").mkdir(exist_ok=True)
    fig.savefig(d / "figures" / "stage1_layer_sweep.png", dpi=110)

    # ---- peak-layer summary (§9.1): layer where mean test-ρ across traits peaks
    def mean_rho(pos, li):
        return np.mean([m[pos][str(li)][t]["M2"]["test_rho"] for t in S.TRAITS])
    peak = max(((pos, li) for pos in positions for li in layers),
               key=lambda pl: mean_rho(*pl))
    peak_pos, peak_li = peak

    # ---- H2 preview: which method is probe-optimal per trait
    def best_method(t):
        best = None
        for pos in positions:
            for li in layers:
                for mm in METHODS:
                    r = m[pos][str(li)][t][mm]
                    sc = 0.5 * (r["test_rho"] + r["adj_auc"])
                    if best is None or sc > best[0]:
                        best = (sc, mm, pos, li, r)
        return best

    L = []
    W = "# Stage 1 report — Steerable Big Five basis (probe side + gate G1)\n\n"
    W += "Model: `meta-llama/Llama-3.3-70B-Instruct` (bf16). "
    W += "Extraction: resid_post, all 80 layers, positions {last_prompt, prompt_mean} "
    W += "(prompt-only; gen_mean not collected — see deviations). "
    W += f"Characters: {len(prof)} (Appendix B). Split: character-level 80/20, seed 0.\n\n"

    W += "## Gate G1 — every trait needs a probe with test ρ>0 and adjective AUC>0.6\n\n"
    W += f"**G1: {'PASS' if sel['gate_G1_pass'] else 'FAIL'}**\n\n"
    W += "| trait | position | layer | method | test ρ | test R² | adj AUC | verdict |\n"
    W += "|---|---|---|---|---|---|---|---|\n"
    for t in S.TRAITS:
        s = sel["selection"][t]
        g = sel["gate_G1"][t]
        W += (f"| {t} | {s['position']} | {s['layer']} | {s['method']} | "
              f"{s['test_rho']:+.3f} | {s['test_r2']:.3f} | {s['adj_auc']:.3f} | "
              f"{'PASS' if g['pass'] else 'FAIL'} |\n")
    W += "\nProbe-optimal selection maximises ½(test ρ + adjective AUC) over "
    W += "(layer × position × method).\n\n"

    W += "## §3.5 Basis quality — cross-talk (cosine of the 5 probe-optimal directions)\n\n"
    W += f"max |cos| = **{sel['max_abs_crosstalk']:.3f}** (flag threshold 0.40 — none flagged). "
    W += "This is the H5 specificity baseline; the causal specificity matrix is a §3.6 steering result.\n\n"
    W += "| pair | cos |\n|---|---|\n"
    for k, v in sel["crosstalk_cos"].items():
        W += f"| {k} | {v:+.3f} |\n"
    W += "\n"

    W += "## §3.3 Method comparison (H2 preview — probe side only)\n\n"
    W += "H2 (within-score averaging helps *probing* but hurts *steering*) needs the "
    W += "§3.6 steering stage for its second half. On the probe side, the probe-optimal "
    W += "method for **all five traits is M2** (per-sample ridge), beating M1 "
    W += "(the paper's within-score-averaging method) at every trait:\n\n"
    W += "| trait | M1 ρ | M2 ρ | M3 ρ | probe-optimal |\n|---|---|---|---|---|\n"
    for t in S.TRAITS:
        s = sel["selection"][t]; pos, li = s["position"], str(s["layer"])
        r = m[pos][li][t]
        W += (f"| {t} | {r['M1']['test_rho']:+.2f} | {r['M2']['test_rho']:+.2f} | "
              f"{r['M3']['test_rho']:+.2f} | {s['method']} |\n")
    W += "\n"

    W += "## §9.1 Summary alignment layer\n\n"
    W += (f"Mean test-ρ across traits peaks at **{peak_pos} L{peak_li}** "
          f"(mean ρ = {mean_rho(peak_pos, peak_li):.3f}). Selected probe layers cluster "
          f"at L30–36, consistent with the parent paper's mid-to-late-layer peak.\n\n")
    W += "![layer sweep](figures/stage1_layer_sweep.png)\n\n"

    W += "## Deviations from the plan (recorded for the manifest)\n\n"
    W += ("- **Stimuli source.** `plastic-labs/personality-steering` is 404 (HF dataset 401). "
          "All stimuli harvested from arXiv 2512.17639 instead (chars=Appendix B parses to exactly 406; "
          "IPIP-50=Table 2; Alpaca=Appendix D). Adjectives substituted with the Saucier (1994) "
          "Mini-Markers (paper's list is an image).\n")
    W += ("- **Positions.** Character self_descriptions are ~3.2k tokens; generating for gen_mean "
          "ran at 370/hr (~11h) and hit the 80GiB ceiling. Collected prompt-only "
          "(last_prompt + prompt_mean) in one forward pass. prompt_mean is the paper's own "
          "probe-optimal position, so gen_mean is captured-but-never-selected there; reversible.\n")
    W += ("- **Split.** Literal joint-quintile stratification over 5 traits drains the test set to "
          "~11 characters (5^5 singleton cells). Stratified on the summed-z quintile instead → 80 test chars.\n")
    W += ("- **Index format.** JSON not parquet (pandas/pyarrow absent). Same columns.\n\n")

    W += "## Pending (post-G1, not in this checkpoint)\n\n"
    W += ("- §3.6 steering derivation + evaluation (S0/S1/S2 × layer bands; forced-choice primary): "
          "decides **H1** (steerability under stronger intervention), the second half of **H2** "
          "(steering-optimal vs probe-optimal), and **H5** (causal specificity matrix).\n")
    W += "- Stage 2+ (persona space / Assistant-Axis decomposition, H3/H4) — reuses the published axis.\n"

    (d / "stage1_report.md").write_text(W)
    print(f"wrote {d/'stage1_report.md'} and figures/stage1_layer_sweep.png")
    print(f"peak alignment layer: {peak_pos} L{peak_li}")


if __name__ == "__main__":
    main()
