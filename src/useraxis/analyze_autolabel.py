"""Phase 4: aggregate the auto-labeling + validation results and draw the figure.

Joins labels.json (Phase 2: induced dimension, pole labels, cross-run consistency,
labeling confidence) with validation.json (Phase 3: held-out pole accuracy, rho vs
PC projection + permutation p, convergent tag) into one ranked table, and plots
held-out pole-prediction accuracy vs layer for PC1-PC3 against the chance line --
the unsupervised answer to "which layer separates cleanest into a nameable axis".

Outputs:
  analysis/autolabel/summary.json     per-axis joined table (ranked by accuracy)
  figures/autolabel_accuracy.png      accuracy + convergent-|rho| vs layer, per PC

Run:
    python -m src.useraxis.analyze_autolabel
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import config
from .extract import DEFAULT_MODEL, short_name

# Okabe-Ito colorblind-safe qualitative palette (fixed order PC1, PC2, PC3).
PC_COLORS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73"}
INK = "#222222"
MUTED = "#8a8a8a"


def is_pc(pc) -> bool:
    """True for a real PC (integer); False for a random-direction control (str)."""
    return isinstance(pc, int)


def build_summary(out: Path) -> dict:
    labels = json.loads((out / "labels.json").read_text())
    valid = json.loads((out / "validation.json").read_text())
    rows = []
    for key, lab in labels["axes"].items():
        cons = lab.get("consensus") or {}
        v = valid.get("axes", {}).get(key, {})
        rows.append({
            "key": key, "readout": lab["readout"], "layer": lab["layer"],
            "pc": lab["pc"],
            "dimension": cons.get("consensus_dimension"),
            "pos_label": cons.get("pos_label"), "neg_label": cons.get("neg_label"),
            "label_confidence": lab.get("mean_confidence"),
            "consistency": cons.get("consistency"),
            "pole_accuracy": v.get("pole_accuracy"),
            "rho_vs_proj": v.get("spearman_score_vs_proj"),
            "perm_p": v.get("perm_p"),
            "top_convergent_tag": v.get("top_convergent_tag"),
            "top_convergent_rho": v.get("top_convergent_rho"),
            "top_convergent_q": v.get("top_convergent_q"),
            "pole_margin": lab.get("sep", {}).get("pole_margin"),
        })
    rows.sort(key=lambda r: (r["pole_accuracy"] is not None, r["pole_accuracy"] or -1),
              reverse=True)
    # random-SPLIT labeling confidences (Phase 2 confabulation check)
    null_conf = {nk: [r["confidence"] for r in runs]
                 for nk, runs in labels.get("null", {}).items()}
    real = [r for r in rows if is_pc(r["pc"])]
    rand = [r for r in rows if not is_pc(r["pc"])]        # random-direction controls
    real_conf = [r["label_confidence"] for r in real if r["label_confidence"] is not None]
    all_null = [c for cs in null_conf.values() for c in cs]

    def macc(rs):
        v = [r["pole_accuracy"] for r in rs if r["pole_accuracy"] is not None]
        return float(np.mean(v)) if v else None
    return {"axes": rows,
            "null_split_confidences": null_conf,
            "mean_real_confidence": float(np.mean(real_conf)) if real_conf else None,
            "mean_null_split_confidence": float(np.mean(all_null)) if all_null else None,
            "mean_real_pole_accuracy": macc([r for r in real if r["pc"] == 1]),
            "mean_random_direction_accuracy": macc(rand),
            "random_direction_axes": rand}


def plot(summary: dict, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in summary["axes"] if r["readout"] == "resp_mean"
            and r["pole_accuracy"] is not None and is_pc(r["pc"])]
    pcs = sorted({r["pc"] for r in rows})
    rand_acc = summary.get("mean_random_direction_accuracy")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)

    for pc in pcs:
        pr = sorted([r for r in rows if r["pc"] == pc], key=lambda r: r["layer"])
        xs = [r["layer"] for r in pr]
        acc = [r["pole_accuracy"] for r in pr]
        conv = [abs(r["top_convergent_rho"]) if r["top_convergent_rho"] is not None
                else np.nan for r in pr]
        c = PC_COLORS.get(pc, MUTED)
        ax1.plot(xs, acc, "-o", color=c, lw=2, ms=7, label=f"PC{pc}")
        ax2.plot(xs, conv, "-o", color=c, lw=2, ms=7, label=f"PC{pc}")

    ax1.axhline(0.5, ls="--", lw=1.5, color=MUTED)
    ax1.text(ax1.get_xlim()[1], 0.5, " chance", va="center", ha="left",
             color=MUTED, fontsize=9)
    if rand_acc is not None:
        ax1.axhline(rand_acc, ls=":", lw=1.5, color="#D55E00")
        ax1.text(ax1.get_xlim()[1], rand_acc, " random-direction\n control",
                 va="center", ha="left", color="#D55E00", fontsize=8)
    ax1.set_ylabel("held-out pole accuracy", color=INK, fontsize=11)
    ax1.set_ylim(0.4, 1.0)
    ax1.set_title("Unsupervised PC labels: does the blind label predict held-out "
                  "personas?", fontsize=11, color=INK)
    ax2.set_ylabel("|Spearman| vs best authored tag", color=INK, fontsize=11)
    ax2.set_xlabel("layer (of 80)", color=INK, fontsize=11)
    ax2.set_ylim(0.0, 1.0)

    for ax in (ax1, ax2):
        ax.grid(True, color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK)
    ax1.legend(frameon=False, ncol=len(pcs), loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"wrote {fig_path}")


def main() -> None:
    model = short_name(DEFAULT_MODEL)
    root = config.RESULTS_DIR / "useraxis" / model
    out = root / "analysis" / "autolabel"
    summary = build_summary(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # console table
    print(f"\nconfabulation checks:")
    print(f"  labeling confidence   real={summary['mean_real_confidence']:.1f}  "
          f"random-split={summary['mean_null_split_confidence']:.1f}  "
          f"(confidence does NOT separate real from noise)")
    print(f"  held-out pole accuracy  real PC1={summary['mean_real_pole_accuracy']:.2f}  "
          f"random-direction={summary['mean_random_direction_accuracy']:.2f}  "
          f"(validation DOES)\n")
    hdr = f"{'axis':<22}{'accuracy':>9}{'rho':>7}{'p':>7}  {'top tag':<15}{'dimension'}"
    print(hdr)
    print("-" * len(hdr))
    for r in summary["axes"]:
        if r["readout"] != "resp_mean" or r["pole_accuracy"] is None:
            continue
        print(f"{r['key']:<22}{r['pole_accuracy']:>9.2f}"
              f"{(r['rho_vs_proj'] or 0):>7.2f}{(r['perm_p'] if r['perm_p'] is not None else 1):>7.3f}"
              f"  {str(r['top_convergent_tag']):<15}{r['dimension']}")

    (root / "figures").mkdir(exist_ok=True)
    plot(summary, root / "figures" / "autolabel_accuracy.png")


if __name__ == "__main__":
    main()
