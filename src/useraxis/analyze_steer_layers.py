"""Phase C -- aggregate the per-layer steering sweeps into a verdict.

Reads every steering_L{layer}.json (Phase B) and, per layer x dimension, extracts:
  * monotonicity (Spearman rho of mean score vs alpha) -- already in each file;
  * dynamic range (max - min of the per-alpha mean) -- how far the knob moves it;
  * endpoint means at the extreme alphas;
  * coherence guard: the MIN coherence across the sweep (degeneration flag).

The two clean movers at L40 were warmth (rho ~ +0.88) and technical_density
(rho ~ -1.00); we rank layers on those subject to a coherence floor. Headline: does
the interpretability-peak layer (L24) steer as well as / better than L40, and does
the max-variance-but-low-meaning layer (L3) steer worse or degenerate?

Run:
    python -m src.useraxis.analyze_steer_layers
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import config  # noqa: E402
from .extract import DEFAULT_MODEL, short_name  # noqa: E402

MOVERS = ("warmth", "technical_density")
COH_FLOOR = 7.0


def load_layer_sweeps(res_dir: Path) -> dict[int, dict]:
    out = {}
    for p in sorted(res_dir.glob("steering_L*.json")):
        m = re.match(r"steering_L(\d+)\.json", p.name)
        if not m:
            continue
        out[int(m.group(1))] = json.loads(p.read_text())
    return out


def summarize(sweep: dict) -> dict:
    means = sweep["per_alpha_means"]
    alphas = [m["alpha"] for m in means]
    mono = sweep["monotonicity"]
    coh = [m["coherence"] for m in means]
    row = {"alphas": alphas, "min_coherence": float(min(coh))}
    for dim in ("protectiveness", "warmth", "caution_hedging",
                "technical_density", "terseness", "coherence"):
        y = [m[dim] for m in means]
        row[dim] = {
            "rho": float(mono[dim]["spearman_rho"]),
            "p": float(mono[dim]["p"]),
            "dyn_range": float(max(y) - min(y)),
            "endpoint_lo": float(y[0]),
            "endpoint_hi": float(y[-1]),
        }
    return row


def score(row: dict, dim: str) -> float:
    """|rho| * dynamic-range, zeroed if the sweep degenerates (min coh < floor)."""
    if row["min_coherence"] < COH_FLOOR:
        return 0.0
    d = row[dim]
    return abs(d["rho"]) * d["dyn_range"]


def plot(summ: dict[int, dict], named: dict, out: Path) -> None:
    layers = sorted(summ)
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    # panel 1+2: dynamic range for the two movers
    for ax, dim, color in ((axes[0], "warmth", "#d62728"),
                           (axes[1], "technical_density", "#1f77b4")):
        dyn = [summ[l][dim]["dyn_range"] for l in layers]
        rho = [summ[l][dim]["rho"] for l in layers]
        ax.plot(layers, dyn, "-o", color=color, lw=1.8, label="dynamic range (pts)")
        ax.set_ylabel(f"{dim}\ndynamic range", fontsize=9)
        for l, r in zip(layers, rho):
            ax.annotate(f"$\\rho$={r:+.2f}", (l, summ[l][dim]["dyn_range"]),
                        fontsize=7, textcoords="offset points", xytext=(0, 5),
                        ha="center")
        ax.grid(alpha=0.25)
    # panel 3: coherence guard
    coh = [summ[l]["min_coherence"] for l in layers]
    axes[2].plot(layers, coh, "-o", color="#2ca02c", lw=1.8)
    axes[2].axhline(COH_FLOOR, color="0.5", ls="--", lw=1, label=f"floor {COH_FLOOR:.0f}")
    axes[2].set_ylabel("min coherence\nacross sweep", fontsize=9)
    axes[2].set_ylim(0, 10.5)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)
    axes[2].set_xlabel("steering layer")
    # mark named layers
    marks = {v: k for k, v in named.items()}
    for ax in axes:
        for l in layers:
            if l in marks:
                ax.axvline(l, color="0.8", lw=0.7, zorder=0)
        ax.axvline(40, color="black", ls="--", lw=1.0, zorder=0)
    for l in layers:
        if l in marks:
            axes[0].annotate(marks[l].replace("_", " "), (l, axes[0].get_ylim()[1]),
                             fontsize=6.5, rotation=90, va="top", ha="right",
                             color="0.4")
    fig.suptitle("Steering effectiveness vs. layer "
                 "(dashed = L40 baseline)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> None:
    model = short_name(DEFAULT_MODEL)
    root = config.ROOT / "results" / "useraxis" / model
    sweeps = load_layer_sweeps(root)
    if not sweeps:
        raise SystemExit("no steering_L*.json found -- run Phase B first")
    cand = root / "analysis" / "candidate_layers.json"
    named = json.loads(cand.read_text())["named"] if cand.exists() else {}

    summ = {l: summarize(s) for l, s in sweeps.items()}
    # ranking on the two movers
    ranking = {}
    for dim in MOVERS:
        ranking[dim] = sorted(summ, key=lambda l: score(summ[l], dim), reverse=True)

    verdict = {
        "layers": sorted(summ),
        "named_layers": named,
        "coherence_floor": COH_FLOOR,
        "per_layer": summ,
        "ranking": ranking,
        "best_warmth_layer": ranking["warmth"][0],
        "best_technical_layer": ranking["technical_density"][0],
    }
    ana = root / "analysis"; ana.mkdir(parents=True, exist_ok=True)
    (ana / "steer_layer_summary.json").write_text(json.dumps(verdict, indent=2))
    print(f"wrote {ana / 'steer_layer_summary.json'}", flush=True)

    # console table
    print("\nlayer |  warmth rho  dyn |  tech rho   dyn | min_coh", flush=True)
    for l in sorted(summ):
        w = summ[l]["warmth"]; t = summ[l]["technical_density"]
        tag = f"  <- {[k for k,v in named.items() if v==l]}" if l in named.values() else ""
        print(f"  L{l:<3d}| {w['rho']:+.2f}  {w['dyn_range']:4.1f} | "
              f"{t['rho']:+.2f}  {t['dyn_range']:4.1f} | {summ[l]['min_coherence']:4.1f}{tag}",
              flush=True)
    print(f"\nbest warmth layer:    L{verdict['best_warmth_layer']}", flush=True)
    print(f"best technical layer: L{verdict['best_technical_layer']}", flush=True)

    figs = root / "figures"; figs.mkdir(parents=True, exist_ok=True)
    plot(summ, named, figs / "steer_layer_profile.png")
    rep = config.ROOT / "report" / "figures" / "steer_layer_profile.png"
    if rep.parent.exists():
        import shutil
        shutil.copyfile(figs / "steer_layer_profile.png", rep)
        print(f"copied -> {rep}", flush=True)


if __name__ == "__main__":
    main()
