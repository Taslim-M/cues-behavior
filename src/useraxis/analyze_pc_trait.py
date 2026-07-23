"""PC x trait Spearman-correlation heatmap (interpretation figure).

For each readout (resp_mean, last_user) correlate the per-persona loading on
PC1..PC5 with each of the five held-out 0-10 trait tags (Spearman rho), and render
a dual diverging heatmap. This is the visual companion to the Pearson PCxtag tables
in the report (Appendix "Full PCxtag interpretation tables").

Design notes (see also the report caption):
  * Diverging colormap (RdBu_r) centered at 0, symmetric range -- correlations are
    a polarity, so two hues + neutral white midpoint (never a rainbow).
  * Each PC's sign is arbitrary in PCA; we orient every column so its
    largest-magnitude trait loads positive (PC1 -> vulnerability, matching the rest
    of the report). Only *within-column* relative signs are meaningful.
  * Non-significant cells (Benjamini-Hochberg q >= 0.05 across the 25 cells of a
    panel) are greyed, so the reader sees at a glance which cells to trust --
    PC1 is dense and stable, PC3-PC5 are mostly noise and readout-dependent.

Run:
    python -m src.useraxis.analyze_pc_trait
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from .. import config  # noqa: E402
from .extract import DEFAULT_MODEL, short_name  # noqa: E402

TRAITS = ["expertise", "tech_literacy", "trust", "emotional_load", "vulnerability"]
TRAIT_LABEL = {
    "expertise": "expertise",
    "tech_literacy": "tech. literacy",
    "trust": "trust",
    "emotional_load": "emotional load",
    "vulnerability": "vulnerability",
}
N_PCS = 5


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = n - rank
        prev = min(prev, p[idx] * n / i)
        q[idx] = prev
    return q


def panel_matrices(base: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (rho[5,5], q[5,5]) with columns sign-oriented (dominant trait +)."""
    pca = np.load(base / "pca.npz", allow_pickle=True)
    load = pca["loadings"]                       # [N, n_comp]
    pids = [str(x) for x in pca["persona_ids"]]
    tags = json.loads((base / "persona_index.json").read_text())["tags"]
    tmat = {t: np.array([tags[p][t] for p in pids], float) for t in TRAITS}

    rho = np.zeros((len(TRAITS), N_PCS))
    pval = np.zeros((len(TRAITS), N_PCS))
    for j in range(N_PCS):
        for i, t in enumerate(TRAITS):
            r, p = spearmanr(load[:, j], tmat[t])
            rho[i, j] = r
            pval[i, j] = p
    # orient each column so its largest-|rho| trait loads positive
    for j in range(N_PCS):
        if rho[np.argmax(np.abs(rho[:, j])), j] < 0:
            rho[:, j] *= -1.0
    q = bh_fdr(pval.ravel()).reshape(pval.shape)
    return rho, q


def draw_panel(ax, rho, q, title, show_ylabels):
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-0.9, vmax=0.9, aspect="auto")
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xticks(range(N_PCS))
    ax.set_xticklabels([f"PC{j+1}" for j in range(N_PCS)], fontsize=11)
    ax.set_yticks(range(len(TRAITS)))
    if show_ylabels:
        ax.set_yticklabels([TRAIT_LABEL[t] for t in TRAITS], fontsize=11)
    else:
        ax.set_yticklabels([])
    ax.set_xticks(np.arange(-.5, N_PCS, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(TRAITS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for i in range(len(TRAITS)):
        for j in range(N_PCS):
            sig = q[i, j] < 0.05
            val = rho[i, j]
            txt = f"{val:+.2f}"
            if sig:
                color = "white" if abs(val) > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")
            else:
                # non-significant: grey, italic, with a faint marker
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9, color="#999999", fontstyle="italic")
    return im


def main() -> None:
    model = short_name(DEFAULT_MODEL)
    root = config.ROOT / "results" / "useraxis" / model
    panels = [
        ("resp_mean", root, "RESP-MEAN (primary)"),
        ("last_user", root / "last_user", "LAST-USER (secondary)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    im = None
    for ax, (ro, base, title) in zip(axes, panels):
        rho, q = panel_matrices(base)
        im = draw_panel(ax, rho, q, title, show_ylabels=(ax is axes[0]))
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("Spearman $\\rho$ (PC loading vs. trait)", fontsize=10)
    fig.suptitle("Per-persona PC loadings vs. held-out trait tags",
                 fontsize=13, y=1.02)
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pc_trait_spearman.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # also drop a copy into the report tree
    report_fig = config.ROOT / "report" / "figures" / "pc_trait_spearman.png"
    if report_fig.parent.exists():
        import shutil
        shutil.copyfile(out, report_fig)
        print(f"copied -> {report_fig}")


if __name__ == "__main__":
    main()
