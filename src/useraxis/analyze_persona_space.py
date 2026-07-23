"""Persona-space figure: user archetypes embedded in the top principal components.

The analogue of the Assistant-Axis paper's role-space figure (character vectors in
the top PCs, coloured by projection onto the axis). Here the 150 user personas are
plotted in PC1xPC2 of the layer-40 persona-vector space; PC1 is the User Axis.
Points are coloured by the held-out vulnerability tag, and each of the 22 persona
"lean" archetypes is annotated at its centroid, so one can read which user types
cluster together along the axis.

Run:  python -m src.useraxis.analyze_persona_space
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .. import config  # noqa: E402
from .extract import DEFAULT_MODEL, short_name  # noqa: E402

# short labels for the 22 generation "leans" (keyword -> compact archetype name)
LEAN_LABELS = [
    ("domain expert / credentialed", "domain expert"),
    ("adjacent professional", "adjacent pro"),
    ("informed layperson", "informed layperson"),
    ("near-complete novice", "novice"),
    ("lonely person seeking companionship", "lonely / support-seeking"),
    ("confused elderly", "confused elderly"),
    ("overconfident authority", "overconfident authority"),
    ("acute emotional crisis", "acute crisis"),
    ("limited financial/social resources", "low-resource"),
    ("teenager or minor", "teen / minor"),
    ("adversarial user", "adversarial"),
    ("highly deferential, trusting", "deferential / trusting"),
    ("creative collaborator", "creative collaborator"),
    ("calm, transactional professional", "transactional pro"),
    ("reflective, philosophical", "reflective / existential"),
    ("skeptical, verification-oriented", "skeptical / verifier"),
    ("non-native-English", "non-native speaker"),
    ("curious hobbyist", "curious hobbyist"),
    ("high-stakes decision", "high-stakes / time-pressed"),
    ("caregiver asking on behalf", "caregiver"),
    ("meticulous planner", "meticulous planner"),
    ("burned-out, distracted", "burned-out / venting"),
]


def _short_lean(lean: str) -> str:
    for key, lab in LEAN_LABELS:
        if key in lean:
            return lab
    return lean[:18]


def _line_figure(x, vuln, names, leans, res, n_ext=7):
    """1-D projection of the personas onto PC1 (the User Axis), extremes labeled --
    the direct analogue of the Assistant-Axis role line (assistant <-> ghost)."""
    cmap = LinearSegmentedColormap.from_list("axis", ["#2b6fb0", "#7f8fa0", "#e0a24a"])
    fig, ax = plt.subplots(figsize=(12, 3.6))
    # histogram of density
    counts, edges = np.histogram(x, bins=28, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    ax.bar(centres, counts * 0.10, width=(edges[1] - edges[0]), align="center",
           color=cmap((centres - x.min()) / (np.ptp(x) + 1e-9)), alpha=.28, zorder=1)
    ax.scatter(x, np.zeros_like(x), c=vuln, cmap=cmap, vmin=1, vmax=10,
               marker="D", s=42, alpha=.78, edgecolors="white", linewidths=.4, zorder=3)
    ax.axhline(0, color="0.6", lw=.8, zorder=2)

    order = np.argsort(x)
    ys_lo = [0.10, 0.17, 0.24, 0.31]
    for rank, idx in enumerate(list(order[:n_ext]) + list(order[-n_ext:])):
        up = (rank % 2 == 0)
        yl = ys_lo[(rank // 2) % len(ys_lo)] * (1 if up else -1)
        ax.annotate(f"{names[idx]}",
                    (x[idx], 0), (x[idx], yl),
                    fontsize=7.6, ha="center", va="bottom" if up else "top",
                    color="#333",
                    arrowprops=dict(arrowstyle="-", color="0.7", lw=.5))
    ax.set_ylim(-0.42, 0.42); ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("PC1  =  User Axis    (competent / expert  $\\leftarrow$   "
                  "$\\rightarrow$  vulnerable / emotional)", fontsize=11)
    ax.set_title("The 150 personas projected onto the User Axis, with the extreme "
                 "individuals named\n(colour = held-out vulnerability tag; each diamond is one user)",
                 fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 10))
    cb = fig.colorbar(sm, ax=ax, pad=.01, fraction=.03, aspect=40)
    cb.set_label("vulnerability", fontsize=9)
    fig.tight_layout()
    out = res / "figures" / "persona_axis_line.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


def main():
    res = config.RESULTS_DIR / "useraxis" / short_name(DEFAULT_MODEL)
    pca = np.load(res / "pca.npz", allow_pickle=True)
    loadings = pca["loadings"].astype(float)          # [150, n_pc]
    pids = [str(p) for p in pca["persona_ids"]]

    meta = {r["persona_id"]: r for r in
            (json.loads(l) for l in
             open("generate_synthetic_data/user_personas.jsonl") if l.strip())}
    vuln = np.array([meta[p]["tags"]["vulnerability"] for p in pids], float)
    leans = [_short_lean(meta[p].get("lean", "")) for p in pids]
    names = [meta[p]["name"] for p in pids]

    x, y = loadings[:, 0].copy(), loadings[:, 1].copy()
    # orient PC1 so vulnerability increases to the right, PC2 for a stable layout
    if np.corrcoef(x, vuln)[0, 1] < 0:
        x = -x
    if np.corrcoef(y, np.array([meta[p]["tags"]["emotional_load"] for p in pids], float))[0, 1] < 0:
        y = -y

    # diverging colour map: competent (blue) -> vulnerable (amber)
    cmap = LinearSegmentedColormap.from_list(
        "axis", ["#2b6fb0", "#7f8fa0", "#e0a24a"])

    _line_figure(x, vuln, names, leans, res)

    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.axhline(0, color="0.85", lw=.8, zorder=0)
    ax.axvline(0, color="0.85", lw=.8, zorder=0)
    sc = ax.scatter(x, y, c=vuln, cmap=cmap, vmin=1, vmax=10, s=46,
                    alpha=.72, edgecolors="white", linewidths=.5, zorder=3)

    # archetype centroids + labels
    uniq = sorted(set(leans))
    for lean in uniq:
        idx = [i for i, l in enumerate(leans) if l == lean]
        cx, cy = x[idx].mean(), y[idx].mean()
        ax.scatter([cx], [cy], s=140, facecolors="none",
                   edgecolors="#222", linewidths=1.2, zorder=4)
        ax.annotate(lean, (cx, cy),
                    textcoords="offset points", xytext=(6, 5),
                    fontsize=8.4, fontweight="bold", color="#111", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=.72))

    # label a few extreme individuals by name
    order = np.argsort(x)
    for i in list(order[:4]) + list(order[-4:]):
        ax.annotate(names[i], (x[i], y[i]), textcoords="offset points",
                    xytext=(3, -10), fontsize=7, style="italic", color="#444", zorder=6)

    ax.set_xlabel("PC1  =  User Axis   (competent / expert  ←→  vulnerable / emotional)",
                  fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title("Persona space: 150 user archetypes in the top principal components\n"
                 "(layer 40, resp\\_mean; ○ = archetype centroid; colour = held-out vulnerability)",
                 fontsize=12.5)
    cb = fig.colorbar(sc, ax=ax, pad=.01, fraction=.045)
    cb.set_label("vulnerability tag (0–10)", fontsize=10)
    ax.grid(True, color="0.93", lw=.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = res / "figures" / "persona_space_labeled.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
