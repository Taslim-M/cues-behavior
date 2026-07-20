"""Stage F4 - figures for the Assistant-User Axis link.

Reads the Stage-F artifacts (analysis/projections.npz + the tier JSONs) and
renders the six figures referenced by the report's Stage-F section into
results/useraxis/<model>/figures/link_*.png.

Run:
    python -m src.useraxis.analyze_link
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import config  # noqa: E402
from .compute_axis import TAG_SCALES  # noqa: E402
from .extract import DEFAULT_MODEL, short_name  # noqa: E402

TARGET_LAYER = 40
LAYER_GRID = (20, 30, 40, 50, 60)
ARMS = ("explicit", "implicit")
USERPOS_VARIANTS = ("userpos_resp", "userpos_lastA", "userpos_lastB")
VAR_LABEL = {"userpos_resp": "resp_mean\n(circular)",
             "userpos_lastA": "last_user\n(resp axis)",
             "userpos_lastB": "last_user\n(cross-rep)"}


def _res_dir() -> Path:
    return config.RESULTS_DIR / "useraxis" / short_name(DEFAULT_MODEL)


def _fig_dir() -> Path:
    d = _res_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load():
    ad = _res_dir() / "analysis"
    P = {k: v for k, v in np.load(ad / "projections.npz", allow_pickle=False).items()}
    geo = json.load(open(ad / "geometry.json"))
    t2 = json.load(open(ad / "tier2_tags.json"))
    t3 = json.load(open(ad / "tier3_crossrep.json"))
    return P, geo, t2, t3


def _persona_means(P, layer):
    keep = P["keep"]
    pids = sorted(set(P["persona_id"][keep].tolist()))
    drift = P[f"drift_L{layer}"]
    out = {"pid": pids}
    for name, src in [("drift", drift), ("vulnerability", P["vulnerability"])] + \
                     [(v, P[f"{v}_L{layer}"]) for v in USERPOS_VARIANTS]:
        out[name] = np.array([src[keep & (P["persona_id"] == p)].mean() for p in pids])
    return out


def fig_cosine_per_layer(geo):
    cu = geo["per_layer_cos_u_a"]
    cul = geo["per_layer_cos_ulast_a"]
    band = geo["random_cosine_band"]
    x = np.arange(len(cu))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhspan(-band, band, color="0.85", label=f"random band ±{band:.3f}")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.plot(x, cu, label="cos(User Axis, Assistant Axis)", color="C0")
    ax.plot(x, cul, label="cos(User Axis[last_user], Assistant Axis)",
            color="C1", ls="--")
    ax.axvline(TARGET_LAYER, color="C3", lw=1, ls=":")
    ax.text(TARGET_LAYER + 0.5, ax.get_ylim()[1] * 0.9, "ℓ=40", color="C3")
    ax.set_xlabel("layer"); ax.set_ylabel("cosine")
    ax.set_title("Geometry: the two axes are near-orthogonal (small mechanical floor)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(_fig_dir() / "link_cosine_per_layer.png", dpi=140)
    plt.close(fig)


def fig_persona_vuln_vs_drift(P):
    pm = _persona_means(P, TARGET_LAYER)
    x, y = pm["vulnerability"], pm["drift"]
    b1, b0 = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, s=18, alpha=0.6, color="C0")
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color="C3", lw=2,
            label=f"OLS slope {b1:+.3f}, r={r:+.2f}")
    ax.set_xlabel("held-out vulnerability tag (0–10)")
    ax.set_ylabel(f"mean Assistant-Axis drift  −(â·r) @ℓ{TARGET_LAYER}")
    ax.set_title("Independent IV (n=150 personas): vulnerability vs drift — null")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(_fig_dir() / "link_persona_vuln_vs_drift.png", dpi=140)
    plt.close(fig)


def fig_crossrep_scatter(P):
    keep = P["keep"]
    x = P[f"userpos_lastB_L{TARGET_LAYER}"][keep]
    y = P[f"drift_L{TARGET_LAYER}"][keep]
    arm = P["arm"][keep]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for a, c in zip(ARMS, ("C0", "C1")):
        m = arm == a
        ax.scatter(x[m], y[m], s=6, alpha=0.25, color=c, label=a)
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b0 + b1 * xs, color="k", lw=2, label=f"slope {b1:+.3f}")
    ax.set_xlabel(f"User-Axis position on pre-response token (last_user) @ℓ{TARGET_LAYER}")
    ax.set_ylabel(f"drift −(â·r) @ℓ{TARGET_LAYER}")
    ax.set_title("Cross-representation (de-circularized): weak & negative")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(_fig_dir() / "link_crossrep_scatter.png", dpi=140)
    plt.close(fig)


def fig_circular_vs_crossrep_bar(t3):
    r = t3["layers"][str(TARGET_LAYER)]["rollout"]
    vals = [r[v]["partial_r"] for v in USERPOS_VARIANTS]
    cis = [r[v]["boot_ci95"] for v in USERPOS_VARIANTS]
    err = [[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]]
    x = np.arange(len(vals))
    colors = ["C7", "C9", "C3"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x, vals, yerr=err, capsize=5, color=colors)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([VAR_LABEL[v] for v in USERPOS_VARIANTS], fontsize=9)
    ax.set_ylabel(f"partial r (userpos → drift) @ℓ{TARGET_LAYER}\n(control length, arm, probe)")
    ax.set_title("De-circularization: positive link collapses & flips sign")
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.01 if v >= 0 else -0.03), f"{v:+.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(_fig_dir() / "link_circular_vs_crossrep_bar.png", dpi=140)
    plt.close(fig)


def fig_tag_forest(t2):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    tags = list(TAG_SCALES)
    y = np.arange(len(tags))
    for off, arm, c in [(-0.18, "explicit", "C0"), (0.18, "implicit", "C1")]:
        betas, ps = [], []
        for t in tags:
            jo = t2["arms"][arm]["persona"]["joint_ols"][t]
            betas.append(jo["beta"]); ps.append(jo["p"])
        ax.scatter(betas, y + off, color=c, label=arm, zorder=3, s=30)
        for b, yy, p in zip(betas, y + off, ps):
            ax.plot([0, b], [yy, yy], color=c, lw=1, alpha=0.5)
    ax.axvline(0, color="0.4", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(tags)
    ax.set_xlabel(f"joint-OLS β (tag → drift) @ℓ{TARGET_LAYER}, per arm")
    ax.set_title("Which user tags predict drift (independent IVs)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(_fig_dir() / "link_tag_forest.png", dpi=140)
    plt.close(fig)


def fig_layer_sweep(t2, t3):
    layers = list(LAYER_GRID)
    vuln = [t2["layers"][str(L)]["rollout"]["beta_vuln"] for L in layers]
    resp = [t3["layers"][str(L)]["rollout"]["userpos_resp"]["partial_r"] for L in layers]
    lastB = [t3["layers"][str(L)]["rollout"]["userpos_lastB"]["partial_r"] for L in layers]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="0.5", lw=0.8)
    ax.plot(layers, resp, "-o", color="C7", label="userpos_resp → drift (circular)")
    ax.plot(layers, lastB, "-o", color="C3", label="userpos_lastB → drift (cross-rep)")
    ax.plot(layers, vuln, "-s", color="C2", label="β vulnerability tag → drift")
    ax.axvline(TARGET_LAYER, color="0.6", ls=":", lw=1)
    ax.set_xlabel("layer"); ax.set_ylabel("effect (partial r / β)")
    ax.set_title("Effect vs depth: circular signal decays, clean signal stays ≤0")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(_fig_dir() / "link_layer_sweep.png", dpi=140)
    plt.close(fig)


def main():
    P, geo, t2, t3 = _load()
    fig_cosine_per_layer(geo)
    fig_persona_vuln_vs_drift(P)
    fig_crossrep_scatter(P)
    fig_circular_vs_crossrep_bar(t3)
    fig_tag_forest(t2)
    fig_layer_sweep(t2, t3)
    print(f"wrote 6 link_*.png to {_fig_dir()}")


if __name__ == "__main__":
    main()
