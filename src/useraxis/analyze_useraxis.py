"""Interpretation plots + report for the User Axis (CPU, matplotlib).

Consumes the Stage-E artifacts (persona_vectors.npy, pca.npz, user_axis.npy,
axis_validation.json, interpretation.json) and writes figures/ + a markdown
report under results/useraxis/<model>/.

Figures (PLAN.md Stage E):
  variance_curve.png       per-PC + cumulative explained variance
  pc_scatter_3d.png        personas in PC1-3, colored by vulnerability tag
  pc1_vs_tags.png          PC1 loading vs each 0-10 tag (r, q annotated)
  cosine_per_layer.png     PC1 <-> vulnerability-contrast cosine per layer
  arm_agreement.png        explicit-arm vs implicit-arm PC1 loading, per persona

Run:
    python -m src.useraxis.analyze_useraxis
    python -m src.useraxis.analyze_useraxis --readout last_user
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import config
from .compute_axis import TAG_SCALES
from .extract import DEFAULT_MODEL, short_name


def _dir_for(res_dir: Path, readout: str) -> Path:
    return res_dir if readout == "resp_mean" else res_dir / readout


def plot_variance(fig_dir: Path, pca: dict, val: dict) -> None:
    var = pca["explained_variance_ratio"]
    n = min(30, len(var))
    xs = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(xs, var[:n] * 100, alpha=0.6, label="per PC")
    ax.plot(xs, np.cumsum(var[:n]) * 100, "o-", ms=3, color="C1", label="cumulative")
    ax.axhline(70, ls="--", lw=0.8, c="gray")
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance explained (%)")
    ax.set_title(f"User-persona space PCA (layer {val['chosen_layer']}); "
                 f"70% at {val['dims_for_70pct']} PCs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "variance_curve.png", dpi=150)
    plt.close(fig)


def plot_scatter3(fig_dir: Path, loadings: np.ndarray, tags: list[dict]) -> None:
    vuln = np.array([t["vulnerability"] for t in tags], float)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    k = min(3, loadings.shape[1])
    coords = [loadings[:, i] if i < k else np.zeros(len(loadings)) for i in range(3)]
    sc = ax.scatter(*coords, c=vuln, cmap="coolwarm", s=35)
    fig.colorbar(sc, label="vulnerability tag", shrink=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Personas in top-3 PCs")
    fig.tight_layout()
    fig.savefig(fig_dir / "pc_scatter_3d.png", dpi=150)
    plt.close(fig)


def plot_pc1_tags(fig_dir: Path, loadings: np.ndarray, tags: list[dict],
                  interp: list[dict]) -> None:
    stats = {(r["pc"], r["tag"]): r for r in interp}
    fig, axes = plt.subplots(1, len(TAG_SCALES), figsize=(4 * len(TAG_SCALES), 3.6),
                             sharey=True)
    for ax, tag in zip(axes, TAG_SCALES):
        y = np.array([t[tag] for t in tags], float)
        ax.scatter(y, loadings[:, 0], s=18, alpha=0.7)
        m, b = np.polyfit(y, loadings[:, 0], 1)
        xs = np.linspace(y.min(), y.max(), 10)
        ax.plot(xs, m * xs + b, c="C1", lw=1)
        r = stats.get((1, tag), {})
        ax.set_title(f"{tag}\nr={r.get('r', float('nan')):+.2f} "
                     f"q={r.get('q_FDR', float('nan')):.1e}")
        ax.set_xlabel("tag (0-10)")
    axes[0].set_ylabel("PC1 loading")
    fig.tight_layout()
    fig.savefig(fig_dir / "pc1_vs_tags.png", dpi=150)
    plt.close(fig)


def plot_cosine(fig_dir: Path, val: dict) -> None:
    cos = np.array(val["pc1_contrast_cosine_per_layer"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(cos)), cos)
    ax.axvline(val["chosen_layer"], ls="--", lw=0.8, c="gray",
               label=f"chosen layer {val['chosen_layer']}")
    ax.axhline(0, lw=0.6, c="black")
    ax.set_xlabel("layer")
    ax.set_ylabel("cos(PC1, vulnerability contrast)")
    ax.set_title("PC1 vs contrast-axis agreement per layer "
                 f"(max |cos| {np.abs(cos).max():.2f} @ L{np.abs(cos).argmax()})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "cosine_per_layer.png", dpi=150)
    plt.close(fig)


def plot_arm_agreement(fig_dir: Path, out_dir: Path, axis: np.ndarray,
                       layer: int, tags: list[dict]) -> float | None:
    fe = out_dir / "persona_vectors_explicit.npy"
    fi = out_dir / "persona_vectors_implicit.npy"
    if not (fe.exists() and fi.exists()):
        return None
    Xe = np.load(fe)[:, layer, :]
    Xi = np.load(fi)[:, layer, :]
    le = (Xe - Xe.mean(0)) @ axis[layer]
    li = (Xi - Xi.mean(0)) @ axis[layer]
    r = float(np.corrcoef(le, li)[0, 1])
    vuln = np.array([t["vulnerability"] for t in tags], float)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(le, li, c=vuln, cmap="coolwarm", s=30)
    fig.colorbar(sc, label="vulnerability tag")
    ax.set_xlabel("PC1 loading (explicit arm)")
    ax.set_ylabel("PC1 loading (implicit arm)")
    ax.set_title(f"Elicitation-arm agreement on the User Axis (r={r:+.2f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "arm_agreement.png", dpi=150)
    plt.close(fig)
    return r


def write_report(out_dir: Path, val: dict, interp: list[dict], arm_r,
                 readout: str, n_personas: int) -> None:
    sig = sorted([r for r in interp if r["q_FDR"] < 0.05], key=lambda r: r["q_FDR"])
    lines = [
        f"# User Axis report - {readout}",
        "",
        f"- personas: **{n_personas}**, Stage-D keep fraction "
        f"**{val['keep_fraction']:.1%}**",
        f"- chosen layer: **{val['chosen_layer']}** | 70% variance in "
        f"**{val['dims_for_70pct']}** PCs (target <= ~20)",
        f"- PC1 <-> vulnerability-contrast cosine at chosen layer: "
        f"**{val['pc1_contrast_cosine_at_layer']:+.3f}** "
        f"(best layer L{val['best_cosine_layer']}; paper analogue > 0.71, "
        "sanity bar > ~0.6)",
        f"- explicit <-> implicit PC1 loading correlation: "
        f"**{val['explicit_implicit_pc1_loading_corr']}**",
        "",
        "## Significant PC x tag correlations (FDR q < 0.05)",
        "",
        "| PC | tag | r | p | q |",
        "|---|---|---|---|---|",
    ]
    for r in sig:
        lines.append(f"| PC{r['pc']} | {r['tag']} | {r['r']:+.2f} "
                     f"| {r['p']:.1e} | {r['q_FDR']:.1e} |")
    if not sig:
        lines.append("| - | (none significant) | | | |")
    lines += ["", "## Figures", ""]
    for f in ("variance_curve", "pc_scatter_3d", "pc1_vs_tags",
              "cosine_per_layer", "arm_agreement"):
        lines.append(f"![{f}](figures/{f}.png)")
    (out_dir / "report.md").write_text("\n".join(lines))


def main():
    args = parse_args()
    res_dir = config.RESULTS_DIR / "useraxis" / short_name(
        {"llama-3.3-70b": DEFAULT_MODEL}.get(args.model, args.model))
    out_dir = _dir_for(res_dir, args.readout)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pca = dict(np.load(out_dir / "pca.npz", allow_pickle=True))
    val = json.loads((out_dir / "axis_validation.json").read_text())
    interp = json.loads((out_dir / "interpretation.json").read_text())
    index = json.loads((out_dir / "persona_index.json").read_text())
    tags = [index["tags"][p] for p in index["persona_ids"]]
    loadings = pca["loadings"]
    axis = np.load(out_dir / "user_axis.npy")[0]  # PC1 per layer

    plot_variance(fig_dir, pca, val)
    plot_scatter3(fig_dir, loadings, tags)
    plot_pc1_tags(fig_dir, loadings, tags, interp)
    plot_cosine(fig_dir, val)
    arm_r = plot_arm_agreement(fig_dir, out_dir, axis, val["chosen_layer"], tags)
    write_report(out_dir, val, interp, arm_r, args.readout, len(tags))
    print(f"Wrote figures + report.md -> {out_dir}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="User-Axis interpretation plots/report")
    ap.add_argument("--model", default="llama-3.3-70b")
    ap.add_argument("--readout", choices=("resp_mean", "last_user"),
                    default="resp_mean")
    return ap.parse_args()


if __name__ == "__main__":
    main()
