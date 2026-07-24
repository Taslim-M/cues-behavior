"""Phase A -- all-layer PCA map for the User Axis (CPU only, no GPU).

We saved per-persona mean activations at every one of the 80 layers (Stage E), so
we can ask where in the network the User-Axis signal lives -- for free. For each
readout and each layer we compute:

  * PC1 / PC2 variance-explained, and dims for 70% / 90% cumulative variance;
  * PC1 <-> vulnerability-contrast cosine (does PCA recover the vuln direction?);
  * PC1 ~ each held-out tag (Pearson + Spearman + BH-FDR) -> interpretability;
  * explicit <-> implicit arm agreement of PC1 loadings (robustness).

The crux: PC1 *variance* peaks at trivial early layers (surface/token features),
whereas the axis is most *meaningful* (recovers vulnerability, agrees across arms)
mid-network. This script quantifies that and picks the Phase-B steering shortlist:
a max-variance "variance arm" and interpretability/cosine-peak "meaning arm(s)".

Reuses helpers from compute_axis: pc1_per_layer, contrast_axis, cosine_rows, bh_fdr.

Run:
    python -m src.useraxis.analyze_layers
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from .. import config  # noqa: E402
from .extract import DEFAULT_MODEL, short_name  # noqa: E402
from .compute_axis import (  # noqa: E402
    TAG_SCALES, pc1_per_layer, contrast_axis, cosine_rows, bh_fdr,
)

READOUTS = [("resp_mean", ""), ("last_user", "/last_user")]
# mid-network window for the "meaning arm": exclude the first ~15% (surface/token)
# and last ~12% (output-head) layers where PC1 is dominated by non-persona structure.
WINDOW = (12, 64)


def orient_toward_vuln(pc1: np.ndarray, X: np.ndarray, vuln: np.ndarray) -> np.ndarray:
    """Flip each layer's PC1 so +proj correlates with higher vulnerability."""
    vuln_c = vuln - vuln.mean()
    out = pc1.copy()
    for l in range(pc1.shape[0]):
        proj = (X[:, l, :] - X[:, l, :].mean(0)) @ pc1[l]
        if np.dot(proj, vuln_c) < 0:
            out[l] = -pc1[l]
    return out


def full_var_spectrum(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-layer PC1 var, PC2 var, dims-for-70%, dims-for-90%."""
    n, L, d = X.shape
    pc1v = np.zeros(L); pc2v = np.zeros(L)
    d70 = np.zeros(L, int); d90 = np.zeros(L, int)
    for l in range(L):
        Xl = X[:, l, :] - X[:, l, :].mean(0, keepdims=True)
        s = np.linalg.svd(Xl, compute_uv=False)
        ev = (s ** 2) / (s ** 2).sum()
        pc1v[l], pc2v[l] = ev[0], ev[1]
        cum = np.cumsum(ev)
        d70[l] = int(np.argmax(cum >= 0.70) + 1)
        d90[l] = int(np.argmax(cum >= 0.90) + 1)
    return pc1v, pc2v, d70, d90


def analyze_readout(base: Path) -> dict:
    X = np.load(base / "persona_vectors.npy")            # [N, L, D], full into RAM
    idx = json.loads((base / "persona_index.json").read_text())
    pids = idx["persona_ids"]; tags = idx["tags"]
    tagmat = {t: np.array([tags[p][t] for p in pids], float) for t in TAG_SCALES}
    vuln = tagmat["vulnerability"]

    pc1, _ = pc1_per_layer(X)
    pc1 = orient_toward_vuln(pc1, X, vuln)
    cax, _ = contrast_axis(X, [tags[p] for p in pids])   # [L, D]
    cos = cosine_rows(pc1, cax)                            # [L]
    pc1v, pc2v, d70, d90 = full_var_spectrum(X)

    L = X.shape[1]
    # per-layer PC1 loadings -> interpretability vs each tag (Pearson + Spearman)
    tag_pearson = {t: np.zeros(L) for t in TAG_SCALES}
    tag_spear = {t: np.zeros(L) for t in TAG_SCALES}
    tag_p = {t: np.zeros(L) for t in TAG_SCALES}
    for l in range(L):
        load = (X[:, l, :] - X[:, l, :].mean(0)) @ pc1[l]
        for t in TAG_SCALES:
            tag_pearson[t][l], tag_p[t][l] = stats.pearsonr(load, tagmat[t])
            tag_spear[t][l], _ = stats.spearmanr(load, tagmat[t])

    # arm agreement per layer: explicit-only vs implicit-only PC1 loadings
    Xe = np.load(base / "persona_vectors_explicit.npy")
    Xi = np.load(base / "persona_vectors_implicit.npy")
    arm = np.zeros(L)
    for l in range(L):
        le = (Xe[:, l, :] - Xe[:, l, :].mean(0)) @ pc1[l]
        li = (Xi[:, l, :] - Xi[:, l, :].mean(0)) @ pc1[l]
        arm[l] = np.corrcoef(le, li)[0, 1]

    return {
        "pc1_var": pc1v.tolist(), "pc2_var": pc2v.tolist(),
        "dims70": d70.tolist(), "dims90": d90.tolist(),
        "contrast_cos": cos.tolist(),
        "pc1_vuln_pearson": tag_pearson["vulnerability"].tolist(),
        "pc1_vuln_spearman": tag_spear["vulnerability"].tolist(),
        "pc1_tag_pearson": {t: tag_pearson[t].tolist() for t in TAG_SCALES},
        "arm_agreement": arm.tolist(),
    }


def validate(base: Path, prof: dict) -> None:
    """Gate: per-layer PC1-var & contrast-cosine reproduce the shipped arrays."""
    v = json.loads((base / "axis_validation.json").read_text())
    a = np.array(prof["pc1_var"]); b = np.array(v["pc1_var_ratio_per_layer"])
    c = np.abs(prof["contrast_cos"]); d = np.abs(v["pc1_contrast_cosine_per_layer"])
    dv = float(np.max(np.abs(a - b))); dc = float(np.max(np.abs(c - d)))
    print(f"  [gate] max|PC1var diff|={dv:.2e}  max|cos diff|={dc:.2e}", flush=True)
    assert dv < 1e-4 and dc < 1e-4, "per-layer arrays diverge from axis_validation.json"


def pick_candidates(profiles: dict) -> dict:
    """Choose the Phase-B steering shortlist from the resp_mean profile."""
    p = profiles["resp_mean"]
    var = np.array(p["pc1_var"]); cos = np.abs(p["contrast_cos"])
    interp = np.abs(p["pc1_vuln_pearson"])
    lo, hi = WINDOW
    win = np.arange(len(var))
    inwin = (win >= lo) & (win < hi)

    def argmax_in(arr):
        m = np.where(inwin, arr, -np.inf)
        return int(np.argmax(m))

    cands = {
        "variance_arm_global": int(np.argmax(var)),        # expect L3 (surface)
        "interp_peak": argmax_in(interp),                  # meaning arm
        "cosine_peak": argmax_in(cos),                     # meaning arm
        "variance_peak_in_window": argmax_in(var),
        "baseline_L40": 40,
    }
    # spread a few extra layers across the window for a dose-response curve
    spread = [16, 24, 32, 48, 56]
    layers = sorted(set(list(cands.values()) + spread))
    return {"named": cands, "layers": layers, "window": list(WINDOW)}


def plot_profiles(profiles: dict, cands: dict, out: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    colors = {"resp_mean": "#1f77b4", "last_user": "#d62728"}
    panels = [
        ("pc1_var", "PC1 variance explained", None),
        ("contrast_cos", "PC1 $\\leftrightarrow$ contrast cosine", (-0.1, 1.0)),
        ("pc1_vuln_pearson", "PC1 $\\sim$ vulnerability (Pearson $r$)", (-0.1, 1.0)),
        ("arm_agreement", "explicit $\\leftrightarrow$ implicit arm agreement", (-0.1, 1.0)),
    ]
    L = len(profiles["resp_mean"]["pc1_var"])
    for ax, (key, title, ylim) in zip(axes, panels):
        for ro in ("resp_mean", "last_user"):
            y = np.array(profiles[ro][key])
            if key == "contrast_cos":
                y = np.abs(y)
            ax.plot(range(L), y, color=colors[ro], lw=1.6,
                    label=ro.replace("_", "-"))
        ax.set_ylabel(title, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.axvspan(cands["window"][0], cands["window"][1], color="0.9", zorder=0)
        for l in cands["layers"]:
            ax.axvline(l, color="0.75", lw=0.6, zorder=0)
        ax.axvline(40, color="black", lw=1.0, ls="--", zorder=1)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=9, loc="upper right")
    # annotate named candidates on the top panel
    var = np.array(profiles["resp_mean"]["pc1_var"])
    for name, l in cands["named"].items():
        axes[0].annotate(f"L{l}", (l, var[l]), fontsize=7,
                         textcoords="offset points", xytext=(0, 4), ha="center")
    axes[-1].set_xlabel("layer")
    fig.suptitle("User-Axis PCA signal across network depth "
                 "(dashed = L40; shaded = meaning window)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> None:
    model = short_name(DEFAULT_MODEL)
    root = config.ROOT / "results" / "useraxis" / model
    profiles = {}
    for ro, sub in READOUTS:
        base = root / sub.lstrip("/") if sub else root
        print(f"[{ro}] analyzing all layers ...", flush=True)
        prof = analyze_readout(base)
        validate(base, prof)
        profiles[ro] = prof

    ana = root / "analysis"; ana.mkdir(parents=True, exist_ok=True)
    (ana / "layer_profile.json").write_text(json.dumps(profiles, indent=2))
    print(f"wrote {ana / 'layer_profile.json'}", flush=True)

    cands = pick_candidates(profiles)
    (ana / "candidate_layers.json").write_text(json.dumps(cands, indent=2))
    print("candidate layers:", cands["layers"], flush=True)
    print("named:", json.dumps(cands["named"]), flush=True)

    figs = root / "figures"; figs.mkdir(parents=True, exist_ok=True)
    plot_profiles(profiles, cands, figs / "layer_profile.png")

    # report copy
    rep = config.ROOT / "report" / "figures" / "layer_profile.png"
    if rep.parent.exists():
        import shutil
        shutil.copyfile(figs / "layer_profile.png", rep)
        print(f"copied -> {rep}", flush=True)


if __name__ == "__main__":
    main()
