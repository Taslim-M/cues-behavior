"""Project the no-persona baseline onto the User Axis and locate it (CPU, no API).

Reads the baseline vectors captured by `baseline_axis.py`, centers them by the SAME
150-persona mean used to build the shipped axis, and projects onto PC1 (and PC2/PC3)
at the analysis layers. Reports, per variant x readout x layer:

  * z-score + percentile of the baseline projection within the 150-persona
    distribution  -- the headline "where does the default user land";
  * implied vulnerability: the projection mapped through the persona PC1~vulnerability
    regression -> "the default user reads as vulnerability ~ X/10";
  * a bootstrap CI on the projection (from the per-sample vectors);
  * the nearest personas.

Gate: the recomputed persona PC1~vulnerability correlation must reproduce the shipped
value (+0.78 resp_mean / +0.86 last_user at L40), confirming centering + orientation.

Outputs analysis/baseline/placement.json + figures/baseline_axis.png.

Run:
    python -m src.useraxis.analyze_baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from .. import config
from .extract import DEFAULT_MODEL, short_name
from .baseline_axis import ANALYSIS_LAYERS
from .compute_axis import TAG_SCALES

READOUTS = [("resp_mean", ""), ("last_user", "last_user")]
PRIMARY_LAYER = 40
VAR_COLORS = {"none": "#0072B2", "neutral_sys": "#E69F00"}


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def pcs_from(X: np.ndarray, layer: int, k: int = 3):
    """Economy SVD of centered X[:,layer] -> (mu, components[k,D])."""
    Xl = X[:, layer, :].astype(np.float64)
    mu = Xl.mean(0)
    _, _, Vt = np.linalg.svd(Xl - mu, full_matrices=False)
    return mu, Vt[:k]


def boot_ci(x: np.ndarray, B: int = 2000):
    rng = np.random.default_rng(0)
    means = np.array([rng.choice(x, size=len(x), replace=True).mean()
                      for _ in range(B)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_names() -> dict[str, str]:
    out = {}
    for line in (config.ROOT / "generate_synthetic_data" / "user_personas.jsonl"
                 ).read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            out[p["persona_id"]] = p["name"]
    return out


def main() -> None:
    model = short_name(DEFAULT_MODEL)
    root = config.RESULTS_DIR / "useraxis" / model
    base = root / "analysis" / "baseline"
    store = np.load(base / "vectors.npz")
    meta = json.loads((base / "meta.json").read_text())
    variants = meta["variants"]
    names = load_names()

    result = {"variants": variants, "primary_layer": PRIMARY_LAYER, "readouts": {}}
    for ro, sub in READOUTS:
        rroot = root / sub if sub else root
        X = np.load(rroot / "persona_vectors.npy")
        axis = np.load(rroot / "user_axis.npy")            # [2, L, D]
        idx = json.loads((rroot / "persona_index.json").read_text())
        pids = idx["persona_ids"]
        tags = idx["tags"]
        vuln = np.array([tags[p]["vulnerability"] for p in pids], float)

        result["readouts"][ro] = {}
        for L in ANALYSIS_LAYERS:
            mu = X[:, L, :].astype(np.float64).mean(0)
            # PC1 from the SHIPPED axis (oriented + -> high vulnerability)
            pc1 = unit(axis[0, L].astype(np.float64))
            p1 = (X[:, L, :].astype(np.float64) - mu) @ pc1
            r_vuln, _ = stats.pearsonr(p1, vuln)
            if r_vuln < 0:                # keep + = vulnerable for reporting
                pc1, p1, r_vuln = -pc1, -p1, -r_vuln
            # PC2/PC3 from a fresh SVD (sign oriented toward emotional_load / by vuln)
            _, comps = pcs_from(X, L, 3)
            emo = np.array([tags[p]["emotional_load"] for p in pids], float)
            extra = {}
            for kk, y, nm in ((2, emo, "PC2"), (3, vuln, "PC3")):
                c = unit(comps[kk - 1])
                pp = (X[:, L, :].astype(np.float64) - mu) @ c
                if stats.spearmanr(pp, y)[0] < 0:
                    c, pp = -c, -pp
                extra[nm] = (c, pp)
            # PC1 -> vulnerability linear map
            a, b0 = np.polyfit(p1, vuln, 1)

            layer_rec = {"pc1_vuln_pearson": float(r_vuln),
                         "persona_proj_sd": float(p1.std()), "variants": {}}
            for variant in variants:
                bvec = store[f"{variant}__{ro}__mean"][L].astype(np.float64)
                bproj = float((bvec - mu) @ pc1)
                z = (bproj - p1.mean()) / (p1.std() or 1.0)
                pct = float((p1 < bproj).mean() * 100)
                implied = float(np.clip(a * bproj + b0, 0, 10))
                # per-sample CI
                li = list(ANALYSIS_LAYERS).index(L)
                svec = store[f"{variant}__{ro}__samples"][:, li, :].astype(np.float64)
                sproj = (svec - mu) @ pc1
                lo, hi = boot_ci(sproj)
                z_lo, z_hi = (lo - p1.mean()) / p1.std(), (hi - p1.mean()) / p1.std()
                order = np.argsort(np.abs(p1 - bproj))[:5]
                rec = {
                    "pc1_proj": bproj, "pc1_z": float(z), "pc1_percentile": pct,
                    "implied_vulnerability": implied,
                    "proj_ci95": [lo, hi], "z_ci95": [float(z_lo), float(z_hi)],
                    "nearest_personas": [
                        {"pid": pids[i], "name": names.get(pids[i], "?"),
                         "proj": float(p1[i]), "vuln": tags[pids[i]]["vulnerability"]}
                        for i in order],
                }
                for nm, (c, pp) in extra.items():
                    bp = float((bvec - mu) @ c)
                    rec[f"{nm.lower()}_z"] = float((bp - pp.mean()) / (pp.std() or 1))
                    rec[f"{nm.lower()}_percentile"] = float((pp < bp).mean() * 100)
                layer_rec["variants"][variant] = rec
            result["readouts"][ro][f"L{L}"] = layer_rec
        del X

    (base / "placement.json").write_text(json.dumps(result, indent=2))

    # ---- console summary (primary readout/layer) ---- #
    prim = result["readouts"]["resp_mean"][f"L{PRIMARY_LAYER}"]
    print(f"\n=== No-persona baseline on PC1 (resp_mean, L{PRIMARY_LAYER}) ===")
    print(f"gate: PC1~vulnerability r = {prim['pc1_vuln_pearson']:+.2f} "
          f"(reported +0.78)\n")
    for variant in variants:
        r = prim["variants"][variant]
        print(f"[{variant:11s}] PC1 z={r['pc1_z']:+.2f} "
              f"(95% CI {r['z_ci95'][0]:+.2f}..{r['z_ci95'][1]:+.2f}) "
              f"| percentile {r['pc1_percentile']:.0f} "
              f"| implied vuln {r['implied_vulnerability']:.1f}/10 "
              f"| PC2 z={r['pc2_z']:+.2f} PC3 z={r['pc3_z']:+.2f}")
        print("   nearest:", ", ".join(f"{p['name']}(v{p['vuln']})"
                                        for p in r["nearest_personas"]))
    plot(result, root / "figures" / "baseline_axis.png")


def plot(result: dict, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = short_name(DEFAULT_MODEL)
    root = config.RESULTS_DIR / "useraxis" / model
    X = np.load(root / "persona_vectors.npy")
    axis = np.load(root / "user_axis.npy")
    idx = json.loads((root / "persona_index.json").read_text())
    tags = idx["tags"]
    vuln = np.array([tags[p]["vulnerability"] for p in idx["persona_ids"]], float)
    L = PRIMARY_LAYER
    mu = X[:, L, :].astype(np.float64).mean(0)
    pc1 = unit(axis[0, L].astype(np.float64))
    p1 = (X[:, L, :].astype(np.float64) - mu) @ pc1
    if stats.pearsonr(p1, vuln)[0] < 0:
        p1 = -p1

    store = np.load(root / "analysis" / "baseline" / "vectors.npz")
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.hist(p1, bins=24, color="#b8c4d0", edgecolor="white")
    ax.axvline(p1.mean(), color="#555", ls="--", lw=1.5)
    ax.text(p1.mean(), ax.get_ylim()[1] * 0.96, " persona mean", color="#555",
            fontsize=9, va="top")
    for variant, c in VAR_COLORS.items():
        key = f"{variant}__resp_mean__mean"
        if key not in store:
            continue
        b = float((store[key][L].astype(np.float64) - mu) @ pc1)
        ax.axvline(b, color=c, lw=2.5)
        ax.text(b, ax.get_ylim()[1] * (0.86 if variant == "none" else 0.74),
                f" {variant}", color=c, fontsize=9, va="top")
    ax.set_xlabel("PC1 projection  (← competent / expert      vulnerable / crisis →)",
                  fontsize=10)
    ax.set_ylabel("personas", fontsize=10)
    ax.set_title(f"Where the profile-less user lands (resp_mean, L{L})", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
