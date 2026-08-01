"""Track A analysis: the full 275-role Big Five atlas.

Reads the atlas profiles (per-role Big Five z + Assistant-Axis projection, all from
our own rollouts) and produces:
  - H3 redo (self-consistent): regress each role's Assistant-Axis projection on its
    5 Big Five coordinates across all 275 roles -> R^2, standardised beta, residual.
  - correlation of each Big Five with the Assistant Axis across roles.
  - k-means clustering of roles in Big Five space -> persona archetypes.
  - atlas figure (roles in Big Five PC space, coloured by Assistant-Axis position).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.bigfive import stimuli as S

ATLAS = Path("results/bigfive/llama-3.3-70b/role_profiles_atlas")


def main():
    prof = json.loads((ATLAS / "role_bigfive_profiles.json").read_text())["per_role"]
    names = [n for n in prof if "aa_proj" in prof[n]]
    Z = np.array([[prof[n]["z_vs_roles"][t] for t in S.TRAITS] for n in names])   # [R,5]
    AA = np.array([prof[n]["aa_proj"] for n in names])
    out = {"n_roles": len(names)}

    # H3 redo: AA ~ Big Five
    Zc = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    aaz = (AA - AA.mean()) / (AA.std() + 1e-9)
    reg = LinearRegression().fit(Zc, aaz)
    out["H3_regression"] = {"r2": round(float(reg.score(Zc, aaz)), 3),
                            "beta": {t: round(float(b), 3) for t, b in zip(S.TRAITS, reg.coef_)},
                            "residual_ai_ness": round(float(1 - reg.score(Zc, aaz)), 3)}
    out["trait_AA_corr"] = {t: round(float(pearsonr(Z[:, i], AA)[0]), 3) for i, t in enumerate(S.TRAITS)}

    # clustering in Big Five space
    k = 6
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z)
    clusters = {}
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        centroid = {t: round(float(Z[idx, i].mean()), 2) for i, t in enumerate(S.TRAITS)}
        # roles nearest the centroid
        d = np.linalg.norm(Z[idx] - Z[idx].mean(0), axis=1)
        rep = [names[idx[j]] for j in np.argsort(d)[:6]]
        clusters[f"cluster_{c}"] = {"n": int(len(idx)), "centroid_z": centroid,
                                    "mean_aa": round(float(AA[idx].mean()), 2), "roles": rep}
    out["clusters"] = clusters

    # face validity extremes per trait
    out["extremes"] = {}
    for i, t in enumerate(S.TRAITS):
        o = np.argsort(Z[:, i])
        out["extremes"][t] = {"low": [names[j] for j in o[:4]], "high": [names[j] for j in o[-4:][::-1]]}

    (ATLAS / "atlas_analysis.json").write_text(json.dumps(out, indent=1))
    print(f"=== H3 (self-consistent, {len(names)} roles): R2={out['H3_regression']['r2']} "
          f"residual={out['H3_regression']['residual_ai_ness']}")
    print("  beta:", out["H3_regression"]["beta"])
    print("  trait~AA corr:", out["trait_AA_corr"])
    print("=== clusters (centroid z, mean AA) ===")
    for c, d in clusters.items():
        print(f"  {c} n={d['n']:3} AA={d['mean_aa']:+.2f} {d['centroid_z']}  e.g. {d['roles'][:4]}")

    # ---- atlas figure: roles in Big Five PC space, coloured by AA ----
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    U, s, Vt = np.linalg.svd(Zs, full_matrices=False)
    pc = U[:, :2] * s[:2]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=AA, cmap="RdBu_r", s=26, alpha=.85,
                    vmin=-np.abs(AA).max(), vmax=np.abs(AA).max())
    # label a few extremes on each PC
    lab = set(np.argsort(pc[:, 0])[:3]) | set(np.argsort(pc[:, 0])[-3:]) | \
          set(np.argsort(pc[:, 1])[:3]) | set(np.argsort(pc[:, 1])[-3:]) | \
          set(np.argsort(AA)[:4]) | set(np.argsort(AA)[-4:])
    for j in lab:
        ax.annotate(names[j], (pc[j, 0], pc[j, 1]), fontsize=7, alpha=.8)
    ax.set_xlabel("Big Five PC1"); ax.set_ylabel("Big Five PC2")
    ax.set_title(f"Personality atlas of {len(names)} LLM personas (colour = Assistant-Axis position)")
    fig.colorbar(sc, label="Assistant-Axis projection"); fig.tight_layout()
    (ATLAS / "figures").mkdir(exist_ok=True)
    fig.savefig(ATLAS / "figures" / "atlas_map.png", dpi=120)
    print("wrote atlas_analysis.json + atlas_map.png")


if __name__ == "__main__":
    main()
