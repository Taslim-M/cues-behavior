"""Analysis for `user_persona_mapping` (plan §5).

  - user-tag -> readout correlations (Q1, Q3): tags vs Big Five z, AA-projection
  - explicit vs implicit agreement (Q2)
  - evoked-role frequency / clustering (Q4)
  - figures + report.md
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.bigfive import stimuli as BF

R = Path("results/user_persona_mapping/llama-3.3-70b")
NUM_TAGS = ["expertise", "vulnerability", "trust", "emotional_load", "tech_literacy"]


def bh_fdr(pvals):
    p = np.asarray(pvals); n = len(p); order = np.argsort(p); ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]; out = np.empty(n); out[order] = np.clip(q, 0, 1); return out


def main():
    m = json.loads((R / "persona_map.json").read_text())
    P = m["personas"]
    ids = sorted(P)
    tags = {tg: np.array([P[i]["tags"].get(tg, np.nan) for i in ids], float) for tg in NUM_TAGS}
    bfz = {t: np.array([P[i]["bigfive_z"][t] for i in ids]) for t in BF.TRAITS}
    aap = np.array([P[i]["aa_proj"] for i in ids])

    out = {"n": len(ids), "tag_bigfive_corr": {}, "tag_aa_corr": {}, "explicit_implicit": {}, "roles": {}}

    # Q1: tag x Big Five correlation (+ FDR over the whole grid)
    grid_p = []
    for tg in NUM_TAGS:
        out["tag_bigfive_corr"][tg] = {}
        for t in BF.TRAITS:
            r, p = pearsonr(tags[tg], bfz[t])
            out["tag_bigfive_corr"][tg][t] = {"r": round(float(r), 3), "p": float(p)}
            grid_p.append((tg, t, p))
    qs = bh_fdr([p for *_, p in grid_p])
    for (tg, t, _), q in zip(grid_p, qs):
        out["tag_bigfive_corr"][tg][t]["q"] = round(float(q), 3)

    # Q3: tag x AA-projection
    for tg in NUM_TAGS:
        r, p = pearsonr(tags[tg], aap)
        out["tag_aa_corr"][tg] = {"r": round(float(r), 3), "p": round(float(p), 4)}

    # Q2: explicit vs implicit agreement (per persona, correlate the two 5-vectors)
    agrees = []
    for i in ids:
        e = P[i].get("bigfive_by_arm", {})
        if "explicit" in e and "implicit" in e:
            ve = [e["explicit"][t] for t in BF.TRAITS]; vi = [e["implicit"][t] for t in BF.TRAITS]
            if np.std(ve) > 1e-9 and np.std(vi) > 1e-9:
                agrees.append(spearmanr(ve, vi).statistic)
    out["explicit_implicit"] = {"mean_bigfive_profile_agreement_rho": round(float(np.nanmean(agrees)), 3),
                                "n": len(agrees)}

    # Q4: evoked-role frequency
    top1 = Counter(P[i]["top_role_vote"][0][0] for i in ids if P[i].get("top_role_vote"))
    out["roles"] = {"distinct_top1": len(top1), "coverage_top15": round(
        sum(c for _, c in top1.most_common(15)) / len(ids), 3), "most_common": top1.most_common(20)}

    (R / "analysis.json").write_text(json.dumps(out, indent=1))

    # ---- prints ----
    print("=== Q1 tag x Big Five (r; * = q<.05) ===")
    print(f"{'tag':14} " + " ".join(f"{t:>7}" for t in BF.TRAITS))
    for tg in NUM_TAGS:
        row = out["tag_bigfive_corr"][tg]
        print(f"{tg:14} " + " ".join(f"{row[t]['r']:+.2f}{'*' if row[t]['q']<.05 else ' '}" for t in BF.TRAITS))
    print("=== Q3 tag x AA-projection ===", {tg: out["tag_aa_corr"][tg]["r"] for tg in NUM_TAGS})
    print(f"=== Q2 explicit/implicit agreement: rho={out['explicit_implicit']['mean_bigfive_profile_agreement_rho']}")
    print(f"=== Q4 roles: {out['roles']['distinct_top1']} distinct top-1; top-15 cover "
          f"{out['roles']['coverage_top15']*100:.0f}% ===")
    print("  most-evoked:", out["roles"]["most_common"][:10])

    # ---- figures ----
    (R / "figures").mkdir(exist_ok=True)
    # Fig 1: tag x BigFive heatmap
    M = np.array([[out["tag_bigfive_corr"][tg][t]["r"] for t in BF.TRAITS] for tg in NUM_TAGS])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-.6, vmax=.6, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels(BF.TRAITS)
    ax.set_yticks(range(len(NUM_TAGS))); ax.set_yticklabels(NUM_TAGS)
    for i in range(len(NUM_TAGS)):
        for j in range(5):
            q = out["tag_bigfive_corr"][NUM_TAGS[i]][BF.TRAITS[j]]["q"]
            ax.text(j, i, f"{M[i,j]:+.2f}" + ("*" if q < .05 else ""), ha="center", va="center",
                    fontsize=8, color="w" if abs(M[i, j]) > .35 else "#222")
    ax.set_title("User tag × evoked Big Five (r; * FDR<.05)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(R / "figures" / "upm_tag_bigfive.png", dpi=120)

    # Fig 2: vulnerability vs AA-projection scatter
    fig2, ax2 = plt.subplots(figsize=(6, 4.2))
    ax2.scatter(tags["vulnerability"], aap, c=tags["expertise"], cmap="viridis", s=28, alpha=.8)
    ax2.set_xlabel("user vulnerability (tag)"); ax2.set_ylabel("model Assistant-Axis projection")
    ax2.set_title(f"Does a vulnerable user move the model? r={out['tag_aa_corr']['vulnerability']['r']:+.2f}")
    fig2.colorbar(ax2.collections[0], label="expertise"); fig2.tight_layout()
    fig2.savefig(R / "figures" / "upm_vuln_aa.png", dpi=120)

    # Fig 3: most-evoked roles
    mc = out["roles"]["most_common"][:15]
    fig3, ax3 = plt.subplots(figsize=(7, 4.2))
    ax3.barh([r for r, _ in mc][::-1], [c for _, c in mc][::-1], color="#4b57c9")
    ax3.set_xlabel("# personas (top-1 evoked)"); ax3.set_title("Which LLM personas do users evoke?")
    fig3.tight_layout(); fig3.savefig(R / "figures" / "upm_roles.png", dpi=120)
    print("wrote analysis.json + 3 figures")


if __name__ == "__main__":
    main()
