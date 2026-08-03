"""Factor analysis for the expanded user-space mapping.

The 289 users vary over 10 near-independent factors (max Cramer's V 0.09). For
each factor we ask how much of the model's evoked trait / Assistant-Axis /
persona it drives -- a clean variance decomposition that the correlated 5-tag
150-user study could not do.

  - one-way ANOVA eta^2 per (factor x readout): fraction of variance in the evoked
    readout attributable to that factor.
  - joint OLS (all one-hot factors) R^2 per readout; compare to the sum of single
    eta^2 (near-additive because the factors are decorrelated).
  - per-level means for the strongest factors; evoked-role associations per level.

    python -m src.useraxis.expanded_factor_analysis
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.bigfive import stimuli as BF

EXP = config.ROOT / "experiments" / "expanded_userspace"
MAP = EXP / "mapping"
FACTORS = ["competence", "vulnerability", "emotional_load", "urgency", "trust",
           "intent", "domain", "comm_style", "age", "on_behalf"]


def eta2_anova(y, groups):
    """one-way ANOVA: eta^2, F, p. y: array; groups: list of level labels."""
    levels = sorted(set(groups))
    gm = y.mean()
    ss_tot = float(np.sum((y - gm) ** 2))
    ss_bet = 0.0
    samples = []
    for lv in levels:
        yl = y[np.array(groups) == lv]
        samples.append(yl)
        ss_bet += len(yl) * (yl.mean() - gm) ** 2
    k, N = len(levels), len(y)
    ss_wit = ss_tot - ss_bet
    F = (ss_bet / (k - 1)) / (ss_wit / (N - k) + 1e-12) if k > 1 and N > k else float("nan")
    p = float(stats.f.sf(F, k - 1, N - k)) if np.isfinite(F) else float("nan")
    return ss_bet / (ss_tot + 1e-12), F, p, {lv: float(s.mean()) for lv, s in zip(levels, samples)}


def onehot(rows, factors):
    """design matrix (intercept + one-hot, drop first level per factor)."""
    cols = [np.ones(len(rows))]
    names = ["intercept"]
    for f in factors:
        levels = sorted({r["factors"][f] for r in rows})
        for lv in levels[1:]:
            cols.append(np.array([1.0 if r["factors"][f] == lv else 0.0 for r in rows]))
            names.append(f"{f}={lv}")
    return np.column_stack(cols), names


def main():
    data = json.loads((MAP / "persona_map.json").read_text())
    pmap = data["personas"]
    rows = []
    for pid, e in pmap.items():
        if not e.get("factors"):
            continue
        rows.append({"pid": pid, "factors": e["factors"],
                     "aa": e["aa_proj"], "role": e["top_role_vote"][0][0],
                     **{t: e["bigfive_z"][t] for t in BF.TRAITS}})
    N = len(rows)
    READOUTS = ["aa"] + list(BF.TRAITS)
    Y = {r: np.array([row[r] for row in rows]) for r in READOUTS}
    print(f"[factor] {N} personas; readouts {READOUTS}")

    # ---- eta^2 matrix (factor x readout) + per-factor detail ----
    eta = {f: {} for f in FACTORS}
    detail = {}
    for f in FACTORS:
        groups = [row["factors"][f] for row in rows]
        detail[f] = {}
        for r in READOUTS:
            e2, F, p, means = eta2_anova(Y[r], groups)
            eta[f][r] = {"eta2": round(e2, 3), "F": round(F, 2), "p": round(p, 4)}
            detail[f][r] = {lv: round(m, 3) for lv, m in means.items()}

    # ---- joint OLS R^2 per readout + sum of single eta^2 ----
    X, xnames = onehot(rows, FACTORS)
    joint = {}
    for r in READOUTS:
        beta, *_ = np.linalg.lstsq(X, Y[r], rcond=None)
        yhat = X @ beta
        ss_res = float(np.sum((Y[r] - yhat) ** 2))
        ss_tot = float(np.sum((Y[r] - Y[r].mean()) ** 2))
        R2 = 1 - ss_res / (ss_tot + 1e-12)
        joint[r] = {"joint_R2": round(R2, 3),
                    "sum_single_eta2": round(sum(eta[f][r]["eta2"] for f in FACTORS), 3)}

    # ---- evoked-role association per factor level ----
    role_assoc = {}
    for f in FACTORS:
        role_assoc[f] = {}
        for lv in sorted({row["factors"][f] for row in rows}):
            sub = [row for row in rows if row["factors"][f] == lv]
            c = Counter(row["role"] for row in sub)
            role_assoc[f][lv] = {"n": len(sub),
                                 "mean_aa": round(float(np.mean([row["aa"] for row in sub])), 2),
                                 "top_roles": c.most_common(3)}

    out = {"n": N, "eta2": eta, "joint": joint, "level_means": detail,
           "role_assoc": role_assoc,
           "factor_rank_by_aa": sorted(FACTORS, key=lambda f: -eta[f]["aa"]["eta2"]),
           "factor_rank_by_AGR": sorted(FACTORS, key=lambda f: -eta[f]["AGR"]["eta2"])}
    (MAP / "factor_analysis.json").write_text(json.dumps(out, indent=1))

    # ---- console summary ----
    print("\n=== eta^2 (fraction of evoked-readout variance from each factor) ===")
    hdr = "factor           " + " ".join(f"{r:>6}" for r in READOUTS)
    print(hdr)
    for f in FACTORS:
        print(f"{f:16} " + " ".join(f"{eta[f][r]['eta2']:6.2f}" for r in READOUTS))
    print("joint R^2        " + " ".join(f"{joint[r]['joint_R2']:6.2f}" for r in READOUTS))
    print("sum single eta^2 " + " ".join(f"{joint[r]['sum_single_eta2']:6.2f}" for r in READOUTS))
    print(f"\ntop factors driving Assistant-Axis: {out['factor_rank_by_aa'][:4]}")
    print(f"top factors driving Agreeableness : {out['factor_rank_by_AGR'][:4]}")
    for f in out["factor_rank_by_aa"][:3]:
        levs = detail[f]["aa"]
        print(f"  {f} -> AA by level: " + ", ".join(f"{lv} {v:+.2f}" for lv, v in sorted(levs.items(), key=lambda x: x[1])))

    # ---- figure: eta^2 heatmap ----
    M = np.array([[eta[f][r]["eta2"] for r in READOUTS] for f in FACTORS])
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=max(0.15, M.max()))
    ax.set_xticks(range(len(READOUTS))); ax.set_xticklabels([r.upper() for r in READOUTS])
    ax.set_yticks(range(len(FACTORS))); ax.set_yticklabels(FACTORS)
    for i in range(len(FACTORS)):
        for j in range(len(READOUTS)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < M.max() * 0.6 else "black", fontsize=8)
    ax.set_title("eta$^2$: variance in evoked readout explained by each user factor")
    fig.colorbar(im, label="eta$^2$"); fig.tight_layout()
    (MAP / "figures").mkdir(exist_ok=True)
    fig.savefig(MAP / "figures" / "factor_eta2.png", dpi=130)
    print("wrote", MAP / "factor_analysis.json", "+ figures/factor_eta2.png")


if __name__ == "__main__":
    main()
