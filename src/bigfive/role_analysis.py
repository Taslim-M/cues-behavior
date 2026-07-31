"""Analyze role Big Five profiles: Assistant-like vs drifted, central tendency + spread.

Consumes results/bigfive/llama-3.3-70b/role_profiles/*.jsonl (per-response readings)
and produces group comparisons, an AA-projection correlation, a within-role spread
analysis, and figures for the report/explainer.
"""
from __future__ import annotations
import glob, json, os
from pathlib import Path
import numpy as np
import torch
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.bigfive import stimuli as S

R = Path("results/bigfive/llama-3.3-70b")
PR = R / "role_profiles"
ASSISTANT_LIKE = ["assistant", "summarizer", "consultant", "instructor", "planner",
                  "organizer", "analyst", "researcher"]
DRIFTED = ["eldritch", "leviathan", "void", "wraith", "ghost",
           "tree", "vampire", "absurdist"]


def aa_proj():
    AA = np.load("results/useraxis/llama-3.3-70b/assistant_axis.npy").astype(np.float32)
    u = AA[40] / np.linalg.norm(AA[40])
    out = {}
    for f in glob.glob("/dev/shm/aa_vectors/role_vectors/*.pt"):
        out[Path(f).stem] = float(torch.load(f).float().numpy()[40] @ u)
    out["default"] = float(torch.load("/dev/shm/aa_vectors/default_vector.pt").float().numpy()[40] @ u)
    return out


def load():
    roles = {}
    for f in sorted(glob.glob(str(PR / "*.jsonl"))):
        recs = [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
        roles[Path(f).stem] = {t: np.array([r["read"][t] for r in recs]) for t in S.TRAITS}
    return roles


def main():
    roles = load()
    if not roles:
        print("no role files yet"); return
    AAp = aa_proj()
    # pooled standardization across all readings so numbers are comparable/interpretable
    pooled = {t: np.concatenate([roles[r][t] for r in roles]) for t in S.TRAITS}
    mu = {t: pooled[t].mean() for t in S.TRAITS}; sd = {t: pooled[t].std() or 1 for t in S.TRAITS}
    def z(r, t): return (roles[r][t] - mu[t]) / sd[t]

    summary = {"roles": {}, "groups": {}, "aa_correlation": {}, "spread": {}}
    for r in roles:
        summary["roles"][r] = {"n": int(len(roles[r][S.TRAITS[0]])), "aa_proj": round(AAp.get(r, float("nan")), 3),
            "traits": {t: {"mean_z": round(float(z(r, t).mean()), 3),
                           "std": round(float(roles[r][t].std()), 3),
                           "range": round(float(roles[r][t].max() - roles[r][t].min()), 3)}
                       for t in S.TRAITS}}

    # group means (Assistant-like vs drifted)
    for grp, names in [("assistant_like", ASSISTANT_LIKE), ("drifted", DRIFTED)]:
        present = [n for n in names if n in roles]
        summary["groups"][grp] = {
            "roles": present,
            "trait_mean_z": {t: round(float(np.mean([z(n, t).mean() for n in present])), 3) for t in S.TRAITS},
            "trait_mean_spread": {t: round(float(np.mean([roles[n][t].std() for n in present])), 3) for t in S.TRAITS},
        }

    # AA-projection <-> trait mean correlation across roles
    common = [r for r in roles if r in AAp]
    for t in S.TRAITS:
        xs = np.array([AAp[r] for r in common]); ys = np.array([z(r, t).mean() for r in common])
        rr, pp = pearsonr(xs, ys) if len(common) > 2 else (float("nan"), float("nan"))
        summary["aa_correlation"][t] = {"r": round(float(rr), 3), "p": round(float(pp), 4)}

    # spread: is within-role Big Five spread wider for drifted roles?
    for grp, names in [("assistant_like", ASSISTANT_LIKE), ("drifted", DRIFTED)]:
        present = [n for n in names if n in roles]
        allstd = [roles[n][t].std() for n in present for t in S.TRAITS]
        summary["spread"][grp] = round(float(np.mean(allstd)), 3)

    (R / "role_analysis.json").write_text(json.dumps(summary, indent=1))
    print("=== group Big Five (mean z) ===")
    for grp in ("assistant_like", "drifted"):
        print(f"  {grp:14}", summary["groups"][grp]["trait_mean_z"])
    print("=== AA-proj <-> trait correlation ===", {t: summary["aa_correlation"][t]["r"] for t in S.TRAITS})
    print(f"=== mean within-role spread: assistant_like={summary['spread']['assistant_like']} "
          f"drifted={summary['spread']['drifted']} ===")

    # ---- figures ----
    (R / "figures").mkdir(exist_ok=True)
    # Fig 1: grouped profile bars
    fig, ax = plt.subplots(figsize=(8, 4.4)); x = np.arange(5); w = 0.38
    al = [summary["groups"]["assistant_like"]["trait_mean_z"][t] for t in S.TRAITS]
    dr = [summary["groups"]["drifted"]["trait_mean_z"][t] for t in S.TRAITS]
    ax.bar(x - w/2, al, w, label="Assistant-like roles", color="#d5703f")
    ax.bar(x + w/2, dr, w, label="Drifted roles", color="#1f938c")
    ax.set_xticks(x); ax.set_xticklabels(S.TRAITS); ax.axhline(0, c="k", lw=.6)
    ax.set_ylabel("Big Five reading (z, pooled)"); ax.legend()
    ax.set_title("Big Five profile: Assistant-like vs drifted roles")
    fig.tight_layout(); fig.savefig(R / "figures" / "bf_roles_profile.png", dpi=120)

    # Fig 2: per-role spread (mean std across traits), ordered by AA proj
    fig2, ax2 = plt.subplots(figsize=(8, 4.4))
    order = sorted(roles, key=lambda r: AAp.get(r, 0))
    spreads = [np.mean([roles[r][t].std() for t in S.TRAITS]) for r in order]
    cols = ["#1f938c" if r in DRIFTED else ("#d5703f" if r in ASSISTANT_LIKE else "#888") for r in order]
    ax2.bar(range(len(order)), spreads, color=cols)
    ax2.set_xticks(range(len(order))); ax2.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("within-role Big Five spread (mean std)")
    ax2.set_title("Personality consistency by role (left = drifted, right = Assistant-like)")
    fig2.tight_layout(); fig2.savefig(R / "figures" / "bf_roles_spread.png", dpi=120)
    print("wrote role_analysis.json + 2 figures")


if __name__ == "__main__":
    main()
