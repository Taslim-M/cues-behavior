"""Analyse the trait_morph run: morph fractions, specificity, manipulation check.

    python -m src.bigfive.trait_morph_analysis --dir /dev/shm/trait_morph
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.bigfive import stimuli as S


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/dev/shm/trait_morph")
    args = ap.parse_args()
    d = Path(args.dir)
    V = np.load(d / "vectors.npz")
    rec = json.loads((d / "records.json").read_text())
    pairs = rec["pairs"]

    # distractor pool: every character's reference vector (base_A of its pair, ref_B)
    refs = {}
    for i, p in enumerate(pairs):
        refs[(i, "A")] = V[f"p{i}:base_A"]
        refs[(i, "B")] = V[f"p{i}:ref_B"]

    CONDS = ["target", "off_target", "wrong_dir"]
    out = {"pairs": [], "summary": {}}
    agg = {c: {"frac": [], "cosgain": []} for c in CONDS}
    dist_fracs = []

    for i, p in enumerate(pairs):
        a = V[f"p{i}:base_A"].astype(np.float64)
        b = V[f"p{i}:ref_B"].astype(np.float64)
        u = b - a
        L = np.linalg.norm(u)
        uhat = u / (L + 1e-9)
        T = p["trait"]

        # distractor baseline: fraction the TARGET edit travels toward OTHER chars
        vt = V[f"p{i}:target"].astype(np.float64)
        dfr = []
        for (j, side), rv in refs.items():
            if j == i:
                continue
            rv = rv.astype(np.float64)
            w = rv - a
            Lw = np.linalg.norm(w)
            if Lw < 1e-6:
                continue
            dfr.append(float((vt - a) @ (w / Lw) / Lw))
        dist_mean = float(np.mean(dfr)) if dfr else float("nan")
        dist_fracs.append(dist_mean)

        pr = {"pair": i, "trait": T, "A": p["A"], "B": p["B"], "note": p["note"],
              "dz_target": p["dz_target"], "off_trait": p["off_trait"],
              "AB_dist": round(L, 2), "cos_A_B": round(cos(a, b), 3),
              "dist_baseline_frac": round(dist_mean, 3), "cond": {}}

        # manipulation check reference levels
        readA = p["cond"]["base_A"]["readings"][T]
        readB = p["cond"]["ref_B"]["readings"][T]
        aaA = p["cond"]["base_A"]["aa_proj"]
        aaB = p["cond"]["ref_B"]["aa_proj"]

        for c in CONDS:
            v = V[f"p{i}:{c}"].astype(np.float64)
            frac = float((v - a) @ uhat / (L + 1e-9))
            cg = cos(v, b) - cos(a, b)
            readC = p["cond"][c]["readings"][T]
            aaC = p["cond"][c]["aa_proj"]
            # trait manipulation: fraction of A->B trait-reading gap traversed
            mfrac = ((readC - readA) / (readB - readA)) if abs(readB - readA) > 1e-6 else float("nan")
            pr["cond"][c] = {"morph_frac": round(frac, 3), "cos_gain": round(cg, 3),
                             "trait_read": round(readC, 3),
                             "trait_manip_frac": round(mfrac, 3),
                             "aa_proj": round(aaC, 3)}
            agg[c]["frac"].append(frac)
            agg[c]["cosgain"].append(cg)
        pr["read_A"] = round(readA, 3)
        pr["read_B"] = round(readB, 3)
        pr["aa_A"] = round(aaA, 3)
        pr["aa_B"] = round(aaB, 3)
        # specificity excess = target movement toward B minus generic drift baseline
        pr["specificity_excess"] = round(pr["cond"]["target"]["morph_frac"] - dist_mean, 3)
        # AA-morph fraction: how far the target edit slid the AA projection A->B
        aaT = p["cond"]["target"]["aa_proj"]
        pr["aa_morph_frac"] = round((aaT - aaA) / (aaB - aaA), 3) if abs(aaB - aaA) > 1e-6 else None
        out["pairs"].append(pr)

    for c in CONDS:
        out["summary"][c] = {"mean_morph_frac": round(float(np.mean(agg[c]["frac"])), 3),
                             "mean_cos_gain": round(float(np.mean(agg[c]["cosgain"])), 3)}
    out["summary"]["distractor_baseline_frac"] = round(float(np.nanmean(dist_fracs)), 3)

    # clean pairs = exclude the caveated EXT pair (trait EXT)
    clean = [pr for pr in out["pairs"] if pr["trait"] != "EXT"]
    n_pos = sum(1 for pr in clean if pr["cond"]["target"]["morph_frac"] > 0)
    manip_ok = sum(1 for pr in clean
                   if 0 < pr["cond"]["target"]["trait_manip_frac"])
    out["summary"]["clean_pairs"] = len(clean)
    out["summary"]["target_moves_toward_B"] = f"{n_pos}/{len(clean)}"
    out["summary"]["manipulation_worked"] = f"{manip_ok}/{len(clean)}"
    out["summary"]["mean_specificity_excess"] = round(
        float(np.mean([pr["specificity_excess"] for pr in clean])), 3)
    out["summary"]["mean_aa_morph_frac"] = round(
        float(np.mean([pr["aa_morph_frac"] for pr in clean if pr["aa_morph_frac"] is not None])), 3)
    n_aa_pos = sum(1 for pr in clean if pr["aa_morph_frac"] and pr["aa_morph_frac"] > 0.1)
    out["summary"]["aa_moves_toward_B"] = f"{n_aa_pos}/{len(clean)}"

    (d / "morph.json").write_text(json.dumps(out, indent=1))

    # ---- figure: morph fraction per pair, target vs controls ----
    labels = [f"{pr['A']}→{pr['B']}\n({pr['trait']})" for pr in out["pairs"]]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    cols = {"target": "#2f63d8", "off_target": "#9aa3b2", "wrong_dir": "#d5703f"}
    for k, c in enumerate(CONDS):
        vals = [pr["cond"][c]["morph_frac"] for pr in out["pairs"]]
        ax.bar(x + (k - 1) * w, vals, w, label=c, color=cols[c])
    dbase = [pr["dist_baseline_frac"] for pr in out["pairs"]]
    ax.plot(x, dbase, "k.", label="distractor baseline")
    ax.axhline(0, color="#333", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("morph fraction toward B (0=A, 1=B)")
    ax.set_title("Single-trait edit moves the L40 persona vector toward the target character")
    ax.legend(fontsize=9)
    fig.tight_layout()
    (d / "figures").mkdir(exist_ok=True)
    fig.savefig(d / "figures" / "morph_fractions.png", dpi=130)

    # console summary
    print("=== trait_morph results ===")
    print(f"{'pair':26} {'targ':>6} {'dist':>6} {'excess':>7} {'wrong':>6} {'aa_frac':>8}  aaA→aaB(aa*)")
    for pr in out["pairs"]:
        t = pr["cond"]["target"]; w_ = pr["cond"]["wrong_dir"]
        flag = "  (EXT caveat)" if pr["trait"] == "EXT" else ""
        print(f"{pr['A'][:11]+'->'+pr['B'][:11]:26} {t['morph_frac']:+6.2f} {pr['dist_baseline_frac']:+6.2f} "
              f"{pr['specificity_excess']:+7.2f} {w_['morph_frac']:+6.2f} "
              f"{(pr['aa_morph_frac'] or 0):+8.2f}  {pr['aa_A']:+.2f}→{pr['aa_B']:+.2f}({t['aa_proj']:+.2f}){flag}")
    s = out["summary"]
    print(f"\nmean morph frac: target {s['target']['mean_morph_frac']:+.2f} | "
          f"distractor {s['distractor_baseline_frac']:+.2f} | "
          f"SPECIFICITY EXCESS {s['mean_specificity_excess']:+.2f} | "
          f"wrong_dir {s['wrong_dir']['mean_morph_frac']:+.2f}")
    print(f"AA-morph: mean frac {s['mean_aa_morph_frac']:+.2f}, moves toward B {s['aa_moves_toward_B']}")
    print(f"clean pairs moving toward B (raw): {s['target_moves_toward_B']}")
    print("wrote", d / "morph.json", "+ figures/morph_fractions.png")


if __name__ == "__main__":
    main()
