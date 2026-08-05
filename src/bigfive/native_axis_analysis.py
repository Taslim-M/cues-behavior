"""Native behavioral factors vs Big Five for explaining the Assistant Axis.

Projects every one of the 275 role vectors (same L40 basis) onto (a) the Assistant
Axis, (b) the Big Five probes, and (c) the native behavioral directions, then
regresses AA on each block and reports R^2 / incremental R^2 / per-factor r.

    python -m src.bigfive.native_axis_analysis
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.bigfive import stimuli as BF

RES = Path("native_axis/results")
BF_DIR = Path("results/bigfive/llama-3.3-70b")
AA_VEC_DIR = Path("/dev/shm/aa_vectors")
AA_PATH = Path("results/useraxis/llama-3.3-70b/assistant_axis.npy")
AA_LAYER = 40


def r2_ols(X, y):
    """R^2 of OLS y ~ [1, X] (X standardized columns)."""
    Xb = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    yhat = Xb @ beta
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / (ss_tot + 1e-12)


def main():
    # ---- probes / axes ----
    bank = {t: np.load(BF_DIR / "direction_bank.npz")[t] for t in BF.TRAITS}
    sel = json.loads((BF_DIR / "stage1_selection.json").read_text())["selection"]
    probe_layer = {t: int(sel[t]["layer"]) for t in BF.TRAITS}
    aa = np.load(AA_PATH).astype(np.float32); aa_unit = aa[AA_LAYER] / (np.linalg.norm(aa[AA_LAYER]) + 1e-8)
    nd = np.load(RES / "native_dirs.npz")
    NATIVE = list(nd.keys())

    # ---- per-role projections from one consistent [80,8192] vector ----
    names, AAv, BFm, NATm = [], [], [], []
    for f in sorted(glob.glob(str(AA_VEC_DIR / "role_vectors" / "*.pt"))):
        name = os.path.basename(f)[:-3]
        v = torch.load(f).float().numpy()                    # [80,8192]
        names.append(name)
        AAv.append(float(v[AA_LAYER] @ aa_unit))
        BFm.append([float(v[probe_layer[t]] @ bank[t][probe_layer[t]]) for t in BF.TRAITS])
        NATm.append([float(v[AA_LAYER] @ nd[k]) for k in NATIVE])
    AAv = np.array(AAv); BFm = np.array(BFm); NATm = np.array(NATm)
    N = len(names)

    def z(M):
        return (M - M.mean(0)) / (M.std(0) + 1e-9)
    y = (AAv - AAv.mean()) / (AAv.std() + 1e-9)
    Xbf, Xnat = z(BFm), z(NATm)

    R2_bf = r2_ols(Xbf, y)
    R2_nat = r2_ols(Xnat, y)
    R2_all = r2_ols(np.column_stack([Xbf, Xnat]), y)
    out = {"n_personas": N, "native_factors": NATIVE,
           "R2_bigfive_probe_layers": round(R2_bf, 3), "R2_native": round(R2_nat, 3),
           "R2_combined": round(R2_all, 3),
           "incremental_native_over_bigfive": round(R2_all - R2_bf, 3),
           "incremental_bigfive_over_native": round(R2_all - R2_nat, 3)}

    # ---- CONTROL: Big Five built at L40 with the SAME contrastive method ----
    # isolates whether the native advantage is factor-choice or just layer/method.
    bf_l40_path = RES / "bigfive_l40_dirs.npz"
    if bf_l40_path.exists():
        bfl = np.load(bf_l40_path); BF5 = list(bfl.keys())
        BFL40 = np.array([[float(torch.load(f).float().numpy()[AA_LAYER] @ bfl[k]) for k in BF5]
                          for f in sorted(glob.glob(str(AA_VEC_DIR / "role_vectors" / "*.pt")))])
        R2_bf_l40 = r2_ols(z(BFL40), y)
        # also: single-factor groundedness (transparency about what drives native R^2)
        gi = NATIVE.index("groundedness") if "groundedness" in NATIVE else 0
        R2_ground = r2_ols(z(NATm[:, [gi]]), y)
        out["R2_bigfive_L40_contrastive_control"] = round(R2_bf_l40, 3)
        out["R2_groundedness_alone"] = round(R2_ground, 3)
        out["control_note"] = ("Big Five re-built at L40 with the identical contrastive method; "
                               "if this stays low while native is high, the advantage is factor-choice, "
                               "not the layer/method.")

    # per-factor correlation with AA + cosine of native dir to AA
    meta = json.loads((RES / "native_dirs_meta.json").read_text())["meta"]
    out["native_factor_r_with_AA"] = {
        k: {"r": round(float(stats.pearsonr(NATm[:, i], AAv)[0]), 3),
            "cos_dir_AA": meta[k]["cos_with_AA"]}
        for i, k in enumerate(NATIVE)}
    out["bigfive_r_with_AA"] = {t: round(float(stats.pearsonr(BFm[:, i], AAv)[0]), 3)
                                for i, t in enumerate(BF.TRAITS)}
    # face validity: top/bottom roles per native factor
    out["native_face_validity"] = {}
    for i, k in enumerate(NATIVE):
        order = np.argsort(NATm[:, i])
        out["native_face_validity"][k] = {
            "low": [names[j] for j in order[:5]], "high": [names[j] for j in order[-5:][::-1]]}

    (RES / "regression.json").write_text(json.dumps(out, indent=1))

    # native scores csv
    with open(RES / "native_scores.csv", "w") as fh:
        fh.write("role,AA," + ",".join(BF.TRAITS) + "," + ",".join(NATIVE) + "\n")
        for r in range(N):
            fh.write(f"{names[r]},{AAv[r]:.4f}," +
                     ",".join(f"{x:.4f}" for x in BFm[r]) + "," +
                     ",".join(f"{x:.4f}" for x in NATm[r]) + "\n")

    # ---- console summary ----
    print(f"=== native vs Big Five for explaining the Assistant Axis (N={N}) ===")
    print(f"  R^2  Big Five (probe layers) = {R2_bf:.3f}")
    if "R2_bigfive_L40_contrastive_control" in out:
        print(f"  R^2  Big Five (L40 contrastive control) = {out['R2_bigfive_L40_contrastive_control']:.3f}")
    print(f"  R^2  native (7 factors)      = {R2_nat:.3f}")
    if "R2_groundedness_alone" in out:
        print(f"  R^2  groundedness alone      = {out['R2_groundedness_alone']:.3f}")
    print(f"  R^2  combined                = {R2_all:.3f}")
    print(f"  incremental native over Big Five = {R2_all-R2_bf:+.3f}")
    print(f"  incremental Big Five over native = {R2_all-R2_nat:+.3f}")
    print("  native factor  r(AA)  cos(dir,AA)")
    for k in sorted(NATIVE, key=lambda k: -abs(out['native_factor_r_with_AA'][k]['r'])):
        e = out["native_factor_r_with_AA"][k]
        print(f"    {k:12} {e['r']:+.2f}   {e['cos_dir_AA']:+.2f}")
    print("  Big Five r(AA):", {t: out["bigfive_r_with_AA"][t] for t in BF.TRAITS})
    print("  face validity (high pole):")
    for k in NATIVE:
        print(f"    {k:12} high={out['native_face_validity'][k]['high']}")

    # ---- figure: R^2 comparison + per-factor r with AA ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.5))
    blabels = ["Big Five\n(probe L)", "Big Five\n(L40 ctrl)", "groundedness\nalone", "Native\n(7)", "Combined\n(12)"]
    bvals = [R2_bf, out.get("R2_bigfive_L40_contrastive_control", 0), out.get("R2_groundedness_alone", 0), R2_nat, R2_all]
    bcols = ["#9aa3b2", "#c2c8d2", "#7aa7e8", "#2f63d8", "#4f9e63"]
    a1.bar(blabels, bvals, color=bcols)
    for i, v in enumerate(bvals):
        a1.text(i, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
    a1.set_ylabel("R$^2$ (variance of Assistant-Axis explained)"); a1.set_ylim(0, 1.05)
    a1.set_title("Explaining the Assistant Axis")
    labels = NATIVE + list(BF.TRAITS)
    rs = [out["native_factor_r_with_AA"][k]["r"] for k in NATIVE] + \
         [out["bigfive_r_with_AA"][t] for t in BF.TRAITS]
    cols = ["#2f63d8"] * len(NATIVE) + ["#9aa3b2"] * len(BF.TRAITS)
    order = np.argsort(np.abs(rs))
    a2.barh([labels[i] for i in order], [rs[i] for i in order], color=[cols[i] for i in order])
    a2.axvline(0, color="#333", lw=.8); a2.set_xlabel("Pearson r with Assistant-Axis")
    a2.set_title("Per-factor correlation (blue=native, grey=Big Five)")
    fig.tight_layout(); (RES / "figures").mkdir(exist_ok=True)
    fig.savefig(RES / "figures" / "native_vs_bigfive.png", dpi=130)
    print("wrote", RES / "regression.json", "+ native_scores.csv + figures/native_vs_bigfive.png")


if __name__ == "__main__":
    main()
