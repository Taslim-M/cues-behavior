"""Stage 3 §5.3 -- decompose the published Assistant Axis into Big Five (H3).

Pure vector algebra on same-space resid_post directions (no model needed):

  1. cos(AA_L, w_trait_L) per trait per layer -- which Big Five axes the Assistant
     Axis aligns with. (Both are [80,8192] resid_post directions; Stage F validated
     the published axis matches our extraction convention at cos 0.99997.)
  2. Assistant fingerprint: the default-Assistant vector's Big Five profile,
     z-scored against the 274 role vectors (same published convention) -- H3 predicts
     high AGR/CSN/EST.
  3. Regression of each role's AA-projection on its 5 Big Five coordinates -> R² and
     standardised beta per trait ("is Assistant-ness a linear combo of Big Five?"),
     with residual (1 - R²) = the AI-ness component orthogonal to human personality.

    python -m src.bigfive.decompose_aa --dir results/bigfive/llama-3.3-70b \
        --aa results/useraxis/llama-3.3-70b/assistant_axis.npy \
        --aa-vectors /dev/shm/aa_vectors
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from src.bigfive import stimuli as S


def unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--aa", required=True)
    ap.add_argument("--aa-vectors", default="/dev/shm/aa_vectors")
    ap.add_argument("--summary-layer", type=int, default=40)
    args = ap.parse_args()
    d = Path(args.dir)

    AA = np.load(args.aa).astype(np.float32)                       # [80, D]
    bank = {t: np.load(d / "direction_bank.npz")[t] for t in S.TRAITS}  # each [80, D]
    default_vec = torch.load(Path(args.aa_vectors) / "default_vector.pt").float().numpy()
    role_files = sorted(glob.glob(str(Path(args.aa_vectors) / "role_vectors" / "*.pt")))
    role_names = [os.path.basename(f)[:-3] for f in role_files]
    roles = np.stack([torch.load(f).float().numpy() for f in role_files])  # [R, 80, D]
    n_layers = AA.shape[0]
    probe_opt = json.loads((d / "stage1_selection.json").read_text())["selection"]

    # -- 1. cos(AA_L, w_trait_L) per layer --
    cos_by_layer = {t: [] for t in S.TRAITS}
    for li in range(n_layers):
        ah = unit(AA[li])
        for t in S.TRAITS:
            cos_by_layer[t].append(float(ah @ unit(bank[t][li])))

    # -- 2. Assistant fingerprint (z vs 274 roles), per layer + at summary layer --
    fingerprint = {}
    for li_name, li in [("summary_L%d" % args.summary_layer, args.summary_layer)]:
        prof = {}
        for t in S.TRAITS:
            w = unit(bank[t][li])
            role_proj = roles[:, li, :] @ w
            dproj = float(default_vec[li] @ w)
            z = (dproj - role_proj.mean()) / (role_proj.std() + 1e-8)
            prof[t] = {"z_vs_roles": float(z), "raw_proj": dproj}
        fingerprint[li_name] = prof
    # also at each trait's own probe-optimal layer
    fp_probeopt = {}
    for t in S.TRAITS:
        li = probe_opt[t]["layer"]
        w = unit(bank[t][li])
        role_proj = roles[:, li, :] @ w
        fp_probeopt[t] = {"layer": li,
                          "z_vs_roles": float((default_vec[li] @ w - role_proj.mean())
                                              / (role_proj.std() + 1e-8))}

    # -- 3. regression: role AA-projection ~ role Big Five coords, at summary layer --
    def regress_at(li):
        aa_u = unit(AA[li])
        y = roles[:, li, :] @ aa_u                      # role AA-projection [R]
        Xf = np.stack([roles[:, li, :] @ unit(bank[t][li]) for t in S.TRAITS], axis=1)  # [R,5]
        Xz = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-8)
        yz = (y - y.mean()) / (y.std() + 1e-8)
        reg = LinearRegression().fit(Xz, yz)
        r2 = reg.score(Xz, yz)
        beta = {t: float(b) for t, b in zip(S.TRAITS, reg.coef_)}
        # residual roles (top |resid|)
        resid = yz - reg.predict(Xz)
        order = np.argsort(-np.abs(resid))[:8]
        top_resid = [{"role": role_names[i], "resid": float(resid[i]),
                      "aa_proj_z": float(yz[i])} for i in order]
        return {"r2": float(r2), "beta": beta, "residual_frac": float(1 - r2),
                "top_residual_roles": top_resid}

    reg_summary = regress_at(args.summary_layer)
    reg_grid = {li: regress_at(li)["r2"] for li in range(0, n_layers, 4)}

    # role Big Five profiles (face-validity spot check) at summary layer
    li = args.summary_layer
    role_bigfive = {}
    Xf = {t: roles[:, li, :] @ unit(bank[t][li]) for t in S.TRAITS}
    for t in S.TRAITS:
        z = (Xf[t] - Xf[t].mean()) / (Xf[t].std() + 1e-8)
        order = np.argsort(z)
        role_bigfive[t] = {"highest": [role_names[i] for i in order[-5:][::-1]],
                           "lowest": [role_names[i] for i in order[:5]]}

    # H3 verdict
    fp = fingerprint["summary_L%d" % args.summary_layer]
    h3_fingerprint_ok = all(fp[t]["z_vs_roles"] > 0 for t in ("AGR", "CSN", "EST"))
    h3_regression_ok = reg_summary["r2"] > 0.5
    H3 = {"fingerprint_AGR_CSN_EST_positive": h3_fingerprint_ok,
          "regression_r2_gt_0.5": h3_regression_ok,
          "confirmed": h3_fingerprint_ok and h3_regression_ok}

    out = {
        "summary_layer": args.summary_layer,
        "cos_AA_bigfive": {"per_layer": cos_by_layer,
                           "at_summary": {t: cos_by_layer[t][args.summary_layer]
                                          for t in S.TRAITS}},
        "assistant_fingerprint": {"at_summary": fp, "at_probe_optimal": fp_probeopt},
        "aa_regression_on_bigfive": {"at_summary": reg_summary, "r2_by_layer": reg_grid},
        "role_bigfive_facevalidity": role_bigfive,
        "H3": H3,
    }
    (d / "stage3_decomposition.json").write_text(json.dumps(out, indent=1))

    print("==== H3: Assistant Axis in Big Five coordinates ====")
    print(f"summary layer L{args.summary_layer}")
    print("cos(AA, trait):", {t: round(cos_by_layer[t][args.summary_layer], 3) for t in S.TRAITS})
    print("Assistant fingerprint (z vs roles):",
          {t: round(fp[t]["z_vs_roles"], 2) for t in S.TRAITS})
    print(f"AA ~ BigFive regression: R2={reg_summary['r2']:.3f}  "
          f"residual(AI-ness)={reg_summary['residual_frac']:.3f}")
    print("  beta:", {t: round(v, 2) for t, v in reg_summary["beta"].items()})
    print(f"H3 {'CONFIRMED' if H3['confirmed'] else 'NOT confirmed'} "
          f"(fingerprint AGR/CSN/EST>0: {h3_fingerprint_ok}, R2>0.5: {h3_regression_ok})")
    print("wrote", d / "stage3_decomposition.json")


if __name__ == "__main__":
    main()
