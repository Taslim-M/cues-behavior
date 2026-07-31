"""Does unsupervised PCA of the character activations recover the Big Five?

Mirrors the Assistant-Axis / User-Axis recipe (one vector per persona -> center
across personas -> PCA per layer), but on our 406 character-mean vectors, and asks:
  Q1  Is the activation cloud low-dimensional? (variance curve)
  Q2  Do Big Five traits emerge as top PCs? (PC <-> trait z-score correlations)
  Q3  Are our SUPERVISED M2 probe directions aligned with the top PCs, or
      orthogonal to them (the Big Five paper's SVD != regression finding)?
  Q4  Is any single PC a clean personality axis (AA-style) or is each trait
      smeared across several PCs (needs supervision)?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

from src.bigfive import stimuli as S

D = Path("results/bigfive/llama-3.3-70b")
ACT = Path("/dev/shm/bf_acts/acts_characters")


def main():
    idx = json.loads((ACT / "index.json").read_text())
    prof = json.loads((D / "character_profiles.json").read_text())
    z_by_id = {p["id"]: p["z"] for p in prof}
    bank = {t: np.load(D / "direction_bank.npz")[t] for t in S.TRAITS}   # [80,8192] unit M2 dirs
    char_of_row = [r["character_id"] for r in idx]
    order = list(dict.fromkeys(char_of_row))                              # 406, stable order
    row_of_char = {c: [] for c in order}
    for i, c in enumerate(char_of_row):
        row_of_char[c].append(i)
    Z = {t: np.array([z_by_id[c][t] for c in order]) for t in S.TRAITS}

    layers = [24, 30, 31, 36, 40, 50]
    report = {"layers": {}, "recipe": "character-mean vectors, center-per-layer, PCA (AA-style)"}

    for L in layers:
        acts = np.load(ACT / "acts_prompt_mean.npy", mmap_mode="r")[:, L, :]  # [4060,8192]
        # one vector per character = mean over its 10 instruction rows (AA "role vector")
        Xc = np.stack([np.asarray(acts[row_of_char[c]]).mean(0) for c in order])  # [406,8192]
        Xc = Xc - Xc.mean(0, keepdims=True)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = s**2 / (s**2).sum()
        scores = U * s                                     # [406, n_pc] PC coordinates
        n70 = int(np.searchsorted(np.cumsum(var), 0.70) + 1)
        n90 = int(np.searchsorted(np.cumsum(var), 0.90) + 1)

        # Q2: correlate each of top-10 PCs with each trait z-score
        pc_trait = {}
        for t in S.TRAITS:
            cors = [abs(pearsonr(scores[:, k], Z[t])[0]) for k in range(min(10, scores.shape[1]))]
            best_pc = int(np.argmax(cors))
            pc_trait[t] = {"best_pc": best_pc + 1, "best_abs_r": round(cors[best_pc], 3),
                           "r_top5_pcs": [round(c, 2) for c in cors[:5]]}

        # Q3: alignment of supervised M2 probe with the PC basis
        probe_align = {}
        for t in S.TRAITS:
            w = bank[t][L]; w = w / (np.linalg.norm(w) + 1e-9)
            cos_pcs = Vt @ w                                # cos with each PC (Vt rows unit)
            k = min(10, len(cos_pcs))
            # fraction of the probe captured by the top-k PC subspace
            frac_top5 = float(np.sqrt((cos_pcs[:5] ** 2).sum()))
            frac_top20 = float(np.sqrt((cos_pcs[:min(20, len(cos_pcs))] ** 2).sum()))
            probe_align[t] = {"max_abs_cos_top10": round(float(np.abs(cos_pcs[:k]).max()), 3),
                              "cos_pc1": round(float(cos_pcs[0]), 3),
                              "frac_in_top5_pcs": round(frac_top5, 3),
                              "frac_in_top20_pcs": round(frac_top20, 3)}

        report["layers"][L] = {
            "var_pc1_5": [round(float(v), 3) for v in var[:5]],
            "n_pcs_for_70pct": n70, "n_pcs_for_90pct": n90,
            "Q2_pc_recovers_trait": pc_trait,
            "Q3_supervised_probe_vs_PCs": probe_align,
        }
        print(f"[L{L}] var PC1-5={np.round(var[:5],3)}  70%@{n70}PCs 90%@{n90}PCs")
        print(f"      trait best-PC |r|: " +
              " ".join(f"{t}=PC{pc_trait[t]['best_pc']}({pc_trait[t]['best_abs_r']})" for t in S.TRAITS))
        print(f"      probe frac in top-5 PCs: " +
              " ".join(f"{t}={probe_align[t]['frac_in_top5_pcs']}" for t in S.TRAITS))

    # Q4 contrast: raw per-sample PCA at L40 -- do top PCs capture instruction/topic?
    L = 40
    acts = np.asarray(np.load(ACT / "acts_prompt_mean.npy", mmap_mode="r")[:, L, :])  # [4060,8192]
    Xr = acts - acts.mean(0, keepdims=True)
    _, sr, Vtr = np.linalg.svd(Xr, full_matrices=False)
    varr = sr**2 / (sr**2).sum()
    # instruction id per row
    instr = [r["instruction_id"] for r in idx]
    uinstr = sorted(set(instr))
    scores_r = Xr @ Vtr[:5].T
    # eta^2 of PC1 explained by instruction identity (between-group var / total)
    def eta2(pc):
        g = {u: [] for u in uinstr}
        for v, ins in zip(pc, instr): g[ins].append(v)
        gm = {u: np.mean(g[u]) for u in uinstr}; tot = np.var(pc)
        bet = np.mean([(gm[ins] - pc.mean())**2 for ins in instr])
        return float(bet / (tot + 1e-12))
    report["Q4_raw_persample_L40"] = {
        "var_pc1_5": [round(float(v), 3) for v in varr[:5]],
        "pc1_variance_explained_by_instruction_eta2": round(eta2(scores_r[:, 0]), 3),
        "pc2_instruction_eta2": round(eta2(scores_r[:, 1]), 3),
        "note": "high eta2 => top PC of raw activations tracks the ALPACA INSTRUCTION, not personality",
    }
    print(f"[raw L40] PC1 instruction-eta2={report['Q4_raw_persample_L40']['pc1_variance_explained_by_instruction_eta2']}"
          f" PC2={report['Q4_raw_persample_L40']['pc2_instruction_eta2']}")

    (D / "pca_analysis.json").write_text(json.dumps(report, indent=1))
    print("wrote", D / "pca_analysis.json")


if __name__ == "__main__":
    main()
