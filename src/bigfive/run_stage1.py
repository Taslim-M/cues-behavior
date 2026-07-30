"""Stage 1 §3.3-3.5 + gate G1 -- derive Big Five probes, evaluate, select, gate.

Reads the character activations (Stage 1 §3.2) and character profiles (Stage 0b),
derives a probe direction per (trait x layer x position x method), evaluates each
on held-out characters and on the never-seen adjective set, selects a
probe-optimal direction per trait, checks basis cross-talk, and runs gate G1.

Math is the dual-form ridge validated in `derive.py` (w = X^T (K+lambda I)^-1 y),
but the heavy loop runs in torch on GPU: for each (layer, position) we
eigendecompose the train Gram (and the 5 CV-fold Grams) ONCE and reuse the
factorization across all five traits and the whole lambda grid. Kernel-space
prediction (K_test_train @ dual) means test evaluation never touches the
8192-dim space.

Samples are the (character x instruction) rows; labels are the character's trait
z-score. Splits are at CHARACTER level so a character's 10 instruction rows never
straddle train/test. Test metrics are reported at character level (projections
averaged over a character's instructions, then correlated with its z-score).

Usage
-----
    python -m src.bigfive.run_stage1 --acts-dir /dev/shm/bf_acts \
        --profiles results/bigfive/llama-3.3-70b/character_profiles.json \
        --out-dir results/bigfive/llama-3.3-70b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.bigfive import stimuli as S
from src.bigfive.derive import ALPHAS, character_split

POSITIONS_EVAL = ("last_prompt", "prompt_mean")   # gen_mean is zero in no-gen mode
METHODS = ("M1", "M2", "M3")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
def _eig(K: torch.Tensor):
    evals, evecs = torch.linalg.eigh(K)
    return evals, evecs


def _dual_weights(Vty, evals, evecs, alpha):
    """dual = evecs @ ((evecs^T y)/(evals+a)); returns the [n] dual vector."""
    return evecs @ (Vty / (evals + alpha))


def derive_and_eval(Xtr, ytr_by_trait, Xte, char_of_test, z_te_by_trait,
                    Xadj_by_trait):
    """All methods x traits for one (layer, position). Torch tensors on DEV.

    Xtr [ntr,d], Xte [nte,d]; ytr/z_te dicts trait->tensor/np; Xadj dict
    trait->(Xpos,Xneg) tensors. Returns nested dict of metrics + the chosen
    direction vectors (numpy) per (trait, method).
    """
    ntr = Xtr.shape[0]
    # Shared train Gram + eig (reused across traits, alphas, and M2 final fit)
    K = Xtr @ Xtr.T
    evals, evecs = _eig(K)
    Kte = Xte @ Xtr.T                     # [nte, ntr], for kernel-space test pred

    # 5 CV folds over TRAIN ROWS -- but rows are grouped by character. We fold on
    # characters so CV never leaks instructions across the fold boundary.
    g = np.random.default_rng(0)
    tr_chars = list(dict.fromkeys(char_of_test["train_chars"]))
    g.shuffle(tr_chars)
    fold_of = {c: i % 5 for i, c in enumerate(tr_chars)}
    row_char = char_of_test["train_row_char"]
    fold_rows = [np.array([r for r in range(ntr) if fold_of[row_char[r]] == f])
                 for f in range(5)]
    # Pre-eig each fold's train-Gram once (reused across traits + alphas).
    fold_eig = []
    for f in range(5):
        tr_idx = np.concatenate([fold_rows[j] for j in range(5) if j != f])
        ti = torch.as_tensor(tr_idx, device=DEV)
        Kf = Xtr[ti] @ Xtr[ti].T
        fev, fvec = _eig(Kf)
        Kval = Xtr[torch.as_tensor(fold_rows[f], device=DEV)] @ Xtr[ti].T
        fold_eig.append((tr_idx, fold_rows[f], fev, fvec, Kval))

    out = {}
    for trait in S.TRAITS:
        y = ytr_by_trait[trait]                       # [ntr] torch
        ymean = y.mean()
        yc = y - ymean

        # ---- M2 alpha via character-grouped 5-fold CV (kernel space) ----
        best_a, best_score = None, -1e9
        for a in ALPHAS:
            preds, tgts = [], []
            for (tr_idx, val_idx, fev, fvec, Kval) in fold_eig:
                yf = y[torch.as_tensor(tr_idx, device=DEV)]
                yfc = yf - yf.mean()
                dual = _dual_weights(fvec.T @ yfc, fev, fvec, float(a))
                p = (Kval @ dual).cpu().numpy()
                preds.append(p)
                tgts.append(y[torch.as_tensor(val_idx, device=DEV)].cpu().numpy())
            p = np.concatenate(preds); t = np.concatenate(tgts)
            rho = spearmanr(p, t).statistic if np.std(p) > 1e-9 else 0.0
            if rho > best_score:
                best_score, best_a = rho, float(a)

        def eval_dir(w_np, dual_test_pred):
            """character-level test rho/R2 + adjective AUC for a direction."""
            # test projections -> average per character -> correlate with z
            proj = dual_test_pred
            by_char = {}
            for r, c in enumerate(char_of_test["test_row_char"]):
                by_char.setdefault(c, []).append(proj[r])
            chars = list(by_char)
            pv = np.array([np.mean(by_char[c]) for c in chars])
            zv = np.array([z_te_by_trait[trait][c] for c in chars])
            rho = spearmanr(pv, zv).statistic if np.std(pv) > 1e-9 else 0.0
            r2 = float(np.corrcoef(pv, zv)[0, 1] ** 2) if np.std(pv) > 1e-9 else 0.0
            # adjective AUC
            Xpos, Xneg = Xadj_by_trait[trait]
            wp = (Xpos @ torch.as_tensor(w_np, device=DEV)).cpu().numpy()
            wn = (Xneg @ torch.as_tensor(w_np, device=DEV)).cpu().numpy()
            scores = np.concatenate([wp, wn])
            labels = np.concatenate([np.ones(len(wp)), np.zeros(len(wn))])
            auc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else 0.5
            return rho, r2, auc

        res_methods = {}
        # ---- M2 final fit on full train ----
        dualM2 = _dual_weights(evecs.T @ yc, evals, evecs, best_a)
        wM2 = (Xtr.T @ dualM2)
        # sign fix using train projection
        if spearmanr((Xtr @ wM2).cpu().numpy(), y.cpu().numpy()).statistic < 0:
            wM2 = -wM2; dualM2 = -dualM2
        testpredM2 = (Kte @ dualM2).cpu().numpy()
        rho, r2, auc = eval_dir(wM2.cpu().numpy(), testpredM2)
        res_methods["M2"] = dict(alpha=best_a, cv_rho=best_score,
                                 test_rho=rho, test_r2=r2, adj_auc=auc,
                                 w=wM2.cpu().numpy())

        # ---- M1: average within rounded-z bins, ridge ----
        zc = np.round(y.cpu().numpy() * 2) / 2
        uk = np.unique(zc)
        Xa = torch.stack([Xtr[torch.as_tensor(np.where(zc == k)[0], device=DEV)].mean(0)
                          for k in uk])
        ya = torch.as_tensor([y[torch.as_tensor(np.where(zc == k)[0], device=DEV)].mean().item()
                              for k in uk], device=DEV, dtype=Xtr.dtype)
        Ka = Xa @ Xa.T
        aev, avec = _eig(Ka)
        yac = ya - ya.mean()
        # small ridge for M1 (few bins); reuse a mid alpha from grid via quick CV-less pick
        aM1 = float(np.median(ALPHAS))
        dualM1 = _dual_weights(avec.T @ yac, aev, avec, aM1)
        wM1 = Xa.T @ dualM1
        if spearmanr((Xtr @ wM1).cpu().numpy(), y.cpu().numpy()).statistic < 0:
            wM1 = -wM1
        testpredM1 = (Xte @ wM1).cpu().numpy()
        rho, r2, auc = eval_dir(wM1.cpu().numpy(), testpredM1)
        res_methods["M1"] = dict(alpha=aM1, n_bins=int(len(uk)),
                                 test_rho=rho, test_r2=r2, adj_auc=auc,
                                 w=wM1.cpu().numpy())

        # ---- M3: mass-mean (tertiles) ----
        yv = y.cpu().numpy()
        lo, hi = np.quantile(yv, [1/3, 2/3])
        top = Xtr[torch.as_tensor(np.where(yv >= hi)[0], device=DEV)].mean(0)
        bot = Xtr[torch.as_tensor(np.where(yv <= lo)[0], device=DEV)].mean(0)
        wM3 = top - bot
        wM3 = wM3 / (wM3.norm() + 1e-8)
        if spearmanr((Xtr @ wM3).cpu().numpy(), yv).statistic < 0:
            wM3 = -wM3
        testpredM3 = (Xte @ wM3).cpu().numpy()
        rho, r2, auc = eval_dir(wM3.cpu().numpy(), testpredM3)
        res_methods["M3"] = dict(test_rho=rho, test_r2=r2, adj_auc=auc,
                                 w=wM3.cpu().numpy())

        out[trait] = res_methods
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default="/dev/shm/bf_acts")
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="", help="comma list to restrict (debug)")
    args = ap.parse_args()

    profiles = json.loads(Path(args.profiles).read_text())
    prof_by_id = {p["id"]: p for p in profiles}
    z_by_id = {p["id"]: p["z"] for p in profiles}

    char_root = Path(args.acts_dir) / "acts_characters"
    adj_root = Path(args.acts_dir) / "acts_adjectives"
    cidx = json.loads((char_root / "index.json").read_text())
    aidx = json.loads((adj_root / "index.json").read_text())
    meta = json.loads((char_root / "meta.json").read_text())
    n_layers = meta["shape"][1]

    train_ids, test_ids = character_split(profiles, seed=0)
    train_set, test_set = set(train_ids), set(test_ids)
    tr_rows = [i for i, r in enumerate(cidx) if r["character_id"] in train_set]
    te_rows = [i for i, r in enumerate(cidx) if r["character_id"] in test_set]
    tr_char = [cidx[i]["character_id"] for i in tr_rows]
    te_char = [cidx[i]["character_id"] for i in te_rows]
    print(f"[stage1] {len(profiles)} chars -> train {len(train_ids)}/{len(tr_rows)} rows, "
          f"test {len(test_ids)}/{len(te_rows)} rows | layers={n_layers} dev={DEV}")

    # adjective row groups per trait/polarity
    adj_rows = {t: {"pos": [], "neg": []} for t in S.TRAITS}
    for i, r in enumerate(aidx):
        adj_rows[r["trait"]]["pos" if r["polarity"] == "pos" else "neg"].append(i)

    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(n_layers)))

    # labels per trait (train rows) and z per test character
    def ytr(trait):
        return torch.as_tensor([z_by_id[c][trait] for c in tr_char],
                               device=DEV, dtype=torch.float32)
    z_te = {t: {c: z_by_id[c][t] for c in set(te_char)} for t in S.TRAITS}

    char_of_test = {
        "train_chars": tr_char, "train_row_char": tr_char,
        "test_row_char": te_char,
    }

    results = {p: {} for p in POSITIONS_EVAL}
    t0 = time.time()
    for pos in POSITIONS_EVAL:
        Cacts = np.load(char_root / f"acts_{pos}.npy", mmap_mode="r")
        Aacts = np.load(adj_root / f"acts_{pos}.npy", mmap_mode="r")
        for li in layers:
            Xtr = torch.as_tensor(np.ascontiguousarray(Cacts[tr_rows, li, :]),
                                  device=DEV, dtype=torch.float32)
            Xte = torch.as_tensor(np.ascontiguousarray(Cacts[te_rows, li, :]),
                                  device=DEV, dtype=torch.float32)
            Xadj = {}
            for t in S.TRAITS:
                Xp = torch.as_tensor(np.ascontiguousarray(Aacts[adj_rows[t]["pos"], li, :]),
                                     device=DEV, dtype=torch.float32)
                Xn = torch.as_tensor(np.ascontiguousarray(Aacts[adj_rows[t]["neg"], li, :]),
                                     device=DEV, dtype=torch.float32)
                Xadj[t] = (Xp, Xn)
            ybt = {t: ytr(t) for t in S.TRAITS}
            layer_res = derive_and_eval(Xtr, ybt, Xte, char_of_test, z_te, Xadj)
            results[pos][li] = layer_res
            if li % 10 == 0 or li == layers[-1]:
                el = time.time() - t0
                print(f"  {pos} L{li:02d}  {el:.0f}s", flush=True)

    # ------- select probe-optimal per trait; assemble basis; gate G1 ------- #
    selection, basis = {}, {}
    for trait in S.TRAITS:
        best = None
        for pos in POSITIONS_EVAL:
            for li in layers:
                for m in METHODS:
                    r = results[pos][li][trait][m]
                    score = 0.5 * (r["test_rho"] + r["adj_auc"])
                    if best is None or score > best["score"]:
                        best = dict(score=score, position=pos, layer=li, method=m,
                                    test_rho=r["test_rho"], test_r2=r["test_r2"],
                                    adj_auc=r["adj_auc"], w=r["w"])
        selection[trait] = {k: v for k, v in best.items() if k != "w"}
        basis[trait] = best["w"]

    # basis quality: Gram (cosine) of the 5 probe-optimal directions
    W = np.stack([basis[t] / (np.linalg.norm(basis[t]) + 1e-9) for t in S.TRAITS])
    gram = (W @ W.T)
    crosstalk = {f"{S.TRAITS[i]}-{S.TRAITS[j]}": float(gram[i, j])
                 for i in range(5) for j in range(i + 1, 5)}

    g1 = {t: {"test_rho": selection[t]["test_rho"], "adj_auc": selection[t]["adj_auc"],
              "pass": selection[t]["test_rho"] > 0 and selection[t]["adj_auc"] > 0.6}
          for t in S.TRAITS}
    g1_pass = all(v["pass"] for v in g1.values())

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "bigfive_basis.npz", **{t: basis[t] for t in S.TRAITS})
    # compact per-(pos,layer,trait,method) metrics (drop the weight vectors)
    slim = {pos: {li: {t: {m: {k: v for k, v in results[pos][li][t][m].items() if k != "w"}
                           for m in METHODS} for t in S.TRAITS} for li in layers}
            for pos in POSITIONS_EVAL}
    (out / "stage1_metrics.json").write_text(json.dumps(slim, indent=1))
    (out / "stage1_selection.json").write_text(json.dumps(
        {"selection": selection, "crosstalk_cos": crosstalk,
         "max_abs_crosstalk": max(abs(v) for v in crosstalk.values()),
         "gate_G1": g1, "gate_G1_pass": g1_pass}, indent=1))

    print("\n==== GATE G1 ====")
    for t in S.TRAITS:
        s = selection[t]
        print(f"  {t}: pos={s['position']:11s} L{s['layer']:02d} {s['method']} "
              f"test_rho={s['test_rho']:+.3f} R2={s['test_r2']:.3f} "
              f"adj_auc={s['adj_auc']:.3f} -> {'PASS' if g1[t]['pass'] else 'FAIL'}")
    print(f"  max|crosstalk cos| = {max(abs(v) for v in crosstalk.values()):.3f}")
    print(f"  G1 {'PASS' if g1_pass else 'FAIL'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
