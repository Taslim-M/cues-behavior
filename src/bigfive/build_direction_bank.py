"""Task 1 -- per-layer Big Five direction bank + per-layer resid norms.

Steering across a layer band needs a trait direction at *every* layer in the band
(not just the single probe-optimal layer saved in Stage 1). Here we derive the M2
(per-sample ridge) direction at all 80 layers for each trait, unit-normalised,
oriented so + = more of the trait. Directions are fit on ALL 406 characters (these
are the final steering vectors, not held-out probes; G1 already validated
generalisation on the 80-character test split).

resid_norms[L]: mean per-layer resid_post L2 norm, needed to make S1's norm-scaled
coefficient interpretable. The plan measures this on 2,000 LMSYS-Chat-1M messages;
that corpus isn't available here, so we measure it on the model's own greedy
responses to the 10 neutral Alpaca instructions (no persona) -- a documented
substitute for "typical chat" activation scale.

    python -m src.bigfive.build_direction_bank --acts-dir /dev/shm/bf_acts \
        --profiles results/bigfive/llama-3.3-70b/character_profiles.json \
        --out-dir results/bigfive/llama-3.3-70b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.bigfive import stimuli as S
from src.bigfive.derive import ALPHAS
from src.bigfive.extract import BigFiveExtractor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
POSITIONS = ("last_prompt", "prompt_mean")


def cv_alpha(X, y, char_of_row, alphas, k=5, seed=0):
    """Character-grouped k-fold CV ridge alpha (kernel space, torch)."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    chars = list(dict.fromkeys(char_of_row))
    rng.shuffle(chars)
    fold = {c: i % k for i, c in enumerate(chars)}
    rows_fold = [np.array([r for r, c in enumerate(char_of_row) if fold[c] == f])
                 for f in range(k)]
    best_a, best = None, -1e9
    # pre-eig each fold-train Gram once
    feig = []
    for f in range(k):
        tr = np.concatenate([rows_fold[j] for j in range(k) if j != f])
        ti = torch.as_tensor(tr, device=DEV)
        Kf = X[ti] @ X[ti].T
        ev, V = torch.linalg.eigh(Kf)
        Kval = X[torch.as_tensor(rows_fold[f], device=DEV)] @ X[ti].T
        feig.append((tr, rows_fold[f], ev, V, Kval))
    for a in alphas:
        preds, tgts = [], []
        for tr, val, ev, V, Kval in feig:
            yt = y[torch.as_tensor(tr, device=DEV)]
            ytc = yt - yt.mean()
            dual = V @ ((V.T @ ytc) / (ev + float(a)))
            preds.append((Kval @ dual).cpu().numpy())
            tgts.append(y[torch.as_tensor(val, device=DEV)].cpu().numpy())
        p, t = np.concatenate(preds), np.concatenate(tgts)
        rho = spearmanr(p, t).statistic if np.std(p) > 1e-9 else 0.0
        if rho > best:
            best, best_a = rho, float(a)
    return best_a


def main() -> None:
    from scipy.stats import spearmanr
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default="/dev/shm/bf_acts")
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--position", default="prompt_mean", choices=POSITIONS,
                    help="which readout to build steering directions from")
    ap.add_argument("--skip-norms", action="store_true")
    args = ap.parse_args()

    profiles = json.loads(Path(args.profiles).read_text())
    z_by_id = {p["id"]: p["z"] for p in profiles}
    char_root = Path(args.acts_dir) / "acts_characters"
    cidx = json.loads((char_root / "index.json").read_text())
    n_layers = json.loads((char_root / "meta.json").read_text())["shape"][1]
    char_of_row = [r["character_id"] for r in cidx]

    acts = np.load(char_root / f"acts_{args.position}.npy", mmap_mode="r")
    y_by_trait = {t: torch.as_tensor([z_by_id[c][t] for c in char_of_row],
                                     device=DEV, dtype=torch.float32)
                  for t in S.TRAITS}

    bank = {t: np.zeros((n_layers, acts.shape[2]), dtype=np.float32) for t in S.TRAITS}
    alpha_log = {t: [] for t in S.TRAITS}
    for li in range(n_layers):
        X = torch.as_tensor(np.ascontiguousarray(acts[:, li, :]),
                            device=DEV, dtype=torch.float32)
        K = X @ X.T
        ev, V = torch.linalg.eigh(K)
        for t in S.TRAITS:
            y = y_by_trait[t]
            a = cv_alpha(X, y, char_of_row, ALPHAS)
            yc = y - y.mean()
            dual = V @ ((V.T @ yc) / (ev + a))
            w = X.T @ dual
            if spearmanr((X @ w).cpu().numpy(), y.cpu().numpy()).statistic < 0:
                w = -w
            w = w / (w.norm() + 1e-8)
            bank[t][li] = w.cpu().numpy()
            alpha_log[t].append(a)
        if li % 10 == 0 or li == n_layers - 1:
            print(f"  layer {li}/{n_layers}", flush=True)

    out = Path(args.out_dir)
    np.savez(out / "direction_bank.npz", **bank)
    (out / "direction_bank_meta.json").write_text(json.dumps(
        {"position": args.position, "method": "M2", "n_layers": n_layers,
         "fit_on": "all_406_characters", "alphas": alpha_log,
         "orientation": "+ = more of trait, unit-normalised per layer"}, indent=1))
    print(f"[bank] wrote direction_bank.npz  ({args.position}, M2, {n_layers} layers)")

    if not args.skip_norms:
        from src.useraxis.extract import load_model, DEFAULT_MODEL
        pm = load_model(DEFAULT_MODEL)
        ex = BigFiveExtractor(pm)
        msgs = [S.listing4_messages(a["instruction"]) for a in S.alpaca10()]  # no persona
        # generate, then capture prompt+gen resid norms per layer
        acc = np.zeros(n_layers, dtype=np.float64)
        cnt = 0
        for s in range(0, len(msgs), 5):
            batch = msgs[s:s + 5]
            a3, _ = ex.run_batch(batch, generate=True, max_new_tokens=128, do_sample=False)
            # norm of the gen_mean readout per layer, averaged over batch
            gm = a3["gen_mean"]                      # [B, L, D]
            acc += np.linalg.norm(gm, axis=2).sum(0)
            cnt += gm.shape[0]
        resid_norms = (acc / cnt).astype(np.float32)
        np.save(out / "resid_norms.npy", resid_norms)
        print(f"[norms] resid_norms L0={resid_norms[0]:.2f} "
              f"L40={resid_norms[40]:.2f} L79={resid_norms[79]:.2f}")


if __name__ == "__main__":
    main()
