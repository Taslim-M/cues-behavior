"""Stage 1 §3.3-3.5 -- derive Big Five directions, evaluate probes, gate G1.

Three derivation methods per (trait x layer x position), per the plan:

  M1  "original"   -- average activations within each discrete trait-score value,
                      then regress score on the averaged activations.
  M2  per-sample   -- regression on per-sample activations, no within-score
                      averaging (5-fold-CV ridge penalty).
  M3  mass-mean    -- unit-normalised mean(top tertile) - mean(bottom tertile).

Because d_model (8192) >> n (406 characters), an unregularised OLS fit is
underdetermined and would be pure noise. Every "regression" here is therefore
ridge, solved in the **dual form** so cost scales with n, not d:

    w = X^T (X X^T + lambda I)^-1 y

The Gram matrix X X^T is computed once per (layer, position) and reused across
all five traits and the whole lambda grid via one eigendecomposition -- without
this the 5 traits x 80 layers x 3 positions x |lambda| fit grid is intractable.

Splitting is at **character** level (stratified by per-trait score quintile,
80/20, seed=0) so the 10 Alpaca instructions belonging to one character can
never straddle train and test.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.bigfive import stimuli as S
from src.bigfive.extract import POSITIONS

ALPHAS = np.logspace(-1, 6, 15)


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
@dataclass
class ActSet:
    root: Path
    index: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.index = json.loads((self.root / "index.json").read_text())

    def acts(self, position: str) -> np.ndarray:
        """memmap [N, n_layers, d]"""
        return np.load(self.root / f"acts_{position}.npy", mmap_mode="r")


def character_split(profiles: list[dict], test_frac: float = 0.2,
                    seed: int = 0) -> tuple[list[str], list[str]]:
    """80/20 split over characters, stratified on the overall-Big-Five quintile.

    The plan §3.1 asks for quintile stratification, but stratifying on the JOINT
    5-trait signature makes 5^5 cells for 406 characters -- almost all singletons,
    whose round(0.2)=0 test allocation drains the test set to ~11 characters. We
    stratify instead on a single ordinal key -- the quintile of the summed
    z-scores -- which preserves a spread of "big-five-ness" across the split while
    keeping ~81 test characters. Deviation from the literal joint stratification;
    documented in the stage report.
    """
    rng = np.random.default_rng(seed)
    tot = np.array([sum(p["z"][t] for t in S.TRAITS if p["z"][t] is not None)
                    for p in profiles])
    q = np.quantile(tot, [0.2, 0.4, 0.6, 0.8])
    strat = np.digitize(tot, q)
    train, test = [], []
    for s in np.unique(strat):
        idxs = list(np.where(strat == s)[0])
        rng.shuffle(idxs)
        n_test = int(round(len(idxs) * test_frac))
        test += [profiles[i]["id"] for i in idxs[:n_test]]
        train += [profiles[i]["id"] for i in idxs[n_test:]]
    return sorted(train), sorted(test)


# --------------------------------------------------------------------------- #
# dual-form ridge
# --------------------------------------------------------------------------- #
def _dual_ridge_fit(X: np.ndarray, y: np.ndarray, alphas: np.ndarray,
                    eig: tuple | None = None):
    """Return {alpha: w} using w = X^T (K + aI)^-1 y, K = X X^T.

    `eig` lets the caller pass a precomputed eigendecomposition of K so the
    O(n^2 d) Gram build and O(n^3) decomposition are paid once per
    (layer, position) rather than once per (trait, alpha).
    """
    if eig is None:
        K = X @ X.T
        evals, evecs = np.linalg.eigh(K)
    else:
        evals, evecs = eig
    Vty = evecs.T @ y
    out = {}
    for a in alphas:
        dual = evecs @ (Vty / (evals + a))
        out[float(a)] = X.T @ dual
    return out


def _cv_select_alpha(X: np.ndarray, y: np.ndarray, alphas: np.ndarray,
                     k: int = 5, seed: int = 0) -> float:
    """5-fold CV over the lambda grid, scored by Spearman rho on held-out folds."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    score = {float(a): [] for a in alphas}
    for f in range(k):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(k) if j != f])
        Xtr, ytr = X[tr], y[tr]
        ws = _dual_ridge_fit(Xtr, ytr - ytr.mean(), alphas)
        for a, w in ws.items():
            pred = X[te] @ w
            if np.std(pred) < 1e-12:
                score[a].append(0.0)
            else:
                score[a].append(spearmanr(pred, y[te]).statistic)
    means = {a: float(np.nanmean(v)) for a, v in score.items()}
    return max(means, key=means.get)


# --------------------------------------------------------------------------- #
# derivation methods
# --------------------------------------------------------------------------- #
def derive_M1(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Average within each discrete score value, then ridge-regress."""
    keys = np.round(y * 2) / 2          # bin to half-units of the z-score
    uk = np.unique(keys)
    Xa = np.stack([X[keys == k].mean(0) for k in uk])
    ya = np.array([y[keys == k].mean() for k in uk])
    a = _cv_select_alpha(Xa, ya, alphas, k=min(5, len(uk)))
    w = _dual_ridge_fit(Xa, ya - ya.mean(), np.array([a]))[a]
    return w, {"method": "M1", "alpha": a, "n_bins": int(len(uk))}


def derive_M2(X: np.ndarray, y: np.ndarray, alphas: np.ndarray,
              eig=None) -> np.ndarray:
    a = _cv_select_alpha(X, y, alphas)
    w = _dual_ridge_fit(X, y - y.mean(), np.array([a]), eig=eig)[a]
    return w, {"method": "M2", "alpha": a}


def derive_M3(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(y, [1 / 3, 2 / 3])
    top, bot = X[y >= hi], X[y <= lo]
    w = top.mean(0) - bot.mean(0)
    n = np.linalg.norm(w)
    return (w / n if n else w), {"method": "M3", "n_top": int(len(top)),
                                 "n_bot": int(len(bot))}


def fix_sign(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ensure + means more of the trait."""
    p = X @ w
    if np.std(p) < 1e-12:
        return w
    return -w if spearmanr(p, y).statistic < 0 else w
