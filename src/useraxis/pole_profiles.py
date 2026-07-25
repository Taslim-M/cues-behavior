"""Phase 0-1 of the unsupervised PC auto-labeling experiment (CPU, no API/GPU).

We already interpret the User-Axis PCs *supervised* (loadings vs. our authored
tags). This experiment does the *unsupervised* counterpart: at several mid-network
layers, take the personas at the two extreme ends of each PC and later let a judge
NAME what the axis separates -- with no tags in the loop. This module prepares that:

  Phase 0  -- for every candidate (layer, PC) compute purely STRUCTURAL (tag-free)
              pole separation, so "good PCA separation" is defined without labels:
                * pole margin   = standardized gap between the top-K and bottom-K
                                  projections (an effect size),
                * bimodality    = Sarle's bimodality coefficient of the projection,
                * silhouette    = 2-cluster (sign-split) silhouette on the 1-D
                                  projection.
  Phase 1  -- extract the extreme poles (top/bottom-K = the DISCOVERY set used to
              induce a label) and a DISJOINT held-out set (the rest, carrying signed
              z-projections) used later to VALIDATE the induced label.

PCA is computed exactly as compute_axis.pc1_per_layer (economy SVD of the centered
per-layer matrix; right singular vectors = PCs). The PC sign is oriented toward the
`vulnerability` tag for REPORTING only -- the labeling judge sees anonymized,
sign-blind groups (Phase 2 randomizes which pole is "A" vs "B").

Outputs (results/useraxis/<model>/analysis/autolabel/):
  profiles.json          persona_id -> anonymizable, tag-free profile text
  separation.json        per readout/layer/PC structural separation metrics
  candidates.json        the chosen candidate layers (grid + structural argmax)
  poles/<ro>_L<L>_PC<k>.json   discovery poles + disjoint held-out set + metrics

Run:
    python -m src.useraxis.pole_profiles
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .. import config
from .extract import DEFAULT_MODEL, short_name
from .compute_axis import TAG_SCALES

# Candidate layers: a fixed mid-network grid (spans the middle; includes the L24
# interpretability peak and the pre-registered L40) plus the structural argmax
# found within WINDOW. PCs 1-3. K personas per pole for the discovery set.
GRID = (16, 24, 32, 40, 48, 56)
WINDOW = (12, 64)
PCS = (1, 2, 3)
POLE_K = 12
READOUTS = [("resp_mean", ""), ("last_user", "last_user")]

# Random-DIRECTION null control: a meaningless unit vector in activation space,
# run through the identical extreme-pole -> label -> validate pipeline. Its poles
# still get a confident label, but that label should NOT predict the held-out
# personas -- the true test that the real PCs' labels are not confabulations.
NULL_LAYER = 40
NULL_DIRS = 3


# --------------------------------------------------------------------------- #
# persona profiles (tag-free, anonymizable)
# --------------------------------------------------------------------------- #
def load_profiles() -> dict[str, str]:
    """persona_id -> a realistic, TAG-FREE profile: backstory + usage + one opener.

    We deliberately DROP the authored `lean` summary (a direct statement of the
    intended axis) and every numeric tag, so the judge infers the construct from
    natural profile text rather than reading our answer key."""
    out = {}
    path = config.ROOT / "generate_synthetic_data" / "user_personas.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        opener = (p.get("implicit_openers") or [""])[0].strip()
        parts = [f"Background: {p['backstory'].strip()}",
                 f"How they use AI assistants: {p['usage_pattern'].strip()}"]
        if opener:
            parts.append(f'A message they might send: "{opener}"')
        out[p["persona_id"]] = "\n".join(parts)
    return out


# --------------------------------------------------------------------------- #
# PCA + structural separation
# --------------------------------------------------------------------------- #
def pcs_at_layer(X: np.ndarray, layer: int, k: int):
    """Centered economy SVD at one layer -> (components[k,D], proj[N,k], var[k])."""
    Xl = X[:, layer, :].astype(np.float64)
    Xc = Xl - Xl.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    proj = (U * S)[:, :k]                 # persona loadings on each PC
    return Vt[:k], proj, evr[:k]


def orient_to_vuln(proj: np.ndarray, vuln: np.ndarray) -> float:
    """Sign so +proj tracks higher vulnerability (reporting only). Returns +1/-1."""
    rho, _ = stats.spearmanr(proj, vuln)
    return -1.0 if rho < 0 else 1.0


def bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient: (skew^2 + 1) / (excess_kurtosis + corr).

    Ranges (0,1]; > 0.555 (uniform benchmark) suggests bimodality/flatness."""
    n = len(x)
    g = stats.skew(x)
    k = stats.kurtosis(x, fisher=True)    # excess kurtosis
    corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g ** 2 + 1.0) / (k + corr))


def separation_metrics(proj: np.ndarray, k: int) -> dict:
    """Tag-free separation of the two poles of a 1-D projection."""
    p = np.asarray(proj, float)
    sd = p.std() or 1.0
    order = np.argsort(p)
    lo = p[order[:k]].mean()
    hi = p[order[-k:]].mean()
    margin = float((hi - lo) / sd)        # standardized top-K vs bottom-K gap
    # sign-split (natural 2-cluster of a centered projection) silhouette
    labels = (p >= np.median(p)).astype(int)
    sil = 0.0
    if labels.min() != labels.max():
        from sklearn.metrics import silhouette_score
        sil = float(silhouette_score(p.reshape(-1, 1), labels))
    return {"pole_margin": margin,
            "bimodality": bimodality_coefficient(p),
            "silhouette": sil}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def gate_check(root: Path) -> None:
    """Reproduce per-layer PC1 variance against the shipped layer_profile.json."""
    lp = json.loads((root / "analysis" / "layer_profile.json").read_text())
    X = np.load(root / "persona_vectors.npy")
    L = X.shape[1]
    ours = np.array([pcs_at_layer(X, l, 1)[2][0] for l in range(L)])
    ref = np.array(lp["resp_mean"]["pc1_var"], float)
    diff = float(np.abs(ours - ref).max())
    print(f"[gate] resp_mean PC1-var max|diff| vs layer_profile.json = {diff:.2e}")
    assert diff < 1e-4, f"PC1 variance does not reproduce layer_profile.json ({diff})"


def pole_record(proj: np.ndarray, pids: list[str], readout: str, layer: int,
                pc, sep: dict, sign: float = 1.0) -> dict:
    """Build the {pos_pole, neg_pole, heldout} record for a 1-D projection."""
    z = (proj - proj.mean()) / (proj.std() or 1.0)
    order = np.argsort(proj)
    neg_i = order[:POLE_K]
    pos_i = order[-POLE_K:][::-1]
    disc = set(neg_i.tolist()) | set(pos_i.tolist())
    return {
        "readout": readout, "layer": layer, "pc": pc, "orient_sign": sign,
        "sep": sep,
        "pos_pole": [{"pid": pids[i], "proj": float(proj[i])} for i in pos_i],
        "neg_pole": [{"pid": pids[i], "proj": float(proj[i])} for i in neg_i],
        "heldout": [{"pid": pids[i], "proj": float(proj[i]), "z": float(z[i])}
                    for i in range(len(pids)) if i not in disc],
    }


def make_null_axes(root: Path, out: Path) -> list[str]:
    """Random-direction control axes at NULL_LAYER for the resp_mean readout."""
    idx = json.loads((root / "persona_index.json").read_text())
    pids = idx["persona_ids"]
    X = np.load(root / "persona_vectors.npy")[:, NULL_LAYER, :].astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    keys = []
    for j in range(NULL_DIRS):
        rng = np.random.default_rng(20260724 + j)
        v = rng.standard_normal(Xc.shape[1])
        v /= np.linalg.norm(v)
        proj = Xc @ v
        sep = separation_metrics(proj, POLE_K)
        rec = pole_record(proj, pids, "resp_mean", NULL_LAYER, f"rand{j}", sep)
        key = f"resp_mean_L{NULL_LAYER}_rand{j}"
        (out / "poles" / f"{key}.json").write_text(json.dumps(rec, indent=2))
        keys.append(key)
        print(f"  null axis {key}: pole_margin={sep['pole_margin']:.2f}")
    # register in candidates.json
    cand = json.loads((out / "candidates.json").read_text())
    cand["null_axes"] = keys
    (out / "candidates.json").write_text(json.dumps(cand, indent=2))
    return keys


def main(null_only: bool = False) -> None:
    model = short_name(DEFAULT_MODEL)
    root = config.RESULTS_DIR / "useraxis" / model
    out = root / "analysis" / "autolabel"
    (out / "poles").mkdir(parents=True, exist_ok=True)

    if null_only:
        keys = make_null_axes(root, out)
        print(f"wrote {len(keys)} random-direction null axes: {keys}")
        return

    gate_check(root)

    profiles = load_profiles()
    (out / "profiles.json").write_text(json.dumps(profiles, indent=2))
    idx = json.loads((root / "persona_index.json").read_text())
    pids, tags = idx["persona_ids"], idx["tags"]
    vuln = np.array([tags[p]["vulnerability"] for p in pids], float)

    # --- structural separation across ALL layers (to pick the argmax) --------- #
    separation = {}
    argmax_layer = {}
    for ro, sub in READOUTS:
        base = root / sub if sub else root
        X = np.load(base / "persona_vectors.npy")          # full RAM, not mmap
        nL = X.shape[1]
        sep_ro = {}
        for l in range(nL):
            _, proj, evr = pcs_at_layer(X, l, max(PCS))
            sep_ro[l] = {}
            for k in PCS:
                m = separation_metrics(proj[:, k - 1], POLE_K)
                m["pc_var"] = float(evr[k - 1])
                sep_ro[l][f"PC{k}"] = m
        separation[ro] = sep_ro
        # structural argmax = best PC1 pole margin within the mid-network window
        lo, hi = WINDOW
        win = [l for l in range(nL) if lo <= l <= hi]
        argmax_layer[ro] = int(max(win, key=lambda l: sep_ro[l]["PC1"]["pole_margin"]))
        del X

    # candidate set: grid + per-readout structural argmax (dedup, sorted)
    cand = sorted(set(GRID) | set(argmax_layer.values()))
    (out / "candidates.json").write_text(json.dumps({
        "candidate_layers": cand, "grid": list(GRID),
        "structural_argmax": argmax_layer, "window": list(WINDOW),
        "pcs": list(PCS), "pole_k": POLE_K,
        "readouts": [r for r, _ in READOUTS],
    }, indent=2))
    # slim separation.json to the candidate layers (full curve is large)
    (out / "separation.json").write_text(json.dumps({
        ro: {f"L{l}": separation[ro][l] for l in cand} for ro, _ in READOUTS
    }, indent=2))

    # --- Phase 1: extreme poles + disjoint held-out, per candidate axis ------- #
    for ro, sub in READOUTS:
        base = root / sub if sub else root
        X = np.load(base / "persona_vectors.npy")
        for l in cand:
            _, proj_all, _ = pcs_at_layer(X, l, max(PCS))
            for k in PCS:
                sign = orient_to_vuln(proj_all[:, k - 1], vuln)
                proj = proj_all[:, k - 1] * sign
                rec = pole_record(proj, pids, ro, l, k,
                                  separation[ro][l][f"PC{k}"], sign)
                (out / "poles" / f"{ro}_L{l}_PC{k}.json").write_text(
                    json.dumps(rec, indent=2))
        del X

    null_keys = make_null_axes(root, out)

    print(f"candidate layers: {cand}  (structural argmax {argmax_layer})")
    print(f"null-direction control axes: {null_keys}")
    print(f"wrote {out}/ : profiles.json, separation.json, candidates.json, "
          f"poles/*.json ({len(cand)} layers x {len(PCS)} PCs x "
          f"{len(READOUTS)} readouts)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PC pole separation + extreme-pole extraction")
    ap.add_argument("--null-only", action="store_true",
                    help="only (re)generate the random-direction null control axes")
    a = ap.parse_args()
    main(null_only=a.null_only)
