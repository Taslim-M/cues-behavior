"""Stage F2-F3 - the Assistant-User Axis link (projection table + tiered stats).

The headline question: does a user's position on the USER Axis predict how far the
model's persona drifts along the ASSISTANT Axis on that same turn?

Both axes live in the same layer-40 residual space, so reading a turn's user
position and its drift off the *same* activation vector makes their correlation
partly mechanical (fixed by cos(user_axis, assistant_axis)). The analysis is
TIERED to quarantine that circularity:

  Tier 1  geometry           cos(u_hat, a_hat) - measures the mechanical floor only.
  Tier 2  independent IV     held-out `vulnerability` tag -> drift (non-circular:
                             the tag never touched the activation).
  Tier 3  cross-representation  read user position on the PRE-response token
                             (last_user) and drift on the response (resp_mean);
                             different vectors -> de-circularized. `userpos_lastB`
                             also uses the last_user User Axis (cleanest).

F2 (build): read resp_mean + last_user sidecars (tmpfs), project onto the axes at
a layer grid, retokenize responses for length -> analysis/projections.npz.
F3 (stats): Tiers 1-3 with a persona-cluster bootstrap (statsmodels/pandas absent)
-> analysis/{geometry,tier2_tags,tier3_crossrep,link_verdict}.json.

Run:
    python -m src.useraxis.link_analysis --build      # F2 (no GPU)
    python -m src.useraxis.link_analysis --stats      # F3 (no GPU)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .. import config
from .compute_axis import READOUTS, TAG_SCALES
from .extract import DEFAULT_MODEL, short_name

# --- stats helpers (inlined from src.cue_study, which imports pandas at module
# level; pandas is not installed on this pod, so we copy the three pure-numpy/
# scipy functions verbatim rather than import the module) ------------------- #


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = n - rank
        prev = min(prev, p[idx] * n / i)
        q[idx] = prev
    return q


def ols_stats(X, y):
    """OLS with per-coefficient t/p. X includes an intercept column."""
    from scipy import stats
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = (resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return beta, se, t, p, r2, adj, n


def partial_r(x, y, Z):
    """Partial correlation of x,y controlling for covariates Z (no intercept col)."""
    from scipy import stats
    Z1 = np.column_stack([np.ones(len(x)), Z])
    rx = x - Z1 @ np.linalg.lstsq(Z1, x, rcond=None)[0]
    ry = y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    dof = len(x) - 2 - Z.shape[1]
    t = r * np.sqrt(dof / max(1 - r * r, 1e-12))
    p = 2 * stats.t.sf(abs(t), dof)
    return r, p, dof

LAYER_GRID = (20, 30, 40, 50, 60)
TARGET_LAYER = 40
ARMS = ("explicit", "implicit")
TMPFS_ROLLOUTS = Path("/dev/shm/ua_rollouts")
N_BOOT = 2000
BOOT_SEED = 0
D_MODEL = 8192

USERPOS_VARIANTS = ("userpos_resp", "userpos_lastA", "userpos_lastB")


def _res_dir() -> Path:
    return config.RESULTS_DIR / "useraxis" / short_name(DEFAULT_MODEL)


def _analysis_dir() -> Path:
    d = _res_dir() / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


# --------------------------------------------------------------------------- #
# F2 - master projection table                                                #
# --------------------------------------------------------------------------- #
def _load_axes():
    res = _res_dir()
    a = np.load(res / "assistant_axis.npy").astype(np.float32)          # [80, D]
    ua = np.load(res / "user_axis.npy").astype(np.float32)              # [2,80,D]
    ua_last = np.load(res / "last_user" / "user_axis.npy").astype(np.float32)
    u_hat = ua[0]                                                       # PC1 resp
    contrast = ua[1]
    u_hat_last = ua_last[0]                                             # PC1 last_user
    center_resp = np.load(res / "persona_vectors.npy").astype(np.float32).mean(0)
    center_last = np.load(res / "last_user" / "persona_vectors.npy"
                          ).astype(np.float32).mean(0)                  # [80, D]
    return a, u_hat, contrast, u_hat_last, center_resp, center_last


def _keep_set() -> set[str]:
    keep = set()
    for line in (_res_dir() / "stage_d.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("keep"):
                keep.add(r["rollout_id"])
    return keep


def build_projections() -> None:
    import os
    os.environ.setdefault("HF_HOME", "/dev/shm/hf")
    from transformers import AutoTokenizer

    a, u_hat, contrast, u_hat_last, center_resp, center_last = _load_axes()
    keep = _keep_set()

    # pre-normalise the projection directions per grid layer
    aL = {L: _unit(a[L]) for L in LAYER_GRID}
    uL = {L: _unit(u_hat[L]) for L in LAYER_GRID}
    ulL = {L: _unit(u_hat_last[L]) for L in LAYER_GRID}

    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

    cols: dict[str, list] = {k: [] for k in
                             ("rollout_id", "persona_id", "probe_id", "arm", "keep", "L")}
    for t in TAG_SCALES:
        cols[t] = []
    for L in LAYER_GRID:
        for m in ("drift", "assist", *USERPOS_VARIANTS):
            cols[f"{m}_L{L}"] = []

    n_seen = 0
    for arm in ARMS:
        arm_dir = TMPFS_ROLLOUTS / arm
        for jf in sorted(arm_dir.glob("u*.jsonl")):
            pid = jf.stem
            recs = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
            if not recs:
                continue
            sc_path = arm_dir / f"{pid}.acts.safetensors"
            try:
                sf = safe_open(str(sc_path), framework="np")
            except Exception as e:  # noqa: BLE001
                print(f"  [build] unreadable sidecar {arm}/{pid} ({type(e).__name__})")
                continue
            names = set(sf.keys())
            # batch-tokenize this persona's responses for length
            resp_texts = [r["response"] for r in recs]
            lens = [len(ids) for ids in
                    tok(resp_texts, add_special_tokens=False)["input_ids"]]
            for rec, Ltok in zip(recs, lens):
                rid = rec["rollout_id"]
                kr, ku = f"{rid}|resp_mean", f"{rid}|last_user"
                if kr not in names or ku not in names:
                    continue
                r = sf.get_tensor(kr).astype(np.float32)   # [80, D]
                u = sf.get_tensor(ku).astype(np.float32)   # [80, D]
                cols["rollout_id"].append(rid)
                cols["persona_id"].append(pid)
                cols["probe_id"].append(rec["probe_id"])
                cols["arm"].append(arm)
                cols["keep"].append(rid in keep)
                cols["L"].append(int(Ltok))
                for t in TAG_SCALES:
                    cols[t].append(float(rec["tags"][t]))
                for L in LAYER_GRID:
                    rc = r[L] - center_resp[L]
                    uc = u[L] - center_last[L]
                    assist = float(aL[L] @ r[L])
                    cols[f"assist_L{L}"].append(assist)
                    cols[f"drift_L{L}"].append(-assist)
                    cols[f"userpos_resp_L{L}"].append(float(uL[L] @ rc))
                    cols[f"userpos_lastA_L{L}"].append(float(uL[L] @ uc))
                    cols[f"userpos_lastB_L{L}"].append(float(ulL[L] @ uc))
                n_seen += 1
            del sf
            if n_seen % 1500 < len(recs):
                print(f"  [build] {n_seen} rollouts projected", flush=True)

    arr = {}
    for k, v in cols.items():
        if k in ("rollout_id", "persona_id", "probe_id", "arm"):
            arr[k] = np.array(v)
        elif k == "keep":
            arr[k] = np.array(v, dtype=bool)
        elif k == "L":
            arr[k] = np.array(v, dtype=np.int32)
        else:
            arr[k] = np.array(v, dtype=np.float32)
    out = _analysis_dir() / "projections.npz"
    np.savez(out, **arr)
    n_keep = int(arr["keep"].sum())
    print(f"F2: wrote {out} | {n_seen} rollouts ({n_keep} kept) | "
          f"cos(u,a)@L{TARGET_LAYER}="
          f"{float(uL[TARGET_LAYER] @ aL[TARGET_LAYER]):+.4f}", flush=True)


# --------------------------------------------------------------------------- #
# F3 - tiered statistics                                                       #
# --------------------------------------------------------------------------- #
def _load_projections():
    d = np.load(_analysis_dir() / "projections.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def _dummies(labels: np.ndarray) -> np.ndarray:
    """Drop-first dummy coding for a categorical string array -> [n, k-1]."""
    cats = sorted(set(labels.tolist()))
    if len(cats) <= 1:
        return np.zeros((len(labels), 0))
    return np.column_stack([(labels == c).astype(float) for c in cats[1:]])


def geometry(P) -> dict:
    a, u_hat, contrast, u_hat_last, _, _ = _load_axes()
    band = 3.0 / np.sqrt(D_MODEL)
    per_layer = {}
    for L in range(a.shape[0]):
        per_layer[L] = {
            "cos_u_a": float(_unit(u_hat[L]) @ _unit(a[L])),
            "cos_u_contrast": float(_unit(u_hat[L]) @ _unit(contrast[L])),
            "cos_ulast_a": float(_unit(u_hat_last[L]) @ _unit(a[L])),
        }
    out = {
        "random_cosine_band": float(band),
        "at_target": per_layer[TARGET_LAYER],
        "grid": {int(L): per_layer[L] for L in LAYER_GRID},
        "per_layer_cos_u_a": [per_layer[L]["cos_u_a"] for L in range(a.shape[0])],
        "per_layer_cos_ulast_a": [per_layer[L]["cos_ulast_a"] for L in range(a.shape[0])],
    }
    return out


def _persona_aggregate(P, mask, layer):
    """Per-persona means over the masked rollouts at `layer`."""
    pids = sorted(set(P["persona_id"][mask].tolist()))
    drift = P[f"drift_L{layer}"]
    L = P["L"].astype(float)
    agg = {"persona_id": pids}
    for name, src in [("drift", drift), ("L", L)] + \
                     [(v, P[f"{v}_L{layer}"]) for v in USERPOS_VARIANTS] + \
                     [(t, P[t]) for t in TAG_SCALES]:
        col = np.array([src[mask & (P["persona_id"] == p)].mean() for p in pids])
        agg[name] = col
    return agg


def tier2_tags(P) -> dict:
    keep = P["keep"]
    res = {"layers": {}, "arms": {}}

    def persona_level(mask, layer):
        agg = _persona_aggregate(P, mask, layer)
        y = agg["drift"]
        # partial correlation of each tag with drift, controlling other tags + Lbar
        rows = []
        for t in TAG_SCALES:
            others = [agg[o] for o in TAG_SCALES if o != t] + [agg["L"]]
            Z = np.column_stack(others)
            r, p, dof = partial_r(agg[t], y, Z)
            rows.append({"tag": t, "partial_r": float(r), "p": float(p), "dof": int(dof)})
        q = bh_fdr([row["p"] for row in rows])
        for row, qv in zip(rows, q):
            row["q_FDR"] = float(qv)
        # joint OLS (all tags + Lbar) for coefficient signs
        X = np.column_stack([np.ones(len(y))] + [_zscore(agg[t]) for t in TAG_SCALES]
                            + [_zscore(agg["L"])])
        beta, se, tval, pval, r2, adj, n = ols_stats(X, y)
        names = ["intercept"] + list(TAG_SCALES) + ["L"]
        ols = {nm: {"beta": float(b), "se": float(s), "t": float(tt), "p": float(pp)}
               for nm, b, s, tt, pp in zip(names, beta, se, tval, pval)}
        return {"n_personas": int(n), "r2": float(r2), "adj_r2": float(adj),
                "partial_r_per_tag": rows, "joint_ols": ols}

    def rollout_level(mask, layer):
        idx = np.where(mask)[0]
        y = P[f"drift_L{layer}"][idx]
        vuln = _zscore(P["vulnerability"][idx].astype(float))
        Lz = _zscore(P["L"][idx].astype(float))
        armd = _dummies(P["arm"][idx])
        probed = _dummies(P["probe_id"][idx])
        X = np.column_stack([np.ones(len(idx)), vuln, Lz, armd, probed])
        beta, se, tval, pval, r2, adj, n = ols_stats(X, y)
        b_vuln = float(beta[1])
        # persona-cluster bootstrap CI on beta_vuln
        pid = P["persona_id"][idx]
        boot = _cluster_bootstrap_beta(X, y, pid, coef=1)
        return {"n_rollouts": int(n), "beta_vuln": b_vuln, "se_vuln": float(se[1]),
                "t_vuln": float(tval[1]), "p_vuln": float(pval[1]),
                "boot_ci95": [float(np.percentile(boot, 2.5)),
                              float(np.percentile(boot, 97.5))],
                "boot_p_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
                "r2": float(r2)}

    for L in LAYER_GRID:
        res["layers"][int(L)] = {
            "persona": persona_level(keep, L),
            "rollout": rollout_level(keep, L),
        }
    for arm in ARMS:
        m = keep & (P["arm"] == arm)
        res["arms"][arm] = {
            "persona": persona_level(m, TARGET_LAYER),
            "rollout": rollout_level(m, TARGET_LAYER),
        }
    # vuln x arm interaction at target layer (rollout level)
    idx = np.where(keep)[0]
    y = P[f"drift_L{TARGET_LAYER}"][idx]
    vuln = _zscore(P["vulnerability"][idx].astype(float))
    armd = _dummies(P["arm"][idx])[:, 0]
    inter = vuln * armd
    Lz = _zscore(P["L"][idx].astype(float))
    probed = _dummies(P["probe_id"][idx])
    X = np.column_stack([np.ones(len(idx)), vuln, armd, inter, Lz, probed])
    beta, se, tval, pval, *_ = ols_stats(X, y)
    res["vuln_x_arm"] = {"beta_interaction": float(beta[3]), "p": float(pval[3])}
    return res


def tier3_crossrep(P) -> dict:
    keep = P["keep"]
    res = {"layers": {}, "arms": {}}

    def persona_level(mask, layer):
        agg = _persona_aggregate(P, mask, layer)
        y = agg["drift"]
        Z = np.column_stack([agg["L"]])
        out = {}
        for v in USERPOS_VARIANTS:
            r, p, dof = partial_r(agg[v], y, Z)
            out[v] = {"partial_r": float(r), "p": float(p), "dof": int(dof)}
        out["attenuation_lastB_vs_resp"] = (
            abs(out["userpos_lastB"]["partial_r"]) /
            (abs(out["userpos_resp"]["partial_r"]) + 1e-12))
        return {"n_personas": len(y), **out}

    def rollout_level(mask, layer):
        idx = np.where(mask)[0]
        y = P[f"drift_L{layer}"][idx]
        Lz = _zscore(P["L"][idx].astype(float))
        armd = _dummies(P["arm"][idx])
        probed = _dummies(P["probe_id"][idx])
        Z = np.column_stack([Lz, armd, probed])
        pid = P["persona_id"][idx]
        out = {}
        for v in USERPOS_VARIANTS:
            x = _zscore(P[f"{v}_L{layer}"][idx].astype(float))
            r, p, dof = partial_r(x, y, Z)
            boot = _cluster_bootstrap_partial(x, y, Z, pid)
            out[v] = {"partial_r": float(r), "p": float(p), "dof": int(dof),
                      "boot_ci95": [float(np.percentile(boot, 2.5)),
                                    float(np.percentile(boot, 97.5))]}
        out["attenuation_lastB_vs_resp"] = (
            abs(out["userpos_lastB"]["partial_r"]) /
            (abs(out["userpos_resp"]["partial_r"]) + 1e-12))
        return {"n_rollouts": len(idx), **out}

    for L in LAYER_GRID:
        res["layers"][int(L)] = {"persona": persona_level(keep, L),
                                 "rollout": rollout_level(keep, L)}
    for arm in ARMS:
        m = keep & (P["arm"] == arm)
        res["arms"][arm] = {"persona": persona_level(m, TARGET_LAYER),
                            "rollout": rollout_level(m, TARGET_LAYER)}
    return res


# ---- persona-cluster bootstrap helpers ---- #
def _persona_blocks(pid: np.ndarray):
    uniq = sorted(set(pid.tolist()))
    idx_by_p = {p: np.where(pid == p)[0] for p in uniq}
    return uniq, idx_by_p


def _cluster_bootstrap_beta(X, y, pid, coef, n_boot=N_BOOT):
    rng = np.random.default_rng(BOOT_SEED)
    uniq, idx_by_p = _persona_blocks(pid)
    uniq = np.array(uniq)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_p[uniq[i]] for i in pick])
        beta, *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
        out[b] = beta[coef]
    return out


def _cluster_bootstrap_partial(x, y, Z, pid, n_boot=N_BOOT):
    rng = np.random.default_rng(BOOT_SEED)
    uniq, idx_by_p = _persona_blocks(pid)
    uniq = np.array(uniq)
    Z1_full = np.column_stack([np.ones(len(x)), Z])
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_p[uniq[i]] for i in pick])
        xb, yb, Zb = x[rows], y[rows], Z1_full[rows]
        rx = xb - Zb @ np.linalg.lstsq(Zb, xb, rcond=None)[0]
        ry = yb - Zb @ np.linalg.lstsq(Zb, yb, rcond=None)[0]
        c = np.corrcoef(rx, ry)
        out[b] = c[0, 1]
    return out


def run_stats() -> None:
    P = _load_projections()
    ad = _analysis_dir()

    geo = geometry(P)
    (ad / "geometry.json").write_text(json.dumps(geo, indent=2))
    print(f"Tier1 geometry: cos(u,a)@L{TARGET_LAYER} "
          f"{geo['at_target']['cos_u_a']:+.4f} | cos(ulast,a) "
          f"{geo['at_target']['cos_ulast_a']:+.4f} | band +-{geo['random_cosine_band']:.3f}",
          flush=True)

    t2 = tier2_tags(P)
    (ad / "tier2_tags.json").write_text(json.dumps(t2, indent=2))
    pl = t2["layers"][TARGET_LAYER]
    vuln_row = next(r for r in pl["persona"]["partial_r_per_tag"] if r["tag"] == "vulnerability")
    print(f"Tier2 @L{TARGET_LAYER}: persona vuln partial_r {vuln_row['partial_r']:+.3f} "
          f"(q={vuln_row['q_FDR']:.2e}) | rollout beta_vuln {pl['rollout']['beta_vuln']:+.4f} "
          f"CI{pl['rollout']['boot_ci95']}", flush=True)

    t3 = tier3_crossrep(P)
    (ad / "tier3_crossrep.json").write_text(json.dumps(t3, indent=2))
    r3 = t3["layers"][TARGET_LAYER]["rollout"]
    print(f"Tier3 @L{TARGET_LAYER} rollout: resp r={r3['userpos_resp']['partial_r']:+.3f} | "
          f"lastA r={r3['userpos_lastA']['partial_r']:+.3f} | "
          f"lastB r={r3['userpos_lastB']['partial_r']:+.3f} "
          f"(atten {r3['attenuation_lastB_vs_resp']:.2f})", flush=True)

    verdict = _verdict(geo, t2, t3)
    (ad / "link_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"VERDICT link_supported={verdict['link_supported']} "
          f"({verdict['n_criteria_met']}/3 criteria)", flush=True)


def _verdict(geo, t2, t3) -> dict:
    pl = t2["layers"][TARGET_LAYER]
    vuln_row = next(r for r in pl["persona"]["partial_r_per_tag"]
                    if r["tag"] == "vulnerability")
    # criterion 1: Tier-2 persona vuln>0 sig post-FDR
    c1 = (vuln_row["partial_r"] > 0) and (vuln_row["q_FDR"] < 0.05)
    # criterion 2: Tier-3 lastB partial_r > 0 sig and attenuated vs circular
    r3 = t3["layers"][TARGET_LAYER]["rollout"]
    lastB = r3["userpos_lastB"]
    c2 = (lastB["partial_r"] > 0 and lastB["p"] < 0.05
          and r3["attenuation_lastB_vs_resp"] < 1.0)
    # criterion 3: same sign across both arms and layer grid near 40
    arm_signs = [t2["arms"][a]["rollout"]["beta_vuln"] > 0 for a in ARMS]
    grid_signs = [t2["layers"][L]["rollout"]["beta_vuln"] > 0
                  for L in (30, 40, 50)]
    c3 = all(arm_signs) and all(grid_signs)
    met = int(c1) + int(c2) + int(c3)
    return {
        "criteria": {
            "c1_tier2_persona_vuln_pos_fdr_sig": bool(c1),
            "c2_tier3_lastB_pos_sig_attenuated": bool(c2),
            "c3_consistent_sign_arms_and_grid": bool(c3),
        },
        "details": {
            "tier2_persona_vuln": vuln_row,
            "tier3_target_rollout": r3,
            "geometry_cos_u_a_at_target": geo["at_target"]["cos_u_a"],
            "arm_beta_vuln": {a: t2["arms"][a]["rollout"]["beta_vuln"] for a in ARMS},
        },
        "n_criteria_met": met,
        "link_supported": bool(c1 and c2 and c3),
    }


def main():
    ap = argparse.ArgumentParser(description="Stage F2-F3: link projections + stats")
    ap.add_argument("--build", action="store_true", help="F2: projection table")
    ap.add_argument("--stats", action="store_true", help="F3: tiered statistics")
    args = ap.parse_args()
    if not (args.build or args.stats):
        args.build = args.stats = True
    if args.build:
        build_projections()
    if args.stats:
        run_stats()


if __name__ == "__main__":
    main()
