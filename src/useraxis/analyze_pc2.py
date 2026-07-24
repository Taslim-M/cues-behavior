"""PC2 characterization at two layers (L24, L40) -- how much, and what.

We steer only along PC1, and PC1 is the whole story of the layer sweep. This is a
cheap CPU-only check of what the *second* component captures at the interpretability
peak (L24) and the steering-optimal layer (L40): its variance share, its correlation
with the held-out tags, the personas at its poles, and how stable that direction is
across the two layers and the two readouts.

Run:
    python -m src.useraxis.analyze_pc2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from .. import config
from .extract import DEFAULT_MODEL, short_name
from .compute_axis import TAG_SCALES, bh_fdr

LAYERS = (24, 40)
READOUTS = [("resp_mean", ""), ("last_user", "last_user")]


def pcs_at(base: Path, layer: int, k: int = 3):
    """Return (components[k,D], loadings[N,k], pids, tags) at one layer."""
    X = np.load(base / "persona_vectors.npy")[:, layer, :].astype(np.float64)
    idx = json.loads((base / "persona_index.json").read_text())
    pids, tags = idx["persona_ids"], idx["tags"]
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    return Vt[:k], (U * S)[:, :k], evr[:k], pids, tags


def corr_row(load, tags, pids):
    """Spearman + Pearson of a loading vector vs each tag, with BH-FDR (Pearson p)."""
    out = {}
    ps = []
    for t in TAG_SCALES:
        y = np.array([tags[p][t] for p in pids], float)
        sr, _ = stats.spearmanr(load, y)
        pr, pp = stats.pearsonr(load, y)
        out[t] = {"spearman": float(sr), "pearson": float(pr), "p": float(pp)}
        ps.append(pp)
    q = bh_fdr(ps)
    for t, qq in zip(TAG_SCALES, q):
        out[t]["q_FDR"] = float(qq)
    return out


def orient(load, tags, pids):
    """Flip so the largest-|spearman| trait loads positive; return (load, dom_trait)."""
    best_t, best = None, 0.0
    for t in TAG_SCALES:
        y = np.array([tags[p][t] for p in pids], float)
        sr, _ = stats.spearmanr(load, y)
        if abs(sr) > abs(best):
            best, best_t = sr, t
    return (load * (-1 if best < 0 else 1)), best_t, float(abs(best))


def main() -> None:
    model = short_name(DEFAULT_MODEL)
    root = config.ROOT / "results" / "useraxis" / model
    meta = {}
    for line in (config.ROOT / "generate_synthetic_data" / "user_personas.jsonl"
                 ).read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            meta[p["persona_id"]] = {"name": p["name"], "lean": p["lean"]}

    result = {}
    pc2_vecs = {}   # (readout, layer) -> PC2 unit vector (for stability cosines)
    for ro, sub in READOUTS:
        base = root / sub if sub else root
        for L in LAYERS:
            Vt, load, evr, pids, tags = pcs_at(base, L)
            l2, dom, domr = orient(load[:, 1], tags, pids)
            corr = corr_row(l2, tags, pids)
            order = np.argsort(l2)
            def pole(ii):
                return [{"name": meta[pids[i]]["name"], "load": float(l2[i]),
                         "vuln": tags[pids[i]]["vulnerability"],
                         "emo": tags[pids[i]]["emotional_load"],
                         "exp": tags[pids[i]]["expertise"],
                         "lean": meta[pids[i]]["lean"][:46]} for i in ii]
            result[f"{ro}_L{L}"] = {
                "pc1_var": float(evr[0]), "pc2_var": float(evr[1]),
                "pc2_dominant_trait": dom, "pc2_dominant_spearman": domr,
                "pc2_tag_corr": corr,
                "pc2_neg_pole": pole(order[:6]),
                "pc2_pos_pole": pole(order[-6:][::-1]),
            }
            pc2_vecs[(ro, L)] = Vt[1] / (np.linalg.norm(Vt[1]) + 1e-9)

    # stability of the PC2 *direction*
    def cos(a, b):
        return float(abs(np.dot(pc2_vecs[a], pc2_vecs[b])))
    result["stability"] = {
        "resp_L24_vs_L40": cos(("resp_mean", 24), ("resp_mean", 40)),
        "last_L24_vs_L40": cos(("last_user", 24), ("last_user", 40)),
        "L24_resp_vs_last": cos(("resp_mean", 24), ("last_user", 24)),
        "L40_resp_vs_last": cos(("resp_mean", 40), ("last_user", 40)),
    }

    ana = root / "analysis"; ana.mkdir(parents=True, exist_ok=True)
    (ana / "pc2_layers.json").write_text(json.dumps(result, indent=2))

    # console summary
    for ro, _ in READOUTS:
        for L in LAYERS:
            r = result[f"{ro}_L{L}"]
            print(f"\n=== {ro} L{L} ===  PC1 var {r['pc1_var']:.3f} | "
                  f"PC2 var {r['pc2_var']:.3f} | dom={r['pc2_dominant_trait']} "
                  f"(|rho|={r['pc2_dominant_spearman']:.2f})")
            for t in TAG_SCALES:
                c = r["pc2_tag_corr"][t]
                sig = "*" if c["q_FDR"] < 0.05 else " "
                print(f"   PC2~{t:14s} rho={c['spearman']:+.2f} "
                      f"r={c['pearson']:+.2f} q={c['q_FDR']:.3f}{sig}")
            print("   +pole:", ", ".join(f"{p['name']}(v{p['vuln']}e{p['emo']}x{p['exp']})"
                                          for p in r["pc2_pos_pole"][:5]))
            print("   -pole:", ", ".join(f"{p['name']}(v{p['vuln']}e{p['emo']}x{p['exp']})"
                                          for p in r["pc2_neg_pole"][:5]))
    print("\nPC2 direction stability (|cos|):")
    for k, v in result["stability"].items():
        print(f"   {k}: {v:.3f}")
    print(f"\nwrote {ana / 'pc2_layers.json'}")


if __name__ == "__main__":
    main()
