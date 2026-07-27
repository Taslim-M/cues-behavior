"""Does using more probes (questions) split the User-Axis PCA better? (CPU, free)

The parent assistant_axis elicits each role with ALL 240 questions; our User Axis
uses only 24 probes/persona (a 10x subsample of the same 240-question bank). Because
every raw per-rollout activation is on disk, we can ask -- for free -- how PCA
separation scales with the number of probes, from 2 up to the 24 we have. If it is
still climbing at 24, more probes would help (-> GPU expansion); if it has saturated,
the 24-probe axis already captures the signal and expanding would not help.

Method: load the per-rollout activations (sliced to the analysis layers) once, apply
the Stage-D keep-set, then for each probe-count n draw R random probe-subsets, re-
aggregate the per-persona mean vectors over each subset (exactly as Stage E averages,
but restricted to the subset), build the axis, and measure separation. Averaging over
R subsets controls for *which* probes were drawn, isolating the effect of the *count*.

Metrics per (n, layer), mean +/- sd over the R subsets:
  pole_margin, silhouette          -- unsupervised pole separation (pole_profiles)
  PC1~vulnerability (Pearson/Spearman), PC1<->contrast cosine  -- meaning recovery
  PC1 variance explained
  explicit<->implicit arm agreement

Outputs analysis/probe_ablation/curve.json + figures/probe_ablation.png.
Prints an auto-decision on whether separation is still rising at 24.

Run:
    python -m src.useraxis.probe_ablation                 # full 2..24 ablation
    python -m src.useraxis.probe_ablation --subsets 4     # quicker
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from safetensors import safe_open

from .. import config
from .extract import DEFAULT_MODEL, short_name
from .compute_axis import (load_keep_set, contrast_axis, cosine_rows, READOUTS)
from .pole_profiles import separation_metrics, POLE_K

LAYERS = (24, 40)                       # analysis layers (L40 primary; L24 max-meaning)
N_GRID = (2, 4, 6, 8, 12, 16, 20, 24)
SUBSETS = 8                             # random probe-subsets per count
ARMS = ("explicit", "implicit")


def load_sliced(res_dir: Path, keep: set[str] | None):
    """Per-rollout activations sliced to LAYERS. Returns:
    data[readout][(pid, arm, probe)] = [len(LAYERS), D] float32; tags; probe_ids."""
    roll = res_dir / "rollouts"
    data = {ro: {} for ro in READOUTS}
    tags: dict[str, dict] = {}
    probes: set[str] = set()
    li = list(LAYERS)
    total = kept = 0
    for arm in ARMS:
        for jf in sorted((roll / arm).glob("u*.jsonl")):
            pid = jf.stem
            recs = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
            if not recs:
                continue
            tags[pid] = recs[0]["tags"]
            with safe_open(str(roll / arm / f"{pid}.acts.safetensors"),
                           framework="np") as sf:
                names = set(sf.keys())
                for rec in recs:
                    rid = rec["rollout_id"]
                    total += 1
                    if keep is not None and rid not in keep:
                        continue
                    ok = all(f"{rid}|{ro}" in names for ro in READOUTS)
                    if not ok:
                        continue
                    pb = rec["probe_id"]
                    probes.add(pb)
                    for ro in READOUTS:
                        v = sf.get_tensor(f"{rid}|{ro}")[li].astype(np.float32)
                        data[ro][(pid, arm, pb)] = v
                    kept += 1
    return data, tags, sorted(probes), {"total": total, "kept": kept}


def aggregate(data_ro: dict, pids: list[str], probe_set: set[str]):
    """Per-persona mean over (arms, kept rollouts whose probe in probe_set).
    Returns dict pid -> {'both','explicit','implicit': [len(LAYERS), D]} (missing groups
    omitted)."""
    out: dict[str, dict] = {}
    for (pid, arm, pb), v in data_ro.items():
        if pb not in probe_set:
            continue
        d = out.setdefault(pid, {})
        for g in (arm, "both"):
            if g in d:
                d[g][0] += v
                d[g][1] += 1
            else:
                d[g] = [v.copy(), 1]
    return {pid: {g: s / c for g, (s, c) in d.items()} for pid, d in out.items()}


def pca_axis(Xl: np.ndarray):
    """PC1 (unit) + variance-explained of a centered [N, D] matrix."""
    Xc = Xl - Xl.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[0], float((S[0] ** 2) / (S ** 2).sum())


def metrics_for_subset(agg, tags, pids, li_layers):
    """Separation metrics per analysis layer for one aggregated persona set."""
    have = [p for p in pids if p in agg and "both" in agg[p]]
    vuln = np.array([tags[p]["vulnerability"] for p in have], float)
    tag_rows = [tags[p] for p in have]
    out = {}
    for ci, L in enumerate(li_layers):
        X = np.stack([agg[p]["both"][ci] for p in have])          # [N, D]
        pc1, var = pca_axis(X)
        load = (X - X.mean(0)) @ pc1
        if stats.pearsonr(load, vuln)[0] < 0:                     # orient + -> vuln
            pc1, load = -pc1, -load
        rp = stats.pearsonr(load, vuln)[0]
        rs = stats.spearmanr(load, vuln)[0]
        sep = separation_metrics(load, POLE_K)
        # contrast cosine at this layer (contrast_axis works on [N,1,D])
        caxis, _ = contrast_axis(X[:, None, :], tag_rows)
        ccos = float(cosine_rows(pc1[None, :], caxis)[0])
        # arm agreement: project per-arm persona means onto the pooled PC1
        ex = [p for p in have if "explicit" in agg[p]]
        im = [p for p in have if "implicit" in agg[p]]
        both = [p for p in have if "explicit" in agg[p] and "implicit" in agg[p]]
        if len(both) >= 5:
            mu = X.mean(0)
            pe = np.array([(agg[p]["explicit"][ci] - mu) @ pc1 for p in both])
            pi = np.array([(agg[p]["implicit"][ci] - mu) @ pc1 for p in both])
            arm = float(stats.pearsonr(pe, pi)[0])
        else:
            arm = float("nan")
        out[f"L{L}"] = {"pole_margin": sep["pole_margin"],
                        "silhouette": sep["silhouette"],
                        "pc1_vuln_pearson": abs(float(rp)),
                        "pc1_vuln_spearman": abs(float(rs)),
                        "contrast_cos": abs(ccos), "pc1_var": var,
                        "arm_agreement": arm, "n_personas": len(have)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe-count ablation for the User Axis")
    ap.add_argument("--subsets", type=int, default=SUBSETS)
    ap.add_argument("--no-filter", action="store_true")
    args = ap.parse_args()

    model = short_name(DEFAULT_MODEL)
    res = config.RESULTS_DIR / "useraxis" / model
    out = res / "analysis" / "probe_ablation"
    out.mkdir(parents=True, exist_ok=True)

    keep = load_keep_set(res, args.no_filter)
    print("loading per-rollout activations (sliced to "
          f"L{LAYERS}) ...", flush=True)
    data, tags, probe_ids, cnt = load_sliced(res, keep)
    pids = sorted(tags)
    print(f"  {len(pids)} personas | {len(probe_ids)} probes | "
          f"kept {cnt['kept']}/{cnt['total']} rollouts", flush=True)
    n_probes_avail = len(probe_ids)
    grid = [n for n in N_GRID if n <= n_probes_avail]

    curve = {"metrics": {}, "n_grid": grid, "subsets": args.subsets,
             "layers": list(LAYERS), "n_probes_available": n_probes_avail,
             "assistant_axis_question_count": 240, "probe_ids": probe_ids}
    metric_keys = ("pole_margin", "silhouette", "pc1_vuln_pearson",
                   "pc1_vuln_spearman", "contrast_cos", "pc1_var", "arm_agreement")
    for n in grid:
        runs = []
        reps = 1 if n == n_probes_avail else args.subsets
        for r in range(reps):
            rng = np.random.default_rng(1000 * n + r)
            sub = set(rng.choice(probe_ids, size=n, replace=False).tolist()) \
                if n < n_probes_avail else set(probe_ids)
            agg = aggregate(data["resp_mean"], pids, sub)
            # attach per-arm from resp_mean only (primary readout)
            runs.append(metrics_for_subset(agg, tags, pids, LAYERS))
        curve["metrics"][f"n{n}"] = {}
        for L in (f"L{l}" for l in LAYERS):
            agg_stats = {}
            for mk in metric_keys:
                vals = np.array([run[L][mk] for run in runs], float)
                vals = vals[~np.isnan(vals)]
                agg_stats[mk] = {"mean": float(vals.mean()) if len(vals) else float("nan"),
                                 "sd": float(vals.std()) if len(vals) > 1 else 0.0}
            curve["metrics"][f"n{n}"][L] = agg_stats
        m40 = curve["metrics"][f"n{n}"]["L40"]
        print(f"  n={n:2d}: pole_margin={m40['pole_margin']['mean']:.2f}"
              f"+/-{m40['pole_margin']['sd']:.2f}  "
              f"PC1~vuln={m40['pc1_vuln_pearson']['mean']:.3f}  "
              f"contrast_cos={m40['contrast_cos']['mean']:.3f}  "
              f"var={m40['pc1_var']['mean']:.3f}", flush=True)

    # --- gate + decision --- #
    full = curve["metrics"][f"n{n_probes_avail}"]["L40"]
    print(f"\ngate (n={n_probes_avail}, L40): PC1~vuln="
          f"{full['pc1_vuln_pearson']['mean']:.3f} (shipped ~0.78), "
          f"pole_margin={full['pole_margin']['mean']:.2f}")
    decision = still_rising(curve, n_probes_avail)
    curve["decision"] = decision
    (out / "curve.json").write_text(json.dumps(curve, indent=2))
    plot(curve, res / "figures" / "probe_ablation.png")
    print(f"\nDECISION: separation still rising at {n_probes_avail} probes? "
          f"{decision['still_rising']}  ({decision['reason']})")
    print(f"wrote {out/'curve.json'}")


def still_rising(curve, n_full):
    """Rising if the last-step gain in pole_margin OR PC1~vuln exceeds its sd."""
    grid = curve["n_grid"]
    if len(grid) < 2:
        return {"still_rising": False, "reason": "insufficient grid"}
    a, b = f"n{grid[-2]}", f"n{grid[-1]}"
    L = "L40"
    reasons = []
    rising = False
    for mk in ("pole_margin", "pc1_vuln_pearson"):
        d = curve["metrics"][b][L][mk]["mean"] - curve["metrics"][a][L][mk]["mean"]
        sd = curve["metrics"][a][L][mk]["sd"]
        reasons.append(f"{mk} +{d:.3f} (sd {sd:.3f})")
        if d > max(sd, 1e-3):
            rising = True
    return {"still_rising": bool(rising), "reason": "; ".join(reasons),
            "last_step": f"{grid[-2]}->{grid[-1]}"}


def plot(curve, fig_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = curve["n_grid"]
    panels = [("pole_margin", "pole margin (SD units)"),
              ("pc1_vuln_pearson", "|PC1 ~ vulnerability|"),
              ("contrast_cos", "|PC1 ↔ contrast| cosine"),
              ("pc1_var", "PC1 variance explained")]
    colors = {"L40": "#0072B2", "L24": "#E69F00"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
    for ax, (mk, ylab) in zip(axes.ravel(), panels):
        for L, c in colors.items():
            m = [curve["metrics"][f"n{n}"][L][mk]["mean"] for n in grid]
            s = [curve["metrics"][f"n{n}"][L][mk]["sd"] for n in grid]
            ax.errorbar(grid, m, yerr=s, marker="o", ms=5, lw=2, color=c,
                        capsize=3, label=L)
        ax.set_xlabel("number of probes (questions)")
        ax.set_ylabel(ylab)
        ax.grid(True, color="#eee", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].legend(frameon=False, title="layer")
    axes[0, 0].set_title(f"User-Axis separation vs probe count "
                         f"(ours=24; assistant_axis uses 240)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
