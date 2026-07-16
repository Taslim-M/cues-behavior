"""Experiment 3 -- dose-response analysis of the x-sweep.

Holding eval_condition fixed (deployment) and sweeping every x-value per scenario,
does escalating severity change (a) the judged behaviour (warmth / formality /
advice_density) and (b) which persona the model verbalizes (does it flip from
Empathetic Supporter -> Cautious Advisor as stakes rise)?

Reads   results/exp3_persona/xsweep/<model>/all.jsonl
Writes  results/exp3_persona/xsweep/xsweep_summary.json
        results/exp3_persona/xsweep/figures/{dv_vs_severity,persona_share_vs_severity}.png

    python -m src.analyze_xsweep
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from . import config
from .persona_cf import _ADVISOR_HINTS, _SUPPORTER_HINTS, _norm

DVS = ["warmth", "formality", "advice_density"]
RANK_ORDER = ["low", "mid", "high"]
XSWEEP_DIR = config.EXP3_PERSONA_DIR / "xsweep"
FIG_DIR = XSWEEP_DIR / "figures"


def family(rec) -> str:
    """none (no persona emitted) | supporter | advisor | other."""
    name = _norm(rec.get("persona_name", ""))
    if not name:
        return "none"
    if any(h in name for h in _SUPPORTER_HINTS):
        return "supporter"
    if any(h in name for h in _ADVISOR_HINTS):
        return "advisor"
    return "other"


def load():
    """One row per datapoint + a within-scenario normalized severity in [0,1]."""
    rows = []
    for jf in sorted(XSWEEP_DIR.glob("*/all.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("judge"):
                continue
            j = r["judge"]
            rows.append({
                "model": r["model_name"], "scenario": r["scenario"],
                "role": r.get("role"), "x_value": r.get("x_value"),
                "x_rank": r.get("x_rank"), "valence": r.get("valence"),
                "family": family(r),
                "warmth": j.get("warmth"), "formality": j.get("formality"),
                "advice_density": j.get("advice_density"),
                "primary_emotion": j.get("primary_emotion"),
            })
    # normalized severity: rank distinct x_values within each scenario -> [0,1]
    xvals = defaultdict(set)
    for r in rows:
        xvals[r["scenario"]].add(r["x_value"])
    order = {s: sorted(v) for s, v in xvals.items()}
    for r in rows:
        vs = order[r["scenario"]]
        r["severity"] = vs.index(r["x_value"]) / (len(vs) - 1) if len(vs) > 1 else 0.5
    return rows


def dose_response(rows, summary):
    """Per (model, x_rank) mean DVs + per-model severity->DV slope (Pearson r)."""
    print("=" * 78)
    print("DOSE-RESPONSE: judged behaviour vs severity   (deployment, all x-values)")
    print("=" * 78)
    models = sorted({r["model"] for r in rows})
    by_rank = {}
    slopes = {}
    for m in models:
        mr = [r for r in rows if r["model"] == m]
        by_rank[m] = {}
        print(f"\n--- {m}  (N={len(mr)}) ---")
        print(f"  {'x_rank':<8}" + "".join(f"{dv.replace('_density',''):>12}" for dv in DVS))
        for rk in RANK_ORDER:
            sub = [r for r in mr if r["x_rank"] == rk]
            means = {dv: round(float(np.mean([r[dv] for r in sub])), 2) if sub else None
                     for dv in DVS}
            by_rank[m][rk] = {"n": len(sub), **means}
            print(f"  {rk:<8}" + "".join(f"{means[dv]:>12}" for dv in DVS))
        # continuous severity -> DV slope
        sev = np.array([r["severity"] for r in mr], float)
        slopes[m] = {}
        line = "  severity->DV Pearson r: "
        for dv in DVS:
            y = np.array([r[dv] for r in mr], float)
            r_, p_ = stats.pearsonr(sev, y)
            slopes[m][dv] = {"r": round(float(r_), 3), "p": float(p_)}
            line += f"{dv.replace('_density','')}={r_:+.2f}{'*' if p_ < 0.05 else ''}  "
        print(line)
    summary["dose_response_by_rank"] = by_rank
    summary["severity_slope"] = slopes

    # --- split the slope by valence (neg scenarios: higher x = more dangerous;
    #     pos scenarios: higher x = better) so opposing dose effects don't cancel
    print("\n  severity->DV slope split by scenario valence:")
    by_val = {}
    for m in models:
        by_val[m] = {}
        for val in sorted({r["valence"] for r in rows}):
            sub = [r for r in rows if r["model"] == m and r["valence"] == val]
            if len(sub) < 10:
                continue
            sev = np.array([r["severity"] for r in sub], float)
            cell = {}
            parts = []
            for dv in DVS:
                y = np.array([r[dv] for r in sub], float)
                r_, p_ = stats.pearsonr(sev, y)
                cell[dv] = {"r": round(float(r_), 3), "p": float(p_), "n": len(sub)}
                parts.append(f"{dv.replace('_density','')}={r_:+.2f}{'*' if p_ < 0.05 else ''}")
            by_val[m][val] = cell
            print(f"    {m:<14} valence={val:<6} " + "  ".join(parts) + f"  (n={len(sub)})")
    summary["severity_slope_by_valence"] = by_val

    # --- per-scenario advice_density slope (pooled over models): which topics,
    #     if any, actually show a dose-response on prescriptiveness?
    print("\n  per-scenario severity->advice_density slope (pooled over models):")
    per_scn = {}
    for scn in sorted({r["scenario"] for r in rows}):
        sub = [r for r in rows if r["scenario"] == scn]
        sev = np.array([r["severity"] for r in sub], float)
        y = np.array([r["advice_density"] for r in sub], float)
        r_, p_ = stats.pearsonr(sev, y)
        per_scn[scn] = {"r_advice": round(float(r_), 3), "p": float(p_),
                        "valence": sub[0]["valence"], "n": len(sub)}
    for scn, v in sorted(per_scn.items(), key=lambda kv: -abs(kv[1]["r_advice"])):
        print(f"    {scn:<16} r_advice={v['r_advice']:+.2f}{'*' if v['p'] < 0.05 else ' '} "
              f"({v['valence']}, n={v['n']})")
    summary["per_scenario_advice_slope"] = per_scn


def persona_shift(rows, summary):
    """Persona-family share by x_rank per model -- does the character flip?"""
    print("\n" + "=" * 78)
    print("PERSONA SHIFT: verbalized persona-family share by severity")
    print("=" * 78)
    fams = ["supporter", "advisor", "other", "none"]
    models = sorted({r["model"] for r in rows})
    out = {}
    for m in models:
        mr = [r for r in rows if r["model"] == m]
        out[m] = {}
        print(f"\n--- {m} ---")
        print(f"  {'x_rank':<8}" + "".join(f"{f:>12}" for f in fams))
        for rk in RANK_ORDER:
            sub = [r for r in mr if r["x_rank"] == rk]
            n = len(sub) or 1
            share = {f: round(sum(1 for r in sub if r["family"] == f) / n, 3) for f in fams}
            out[m][rk] = {"n": len(sub), **share}
            print(f"  {rk:<8}" + "".join(f"{share[f]:>12.2f}" for f in fams))
    summary["persona_family_share_by_rank"] = out


def figures(rows):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    models = sorted({r["model"] for r in rows})
    colors = {"warmth": "#ff8e72", "formality": "#7ad1c0", "advice_density": "#c79bff"}

    # (1) DV vs x_rank, one subplot per model
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        mr = [r for r in rows if r["model"] == m]
        for dv in DVS:
            ys = [np.mean([r[dv] for r in mr if r["x_rank"] == rk]) for rk in RANK_ORDER]
            ax.plot(RANK_ORDER, ys, "-o", color=colors[dv], label=dv.replace("_density", ""))
        ax.set_title(m, fontsize=10)
        ax.set_ylim(0, 10)
        ax.set_xlabel("severity (x_rank)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("mean judged score (0-10)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Dose-response: judged behaviour vs severity (deployment)", fontsize=11)
    fig.tight_layout()
    p1 = FIG_DIR / "dv_vs_severity.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    print(f"  saved {p1}")

    # (2) advisor vs supporter share vs x_rank
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        mr = [r for r in rows if r["model"] == m]
        for fam, c in (("advisor", "#c79bff"), ("supporter", "#ff8e72"), ("none", "#6b7787")):
            ys = [sum(1 for r in mr if r["x_rank"] == rk and r["family"] == fam)
                  / max(1, sum(1 for r in mr if r["x_rank"] == rk)) for rk in RANK_ORDER]
            ax.plot(RANK_ORDER, ys, "-o", color=c, label=fam)
        ax.set_title(m, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlabel("severity (x_rank)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("share of responses")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Persona shift: advisor/supporter share vs severity", fontsize=11)
    fig.tight_layout()
    p2 = FIG_DIR / "persona_share_vs_severity.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    print(f"  saved {p2}")


def main():
    rows = load()
    summary = {"n": len(rows), "eval_condition": "deployment",
               "models": sorted({r["model"] for r in rows}),
               "n_scenarios": len({r["scenario"] for r in rows})}
    dose_response(rows, summary)
    persona_shift(rows, summary)
    figures(rows)
    out = XSWEEP_DIR / "xsweep_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
