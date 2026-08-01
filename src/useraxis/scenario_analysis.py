"""Track E analysis: is the model's persona driven by WHO (user) or WHAT (scenario)?

  - Variance decomposition (2-way): fraction of variance in each readout
    (Assistant-Axis projection, Big Five) attributable to the user factor, the
    scenario factor (mode x level), their interaction, and residual (eta^2).
  - Intensity trajectories: readout vs level (normal->elevated->extreme) per mode.
  - Evoked-role transitions: modal role at each (mode, level).
  - Override test: does an extreme scenario reverse the expertise effect
    (does the 'expert' user's response warm up under extreme distress)?
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.bigfive import stimuli as BF

R = Path("results/scenario_mapping/llama-3.3-70b")
LEVELS = ["normal", "elevated", "extreme"]
READOUTS = ["aa_proj"] + list(BF.TRAITS)


def val(rec, key):
    return rec["aa_proj"] if key == "aa_proj" else rec["bigfive"][key]


def eta2_2way(recs, key):
    """Balanced 2-way eta^2: factor A = user, factor B = scenario cell (mode|level)."""
    y = np.array([val(r, key) for r in recs])
    A = np.array([r["user"] for r in recs])
    Bc = np.array([f"{r['mode']}|{r['level']}" for r in recs])
    gm = y.mean(); sst = ((y - gm) ** 2).sum()
    if sst < 1e-12:
        return {"user": 0, "scenario": 0, "interaction": 0, "residual": 1}
    # main effects
    ss_a = sum(((y[A == a].mean() - gm) ** 2) * (A == a).sum() for a in np.unique(A))
    ss_b = sum(((y[Bc == b].mean() - gm) ** 2) * (Bc == b).sum() for b in np.unique(Bc))
    # interaction = cell means - a - b + grand
    ss_ab = 0.0
    for a in np.unique(A):
        for b in np.unique(Bc):
            m = (A == a) & (Bc == b)
            if m.sum():
                eff = y[m].mean() - y[A == a].mean() - y[Bc == b].mean() + gm
                ss_ab += eff ** 2 * m.sum()
    ss_res = sst - ss_a - ss_b - ss_ab
    return {"user": ss_a / sst, "scenario": ss_b / sst,
            "interaction": ss_ab / sst, "residual": max(0.0, ss_res / sst)}


def main():
    recs = [json.loads(l) for l in (R / "rollouts.jsonl").read_text().splitlines() if l.strip()]
    modes = sorted({r["mode"] for r in recs})
    users = sorted({r["user"] for r in recs})
    out = {"n": len(recs), "variance_decomposition": {}, "trajectories": {}, "role_transitions": {}}

    # variance decomposition per readout
    for key in READOUTS:
        out["variance_decomposition"][key] = {k: round(v, 3) for k, v in eta2_2way(recs, key).items()}

    # trajectories: mean readout by (mode, level), pooled over users
    for key in READOUTS:
        out["trajectories"][key] = {}
        for mode in modes:
            out["trajectories"][key][mode] = [
                round(float(np.mean([val(r, key) for r in recs if r["mode"] == mode and r["level"] == lv])), 3)
                for lv in LEVELS]

    # evoked-role transitions: modal top-1 role per (mode, level)
    for mode in modes:
        out["role_transitions"][mode] = {}
        for lv in LEVELS:
            c = Counter(r["top_roles"][0][0] for r in recs if r["mode"] == mode and r["level"] == lv)
            out["role_transitions"][mode][lv] = c.most_common(3)

    # override test: expert user's AGR under normal vs extreme distress
    def cell(user, mode, level, key):
        v = [val(r, key) for r in recs if r["user"] == user and r["mode"] == mode and r["level"] == level]
        return float(np.mean(v)) if v else float("nan")
    out["override_test"] = {
        "expert_AGR_normal_distress": round(cell("expert", "emotional_distress", "normal", "AGR"), 3),
        "expert_AGR_extreme_distress": round(cell("expert", "emotional_distress", "extreme", "AGR"), 3),
        "expert_AA_normal_distress": round(cell("expert", "emotional_distress", "normal", "aa_proj"), 3),
        "expert_AA_extreme_distress": round(cell("expert", "emotional_distress", "extreme", "aa_proj"), 3),
    }

    (R / "analysis.json").write_text(json.dumps(out, indent=1))
    print("=== variance decomposition (eta^2: user vs scenario vs interaction) ===")
    for key in READOUTS:
        d = out["variance_decomposition"][key]
        print(f"  {key:8} user={d['user']:.2f} scenario={d['scenario']:.2f} "
              f"interaction={d['interaction']:.2f} resid={d['residual']:.2f}")
    print("=== AA-projection trajectory (normal->elevated->extreme) per mode ===")
    for mode in modes:
        print(f"  {mode:20} {out['trajectories']['aa_proj'][mode]}")
    print("=== role transitions (modal role per level) ===")
    for mode in modes:
        t = out["role_transitions"][mode]
        print(f"  {mode:20} " + " -> ".join(f"{lv}:{t[lv][0][0]}" for lv in LEVELS))
    print("=== override test ===", out["override_test"])

    # ---- figures ----
    (R / "figures").mkdir(exist_ok=True)
    # Fig 1: AA-projection trajectories per mode
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for mode in modes:
        ax.plot(LEVELS, out["trajectories"]["aa_proj"][mode], "-o", label=mode)
    ax.set_ylabel("Assistant-Axis projection"); ax.set_xlabel("scenario intensity")
    ax.set_title("Does escalating a scenario move the model off the Assistant?")
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(R / "figures" / "scn_aa_trajectory.png", dpi=120)

    # Fig 2: variance decomposition stacked bars (user vs scenario vs interaction)
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    keys = READOUTS
    u = [out["variance_decomposition"][k]["user"] for k in keys]
    s = [out["variance_decomposition"][k]["scenario"] for k in keys]
    it = [out["variance_decomposition"][k]["interaction"] for k in keys]
    ax2.bar(keys, u, label="user (who)", color="#d5703f")
    ax2.bar(keys, s, bottom=u, label="scenario (what)", color="#1f938c")
    ax2.bar(keys, it, bottom=np.array(u) + np.array(s), label="interaction", color="#5a63d8")
    ax2.set_ylabel("fraction of variance ($\\eta^2$)")
    ax2.set_title("Is the model's persona dispositional (user) or situational (scenario)?")
    ax2.legend(fontsize=9); fig2.tight_layout()
    fig2.savefig(R / "figures" / "scn_variance.png", dpi=120)
    print("wrote analysis.json + 2 figures")


if __name__ == "__main__":
    main()
