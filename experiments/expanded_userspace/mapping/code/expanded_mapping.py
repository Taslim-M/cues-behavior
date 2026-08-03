"""Independent mapping of the EXPANDED user-space (289 decorrelated-factorial
users) to the model's internal Big Five traits and Assistant-Axis personas.

This reuses the user_persona_mapping *readout* (project the model's response
activation onto our Big Five probes, the Assistant Axis, and the 275 role
vectors) but runs it over the already-captured `resp_mean` activations from the
expanded_userspace rollouts -- no new generation. Kept fully separate from the
150-user study.

    python -m src.useraxis.expanded_mapping

Stage 1 only (projection + per-persona aggregate). Factor analysis is a separate
module (expanded_factor_analysis.py).
"""
from __future__ import annotations

import glob
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from src import config
from src.bigfive import stimuli as BF

ROOT = config.ROOT
EXP = ROOT / "experiments" / "expanded_userspace"
PERSONAS = EXP / "data" / "personas.jsonl"
ROLL = EXP / "results" / "llama-3.3-70b" / "rollouts"
BF_DIR = ROOT / "results" / "bigfive" / "llama-3.3-70b"
AA_PATH = ROOT / "results" / "useraxis" / "llama-3.3-70b" / "assistant_axis.npy"
AA_VEC_DIR = Path("/dev/shm/aa_vectors")
AA_LAYER = 40
OUT = EXP / "mapping"


def load_role_matrix():
    names, rows = [], []
    for f in sorted(glob.glob(str(AA_VEC_DIR / "role_vectors" / "*.pt"))):
        names.append(os.path.basename(f)[:-3])
        rows.append(torch.load(f).float().numpy()[AA_LAYER])
    names.append("default")
    rows.append(torch.load(AA_VEC_DIR / "default_vector.pt").float().numpy()[AA_LAYER])
    R = np.stack(rows)
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
    return names, R.astype(np.float32)


def main():
    bank = {t: np.load(BF_DIR / "direction_bank.npz")[t] for t in BF.TRAITS}
    sel = json.loads((BF_DIR / "stage1_selection.json").read_text())["selection"]
    probe_layer = {t: int(sel[t]["layer"]) for t in BF.TRAITS}
    probe_vec = {t: bank[t][probe_layer[t]].astype(np.float32) for t in BF.TRAITS}
    AA = np.load(AA_PATH).astype(np.float32)
    aa_unit = AA[AA_LAYER] / (np.linalg.norm(AA[AA_LAYER]) + 1e-8)
    role_names, R = load_role_matrix()
    # only these residual-stream layers are needed; safetensors get_slice supports a
    # contiguous row range, so load the [lo:hi] block covering them (not all 80).
    need_layers = sorted({*probe_layer.values(), AA_LAYER})
    lo, hi = min(need_layers), max(need_layers) + 1
    print(f"[map] probe layers {probe_layer}; {len(role_names)} roles @L{AA_LAYER}; "
          f"reading layer block [{lo}:{hi}]")

    personas = [json.loads(l) for l in open(PERSONAS) if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "personas").mkdir(exist_ok=True)

    pmap = {}
    all_bf = {t: [] for t in BF.TRAITS}
    skipped = []
    for pi, p in enumerate(personas):
        pid = p["persona_id"]
        rows = []
        for arm in ("explicit", "implicit"):
            f = ROLL / arm / f"{pid}.acts.safetensors"
            if not f.exists():
                continue
            try:
                st = safe_open(str(f), framework="np")
            except Exception as e:                              # truncated/corrupt file
                skipped.append(f"{arm}/{pid}: {type(e).__name__}")
                continue
            with st:
                keys = [k for k in st.keys() if k.endswith("|resp_mean")]
                for k in keys:
                    block = st.get_slice(k)[lo:hi, :]              # [hi-lo, 8192]
                    lyr = {L: block[L - lo] for L in need_layers}
                    g40 = lyr[AA_LAYER]
                    g40u = g40 / (np.linalg.norm(g40) + 1e-8)
                    cos = R @ g40u
                    top = np.argsort(-cos)[:5]
                    rows.append({
                        "arm": arm, "rollout": k.split("|")[0],
                        "bigfive": {t: float(lyr[probe_layer[t]] @ probe_vec[t]) for t in BF.TRAITS},
                        "aa_proj": float(g40 @ aa_unit),
                        "top_roles": [[role_names[j], round(float(cos[j]), 4)] for j in top],
                    })
        if not rows:
            continue
        (OUT / "personas" / f"{pid}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")

        entry = {"name": p.get("name"), "factors": p.get("factors", {}),
                 "tags": p.get("tags", {}), "n": len(rows)}
        for t in BF.TRAITS:
            v = np.array([r["bigfive"][t] for r in rows])
            entry.setdefault("bigfive", {})[t] = {"mean": float(v.mean()), "std": float(v.std())}
            all_bf[t].append((pid, float(v.mean())))
        entry["aa_proj"] = float(np.mean([r["aa_proj"] for r in rows]))
        for arm in ("explicit", "implicit"):
            sub = [r for r in rows if r["arm"] == arm]
            if sub:
                entry.setdefault("bigfive_by_arm", {})[arm] = {
                    t: float(np.mean([r["bigfive"][t] for r in sub])) for t in BF.TRAITS}
                entry.setdefault("aa_by_arm", {})[arm] = float(np.mean([r["aa_proj"] for r in sub]))
        votes, cos_sum = {}, {}
        for r in rows:
            top1 = r["top_roles"][0][0]
            votes[top1] = votes.get(top1, 0) + 1
            for name, c in r["top_roles"]:
                cos_sum[name] = cos_sum.get(name, 0.0) + c
        entry["top_role_vote"] = sorted(votes.items(), key=lambda x: -x[1])[:3]
        entry["top_role_meancos"] = sorted(cos_sum.items(), key=lambda x: -x[1])[:5]
        pmap[pid] = entry
        if pi % 40 == 0 or pi == len(personas) - 1:
            print(f"  {pi+1}/{len(personas)} {pid} n={len(rows)} "
                  f"aa={entry['aa_proj']:+.2f} top={entry['top_role_vote'][0][0]}", flush=True)

    mu = {t: float(np.mean([m for _, m in all_bf[t]])) for t in BF.TRAITS}
    sd = {t: float(np.std([m for _, m in all_bf[t]])) or 1.0 for t in BF.TRAITS}
    for pid, e in pmap.items():
        e["bigfive_z"] = {t: round((e["bigfive"][t]["mean"] - mu[t]) / sd[t], 3) for t in BF.TRAITS}

    (OUT / "persona_map.json").write_text(json.dumps(
        {"personas": pmap, "bigfive_pop_mean": mu, "bigfive_pop_std": sd, "n": len(pmap),
         "skipped_arms": skipped}, indent=1))
    print(f"[agg] wrote {OUT}/persona_map.json ({len(pmap)} personas); "
          f"skipped {len(skipped)} corrupt arm-files: {skipped}")
    c = Counter(e["top_role_vote"][0][0] for e in pmap.values() if e.get("top_role_vote"))
    print("  most-evoked roles:", c.most_common(10))
    print("  AA range:", round(min(e["aa_proj"] for e in pmap.values()), 2),
          "..", round(max(e["aa_proj"] for e in pmap.values()), 2))


if __name__ == "__main__":
    main()
