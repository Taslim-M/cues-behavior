"""Experiment `user_persona_mapping` (see user_persona_mapping.md).

For each of the 150 synthetic *user* personas, roll out the model in its plain
default state (no role of its own) under two arms -- explicit (user described in
the system slot + a neutral probe) and implicit (the user's self-revealing
opener) -- and read the model's *response* activation (``gen_mean``) three ways:

  1. Big Five traits activated  -> project onto our M2 probes at probe-optimal layers
  2. Which LLM persona activated -> cosine of gen_mean[L40] vs the 274 Assistant-Axis
                                    role vectors (+ default) -> top-k evoked roles
  3. Assistant-Axis position     -> gen_mean[L40] . unit(AA[L40])

Everything is projected in-flight; only the per-rollout readings are stored.

    python -m src.useraxis.user_persona_mapping --explicit 4 --implicit 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src import config
from src.bigfive import stimuli as BF
from src.bigfive.extract import BigFiveExtractor
from src.useraxis.extract import DEFAULT_MODEL, load_model, short_name
from src.useraxis.run_rollouts import sample_shared_probes

ROOT = config.ROOT
PERSONAS = ROOT / "generate_synthetic_data" / "user_personas.jsonl"
QUESTIONS = ROOT / "generate_synthetic_data" / "extraction_questions.json"
BF_DIR = ROOT / "results" / "bigfive" / "llama-3.3-70b"
AA_PATH = ROOT / "results" / "useraxis" / "llama-3.3-70b" / "assistant_axis.npy"
AA_VEC_DIR = Path("/dev/shm/aa_vectors")
AA_LAYER = 40


def load_personas():
    return [json.loads(l) for l in open(PERSONAS) if l.strip()]


def build_explicit(persona, probes):
    """system = an explicit user-description; user = a shared neutral probe."""
    n_sys = len(persona["explicit_system_prompts"])
    stubs = []
    for j, probe in enumerate(probes):
        e = j % n_sys
        stubs.append({"arm": "explicit", "elicit_idx": e, "probe_id": probe.get("id", j),
                      "messages": [{"role": "system", "content": persona["explicit_system_prompts"][e]},
                                   {"role": "user", "content": probe["text"]}]})
    return stubs


def build_implicit(persona, n):
    """no system; user = a self-revealing opener."""
    openers = persona["implicit_openers"]
    stubs = []
    for j in range(n):
        e = j % len(openers)
        stubs.append({"arm": "implicit", "elicit_idx": e, "probe_id": -1,
                      "messages": [{"role": "user", "content": openers[e]}]})
    return stubs


def load_role_matrix():
    """[names], R[n,8192] unit-normalized role vectors at L40 (+ default)."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--explicit", type=int, default=4)
    ap.add_argument("--implicit", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "user_persona_mapping" / "llama-3.3-70b"))
    args = ap.parse_args()

    # ---- readout artifacts ----
    bank = {t: np.load(BF_DIR / "direction_bank.npz")[t] for t in BF.TRAITS}
    sel = json.loads((BF_DIR / "stage1_selection.json").read_text())["selection"]
    probe_layer = {t: int(sel[t]["layer"]) for t in BF.TRAITS}
    probe_vec = {t: bank[t][probe_layer[t]].astype(np.float32) for t in BF.TRAITS}
    AA = np.load(AA_PATH).astype(np.float32)
    aa_unit = AA[AA_LAYER] / (np.linalg.norm(AA[AA_LAYER]) + 1e-8)
    role_names, R = load_role_matrix()
    print(f"[map] probe layers {probe_layer}; {len(role_names)} role vectors @L{AA_LAYER}")

    personas = load_personas()
    if args.limit:
        personas = personas[:args.limit]
    probes = sample_shared_probes(QUESTIONS, args.explicit, args.seed)  # neutral shared probes

    out_dir = Path(args.out_dir)
    (out_dir / "personas").mkdir(parents=True, exist_ok=True)
    todo = [p for p in personas if not (out_dir / "personas" / f"{p['persona_id']}.jsonl").exists()]
    total = len(personas) * (args.explicit + args.implicit)
    print(f"[map] {len(personas)} personas x ({args.explicit} expl + {args.implicit} impl) "
          f"= {total} rollouts; {len(todo)} personas to run")
    if not todo:
        aggregate(out_dir, personas, role_names); return

    pm = load_model(args.model)
    ex = BigFiveExtractor(pm)
    t0 = time.time(); done = 0
    for pi, persona in enumerate(todo):
        stubs = build_explicit(persona, probes) + build_implicit(persona, args.implicit)
        rows = []
        for s in range(0, len(stubs), args.batch_size):
            batch = stubs[s:s + args.batch_size]
            acts, texts = ex.run_batch([b["messages"] for b in batch], generate=True,
                                       max_new_tokens=args.max_new_tokens,
                                       do_sample=args.temperature > 0,
                                       temperature=args.temperature, top_p=1.0)
            gm = acts["gen_mean"]                                  # [B, n_layers, d]
            for bi, b in enumerate(batch):
                g40 = gm[bi, AA_LAYER]
                g40u = g40 / (np.linalg.norm(g40) + 1e-8)
                cos = R @ g40u                                    # [n_roles]
                top = np.argsort(-cos)[:5]
                rows.append({
                    "persona_id": persona["persona_id"], "arm": b["arm"],
                    "elicit_idx": b["elicit_idx"], "probe_id": b["probe_id"],
                    "bigfive": {t: float(gm[bi, probe_layer[t]] @ probe_vec[t]) for t in BF.TRAITS},
                    "aa_proj": float(g40 @ aa_unit),
                    "top_roles": [[role_names[j], round(float(cos[j]), 4)] for j in top],
                })
            done += len(batch)
        (out_dir / "personas" / f"{persona['persona_id']}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        if pi % 10 == 0 or pi == len(todo) - 1:
            el = time.time() - t0
            print(f"  persona {pi+1}/{len(todo)} {persona['persona_id']} "
                  f"{done/el*3600:.0f} rollouts/hr", flush=True)

    aggregate(out_dir, personas, role_names)


def aggregate(out_dir, personas, role_names):
    """Per-persona map: Big Five mean/std, mean AA-proj, top evoked role (vote), tags."""
    pmap = {}
    all_bf = {t: [] for t in BF.TRAITS}
    for p in personas:
        f = out_dir / "personas" / f"{p['persona_id']}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        entry = {"name": p.get("name"), "lean": p.get("lean"), "tags": p.get("tags", {}), "n": len(recs)}
        # Big Five per arm + overall
        for t in BF.TRAITS:
            v = np.array([r["bigfive"][t] for r in recs])
            entry.setdefault("bigfive", {})[t] = {"mean": float(v.mean()), "std": float(v.std())}
            all_bf[t].append((p["persona_id"], float(v.mean())))
        entry["aa_proj"] = float(np.mean([r["aa_proj"] for r in recs]))
        for arm in ("explicit", "implicit"):
            sub = [r for r in recs if r["arm"] == arm]
            if sub:
                entry.setdefault("bigfive_by_arm", {})[arm] = {
                    t: float(np.mean([r["bigfive"][t] for r in sub])) for t in BF.TRAITS}
        # top evoked role: vote across rollouts by rank-1, plus mean cosine per role
        votes = {}
        cos_sum = {}
        for r in recs:
            top1 = r["top_roles"][0][0]
            votes[top1] = votes.get(top1, 0) + 1
            for name, c in r["top_roles"]:
                cos_sum[name] = cos_sum.get(name, 0.0) + c
        entry["top_role_vote"] = sorted(votes.items(), key=lambda x: -x[1])[:3]
        entry["top_role_meancos"] = sorted(cos_sum.items(), key=lambda x: -x[1])[:5]
        pmap[p["persona_id"]] = entry

    # z-score Big Five means across the persona population
    mu = {t: float(np.mean([m for _, m in all_bf[t]])) for t in BF.TRAITS}
    sd = {t: float(np.std([m for _, m in all_bf[t]])) or 1.0 for t in BF.TRAITS}
    for pid, e in pmap.items():
        e["bigfive_z"] = {t: round((e["bigfive"][t]["mean"] - mu[t]) / sd[t], 3) for t in BF.TRAITS}

    (out_dir / "persona_map.json").write_text(json.dumps(
        {"personas": pmap, "bigfive_pop_mean": mu, "bigfive_pop_std": sd, "n": len(pmap)}, indent=1))
    print(f"[agg] wrote persona_map.json ({len(pmap)} personas)")
    # quick peek: most common evoked roles
    from collections import Counter
    c = Counter(e["top_role_vote"][0][0] for e in pmap.values() if e.get("top_role_vote"))
    print("  most-evoked roles:", c.most_common(8))


if __name__ == "__main__":
    main()
