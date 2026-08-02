"""Single-trait persona morphing (trait_morph experiment).

For character pairs that differ in mostly ONE Big Five trait, steer just that
trait while the model is in character A and test whether the L40 persona vector
(gen_mean at layer 40 -- the Assistant-Axis readout) slides toward character B.

Conditions per pair: base_A, ref_B, target (steer T toward B), off_target (steer
an irrelevant trait), wrong_dir (steer T the wrong way). We save the mean L40
persona vector, the Big Five probe readings (manipulation check) and the AA
projection for every condition, plus a distractor set built from all references.

    python -m src.bigfive.trait_morph --n-sys 3 --n-q 10 --out /dev/shm/trait_morph

See trait_morph/PLAN.md for the design and controls.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.bigfive import stimuli as S
from src.bigfive import steer as ST
from src.bigfive.steer_confirm import cfg_from_rec
from src.bigfive.extract import BigFiveExtractor
from src.bigfive.role_profiles import load_roles, load_questions, conv
from src.useraxis.extract import DEFAULT_MODEL, load_model

RESULTS = Path("results/bigfive/llama-3.3-70b")
ATLAS = RESULTS / "role_profiles_atlas" / "role_bigfive_profiles.json"

# Curated pairs oriented A->B so the target edit uses a band that EXCLUDES L40
# (except the caveated EXT pair). dir is the sign of z_T(B)-z_T(A).
PAIRS = [
    {"trait": "AGR", "A": "vampire",     "B": "tutor",     "note": "AGR up"},
    {"trait": "AGR", "A": "interpreter", "B": "vampire",   "note": "AGR down"},
    {"trait": "OPN", "A": "auditor",     "B": "prodigy",   "note": "OPN up"},
    {"trait": "OPN", "A": "mystic",      "B": "organizer", "note": "OPN down"},
    {"trait": "CSN", "A": "nomad",       "B": "engineer",  "note": "CSN up"},
    {"trait": "EST", "A": "narrator",    "B": "futurist",  "note": "EST up"},
    {"trait": "EXT", "A": "journalist",  "B": "predator",  "note": "EXT down (caveat: S0 hits L40)"},
]
# traits usable as off-target edits (real, non-no-op high poles)
OFFTRAIT_POOL = ["AGR", "CSN", "EST", "OPN"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-sys", type=int, default=3, help="system prompts per character")
    ap.add_argument("--n-q", type=int, default=10, help="extraction questions")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--out", default="/dev/shm/trait_morph")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- probes, steering bank, AA ----
    bank = {t: np.load(RESULTS / "direction_bank.npz")[t] for t in S.TRAITS}
    resid_norms = np.load(RESULTS / "resid_norms.npy")
    sel = json.loads((RESULTS / "stage1_selection.json").read_text())["selection"]
    probe_layer = {t: int(sel[t]["layer"]) for t in S.TRAITS}
    probe_vec = {t: bank[t][probe_layer[t]].astype(np.float32) for t in S.TRAITS}
    steer_sel = json.loads((RESULTS / "steering_results.json").read_text())["selection"]
    _aa = np.load(RESULTS.parent.parent / "useraxis" / "llama-3.3-70b" / "assistant_axis.npy").astype(np.float32)
    aa_unit = _aa[40] / (np.linalg.norm(_aa[40]) + 1e-8)

    atlas = json.loads(ATLAS.read_text())["per_role"]
    Z = {n: {t: atlas[n]["z_vs_roles"][t] for t in S.TRAITS} for n in atlas}

    roles = {r["name"]: r["system_prompts"] for r in load_roles()}
    questions = load_questions(args.n_q)

    # steerer (no S2 configs are selected, so proj_pct is unused)
    steerer = ST.Steerer(None, bank, resid_norms, {t: {} for t in S.TRAITS})

    pm = load_model(args.model)
    ex = BigFiveExtractor(pm)
    steerer.model = pm.model

    def pole_cfg(trait, pole):
        return cfg_from_rec(steer_sel[trait][f"{pole}_pole"]["cfg"])

    def run_condition(char, steer_spec):
        """Generate over the probe set (optionally under steering); return the
        mean L40 persona vector, mean Big Five readings, mean AA proj, n."""
        sysprompts = roles[char][: args.n_sys]
        units = [conv(sp, q["question"]) for sp in sysprompts for q in questions]
        sum40 = np.zeros(aa_unit.shape[0], dtype=np.float64)
        read_acc = {t: [] for t in S.TRAITS}
        aa_acc, n = [], 0
        ctx = (steerer.apply(*steer_spec) if steer_spec else _null())
        with ctx:
            for s in range(0, len(units), args.batch_size):
                batch = units[s:s + args.batch_size]
                acts, _ = ex.run_batch(batch, generate=True,
                                       max_new_tokens=args.max_new_tokens,
                                       do_sample=args.temperature > 0,
                                       temperature=args.temperature, top_p=args.top_p)
                gm = acts["gen_mean"]                      # [B, n_layers, d]
                for bi in range(gm.shape[0]):
                    sum40 += gm[bi, 40].astype(np.float64)
                    for t in S.TRAITS:
                        read_acc[t].append(float(gm[bi, probe_layer[t]] @ probe_vec[t]))
                    aa_acc.append(float(gm[bi, 40] @ aa_unit))
                    n += 1
        vec40 = (sum40 / max(n, 1)).astype(np.float32)
        readings = {t: float(np.mean(read_acc[t])) for t in S.TRAITS}
        return vec40, readings, float(np.mean(aa_acc)), n

    vectors = {}          # "pair{i}:{cond}" -> vec40
    records = []
    t0 = time.time()
    for i, p in enumerate(PAIRS):
        T, A, B = p["trait"], p["A"], p["B"]
        dz = Z[B][T] - Z[A][T]
        target_pole = "high" if dz > 0 else "low"
        wrong_pole = "low" if dz > 0 else "high"
        # off-target trait: smallest |Δz| among the real-lever pool, excluding T
        offT = min([t for t in OFFTRAIT_POOL if t != T],
                   key=lambda t: abs(Z[B][t] - Z[A][t]))
        conds = {
            "base_A":     (A, None),
            "ref_B":      (B, None),
            "target":     (A, (T, pole_cfg(T, target_pole))),
            "off_target": (A, (offT, pole_cfg(offT, "high"))),
            "wrong_dir":  (A, (T, pole_cfg(T, wrong_pole))),
        }
        rec = {"pair": i, "trait": T, "A": A, "B": B, "note": p["note"],
               "dz_target": round(dz, 3), "target_pole": target_pole,
               "off_trait": offT, "off_dz": round(Z[B][offT] - Z[A][offT], 3),
               "z_A": Z[A], "z_B": Z[B], "cond": {}}
        for cond, (char, spec) in conds.items():
            v, reads, aap, n = run_condition(char, spec)
            vectors[f"p{i}:{cond}"] = v
            rec["cond"][cond] = {"readings": reads, "aa_proj": round(aap, 3), "n": n}
            el = time.time() - t0
            print(f"[{el/60:5.1f}m] pair{i} {T} {A}->{B} :: {cond:10s} "
                  f"n={n} aa={aap:+.2f} {T}read={reads[T]:+.3f}", flush=True)
        records.append(rec)

    np.savez(out / "vectors.npz", **vectors)
    (out / "records.json").write_text(json.dumps(
        {"model": args.model, "probe_layer": probe_layer, "pairs": records,
         "config": {"n_sys": args.n_sys, "n_q": args.n_q,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature}}, indent=1))
    print(f"\nwrote {out}/vectors.npz ({len(vectors)} vecs) + records.json "
          f"in {(time.time()-t0)/60:.1f} min")


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
