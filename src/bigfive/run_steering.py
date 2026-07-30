"""Run the forced-choice steering grid for all traits (Stage 1 §3.6, primary metric).

For every (trait x config) it generates N_SEEDS forced-choice completions under the
steering intervention, parses the 5 picks, and records the positive fraction (and a
held-out-only positive fraction) plus a coherence flag (did the model return a valid
5-pick list). Writes results/bigfive/<model>/steering_grid.json.

    python -m src.bigfive.run_steering --acts-dir /dev/shm/bf_acts \
        --dir results/bigfive/llama-3.3-70b [--seeds 3] [--limit-configs 0]
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
from src.useraxis.extract import DEFAULT_MODEL, load_model, DualReadoutExtractor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default="/dev/shm/bf_acts")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--traits", default="", help="comma subset (debug)")
    ap.add_argument("--limit-configs", type=int, default=0)
    ap.add_argument("--out", default="steering_grid.json")
    args = ap.parse_args()

    d = Path(args.dir)
    bank_npz = np.load(d / "direction_bank.npz")
    bank = {t: bank_npz[t] for t in S.TRAITS}
    resid_norms = np.load(d / "resid_norms.npy")
    n_layers = bank[S.TRAITS[0]].shape[0]

    char_root = Path(args.acts_dir) / "acts_characters"
    acts_char = np.load(char_root / "acts_prompt_mean.npy", mmap_mode="r")
    print("[grid] computing per-layer projection percentiles for S2 thresholds...")
    proj_pct = ST.compute_proj_percentiles(acts_char, bank)

    stmt_sets = ST.build_statement_sets()
    configs = ST.build_configs(n_layers)
    if args.limit_configs:
        # keep a representative slice: some S0, S1, S2
        s0 = [c for c in configs if c["kind"] == "S0"][:args.limit_configs]
        s1 = [c for c in configs if c["kind"] == "S1"][:args.limit_configs]
        s2 = [c for c in configs if c["kind"] == "S2"][:args.limit_configs]
        configs = s0 + s1 + s2
    traits = args.traits.split(",") if args.traits else list(S.TRAITS)
    print(f"[grid] {len(configs)} configs x {len(traits)} traits x {args.seeds} seeds")

    pm = load_model(DEFAULT_MODEL)
    ex = DualReadoutExtractor(pm)
    steerer = ST.Steerer(pm.model, bank, resid_norms, proj_pct)

    def run_forced_choice(trait, ctx_factory):
        """Generate seeds under a steering context; return (mean_pos, mean_pos_ext, coherence)."""
        pos_set = set(stmt_sets[trait]["pos"])
        prov = stmt_sets[trait]["provenance"]
        ext_pos = {s for s in pos_set if prov[s] == "ext"}
        ext_all = {s for s in (stmt_sets[trait]["pos"] + stmt_sets[trait]["neg"])
                   if prov[s] == "ext"}
        convs, orders = [], []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            stmts = stmt_sets[trait]["pos"] + stmt_sets[trait]["neg"]
            order = list(stmts)
            rng.shuffle(order)
            orders.append(order)
            convs.append(ST.forced_choice_messages(order))
        with ctx_factory():
            outs = ex.generate_batch(convs, max_new_tokens=args.max_new_tokens,
                                     temperature=0.7, top_p=0.9)
        pfr, pext, ok = [], [], 0
        for order, txt in zip(orders, outs):
            picks = ST.parse_picks(txt, order)
            if picks is None:
                continue
            ok += 1
            pfr.append(ST.positive_fraction(picks, pos_set))
            ext_picks = [p for p in picks if p in ext_all]
            if ext_picks:
                pext.append(sum(1 for p in ext_picks if p in ext_pos) / len(ext_picks))
        return (float(np.mean(pfr)) if pfr else None,
                float(np.mean(pext)) if pext else None,
                ok / args.seeds)

    results = {t: {} for t in traits}
    t0 = time.time()
    # baseline (no steering) per trait
    for trait in traits:
        from contextlib import nullcontext
        mp, me, coh = run_forced_choice(trait, lambda: nullcontext())
        results[trait]["baseline"] = {"pos_frac": mp, "pos_frac_ext": me,
                                      "coherence": coh, "kind": "baseline"}

    total = len(configs) * len(traits)
    done = 0
    for trait in traits:
        for cfg in configs:
            mp, me, coh = run_forced_choice(
                trait, lambda c=cfg, t=trait: steerer.apply(t, c))
            results[trait][cfg["label"]] = {
                **{k: cfg[k] for k in cfg if k != "label"},
                "pos_frac": mp, "pos_frac_ext": me, "coherence": coh}
            done += 1
            if done % 50 == 0 or done == total:
                el = time.time() - t0
                print(f"  {done}/{total}  {el:.0f}s  eta {(total-done)/max(done/el,1e-9)/60:.1f}min",
                      flush=True)

    outp = d / args.out
    outp.write_text(json.dumps(results, indent=1))
    print(f"[grid] wrote {outp} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
