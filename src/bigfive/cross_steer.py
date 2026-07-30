"""Stage 4 §6.3 -- cross-steering causal tie between Big Five and the Assistant Axis (H4).

Converts the Stage 3 *correlational* decomposition into a *causal* map:

  A. Steer along each Big Five direction (selected steering-optimal high pole),
     generate on neutral prompts, extract the response activation, and measure the
     induced change in Assistant-Axis projection (does raising trait X move the model
     on/off the Assistant?).
  B. Steer along the Assistant Axis (+/- addition across a band), and measure the
     induced change in each Big Five forced-choice score (does becoming more/less
     Assistant-like move the Big Five profile?).

Produces cross_steering_matrix.json and an H4 verdict (per-turn Big Five predicts
Assistant-Axis movement). The multi-turn drift study (§6.2, 200 conversations) is
noted as future work; this causal tie is the plan's "strong test -- do not skip".

    python -m src.bigfive.cross_steer --dir results/bigfive/llama-3.3-70b \
        --aa results/useraxis/llama-3.3-70b/assistant_axis.npy --acts-dir /dev/shm/bf_acts
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "/workspace/assistant-axis")
from assistant_axis.steering import ActivationSteering  # noqa: E402

from src.bigfive import stimuli as S
from src.bigfive import steer as ST

AA_LAYER = 40


def neutral_prompts() -> list[list[dict]]:
    return [[{"role": "user", "content": a["instruction"]}] for a in S.alpaca10()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--aa", required=True)
    ap.add_argument("--acts-dir", default="/dev/shm/bf_acts")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    d = Path(args.dir)

    from src.bigfive.steer_confirm import cfg_from_rec
    from src.useraxis.extract import DEFAULT_MODEL, load_model, DualReadoutExtractor

    bank = {t: np.load(d / "direction_bank.npz")[t] for t in S.TRAITS}
    resid_norms = np.load(d / "resid_norms.npy")
    AA = np.load(args.aa).astype(np.float32)
    aa_unit = AA / (np.linalg.norm(AA, axis=1, keepdims=True) + 1e-8)
    acts_char = np.load(Path(args.acts_dir) / "acts_characters" / "acts_prompt_mean.npy",
                        mmap_mode="r")
    steering = json.loads((d / "steering_results.json").read_text())["selection"]

    pm = load_model(DEFAULT_MODEL)
    ex = DualReadoutExtractor(pm)
    proj_pct = ST.compute_proj_percentiles(acts_char, bank)
    steerer = ST.Steerer(pm.model, bank, resid_norms, proj_pct)
    stmt_sets = ST.build_statement_sets()

    # ---- A. steer Big Five -> measure Assistant-Axis projection ----
    convs = neutral_prompts()

    def mean_aa_proj(ctx_factory):
        # Generate under steering, then extract the response activations under the
        # same steering, and project the response-mean at the AA layer onto the axis.
        with ctx_factory():
            resps = ex.generate_batch(convs, max_new_tokens=120, temperature=0.7, top_p=0.9)
            full = [c + [{"role": "assistant", "content": r}] for c, r in zip(convs, resps)]
            reps = ex.extract_batch(full)               # list of {resp_mean:[L,D]} or None
        vals = [float(r["resp_mean"][AA_LAYER] @ aa_unit[AA_LAYER])
                for r in reps if r is not None]
        return float(np.mean(vals)) if vals else float("nan")

    base_aa = mean_aa_proj(lambda: nullcontext())
    A = {}
    for t in S.TRAITS:
        hi = cfg_from_rec(steering[t]["high_pole"]["cfg"])
        lo = cfg_from_rec(steering[t]["low_pole"]["cfg"])
        aa_hi = mean_aa_proj(lambda: steerer.apply(t, hi))
        aa_lo = mean_aa_proj(lambda: steerer.apply(t, lo))
        A[t] = {"aa_proj_base": base_aa, "aa_proj_high": aa_hi, "aa_proj_low": aa_lo,
                "delta_high": aa_hi - base_aa, "delta_low": aa_lo - base_aa,
                "swing": aa_hi - aa_lo}
        print(f"[A] steer {t}: AA base={base_aa:+.2f} high={aa_hi:+.2f} low={aa_lo:+.2f} "
              f"swing={aa_hi-aa_lo:+.2f}", flush=True)

    # ---- B. steer Assistant Axis -> measure Big Five forced-choice ----
    @contextmanager
    def steer_aa(coef, band=(28, 52)):
        lo, hi = band
        layers = list(range(lo, hi))
        vecs = [torch.tensor(aa_unit[l]) for l in layers]
        coefs = [coef * float(resid_norms[l]) for l in layers]
        with ActivationSteering(pm.model, steering_vectors=vecs, coefficients=coefs,
                                layer_indices=layers, intervention_type="addition",
                                positions="all"):
            yield

    def fc_pf(trait_measured, ctx_factory):
        pos_set = set(stmt_sets[trait_measured]["pos"])
        cs, orders = [], []
        for seed in range(args.seeds):
            rng = np.random.default_rng(200 + seed)
            order = stmt_sets[trait_measured]["pos"] + stmt_sets[trait_measured]["neg"]
            order = list(order); rng.shuffle(order); orders.append(order)
            cs.append(ST.forced_choice_messages(order))
        with ctx_factory():
            outs = ex.generate_batch(cs, max_new_tokens=80, temperature=0.7, top_p=0.9)
        pfs = [ST.positive_fraction(p, pos_set)
               for order, txt in zip(orders, outs)
               if (p := ST.parse_picks(txt, order))]
        return float(np.mean(pfs)) if pfs else None

    B = {}
    for t in S.TRAITS:
        base = fc_pf(t, lambda: nullcontext())
        toward = fc_pf(t, lambda: steer_aa(+0.1))   # more Assistant-like
        away = fc_pf(t, lambda: steer_aa(-0.1))      # less Assistant-like
        B[t] = {"pf_base": base, "pf_toward_assistant": toward, "pf_away_assistant": away,
                "delta_toward": (toward - base) if (toward is not None and base is not None) else None}
        print(f"[B] steer AA: {t} base={base} toward={toward} away={away}", flush=True)

    # H4 verdict: does steering Big Five reliably move AA? (A) and does AA steering
    # move Big Five? (B). Report the causal map; "confirmed" if the strongest Big
    # Five->AA effect and AA->Big Five effect both clear a small threshold.
    max_A = max((abs(A[t]["swing"]) for t in S.TRAITS))
    strongest_A = max(S.TRAITS, key=lambda t: abs(A[t]["swing"]))
    H4 = {"strongest_bigfive_to_AA": strongest_A,
          "max_AA_swing": max_A,
          "AA_to_bigfive_effects": {t: B[t]["delta_toward"] for t in S.TRAITS},
          "causal_tie_present": max_A > 0.3}

    out = {"A_bigfive_to_assistant_axis": A, "B_assistant_axis_to_bigfive": B, "H4": H4}
    (d / "cross_steering_matrix.json").write_text(json.dumps(out, indent=1))
    print("\n==== H4 cross-steering causal tie ====")
    print(f"strongest Big Five->AA: {strongest_A} (swing {max_A:+.2f})")
    print(f"AA->Big Five: " + " ".join(
        f"{t}:{B[t]['delta_toward']:+.2f}" if B[t]['delta_toward'] is not None else f"{t}:NA"
        for t in S.TRAITS))
    print("wrote", d / "cross_steering_matrix.json")


if __name__ == "__main__":
    main()
