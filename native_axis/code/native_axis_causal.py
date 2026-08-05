"""Causal test: does steering the native behavioral axes move the Assistant Axis?

Builds per-layer direction banks for the native factors + a Big-Five contrastive
control, then steers each factor's band [24,36] (EXCLUDING the L40 readout, so the
AA shift is a computed downstream effect, not the injected vector's image) at +-c
on neutral prompts and measures the induced Assistant-Axis swing.

    HF_HOME=/dev/shm/hf HF_HUB_OFFLINE=1 python -m src.bigfive.native_axis_causal
"""
from __future__ import annotations

import glob
import json
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/assistant-axis")
from assistant_axis.steering import ActivationSteering  # noqa: E402

from src.bigfive.native_axis_build import NATIVE, BIGFIVE_CONTRAST
from src.bigfive.extract import BigFiveExtractor
from src.bigfive.role_profiles import load_questions, conv
from src.useraxis.extract import DEFAULT_MODEL, load_model

RES = Path("native_axis/results/causal")
AA_LAYER = 40
BAND = (24, 36)
COEF = 0.12
GROUND_SWEEP = [0.06, 0.12, 0.24]


def build_bank(ex, spec, questions):
    """{factor: [80,8192] unit direction per layer} from contrastive high/low."""
    stubs = []
    for fac, (hi, lo) in spec.items():
        for pole, sysp in (("high", hi), ("low", lo)):
            for q in questions:
                stubs.append((fac, pole, conv(sysp, q["question"])))
    acc = {(f, p): [] for f in spec for p in ("high", "low")}
    B = 24
    for s in range(0, len(stubs), B):
        batch = stubs[s:s + B]
        actsb, _ = ex.run_batch([b[2] for b in batch], generate=True, max_new_tokens=128,
                                do_sample=True, temperature=0.8, top_p=0.9)
        gm = actsb["gen_mean"]                                  # [B,80,d]
        for bi, (f, p, _) in enumerate(batch):
            acc[(f, p)].append(gm[bi].astype(np.float64))       # [80,d]
        print(f"  [bank] {min(s+B,len(stubs))}/{len(stubs)}", flush=True)
    bank = {}
    for f in spec:
        hi = np.mean(acc[(f, "high")], axis=0)                  # [80,d]
        lo = np.mean(acc[(f, "low")], axis=0)
        d = (hi - lo)
        bank[f] = (d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    return bank


def main():
    RES.mkdir(parents=True, exist_ok=True)
    resid_norms = np.load("results/bigfive/llama-3.3-70b/resid_norms.npy")
    aa = np.load("results/useraxis/llama-3.3-70b/assistant_axis.npy").astype(np.float32)
    aa_unit = aa[AA_LAYER] / (np.linalg.norm(aa[AA_LAYER]) + 1e-8)
    # role matrix for nearest-role readout
    rnames, rows = [], []
    for f in sorted(glob.glob("/dev/shm/aa_vectors/role_vectors/*.pt")):
        rnames.append(os.path.basename(f)[:-3]); rows.append(torch.load(f).float().numpy()[AA_LAYER])
    R = np.stack(rows); R = (R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    pm = load_model(DEFAULT_MODEL)
    ex = BigFiveExtractor(pm)

    qb = load_questions(12)
    nb = build_bank(ex, NATIVE, qb)
    bb = build_bank(ex, BIGFIVE_CONTRAST, qb)
    banks = {**{f"nat:{k}": v for k, v in nb.items()}, **{f"bf:{k}": v for k, v in bb.items()}}
    np.savez(RES / "banks.npz", **{k.replace(":", "__"): v for k, v in banks.items()})

    @contextmanager
    def steer(bankf, c):
        lo, hi = BAND
        layers = list(range(lo, hi))
        vecs = [torch.tensor(bankf[l]) for l in layers]
        coefs = [c * float(resid_norms[l]) for l in layers]
        with ActivationSteering(pm.model, steering_vectors=vecs, coefficients=coefs,
                                layer_indices=layers, intervention_type="addition", positions="all"):
            yield

    neutral = [conv("", q["question"]) for q in load_questions(16)]

    def measure(ctx_factory, keep_text=0):
        aavals, roles, texts = [], [], []
        B = 16
        for s in range(0, len(neutral), B):
            batch = neutral[s:s + B]
            with ctx_factory():
                actsb, txt = ex.run_batch(batch, generate=True, max_new_tokens=96,
                                          do_sample=True, temperature=0.8, top_p=0.9)
            gm = actsb["gen_mean"]
            for bi in range(gm.shape[0]):
                g = gm[bi, AA_LAYER]; gu = g / (np.linalg.norm(g) + 1e-8)
                aavals.append(float(g @ aa_unit))
                roles.append(rnames[int(np.argmax(R @ gu))])
                if len(texts) < keep_text:
                    texts.append(txt[bi].strip().replace("\n", " ")[:200])
        from collections import Counter
        return (float(np.mean(aavals)), Counter(roles).most_common(3), texts)

    print("[causal] measuring base (no steer)...", flush=True)
    base_aa, base_roles, _ = measure(lambda: nullcontext())
    print(f"  base AA={base_aa:+.2f} roles={base_roles}", flush=True)

    results = {"band": list(BAND), "coef": COEF, "base_aa": round(base_aa, 3),
               "base_roles": base_roles, "factors": {}}
    for key, bankf in banks.items():
        aap, rolp, txtp = measure(lambda: steer(bankf, +COEF), keep_text=2)
        aam, rolm, txtm = measure(lambda: steer(bankf, -COEF), keep_text=2)
        results["factors"][key] = {
            "aa_plus": round(aap, 3), "aa_minus": round(aam, 3),
            "swing": round(aap - aam, 3),
            "delta_plus": round(aap - base_aa, 3), "delta_minus": round(aam - base_aa, 3),
            "roles_plus": rolp, "roles_minus": rolm,
            "text_plus": txtp, "text_minus": txtm}
        print(f"[causal] {key:16} AA {aam:+.2f}(-c) <- {base_aa:+.2f} -> {aap:+.2f}(+c) "
              f"swing={aap-aam:+.2f}", flush=True)

    # dose-response for groundedness
    sweep = {}
    for c in GROUND_SWEEP:
        ap, _, _ = measure(lambda: steer(nb["groundedness"], +c))
        am, _, _ = measure(lambda: steer(nb["groundedness"], -c))
        sweep[str(c)] = {"aa_plus": round(ap, 3), "aa_minus": round(am, 3), "swing": round(ap - am, 3)}
        print(f"[sweep] groundedness c={c}: swing={ap-am:+.2f}", flush=True)
    results["groundedness_dose_response"] = sweep

    (RES / "causal.json").write_text(json.dumps(results, indent=1))
    # ranking summary
    rank = sorted(results["factors"].items(), key=lambda x: -abs(x[1]["swing"]))
    print("\n=== Assistant-Axis swing ranking (steer band [24,36], excludes L40) ===")
    for k, v in rank:
        print(f"  {k:16} swing={v['swing']:+.2f}  (+c {v['delta_plus']:+.2f} / -c {v['delta_minus']:+.2f})")
    print(f"\nwrote {RES}/causal.json + banks.npz")


if __name__ == "__main__":
    main()
