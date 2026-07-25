"""Where does a PROFILE-LESS user land on the User Axis? (GPU capture)

Every rollout in the corpus is persona-conditioned (explicit system prompt or
implicit in-voice opener), so we never measured the model's *default* user -- the
one it assumes when given no profile at all. This module captures that baseline: it
runs the SAME shared probes the personas saw, with no persona, and saves the
dual-readout activations so `analyze_baseline.py` can project them onto the shipped
PC1 and report where the default user sits relative to the 150 personas.

Two variants (they separate two confounds):
  none        -- bare probe, NO system prompt          (the literal profile-less user)
  neutral_sys -- a bland "You are a helpful assistant." (isolates "no USER info" from
                 "no system prompt at all")

Each probe is generated `--repeats` times (temperature 0.7, like the persona
rollouts); we average resp_mean/last_user over probes x repeats into one baseline
vector [n_layers, d] per variant per readout -- the exact analogue of a persona
vector, which is also a mean over the same probes. Per-sample vectors are kept only
at the analysis layers (small) for a bootstrap CI.

Saves results/useraxis/<model>/analysis/baseline/vectors.npz (+ meta.json).

Run (needs the GPU env: HF_HOME=/dev/shm/hf, expandable_segments):
    python -m src.useraxis.baseline_axis                 # 24 probes x 3 repeats
    python -m src.useraxis.baseline_axis --probes 8 --repeats 1   # smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import config
from .extract import (DEFAULT_MODEL, DualReadoutExtractor, load_model, short_name)
from .run_rollouts import QUESTIONS_PATH, sample_shared_probes

# layers kept per-sample (for CIs); the full baseline mean is saved at all layers.
ANALYSIS_LAYERS = (24, 40, 48)
NEUTRAL_SYS = "You are a helpful assistant."


def build_convs(variant: str, probe_text: str) -> list[dict]:
    if variant == "none":
        return [{"role": "user", "content": probe_text}]
    if variant == "neutral_sys":
        return [{"role": "system", "content": NEUTRAL_SYS},
                {"role": "user", "content": probe_text}]
    raise ValueError(f"unknown variant {variant}")


def capture(args) -> None:
    model = short_name(DEFAULT_MODEL)
    out = config.RESULTS_DIR / "useraxis" / model / "analysis" / "baseline"
    out.mkdir(parents=True, exist_ok=True)

    probes = sample_shared_probes(QUESTIONS_PATH, args.probes, args.seed)
    print(f"baseline capture: {len(probes)} probes x {args.repeats} repeats x "
          f"{len(args.variants)} variants", flush=True)
    pm = load_model(DEFAULT_MODEL)
    ex = DualReadoutExtractor(pm)

    store: dict[str, np.ndarray] = {}
    meta = {"probe_ids": [p["id"] for p in probes], "repeats": args.repeats,
            "seed": args.seed, "variants": args.variants,
            "n_layers": ex.n_layers, "d_model": ex.d_model,
            "analysis_layers": list(ANALYSIS_LAYERS),
            "gen": {"temperature": 0.7, "top_p": 0.9,
                    "max_new_tokens": args.max_new_tokens},
            "neutral_sys": NEUTRAL_SYS}

    for variant in args.variants:
        # accumulators over all valid samples
        sums = {ro: np.zeros((ex.n_layers, ex.d_model), np.float64)
                for ro in ("resp_mean", "last_user")}
        n_ok = 0
        per_sample = {ro: [] for ro in ("resp_mean", "last_user")}  # sliced layers
        for rep in range(args.repeats):
            for i in range(0, len(probes), args.batch_size):
                chunk = probes[i:i + args.batch_size]
                convs = [build_convs(variant, p["text"]) for p in chunk]
                resp = ex.generate_batch(convs, max_new_tokens=args.max_new_tokens)
                full = [c + [{"role": "assistant", "content": r}]
                        for c, r in zip(convs, resp)]
                feats = ex.extract_batch(full, max_length=args.max_length)
                for f in feats:
                    if f is None:
                        continue
                    n_ok += 1
                    for ro in ("resp_mean", "last_user"):
                        v = f[ro].numpy().astype(np.float64)      # [L, D]
                        sums[ro] += v
                        per_sample[ro].append(v[list(ANALYSIS_LAYERS)])
            print(f"  [{variant}] repeat {rep + 1}/{args.repeats}: "
                  f"{n_ok} valid samples so far", flush=True)
        if n_ok == 0:
            raise SystemExit(f"variant {variant}: no valid samples captured")
        for ro in ("resp_mean", "last_user"):
            store[f"{variant}__{ro}__mean"] = (sums[ro] / n_ok).astype(np.float32)
            store[f"{variant}__{ro}__samples"] = np.stack(
                per_sample[ro]).astype(np.float32)  # [n, len(ANALYSIS_LAYERS), D]
        meta.setdefault("n_valid", {})[variant] = n_ok
        print(f"  [{variant}] done: {n_ok} valid samples", flush=True)

    np.savez(out / "vectors.npz", **store)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/vectors.npz (+ meta.json)", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="No-persona baseline activation capture")
    ap.add_argument("--variants", default="none,neutral_sys",
                    type=lambda s: s.split(","))
    ap.add_argument("--probes", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-length", type=int, default=4096)
    return ap.parse_args()


if __name__ == "__main__":
    capture(parse_args())
