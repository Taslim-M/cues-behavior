"""Phase B driver -- single-layer steering sweep across candidate layers.

Loads the 70B once, generates the layer-independent alpha=0 baseline once, then for
each candidate layer runs the full alpha sweep (reusing the baseline + the loaded
model) and judges it. Per-layer results land in steering_L{layer}.json.

Candidate layers come from analysis/candidate_layers.json (Phase A). The point is to
compare the max-variance layer (L3, surface) against the interpretability-peak layer
(L24) and a spread across depth, to see where steering is actually effective.

Env (see runpod-env-pitfalls):
    HF_HOME=/dev/shm/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -m src.useraxis.steer_layers --alphas="-8,-4,-2,0,2,4,8"

Flags:
    --layers 3,16,24,32,40,48,56   override candidate list (default: from Phase A)
    --skip L40                      skip a tag (e.g. reuse the shipped L40 sweep)
    --judge-only                    re-judge existing per-layer response files
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from .. import config
from .extract import DEFAULT_MODEL, load_model, short_name
from .steer import generate_steered, judge_all, build_prompt_set, out_paths


def load_candidates(res_dir: Path) -> list[int]:
    p = res_dir / "analysis" / "candidate_layers.json"
    if p.exists():
        return json.loads(p.read_text())["layers"]
    return [3, 16, 24, 32, 40, 48, 56]


def make_args(layer: int, a) -> SimpleNamespace:
    return SimpleNamespace(
        model=a.model, axis=a.axis, layer=layer, alphas=a.alphas,
        n_neutral=a.n_neutral, batch_size=a.batch_size, seed=a.seed,
        tag=f"L{layer}")


def main() -> None:
    a = parse_args()
    res_dir = config.RESULTS_DIR / "useraxis" / short_name(
        {"llama-3.3-70b": DEFAULT_MODEL}.get(a.model, a.model))
    layers = ([int(x) for x in a.layers.split(",")] if a.layers
              else load_candidates(res_dir))
    skip = set(a.skip.split(",")) if a.skip else set()
    layers = [l for l in layers if f"L{l}" not in skip]
    print(f"Phase B: layers={layers} alphas={a.alphas} "
          f"n_neutral={a.n_neutral} skip={sorted(skip) or '-'}", flush=True)

    if not a.judge_only:
        # baseline alpha=0 once (layer-independent); generated at layer 0's axis but
        # the alpha=0 branch never touches the axis, so any layer works.
        pm = load_model({"llama-3.3-70b": DEFAULT_MODEL}.get(a.model, a.model))
        base_args = make_args(layers[0], a)
        base_args.alphas = "0"
        base_rows = generate_steered(res_dir, base_args, pm=pm)
        base_rows = [{k: r[k] for k in ("prompt_id", "kind", "prompt", "response")}
                     for r in base_rows]
        for l in layers:
            generate_steered(res_dir, make_args(l, a), pm=pm,
                             baseline_rows=base_rows)

    if not a.skip_judge:
        for l in layers:
            if out_paths(res_dir, f"L{l}")["responses"].exists():
                print(f"== judging L{l} ==", flush=True)
                asyncio.run(judge_all(res_dir, a.concurrency, f"L{l}"))


def parse_args():
    ap = argparse.ArgumentParser(description="Multi-layer User-Axis steering sweep")
    ap.add_argument("--model", default="llama-3.3-70b")
    ap.add_argument("--axis", choices=("pc1", "contrast"), default="pc1")
    ap.add_argument("--layers", default="", help="comma list; default = Phase A picks")
    ap.add_argument("--skip", default="", help="comma tags to skip, e.g. L40")
    ap.add_argument("--alphas", default="-8,-4,-2,0,2,4,8")
    ap.add_argument("--n-neutral", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--judge-only", action="store_true")
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENCY)
    return ap.parse_args()


if __name__ == "__main__":
    main()
