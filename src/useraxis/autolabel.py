"""Phase 2: unsupervised auto-labeling of the PC poles (API, sonnet 4.5).

For each candidate (readout, layer, PC) we show the judge only the two extreme
DISCOVERY poles (anonymized, no tags, no `lean`) and ask it to NAME the single
dimension that separates them. Guards against confabulation / position bias:

  * blind + order-randomized  -- which physical pole is shown as "Group A" is
    alternated across runs and the within-group order is permuted (seeded, so the
    whole thing is reproducible);
  * stability                 -- R independent runs per axis; a consensus meta-judge
    rates how consistently the runs name the same dimension and emits one consensus
    label per pole;
  * null control              -- ~3 RANDOM (non-PC) splits of the persona pool run
    through the identical prompt; a working method should return low-confidence /
    incoherent labels here.

Reuses the async OpenRouter client (src.client.chat), config.JUDGE_MODEL, and the
robust JSON parser (jsonutil.extract_json_obj) already used by steer.py::judge_one.

Inputs : results/useraxis/<model>/analysis/autolabel/{profiles,candidates}.json
         + poles/<ro>_L<L>_PC<k>.json   (from pole_profiles.py)
Output : results/useraxis/<model>/analysis/autolabel/labels.json

Run:
    python -m src.useraxis.autolabel                       # resp_mean, all candidates
    python -m src.useraxis.autolabel --readout last_user --layers 40 --pcs 1 --no-null
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from .. import config
from ..client import chat, set_concurrency
from .extract import DEFAULT_MODEL, short_name
from .jsonutil import extract_json_obj

RUNS = 5              # independent labeling runs per axis (stability)
NULL_SPLITS = 3       # random (non-PC) control splits per (readout, layer)
RO_OFFSET = {"resp_mean": 0, "last_user": 500000}   # deterministic-seed namespacing

LABEL_SYS = (
    "You are a careful behavioral scientist studying people who use AI assistants. "
    "You will see two groups of anonymized user profiles, Group A and Group B, that "
    "an unsupervised statistical procedure placed at opposite ends of a single latent "
    "dimension. Identify the ONE dimension that best separates them and name each "
    "pole. Reason only from the profile text. Output strict JSON."
)

LABEL_USER_TMPL = """GROUP A:
{group_a}

============================================================
GROUP B:
{group_b}
============================================================

These two groups sit at opposite ends of ONE latent dimension found without any
labels. Identify the single dominant characteristic that separates Group A from
Group B. Return ONLY a JSON object with these keys:
  "dimension": short noun phrase naming the axis (e.g. "emotional vulnerability",
               "technical expertise", "time pressure")
  "group_A_label": at most 6 words describing Group A's end
  "group_B_label": at most 6 words describing Group B's end
  "confidence": integer 0-10 -- how cleanly the two groups separate on this axis
  "evidence": one sentence citing concrete differences you used"""

CONSENSUS_SYS = (
    "You consolidate several independent analyses of the SAME two fixed groups "
    "(call them POS and NEG). Output strict JSON."
)

CONSENSUS_USER_TMPL = """Independent runs each named the dimension separating group
POS from group NEG (the pole identities below are already aligned across runs):

{runs}

Return ONLY a JSON object:
  "consensus_dimension": the single dimension these runs are describing (short noun phrase)
  "pos_label": at most 6 words for the POS end
  "neg_label": at most 6 words for the NEG end
  "consistency": integer 0-10 -- how strongly the runs agree on the SAME underlying dimension"""


def render_group(pids: list[str], profiles: dict[str, str], prefix: str) -> str:
    return "\n\n".join(f"[{prefix}{i + 1}]\n{profiles[pid]}"
                       for i, pid in enumerate(pids))


async def _label_call(group_a: str, group_b: str) -> dict | None:
    msgs = [
        {"role": "system", "content": LABEL_SYS},
        {"role": "user", "content": LABEL_USER_TMPL.format(
            group_a=group_a, group_b=group_b)},
        {"role": "assistant", "content": "{"},
    ]
    raw = await chat(config.JUDGE_MODEL, msgs, temperature=0.5, max_tokens=400)
    try:
        obj = extract_json_obj(raw if raw.lstrip().startswith("{") else "{" + raw)
    except ValueError:
        return None
    try:
        return {"dimension": str(obj["dimension"]),
                "group_A_label": str(obj["group_A_label"]),
                "group_B_label": str(obj["group_B_label"]),
                "confidence": float(obj["confidence"]),
                "evidence": str(obj.get("evidence", ""))}
    except (KeyError, TypeError, ValueError):
        return None


async def label_axis(rec: dict, profiles: dict, seed0: int) -> list[dict]:
    """R blind, order-randomized labeling runs of one (readout, layer, PC) axis."""
    pos = [d["pid"] for d in rec["pos_pole"]]
    neg = [d["pid"] for d in rec["neg_pole"]]

    async def one(run_idx: int) -> dict | None:
        rng = np.random.default_rng(seed0 + run_idx)
        p = [pos[i] for i in rng.permutation(len(pos))]
        n = [neg[i] for i in rng.permutation(len(neg))]
        a_is_pos = (run_idx % 2 == 0)          # 3 pos-as-A / 2 pos-as-B over 5 runs
        a_pids, b_pids = (p, n) if a_is_pos else (n, p)
        obj = await _label_call(render_group(a_pids, profiles, "A"),
                                render_group(b_pids, profiles, "B"))
        if obj is None:
            return None
        # map A/B answers back onto the fixed POS/NEG poles
        pos_lab = obj["group_A_label"] if a_is_pos else obj["group_B_label"]
        neg_lab = obj["group_B_label"] if a_is_pos else obj["group_A_label"]
        return {"run": run_idx, "a_is_pos": a_is_pos,
                "dimension": obj["dimension"], "pos_label": pos_lab,
                "neg_label": neg_lab, "confidence": obj["confidence"],
                "evidence": obj["evidence"]}

    runs = await asyncio.gather(*(one(r) for r in range(RUNS)))
    return [r for r in runs if r is not None]


async def consensus(runs: list[dict]) -> dict | None:
    if not runs:
        return None
    lines = "\n".join(
        f'- dimension="{r["dimension"]}" | POS="{r["pos_label"]}" | '
        f'NEG="{r["neg_label"]}" | confidence={r["confidence"]:.0f}'
        for r in runs)
    msgs = [
        {"role": "system", "content": CONSENSUS_SYS},
        {"role": "user", "content": CONSENSUS_USER_TMPL.format(runs=lines)},
        {"role": "assistant", "content": "{"},
    ]
    raw = await chat(config.JUDGE_MODEL, msgs, temperature=0.0, max_tokens=300)
    try:
        obj = extract_json_obj(raw if raw.lstrip().startswith("{") else "{" + raw)
        return {"consensus_dimension": str(obj["consensus_dimension"]),
                "pos_label": str(obj["pos_label"]),
                "neg_label": str(obj["neg_label"]),
                "consistency": float(obj["consistency"])}
    except (ValueError, KeyError, TypeError):
        return None


async def null_axis(all_pids: list[str], profiles: dict, k: int,
                    seed0: int) -> list[dict]:
    """NULL_SPLITS random (non-PC) splits through the identical labeling prompt."""
    async def one(j: int) -> dict | None:
        rng = np.random.default_rng(seed0 + 7919 * (j + 1))
        pick = rng.choice(len(all_pids), size=2 * k, replace=False)
        a = [all_pids[i] for i in pick[:k]]
        b = [all_pids[i] for i in pick[k:]]
        obj = await _label_call(render_group(a, profiles, "A"),
                                render_group(b, profiles, "B"))
        if obj is None:
            return None
        return {"split": j, "dimension": obj["dimension"],
                "confidence": obj["confidence"], "evidence": obj["evidence"]}

    res = await asyncio.gather(*(one(j) for j in range(NULL_SPLITS)))
    return [r for r in res if r is not None]


async def run(args) -> None:
    model = short_name(DEFAULT_MODEL)
    out = config.RESULTS_DIR / "useraxis" / model / "analysis" / "autolabel"
    profiles = json.loads((out / "profiles.json").read_text())
    cand = json.loads((out / "candidates.json").read_text())
    all_pids = list(profiles)
    k = cand["pole_k"]

    layers = args.layers or cand["candidate_layers"]
    pcs = args.pcs or cand["pcs"]
    set_concurrency(args.concurrency)

    # existing labels.json is updated in place (so last_user winner runs can be
    # appended without clobbering the resp_mean pass)
    path = out / "labels.json"
    store = json.loads(path.read_text()) if path.exists() else {}
    store.setdefault("judge_model", config.JUDGE_MODEL)
    store.setdefault("runs_per_axis", RUNS)
    store.setdefault("axes", {})
    store.setdefault("null", {})

    async def label_one(key: str, rec: dict, seed0: int) -> None:
        runs = await label_axis(rec, profiles, seed0)
        cons = await consensus(runs)
        store["axes"][key] = {
            "readout": rec["readout"], "layer": rec["layer"], "pc": rec["pc"],
            "orient_sign": rec["orient_sign"], "sep": rec["sep"],
            "n_runs": len(runs), "runs": runs, "consensus": cons,
            "mean_confidence": float(np.mean([r["confidence"] for r in runs]))
            if runs else None,
        }
        c = cons or {}
        print(f"[{key}] dim='{c.get('consensus_dimension','?')}' "
              f"POS='{c.get('pos_label','?')}' NEG='{c.get('neg_label','?')}' "
              f"conf={store['axes'][key]['mean_confidence']} "
              f"consist={c.get('consistency','?')}")

    for L in layers:
        for k_pc in pcs:
            key = f"{args.readout}_L{L}_PC{k_pc}"
            rec_path = out / "poles" / f"{key}.json"
            if not rec_path.exists():
                print(f"  skip {rec_path.name} (missing)")
                continue
            await label_one(key, json.loads(rec_path.read_text()),
                            RO_OFFSET[args.readout] + L * 100 + k_pc * 10)

        if not args.no_null:
            nk = f"{args.readout}_L{L}"
            store["null"][nk] = await null_axis(all_pids, profiles, k,
                                                RO_OFFSET[args.readout] + L * 100 + 999)
            confs = [r["confidence"] for r in store["null"][nk]]
            print(f"[null {nk}] confidences={confs} "
                  f"dims={[r['dimension'] for r in store['null'][nk]]}")

    # random-DIRECTION control axes (labeled through the identical pipeline; their
    # labels are validated on held-out in Phase 3 and should fail to generalize)
    if args.null_dirs and args.readout == "resp_mean":
        for j, key in enumerate(cand.get("null_axes", [])):
            rec_path = out / "poles" / f"{key}.json"
            if rec_path.exists():
                await label_one(key, json.loads(rec_path.read_text()), 900000 + j)

    path.write_text(json.dumps(store, indent=2, ensure_ascii=False))
    print(f"wrote {path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Unsupervised PC pole auto-labeling")
    ap.add_argument("--readout", default="resp_mean",
                    choices=("resp_mean", "last_user"))
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="subset of candidate layers (default: all)")
    ap.add_argument("--pcs", type=int, nargs="*", default=None,
                    help="subset of PCs (default: all)")
    ap.add_argument("--no-null", action="store_true",
                    help="skip random-SPLIT null controls")
    ap.add_argument("--null-dirs", action="store_true",
                    help="also label the random-DIRECTION control axes")
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENCY)
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
