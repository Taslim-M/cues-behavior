"""Phase 3: validate each induced label -- the scientific crux (API + CPU).

Auto-labeling (Phase 2) can confabulate: emit a fluent axis name that does not
generalize. This phase tests every consensus label on personas the label was NOT
built from.

  * PREDICTIVE validity -- a FRESH judge, given ONLY the induced dimension + the two
    pole labels (never the discovery personas), scores each held-out persona on that
    bipolar axis (0-100). We then check, on personas held out from the labeling:
       - pole-classification accuracy on the clearly-separated held-out (|z|>0.5),
       - Spearman(judge score, PC projection) over all held-out.
    A label built on 24 extreme personas that still orders the other 126 by their PC
    projection is a real, generalizing construct; near-chance means confabulation.
  * NULL baseline -- a permutation test: shuffle the held-out projections many times
    and recompute Spearman, giving the null band the real rho must beat (chance
    accuracy = 0.5). Complements the random-split *labeling* control in Phase 2.
  * CONVERGENT validity -- Spearman(judge score, each authored tag) + BH-FDR, tying
    the unsupervised discovery back to the report's supervised story (does auto-PC1
    land on `vulnerability`? what, if anything, does auto-PC2 track?).

Processes every axis present in labels.json (so a later last_user pass is picked up
automatically). Reuses src.client.chat / config.JUDGE_MODEL / jsonutil.

Run:
    python -m src.useraxis.validate_labels
    python -m src.useraxis.validate_labels --axes resp_mean_L40_PC1
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .. import config
from ..client import chat, set_concurrency
from .extract import DEFAULT_MODEL, short_name
from .jsonutil import extract_json_obj
from .compute_axis import TAG_SCALES, bh_fdr

BATCH = 10            # held-out profiles scored per judge call
N_PERM = 2000         # permutations for the rho null
Z_CLEAR = 0.5         # |z-projection| threshold for the accuracy subset

SCORE_SYS = (
    "You place AI-assistant users on a defined bipolar dimension. You see ONLY the "
    "dimension definition and the profiles to score -- not any reference examples. "
    "Output strict JSON."
)

SCORE_USER_TMPL = """DIMENSION: {dimension}
  score 100 = {pos_label}
  score 0   = {neg_label}

Score each profile from 0 to 100 for where that person falls on this dimension
(100 = fully the "{pos_label}" end, 0 = fully the "{neg_label}" end, 50 = neutral).

PROFILES:
{profiles}

Return ONLY a JSON object mapping each profile id to an integer 0-100, e.g.
{{"P1": 73, "P2": 8, ...}}. Include every id."""


async def score_batch(dimension: str, pos_label: str, neg_label: str,
                      items: list[tuple[str, str]]) -> dict[str, float]:
    """items = [(id, profile_text)]; returns {id: score 0-100} for parsed ids."""
    block = "\n\n".join(f"[{pid}]\n{txt}" for pid, txt in items)
    msgs = [
        {"role": "system", "content": SCORE_SYS},
        {"role": "user", "content": SCORE_USER_TMPL.format(
            dimension=dimension, pos_label=pos_label, neg_label=neg_label,
            profiles=block)},
        {"role": "assistant", "content": "{"},
    ]
    raw = await chat(config.JUDGE_MODEL, msgs, temperature=0.0, max_tokens=400)
    try:
        obj = extract_json_obj(raw if raw.lstrip().startswith("{") else "{" + raw)
    except ValueError:
        return {}
    out = {}
    for pid, _ in items:
        try:
            out[pid] = max(0.0, min(100.0, float(obj[pid])))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def perm_null(scores: np.ndarray, proj: np.ndarray) -> dict:
    """Permutation test for Spearman(scores, proj): shuffle proj, recompute."""
    rho, _ = stats.spearmanr(scores, proj)
    rng = np.random.default_rng(0)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i], _ = stats.spearmanr(scores, rng.permutation(proj))
    p = float((np.abs(null) >= abs(rho)).mean())
    return {"rho": float(rho), "perm_p": p,
            "null_rho_mean": float(null.mean()),
            "null_rho_abs_p95": float(np.quantile(np.abs(null), 0.95))}


async def validate_axis(rec: dict, profiles: dict, tags: dict) -> dict | None:
    cons = rec.get("consensus")
    if not cons:
        return None
    heldout = json.loads((rec["_poles_path"]).read_text())["heldout"]
    items = [(h["pid"], profiles[h["pid"]]) for h in heldout]
    # batch the held-out scoring
    batches = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]
    results = await asyncio.gather(*(
        score_batch(cons["consensus_dimension"], cons["pos_label"],
                    cons["neg_label"], b) for b in batches))
    scored: dict[str, float] = {}
    for d in results:
        scored.update(d)

    rows = [(h, scored[h["pid"]]) for h in heldout if h["pid"] in scored]
    if len(rows) < 20:
        return {"error": f"only {len(rows)} held-out scored", "n_scored": len(rows)}
    proj = np.array([h["proj"] for h, _ in rows], float)
    z = np.array([h["z"] for h, _ in rows], float)
    sc = np.array([s for _, s in rows], float)

    # predictive: pole accuracy on the clearly-separated subset
    clear = np.abs(z) > Z_CLEAR
    pred_pos = sc[clear] > 50.0
    true_pos = proj[clear] > 0.0
    acc = float((pred_pos == true_pos).mean()) if clear.sum() else float("nan")

    null = perm_null(sc, proj)

    # convergent: judge score vs each authored tag (Spearman + BH-FDR)
    conv, ps = {}, []
    for t in TAG_SCALES:
        y = np.array([tags[h["pid"]][t] for h, _ in rows], float)
        r, p = stats.spearmanr(sc, y)
        conv[t] = {"spearman": float(r), "p": float(p)}
        ps.append(p)
    for t, q in zip(TAG_SCALES, bh_fdr(ps)):
        conv[t]["q_FDR"] = float(q)
    top_tag = max(TAG_SCALES, key=lambda t: abs(conv[t]["spearman"]))

    return {
        "n_scored": len(rows), "n_clear": int(clear.sum()),
        "pole_accuracy": acc, "spearman_score_vs_proj": null["rho"],
        "perm_p": null["perm_p"], "null_abs_rho_p95": null["null_rho_abs_p95"],
        "convergent": conv,
        "top_convergent_tag": top_tag,
        "top_convergent_rho": conv[top_tag]["spearman"],
        "top_convergent_q": conv[top_tag]["q_FDR"],
    }


async def run(args) -> None:
    model = short_name(DEFAULT_MODEL)
    out = config.RESULTS_DIR / "useraxis" / model / "analysis" / "autolabel"
    profiles = json.loads((out / "profiles.json").read_text())
    labels = json.loads((out / "labels.json").read_text())
    idx = json.loads(
        (config.RESULTS_DIR / "useraxis" / model / "persona_index.json").read_text())
    tags = idx["tags"]

    set_concurrency(args.concurrency)
    keys = args.axes or list(labels["axes"])
    vpath = out / "validation.json"
    store = json.loads(vpath.read_text()) if vpath.exists() else {}
    store.setdefault("judge_model", config.JUDGE_MODEL)
    store.setdefault("z_clear", Z_CLEAR)
    store.setdefault("chance_accuracy", 0.5)
    store.setdefault("axes", {})

    for key in keys:
        rec = dict(labels["axes"][key])
        # the pole-file stem IS the axis key (real: resp_mean_L40_PC1;
        # null-direction control: resp_mean_L40_rand0)
        rec["_poles_path"] = out / "poles" / f"{key}.json"
        res = await validate_axis(rec, profiles, tags)
        if res is None:
            print(f"[{key}] no consensus label -- skipped")
            continue
        store["axes"][key] = {"readout": rec["readout"], "layer": rec["layer"],
                              "pc": rec["pc"],
                              "dimension": rec["consensus"]["consensus_dimension"],
                              **res}
        s = store["axes"][key]
        print(f"[{key}] '{s['dimension']}': acc={s.get('pole_accuracy'):.2f} "
              f"rho={s.get('spearman_score_vs_proj'):+.2f} (p={s.get('perm_p'):.3f}) "
              f"-> tag {s.get('top_convergent_tag')} "
              f"rho={s.get('top_convergent_rho'):+.2f} q={s.get('top_convergent_q'):.3f}")

    vpath.write_text(json.dumps(store, indent=2, ensure_ascii=False))
    print(f"wrote {vpath}")


def parse_args():
    ap = argparse.ArgumentParser(description="Validate induced PC labels (held-out)")
    ap.add_argument("--axes", nargs="*", default=None,
                    help="axis keys from labels.json (default: all)")
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENCY)
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
