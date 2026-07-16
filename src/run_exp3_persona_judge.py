"""Experiment 3 -- persona-FAITHFULNESS judging across the two designs.

Pairs the coupled (results/exp3_persona) and decoupled (results/exp3_decoupled)
datapoints on prompt_id (run 0), then for each prompt scores how faithfully the
reply embodies the persona it was paired with, using judge_prompt_persona:

  * coupled : judge(coupled reply, coupled in-context persona)          -> 1 call
  * cold    : judge(cold reply, v1 standalone persona)                  -> 1 call
              judge(cold reply, v3 standalone persona)                  -> 1 call
              numeric scores AVERAGED across whichever of v1/v3 parsed

One self-describing record per prompt_id:

    results/exp3_faith/<model>/<prompt_id>.json
    results/exp3_faith/<model>/all.jsonl              (aggregated)

Resumable: existing per-prompt files are skipped unless --overwrite.

    python -m src.run_exp3_persona_judge --models llama-3.3-70b --limit 4   # smoke
    python -m src.run_exp3_persona_judge --models llama-3.3-70b             # full
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
from datetime import datetime, timezone

from . import config
from .client import set_concurrency
from .exp3_persona_judge import format_persona_spec, judge_persona_faithfulness

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DESIGN_DIRS = {"coupled": config.EXP3_PERSONA_DIR, "decoupled": config.EXP3_DECOUPLED_DIR}
SCORE_KEYS = ("intensity_fidelity", "expression_fidelity", "persona_gestalt",
              "overall_faithfulness")
META_KEYS = ("prompt_id", "scenario", "role", "eval_condition", "domain",
             "valence", "x_value", "x_rank")


def _load_run0(dirpath):
    out = {}
    for jf in glob.glob(str(dirpath / "*.jsonl")):
        for line in open(jf, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("run", 0) == 0:
                out.setdefault(r["prompt_id"], r)
    return out


def _avg_scores(judges):
    """Average the numeric faithfulness fields across a list of judge dicts."""
    present = [j for j in judges if j]
    if not present:
        return None
    avg = {}
    for k in SCORE_KEYS:
        vals = [j[k] for j in present if j.get(k) is not None]
        avg[k] = round(sum(vals) / len(vals), 3) if vals else None
    nu = [j["n_unlisted"] for j in present if j.get("n_unlisted") is not None]
    avg["n_unlisted"] = round(sum(nu) / len(nu), 3) if nu else None
    avg["n_sources"] = len(present)
    return avg


async def _judge_or_none(user, persona, traits, persona_name, response):
    """Format a spec and judge it; returns (judge_dict|None, n_traits, error)."""
    spec, n = format_persona_spec(persona, traits, persona_name)
    if n == 0 or not (response or "").strip():
        return None, n, ("empty_persona" if n == 0 else "empty_response")
    try:
        j = await judge_persona_faithfulness(user, spec, response)
        return j, n, None
    except Exception as e:  # noqa: BLE001
        return None, n, str(e)


async def judge_pair(coupled, decoup):
    user = coupled.get("user") or decoup.get("user") or ""

    # coupled reply vs its in-context persona
    c_task = _judge_or_none(user, coupled.get("persona") or {}, coupled.get("traits") or [],
                            coupled.get("persona_name", ""), coupled.get("response", ""))
    # cold reply vs v1 and v3 standalone personas
    cold_resp = decoup.get("response", "")
    el = decoup.get("elicit") or {}
    v1, v3 = el.get("v1") or {}, el.get("v3") or {}
    v1_task = _judge_or_none(user, v1.get("persona") or {}, v1.get("traits") or [],
                             v1.get("persona_name", ""), cold_resp)
    v3_task = _judge_or_none(user, v3.get("persona") or {}, v3.get("traits") or [],
                             v3.get("persona_name", ""), cold_resp)

    (cj, cn, cerr), (j1, n1, e1), (j3, n3, e3) = await asyncio.gather(c_task, v1_task, v3_task)

    rec = {k: coupled[k] for k in META_KEYS if k in coupled}
    rec.update(
        model_name=coupled.get("model_name"),
        user=user,
        coupled={"n_traits": cn, "error": cerr, "judge": cj},
        cold={
            "v1": {"n_traits": n1, "error": e1, "judge": j1},
            "v3": {"n_traits": n3, "error": e3, "judge": j3},
            "avg": _avg_scores([j1, j3]),
        },
        judged_at=datetime.now(timezone.utc).isoformat(),
    )
    return rec


async def _run_and_write(model, coupled, decoup):
    pid = coupled["prompt_id"]
    d = config.EXP3_FAITH_DIR / model
    d.mkdir(parents=True, exist_ok=True)
    try:
        rec = await judge_pair(coupled, decoup)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {model}/{pid}: {e}", flush=True)
        return False
    (d / f"{pid}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


async def _gather_with_progress(coros, every=20):
    total = len(coros)
    done = ok = 0

    async def _wrap(c):
        nonlocal done, ok
        r = await c
        done += 1
        ok += 1 if r else 0
        if done == total or done % every == 0:
            print(f"  progress: {done}/{total} done ({ok} ok, {done - ok} failed)", flush=True)
        return r
    return await asyncio.gather(*[_wrap(c) for c in coros])


def aggregate(model):
    d = config.EXP3_FAITH_DIR / model
    if not d.exists():
        return 0
    out = d / "all.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(d.glob("*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"  aggregated {n} records -> {out}", flush=True)
    return n


async def main_async(models, limit, overwrite, concurrency):
    set_concurrency(concurrency)
    decoup_all = {m: _load_run0(config.EXP3_DECOUPLED_DIR / m) for m in models}
    coupled_all = {m: _load_run0(config.EXP3_PERSONA_DIR / m) for m in models}

    tasks, skipped = [], 0
    for m in models:
        shared = sorted(set(coupled_all[m]) & set(decoup_all[m]))
        if limit:
            shared = shared[:limit]
        print(f"{m}: {len(shared)} paired prompts "
              f"(coupled {len(coupled_all[m])}, decoupled {len(decoup_all[m])})", flush=True)
        for pid in shared:
            path = config.EXP3_FAITH_DIR / m / f"{pid}.json"
            if path.exists() and not overwrite:
                skipped += 1
                continue
            tasks.append(_run_and_write(m, coupled_all[m][pid], decoup_all[m][pid]))

    print(f"exp3_faith: dispatching {len(tasks)} pairs x up-to-3 judge calls "
          f"(skipping {skipped}, concurrency={concurrency}) ...", flush=True)
    results = await _gather_with_progress(tasks)
    ok = sum(1 for r in results if r)
    for m in models:
        aggregate(m)
    print(f"exp3_faith done: {ok}/{len(tasks)} succeeded ({len(tasks) - ok} failed).", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama-3.3-70b"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    asyncio.run(main_async(a.models, a.limit, a.overwrite, a.concurrency))
