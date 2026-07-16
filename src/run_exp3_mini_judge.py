"""Experiment 3 -- MINI-MARKER persona-FAITHFULNESS judging (two judges, averaged).

Pairs the coupled (results/exp3_mini_coupled) and solo (results/exp3_mini_solo)
datapoints on prompt_id (run 0). For each prompt it scores how faithfully the
reply embodies the Mini-Marker persona it was paired with, using
judge_faithfulness_prompt_mini_marker run through BOTH config.MINI_JUDGE_MODELS
(anthropic/claude-sonnet-5 + openai/gpt-5.6-luna); the numeric scores are AVERAGED
across the two judges:

  * coupled : judge(coupled reply, coupled in-context mini persona)   -> 2 calls avg
  * cold    : judge(cold reply,   solo mini persona)                  -> 2 calls avg

One self-describing record per prompt_id:

    results/exp3_mini_faith/<model>/<prompt_id>.json
    results/exp3_mini_faith/<model>/all.jsonl              (aggregated)

Resumable: existing per-prompt files are skipped unless --overwrite.

    python -m src.run_exp3_mini_judge --models llama-3.3-70b --limit 4   # smoke
    python -m src.run_exp3_mini_judge --models llama-3.3-70b             # full
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
from .exp3_mini_judge import SCORE_KEYS, format_mini_profile, judge_mini_faithfulness

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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


def _avg_across_judges(judges):
    """Average the numeric fields across the (successful) judge dicts."""
    present = [j for j in judges if j]
    if not present:
        return None
    avg = {}
    for k in SCORE_KEYS + ("n_inversions", "n_leakage"):
        vals = [j[k] for j in present if j.get(k) is not None]
        avg[k] = round(sum(vals) / len(vals), 3) if vals else None
    avg["n_judges"] = len(present)
    avg["judge_models"] = [j.get("judge_model") for j in present]
    return avg


async def _judge_all_models(user, profile, n_traits, response):
    """Run every MINI_JUDGE_MODEL on one (profile, response); return per-model dicts
    keyed by model id, plus (error_or_None)."""
    if n_traits == 0 or not (response or "").strip():
        return {}, ("empty_persona" if n_traits == 0 else "empty_response")

    async def _one(model):
        try:
            return model, await judge_mini_faithfulness(user, profile, response, model), None
        except Exception as e:  # noqa: BLE001
            return model, None, str(e)

    results = await asyncio.gather(*[_one(m) for m in config.MINI_JUDGE_MODELS])
    by_model = {m: {"judge": j, "error": err} for m, j, err in results}
    return by_model, None


async def judge_pair(coupled, solo):
    user = coupled.get("user") or solo.get("user") or ""

    c_profile, c_n = format_mini_profile(
        coupled.get("persona") or {}, coupled.get("traits") or [],
        coupled.get("persona_name", ""), coupled.get("factor_profile") or {})
    s_profile, s_n = format_mini_profile(
        solo.get("persona") or {}, solo.get("traits") or [],
        solo.get("persona_name", ""), solo.get("factor_profile") or {})

    (c_models, c_err), (s_models, s_err) = await asyncio.gather(
        _judge_all_models(user, c_profile, c_n, coupled.get("response", "")),
        _judge_all_models(user, s_profile, s_n, solo.get("response", "")),
    )

    rec = {k: coupled[k] for k in META_KEYS if k in coupled}
    rec.update(
        model_name=coupled.get("model_name"),
        user=user,
        coupled={"n_traits": c_n, "error": c_err, "judges": c_models,
                 "avg": _avg_across_judges([v["judge"] for v in c_models.values()])},
        cold={"n_traits": s_n, "error": s_err, "judges": s_models,
              "avg": _avg_across_judges([v["judge"] for v in s_models.values()])},
        judged_at=datetime.now(timezone.utc).isoformat(),
    )
    return rec


async def _run_and_write(model, coupled, solo):
    pid = coupled["prompt_id"]
    d = config.EXP3_MINI_FAITH_DIR / model
    d.mkdir(parents=True, exist_ok=True)
    try:
        rec = await judge_pair(coupled, solo)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {model}/{pid}: {e}", flush=True)
        return False
    (d / f"{pid}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
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
    d = config.EXP3_MINI_FAITH_DIR / model
    if not d.exists():
        return 0
    out = d / "all.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(d.glob("*.json")):
            f.write(json.dumps(json.loads(p.read_text(encoding="utf-8")),
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"  aggregated {n} records -> {out}", flush=True)
    return n


async def main_async(models, limit, overwrite, concurrency):
    set_concurrency(concurrency)
    coupled_all = {m: _load_run0(config.EXP3_MINI_COUPLED_DIR / m) for m in models}
    solo_all = {m: _load_run0(config.EXP3_MINI_SOLO_DIR / m) for m in models}

    tasks, skipped = [], 0
    for m in models:
        shared = sorted(set(coupled_all[m]) & set(solo_all[m]))
        if limit:
            shared = shared[:limit]
        print(f"{m}: {len(shared)} paired prompts "
              f"(coupled {len(coupled_all[m])}, solo {len(solo_all[m])})", flush=True)
        for pid in shared:
            path = config.EXP3_MINI_FAITH_DIR / m / f"{pid}.json"
            if path.exists() and not overwrite:
                skipped += 1
                continue
            tasks.append(_run_and_write(m, coupled_all[m][pid], solo_all[m][pid]))

    print(f"exp3_mini_faith: dispatching {len(tasks)} pairs x {2 * len(config.MINI_JUDGE_MODELS)} "
          f"judge calls (skipping {skipped}, concurrency={concurrency}) ...", flush=True)
    results = await _gather_with_progress(tasks)
    ok = sum(1 for r in results if r)
    for m in models:
        aggregate(m)
    print(f"exp3_mini_faith done: {ok}/{len(tasks)} succeeded ({len(tasks) - ok} failed).",
          flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["llama-3.3-70b", "llama-3.1-8b", "qwen3-235b", "qwen3-30b"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    asyncio.run(main_async(a.models, a.limit, a.overwrite, a.concurrency))
