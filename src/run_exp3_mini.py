"""Experiment 3 -- MINI-MARKER design: closed-vocabulary persona + judged behaviour.

A repeat of the Experiment-3 design comparison, but the persona is described with
the CLOSED 40-word Saucier (1994) Mini-Marker vocabulary (+ an AB5C facet blend
per trait and an overall 5-factor profile) -- see infer_prompts.py's
infer_persona_prompt_mini_marker_v1. Two designs, both on the SAME balanced
432-prompt subsample as every other Experiment-3 run:

  COUPLED  (results/exp3_mini_coupled)
      one generation: the model writes its <persona> (mini-marker traits) and
      then, conditioned on it, its <response>. "paired traits" + in-context reply.

  SOLO     (results/exp3_mini_solo)
      TWO independent calls: (a) the mini-marker persona elicited ALONE (v1
      framing, stop after </persona>) -> "solo traits"; (b) the natural reply with
      NO self-modeling scaffold -> "cold response". Only ONE framing is used (no
      v1/v3 averaging).

Both designs judge the behavioural reply with anthropic/claude-sonnet-4.5 for the
identical DVs (warmth / formality / advice_density / primary_emotion) so coupled
vs cold behaviour is directly comparable. Persona FAITHFULNESS is scored
separately (run_exp3_mini_judge.py).

One self-describing record per (model, eval_condition, prompt, run):

    results/exp3_mini_<design>/<model_name>/<eval_condition>/<prompt_id>__run<k>.json
    results/exp3_mini_<design>/<model_name>/<eval_condition>.jsonl        (aggregated)

Resumable: existing datapoint files are skipped unless --overwrite.

Usage:
    python -m src.run_exp3_mini --design coupled --models llama-3.3-70b --limit 4  # smoke
    python -m src.run_exp3_mini --design solo    --models llama-3.3-70b            # full
    python -m src.run_exp3_mini --design both     --models qwen3-30b               # both
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone

from . import config
from .client import chat, set_concurrency
from .exp2_judge import judge_response
from .inference import parse_persona_mini
from .run_exp3_decoupled import compose_response
from .run_exp3_persona import META_KEYS, load_all_prompts, select_balanced
from .system_prompts import PERSONA_MINI_MARKER_COUPLED, PERSONA_MINI_MARKER_SOLO

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DESIGN_DIR = {"coupled": config.EXP3_MINI_COUPLED_DIR, "solo": config.EXP3_MINI_SOLO_DIR}
SYS_PROMPT_NAME = {"coupled": "persona_mini_coupled", "solo": "persona_mini_solo"}
# room for a mini-marker persona block AND the reply in one coupled generation
MAX_TOKENS_COUPLED = 1500


def _prepend_eval(record, sys_prompt):
    eval_text = record.get("system")
    if eval_text:
        return eval_text.rstrip() + "\n\n" + sys_prompt
    return sys_prompt


def compose_coupled(record):
    return [
        {"role": "system", "content": _prepend_eval(record, PERSONA_MINI_MARKER_COUPLED)},
        {"role": "user", "content": record["user"]},
    ]


def compose_solo_persona(record):
    return [
        {"role": "system", "content": _prepend_eval(record, PERSONA_MINI_MARKER_SOLO)},
        {"role": "user", "content": record["user"]},
    ]


def _datapoint_path(design, model_name, eval_condition, prompt_id, run):
    d = DESIGN_DIR[design] / model_name / eval_condition
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{prompt_id}__run{run}.json"


async def _judge(user, response, do_judge):
    if not (do_judge and (response or "").strip()):
        return None, None
    try:
        return await judge_response(user, response), None
    except Exception as e:  # noqa: BLE001 - judging never sinks the datapoint
        return None, str(e)


async def run_one_coupled(model_name, model_id, prompt, run, temperature, do_judge):
    messages = compose_coupled(prompt)
    raw_output = await chat(model_id, messages, temperature, MAX_TOKENS_COUPLED)
    parsed = parse_persona_mini(raw_output)
    parsed["response"] = re.sub(r"</?response>", "", parsed["response"], flags=re.I).strip()
    judge, judge_error = await _judge(prompt["user"], parsed["response"], do_judge)

    rec = {k: prompt[k] for k in META_KEYS if k in prompt}
    rec.update(
        model_name=model_name, model_id=model_id, design="coupled",
        sys_prompt_name=SYS_PROMPT_NAME["coupled"], run=run, user=prompt["user"],
        messages=messages, raw_output=raw_output, raw_persona=parsed["raw_persona"],
        persona=parsed["persona"], persona_name=parsed["persona_name"],
        traits=parsed["traits"], factor_profile=parsed["factor_profile"],
        response=parsed["response"], judge=judge, judge_error=judge_error,
        meta=dict(temperature=temperature, max_tokens=MAX_TOKENS_COUPLED,
                  timestamp=datetime.now(timezone.utc).isoformat()),
    )
    return rec


async def run_one_solo(model_name, model_id, prompt, run, temperature, max_tokens, do_judge):
    m_persona = compose_solo_persona(prompt)
    m_resp = compose_response(prompt)  # eval frame only, no self-modeling scaffold
    raw_persona, raw_resp = await asyncio.gather(
        chat(model_id, m_persona, temperature, max_tokens),
        chat(model_id, m_resp, temperature, config.MAX_TOKENS_CUES),
    )
    parsed = parse_persona_mini(raw_persona)
    response = re.sub(r"</?(persona|response)>", "", raw_resp, flags=re.I).strip()
    judge, judge_error = await _judge(prompt["user"], response, do_judge)

    rec = {k: prompt[k] for k in META_KEYS if k in prompt}
    rec.update(
        model_name=model_name, model_id=model_id, design="solo",
        sys_prompt_name=SYS_PROMPT_NAME["solo"], run=run, user=prompt["user"],
        elicit_messages=m_persona, raw_persona=parsed["raw_persona"],
        persona=parsed["persona"], persona_name=parsed["persona_name"],
        traits=parsed["traits"], factor_profile=parsed["factor_profile"],
        response_messages=m_resp, raw_response=raw_resp, response=response,
        judge=judge, judge_error=judge_error,
        meta=dict(temperature=temperature, max_tokens_persona=max_tokens,
                  max_tokens_response=config.MAX_TOKENS_CUES,
                  timestamp=datetime.now(timezone.utc).isoformat()),
    )
    return rec


async def _run_and_write(design, model_name, model_id, prompt, run, temperature,
                         max_tokens, do_judge):
    label = f"{design}/{model_name}/{prompt['prompt_id']}/run{run}"
    try:
        if design == "coupled":
            rec = await run_one_coupled(model_name, model_id, prompt, run, temperature, do_judge)
        else:
            rec = await run_one_solo(model_name, model_id, prompt, run, temperature,
                                     max_tokens, do_judge)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label}: {e}", flush=True)
        return False
    path = _datapoint_path(design, model_name, prompt["eval_condition"],
                           prompt["prompt_id"], run)
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


async def _gather_with_progress(coros, tag, every=20):
    total = len(coros)
    done = ok = 0

    async def _wrap(c):
        nonlocal done, ok
        r = await c
        done += 1
        ok += 1 if r else 0
        if done == total or done % every == 0:
            print(f"  [{tag}] progress: {done}/{total} done ({ok} ok, {done - ok} failed)",
                  flush=True)
        return r

    return await asyncio.gather(*[_wrap(c) for c in coros])


def aggregate(design, model_name, eval_condition):
    d = DESIGN_DIR[design] / model_name / eval_condition
    if not d.exists():
        return
    out = DESIGN_DIR[design] / model_name / f"{eval_condition}.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(d.glob("*.json")):
            f.write(json.dumps(json.loads(p.read_text(encoding="utf-8")),
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"  aggregated {n} records -> {out}", flush=True)


async def run_design(design, models, runs, temperature, max_tokens, limit, overwrite,
                     do_judge, prompts):
    tasks, skipped = [], 0
    for model_name in models:
        model_id = config.MODELS[model_name]
        for p in prompts:
            for run in range(runs):
                path = _datapoint_path(design, model_name, p["eval_condition"],
                                       p["prompt_id"], run)
                if path.exists() and not overwrite:
                    skipped += 1
                    continue
                tasks.append(_run_and_write(design, model_name, model_id, p, run,
                                            temperature, max_tokens, do_judge))
    print(f"exp3_mini[{design}]: models={models} runs={runs} judge={do_judge}; "
          f"dispatching {len(tasks)} datapoints (skipping {skipped}) ...", flush=True)
    results = await _gather_with_progress(tasks, design)
    ok = sum(1 for r in results if r)
    for model_name in models:
        for cond in sorted({p["eval_condition"] for p in prompts}):
            aggregate(design, model_name, cond)
    print(f"exp3_mini[{design}] done: {ok}/{len(tasks)} succeeded "
          f"({len(tasks) - ok} failed).", flush=True)


async def main_async(designs, models, runs, temperature, max_tokens, limit, overwrite,
                     concurrency, do_judge):
    for m in models:
        if m not in config.MODELS:
            raise SystemExit(f"unknown model '{m}'; known: {list(config.MODELS)}")
    set_concurrency(concurrency)
    prompts = select_balanced(load_all_prompts())
    if limit:
        prompts = prompts[:limit]
    print(f"exp3_mini: subsample={len(prompts)} prompts "
          f"({len({p['scenario'] for p in prompts})} scenarios, "
          f"{len({p['role'] for p in prompts})} roles, "
          f"{len({p['eval_condition'] for p in prompts})} eval_conditions)", flush=True)
    for design in designs:
        await run_design(design, models, runs, temperature, max_tokens, limit,
                         overwrite, do_judge, prompts)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["coupled", "solo", "both"], default="both")
    ap.add_argument("--models", nargs="+",
                    default=["llama-3.3-70b", "llama-3.1-8b", "qwen3-235b", "qwen3-30b"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int,
                    default=config.MAX_TOKENS_INFERENCE, help="persona elicitation cap (solo)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENCY)
    ap.add_argument("--no-judge", dest="do_judge", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    designs = ["coupled", "solo"] if a.design == "both" else [a.design]
    asyncio.run(main_async(designs, a.models, a.runs, a.temperature, a.max_tokens,
                           a.limit, a.overwrite, a.concurrency, a.do_judge))
