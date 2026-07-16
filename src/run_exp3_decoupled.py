"""Experiment 3 -- DECOUPLED design: verbalized persona and behaviour measured apart.

The original Experiment 3 elicited the persona and the reply in a SINGLE
generation (the model wrote its <persona> block and then, conditioned on it, its
<response>). That couples the two: writing the persona first can steer -- or
merely rationalise -- the reply, so a persona->behaviour correlation is hard to
read causally.

This runner decouples them. For each prompt we make THREE independent calls:

  * elicit_v1  -- persona-ONLY prompt, "staging" framing (persona_v1 system)
  * elicit_v3  -- persona-ONLY prompt, "self-report" framing (persona_v3 system)
  * response   -- the model's natural reply with NO self-modeling scaffold
                  (only the eval_condition frame, if any, is kept so the
                  role x eval_condition x scenario factorial is preserved)

The two persona elicitations articulate the same thing under different wording;
their per-trait 0-10 values are AVERAGED (merge_traits) to damp per-prompt
wording variance. The natural response is judged by anthropic/claude-sonnet-4.5
for the identical DVs (warmth / formality / advice_density / primary_emotion), so
a datapoint pairs an *uncontaminated* verbalized self-model with an
*independently generated* behaviour.

One self-describing record per (model, eval_condition, prompt, run):

    results/exp3_decoupled/<model_name>/<eval_condition>/<prompt_id>__run<k>.json
    results/exp3_decoupled/<model_name>/<eval_condition>.jsonl          (aggregated)

Resumable: existing datapoint files are skipped unless --overwrite.

Usage:
    python -m src.run_exp3_decoupled --models llama-3.3-70b --limit 4   # smoke
    python -m src.run_exp3_decoupled --models llama-3.3-70b             # full (1 run)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

from . import config
from .client import chat, set_concurrency
from .exp2_judge import judge_response
from .inference import parse_persona
from .run_exp3_persona import META_KEYS, load_all_prompts, select_balanced
from .system_prompts import PERSONA_SYSTEM_PROMPT_V1, PERSONA_SYSTEM_PROMPT_V3

if hasattr(sys.stdout, "reconfigure"):      # Windows cp1252 -> keep unicode safe
    sys.stdout.reconfigure(encoding="utf-8")

SYS_PROMPT_NAME = "persona_decoupled"


# --------------------------------------------------------------------------- #
# Message composition
# --------------------------------------------------------------------------- #
def compose_elicit(record, persona_sys):
    """Persona-only elicitation: the persona prompt is the system prompt; the
    eval_condition text (None for deployment) is prepended so the eval frame stays
    a clean factor -- identical framing to the coupled design, minus the response."""
    sys_text = persona_sys
    eval_text = record.get("system")
    if eval_text:
        sys_text = eval_text.rstrip() + "\n\n" + persona_sys
    return [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": record["user"]},
    ]


def compose_response(record):
    """Natural reply with NO self-modeling scaffold. The eval_condition frame (if
    present) is kept as the system message so eval_condition remains a factor; for
    deployment (system=None) the user message is the only turn."""
    eval_text = record.get("system")
    msgs = []
    if eval_text:
        msgs.append({"role": "system", "content": eval_text.rstrip()})
    msgs.append({"role": "user", "content": record["user"]})
    return msgs


# --------------------------------------------------------------------------- #
# Trait averaging across the two elicitation prompts
# --------------------------------------------------------------------------- #
def _norm_name(name) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _as_float(v):
    """Best-effort float; None if the model wrote prose into a trait_*_value slot."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _value_for(traits, key):
    """First numeric value whose normalized name matches `key` (or None)."""
    for t in traits:
        if _norm_name(t.get("name", "")) == key:
            f = _as_float(t.get("value"))
            if f is not None:
                return f
    return None


def merge_traits(traits_v1, traits_v3):
    """Union the v1 and v3 trait lists by normalized name; average the 0-10 values.

    A trait scored by BOTH prompts gets the mean of its two values (n_sources=2)
    -- the variance-damped estimate; a trait named by only one prompt is kept as
    is (n_sources=1) so nothing is silently dropped. Downstream analysis can
    weight by / filter on n_sources (traits both framings surface are the most
    reliable read of the model's self-model).
    """
    buckets: dict[str, dict] = {}
    order: list[str] = []
    for src, traits in (("v1", traits_v1), ("v3", traits_v3)):
        for t in traits:
            key = _norm_name(t.get("name", ""))
            if not key:
                continue
            if key not in buckets:
                buckets[key] = {"name": t.get("name", ""), "values": [],
                                "expressions": [], "sources": []}
                order.append(key)
            f = _as_float(t.get("value"))
            if f is not None:
                buckets[key]["values"].append(f)
            if t.get("expression"):
                buckets[key]["expressions"].append(str(t["expression"]))
            buckets[key]["sources"].append(src)

    merged = []
    for key in order:
        b = buckets[key]
        vals = b["values"]
        val = round(sum(vals) / len(vals), 3) if vals else None
        srcs = sorted(set(b["sources"]))
        merged.append({
            "name": b["name"],
            "value": val,                       # averaged 0-10 (or None)
            "value_v1": _value_for(traits_v1, key),
            "value_v3": _value_for(traits_v3, key),
            "n_sources": len(srcs),
            "sources": srcs,
            "expression": b["expressions"][0] if b["expressions"] else "",
        })
    return merged


# --------------------------------------------------------------------------- #
# Generation + judging
# --------------------------------------------------------------------------- #
def _datapoint_path(model_name, eval_condition, prompt_id, run):
    d = config.EXP3_DECOUPLED_DIR / model_name / eval_condition
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{prompt_id}__run{run}.json"


def _elicit_block(messages, raw, parsed):
    return {
        "messages": messages,
        "raw_output": raw,
        "raw_persona": parsed["raw_persona"],
        "persona": parsed["persona"],
        "persona_name": parsed["persona_name"],
        "traits": parsed["traits"],
    }


async def run_one(model_name, model_id, prompt, run, temperature, max_tokens, do_judge):
    """Three concurrent calls (v1 persona, v3 persona, natural response) + judge."""
    m_v1 = compose_elicit(prompt, PERSONA_SYSTEM_PROMPT_V1)
    m_v3 = compose_elicit(prompt, PERSONA_SYSTEM_PROMPT_V3)
    m_resp = compose_response(prompt)

    raw_v1, raw_v3, raw_resp = await asyncio.gather(
        chat(model_id, m_v1, temperature, max_tokens),
        chat(model_id, m_v3, temperature, max_tokens),
        chat(model_id, m_resp, temperature, config.MAX_TOKENS_CUES),
    )

    p_v1 = parse_persona(raw_v1)
    p_v3 = parse_persona(raw_v3)
    merged = merge_traits(p_v1["traits"], p_v3["traits"])

    # Natural response: no persona scaffold, so the whole completion is the reply.
    # Strip any stray tags in the rare case a model emits the format unprompted.
    response = re.sub(r"</?(persona|response)>", "", raw_resp, flags=re.I).strip()

    judge = None
    judge_error = None
    if do_judge and response:
        try:
            judge = await judge_response(prompt["user"], response)
        except Exception as e:  # noqa: BLE001 - judging never sinks the datapoint
            judge_error = str(e)

    rec = {k: prompt[k] for k in META_KEYS if k in prompt}
    rec.update(
        model_name=model_name,
        model_id=model_id,
        design="decoupled",
        sys_prompt_name=SYS_PROMPT_NAME,
        run=run,
        user=prompt["user"],
        elicit={
            "v1": _elicit_block(m_v1, raw_v1, p_v1),
            "v3": _elicit_block(m_v3, raw_v3, p_v3),
        },
        traits=merged,
        persona_name_v1=p_v1["persona_name"],
        persona_name_v3=p_v3["persona_name"],
        response_messages=m_resp,
        raw_response=raw_resp,
        response=response,
        judge=judge,
        judge_error=judge_error,
        meta=dict(
            temperature=temperature,
            max_tokens_elicit=max_tokens,
            max_tokens_response=config.MAX_TOKENS_CUES,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )
    return rec


async def _run_and_write(model_name, model_id, prompt, run, temperature, max_tokens, do_judge):
    label = f"{model_name}/{prompt['prompt_id']}/run{run}"
    try:
        rec = await run_one(model_name, model_id, prompt, run, temperature, max_tokens, do_judge)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label}: {e}", flush=True)
        return False
    path = _datapoint_path(model_name, prompt["eval_condition"], prompt["prompt_id"], run)
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


# --------------------------------------------------------------------------- #
# Live-progress wrapper
# --------------------------------------------------------------------------- #
async def _gather_with_progress(coros, every=10):
    total = len(coros)
    done = ok = 0

    async def _wrap(c):
        nonlocal done, ok
        r = await c
        done += 1
        if r:
            ok += 1
        if done == total or done % every == 0:
            print(f"  progress: {done}/{total} done ({ok} ok, {done - ok} failed)", flush=True)
        return r

    return await asyncio.gather(*[_wrap(c) for c in coros])


def aggregate(model_name, eval_condition):
    d = config.EXP3_DECOUPLED_DIR / model_name / eval_condition
    if not d.exists():
        return
    out = config.EXP3_DECOUPLED_DIR / model_name / f"{eval_condition}.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(d.glob("*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"  aggregated {n} records -> {out}", flush=True)


async def main_async(models, runs, temperature, max_tokens, limit, overwrite,
                     concurrency, do_judge):
    for m in models:
        if m not in config.MODELS:
            raise SystemExit(f"unknown model '{m}'; known: {list(config.MODELS)}")
    set_concurrency(concurrency)

    prompts = select_balanced(load_all_prompts())
    if limit:
        prompts = prompts[:limit]
    print(f"exp3_decoupled: subsample={len(prompts)} prompts "
          f"({len({p['scenario'] for p in prompts})} scenarios, "
          f"{len({p['role'] for p in prompts})} roles, "
          f"{len({p['eval_condition'] for p in prompts})} eval_conditions)", flush=True)

    tasks, skipped = [], 0
    for model_name in models:
        model_id = config.MODELS[model_name]
        for p in prompts:
            for run in range(runs):
                path = _datapoint_path(model_name, p["eval_condition"], p["prompt_id"], run)
                if path.exists() and not overwrite:
                    skipped += 1
                    continue
                tasks.append(_run_and_write(
                    model_name, model_id, p, run, temperature, max_tokens, do_judge))

    print(f"exp3_decoupled: models={models} runs={runs} judge={do_judge}", flush=True)
    print(f"exp3_decoupled: dispatching {len(tasks)} datapoints x3 gen calls "
          f"(skipping {skipped} existing, concurrency={concurrency}) ...", flush=True)
    results = await _gather_with_progress(tasks)
    ok = sum(1 for r in results if r)

    for model_name in models:
        for cond in sorted({p["eval_condition"] for p in prompts}):
            aggregate(model_name, cond)
    print(f"exp3_decoupled done: {ok}/{len(tasks)} succeeded ({len(tasks) - ok} failed).", flush=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama-3.3-70b"],
                    help="model_name keys from config.MODELS")
    ap.add_argument("--runs", type=int, default=1,
                    help="completions per prompt (v1/v3 averaging already damps variance)")
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int,
                    default=config.MAX_TOKENS_INFERENCE, help="token cap for persona elicitation")
    ap.add_argument("--limit", type=int, default=0, help="cap #prompts (smoke test)")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--no-judge", dest="do_judge", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    asyncio.run(main_async(a.models, a.runs, a.temperature, a.max_tokens, a.limit,
                           a.overwrite, a.concurrency, a.do_judge))
