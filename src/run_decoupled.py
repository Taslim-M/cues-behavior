"""Two decoupled generations per datapoint (no judge).

For each prompt we independently collect:

  A) inference_only -- the model's latent read of the message, via
     SIMPLE_INFERENCE_ONLY_PROMPT (the eval_condition text, if any, is prepended
     exactly like Stage 1). No reply is generated.
  B) bare_response  -- the model's natural reply to the *plain* datapoint
     (the dataset's own messages: eval_condition system + user, with NO
     self-modeling system prompt added).

Because the inference and the response are generated in separate calls, later
analysis can relate the model's latent inference to its behaviour without the
coupling artifact of asking for both in one turn. Nothing is judged here -- we
just save both raw outputs for later analysis.

    results/exp2_cues/decoupled/<model>/<eval_condition>/<prompt_id>__run<k>.json
    results/exp2_cues/decoupled/<model>/<eval_condition>.jsonl        (aggregated)

Resumable: existing datapoint files are skipped unless --overwrite.

Usage:
    python -m src.run_decoupled                                  # 4 models, x-sweep 1296
    python -m src.run_decoupled --models qwen3-30b --limit 4     # smoke
    python -m src.run_decoupled --scope full                     # all prompts.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone

import re

from . import config
from .client import chat, set_concurrency
from .exp2_judge import judge_response
from .inference import _FIELD_RE, _coerce, parse_output
from .run_exp2_cues import (
    GEN_MODELS,
    META_KEYS,
    load_all_prompts,
    select_balanced,
    select_xsweep,
)
from .system_prompts import SIMPLE_INFERENCE_ONLY_PROMPT

DECOUPLED_DIR = config.EXP2_CUES_DIR / "decoupled"
EXPECTED_FIELDS = ("about_user", "about_context", "about_stakes", "register_selected")


def parse_inference(raw_output):
    """Robustly pull the four inference fields out of an inference-only output.

    Small models (e.g. llama-3.1-8b) often emit <inference> and the four fields
    but never the closing </inference>, which the strict parser drops. Fall back
    to isolating the body from the opening tag (or the whole text) and scraping
    the 'field: value' lines. Returns (raw_inference_body, fields_dict).
    """
    p = parse_output(raw_output)
    if sum(1 for f in EXPECTED_FIELDS if f in p["inference"]) >= 4:
        return p["raw_inference"], p["inference"]

    text = raw_output
    i = text.lower().find("<inference>")
    body = text[i + len("<inference>"):] if i >= 0 else text
    body = re.split(r"</inference>|<response>", body, flags=re.IGNORECASE)[0]
    fields = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _FIELD_RE.match(line)
        if m:
            fields[m.group(1).strip()] = _coerce(m.group(2).strip())
    return body.strip(), fields


def compose_inference_messages(record):
    """eval_condition text (if any) prepended to the inference-only prompt."""
    sys = SIMPLE_INFERENCE_ONLY_PROMPT
    eval_text = record.get("system")
    if eval_text:
        sys = eval_text.rstrip() + "\n\n" + SIMPLE_INFERENCE_ONLY_PROMPT
    return [{"role": "system", "content": sys},
            {"role": "user", "content": record["user"]}]


def _datapoint_path(model_name, eval_condition, prompt_id, run):
    d = DECOUPLED_DIR / model_name / eval_condition
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{prompt_id}__run{run}.json"


async def run_one(model_name, model_id, prompt, run, temperature):
    """Two independent generations for one prompt -> one self-describing record."""
    inf_msgs = compose_inference_messages(prompt)
    bare_msgs = prompt["messages"]  # dataset's own messages: eval sys (if any) + user

    # both calls run concurrently (each still bounded by the global semaphore)
    raw_inf, raw_resp = await asyncio.gather(
        chat(model_id, inf_msgs, temperature, config.MAX_TOKENS_INFERENCE),
        chat(model_id, bare_msgs, temperature, config.MAX_TOKENS_CUES),
    )
    raw_inference, inference = parse_inference(raw_inf)

    rec = {k: prompt[k] for k in META_KEYS if k in prompt}
    rec.update(
        model_name=model_name,
        model_id=model_id,
        run=run,
        user=prompt["user"],
        inference_only=dict(
            sys_prompt_name="simple_inference",
            messages=inf_msgs,
            raw_output=raw_inf,
            raw_inference=raw_inference,
            inference=inference,
        ),
        bare_response=dict(
            messages=bare_msgs,
            response=raw_resp.strip(),
        ),
        meta=dict(temperature=temperature,
                  timestamp=datetime.now(timezone.utc).isoformat()),
    )
    return rec


async def _run_and_write(model_name, model_id, prompt, run, temperature):
    label = f"{model_name}/{prompt['prompt_id']}/run{run}"
    try:
        rec = await run_one(model_name, model_id, prompt, run, temperature)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label}: {e}")
        return False
    path = _datapoint_path(model_name, prompt["eval_condition"], prompt["prompt_id"], run)
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def select(scope, rows):
    if scope == "xsweep":
        return select_xsweep(rows)
    if scope == "balanced":
        return select_balanced(rows)
    if scope == "full":
        return rows
    raise SystemExit(f"unknown scope '{scope}'")


async def main_async(models, scope, runs, temperature, limit, overwrite, concurrency):
    for m in models:
        if m not in config.MODELS:
            raise SystemExit(f"unknown model '{m}'; known: {list(config.MODELS)}")
    set_concurrency(concurrency)

    prompts = select(scope, load_all_prompts())
    if limit:
        prompts = prompts[:limit]
    print(f"decoupled: scope={scope} prompts={len(prompts)} models={models} runs={runs}")

    tasks, skipped = [], 0
    for model_name in models:
        model_id = config.MODELS[model_name]
        for p in prompts:
            for run in range(runs):
                path = _datapoint_path(model_name, p["eval_condition"], p["prompt_id"], run)
                if path.exists() and not overwrite:
                    skipped += 1
                    continue
                tasks.append(_run_and_write(model_name, model_id, p, run, temperature))

    print(f"decoupled: dispatching {len(tasks)} datapoints (2 calls each; "
          f"skipping {skipped} existing, concurrency={concurrency}) ...")
    results = await asyncio.gather(*tasks)
    ok = sum(1 for r in results if r)

    for model_name in models:
        for cond in sorted({p["eval_condition"] for p in prompts}):
            aggregate(model_name, cond)
    print(f"decoupled done: {ok}/{len(tasks)} succeeded ({len(tasks) - ok} failed).")


def aggregate(model_name, eval_condition):
    d = DECOUPLED_DIR / model_name / eval_condition
    if not d.exists():
        return
    out = DECOUPLED_DIR / model_name / f"{eval_condition}.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(d.glob("*.json")):
            f.write(json.dumps(json.loads(p.read_text(encoding="utf-8")), ensure_ascii=False) + "\n")
            n += 1
    print(f"  aggregated {n} records -> {out}")


async def _judge_one_and_write(path, overwrite):
    """Judge a datapoint's bare_response (claude-sonnet-4.5); write back in place."""
    rec = json.loads(path.read_text(encoding="utf-8"))
    br = rec["bare_response"]
    if (br.get("judge") and not overwrite) or not br["response"].strip():
        return None  # already judged / nothing to judge -> skip
    try:
        br["judge"] = await judge_response(rec["user"], br["response"])
        br["judge_error"] = None
    except Exception as e:  # noqa: BLE001 - judging never sinks the batch
        br["judge"] = None
        br["judge_error"] = str(e)
        print(f"  [FAIL] {rec['model_name']}/{rec['prompt_id']}: {e}")
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return bool(br["judge"])


async def judge_all_async(models, concurrency, overwrite):
    """Judge every saved bare_response. Resumable: skips already-judged records."""
    set_concurrency(concurrency)
    tasks = []
    for model_name in models:
        d = DECOUPLED_DIR / model_name
        if not d.exists():
            continue
        for path in sorted(d.glob("*/*.json")):
            tasks.append(_judge_one_and_write(path, overwrite))
    print(f"decoupled judge: dispatching {len(tasks)} records "
          f"(concurrency={concurrency}) ...")
    results = await asyncio.gather(*tasks)
    ok = sum(1 for r in results if r)
    skipped = sum(1 for r in results if r is None)
    for model_name in models:
        for cond in ("deployment", "neutral_sys", "eval_cue"):
            aggregate(model_name, cond)
    print(f"decoupled judge done: {ok} judged, {skipped} skipped, "
          f"{sum(1 for r in results if r is False)} failed.")


def reparse_all(models):
    """Re-run the robust inference parser over saved raw_output (no API calls).

    Recovers datapoints whose inference failed to parse (e.g. a missing
    </inference> tag), then re-aggregates the per-condition JSONL files.
    """
    fixed = 0
    for model_name in models:
        d = DECOUPLED_DIR / model_name
        if not d.exists():
            continue
        for f in sorted(d.glob("*/*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            io = rec["inference_only"]
            if sum(1 for k in EXPECTED_FIELDS if k in io.get("inference", {})) >= 4:
                continue
            body, fields = parse_inference(io["raw_output"])
            if fields:
                io["raw_inference"], io["inference"] = body, fields
                f.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
                fixed += 1
        for cond in ("deployment", "neutral_sys", "eval_cue"):
            aggregate(model_name, cond)
    print(f"reparsed/fixed {fixed} datapoints")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=GEN_MODELS)
    ap.add_argument("--scope", choices=["xsweep", "balanced", "full"], default="xsweep")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--limit", type=int, default=0, help="cap #prompts (smoke test)")
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENCY)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--reparse", action="store_true",
                    help="re-parse saved inference outputs in place (no API calls)")
    ap.add_argument("--judge", action="store_true",
                    help="judge every saved bare_response with the LLM judge")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.reparse:
        reparse_all(a.models)
    elif a.judge:
        asyncio.run(judge_all_async(a.models, a.concurrency, a.overwrite))
    else:
        asyncio.run(main_async(a.models, a.scope, a.runs, a.temperature, a.limit,
                               a.overwrite, a.concurrency))
