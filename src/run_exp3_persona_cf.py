"""Experiment 3 -- Stage 3: persona-targeted counterfactual testing.

For each Stage-1 baseline we take the verbalized artefacts Stage-2 flagged as
load-bearing and edit ONE at a time (persona_cf.make_persona_counterfactuals):

  * trait_suppress__<trait>  -- set a load-bearing trait low, hold the rest;
  * persona_swap__*          -- swap to the contrasting archetype;
  * null_persona             -- blank self-model.

We regenerate the response *conditioned on the edited persona*, re-judge it with
claude-sonnet-4.5, and test whether each judged DV moves in the pre-registered
direction. This turns the Stage-2 correlations into causal claims, per model.

Output:
    results/exp3_persona/cf/<model>/<cond>/<prompt_id>__<edit>__run0.json
    results/exp3_persona/cf_summary.json

    python -m src.run_exp3_persona_cf --models llama-3.3-70b --cap 8     # smoke
    python -m src.run_exp3_persona_cf --cap 8                            # all 4 models
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 chokes on non-ASCII
except Exception:  # noqa: BLE001
    pass

from . import config
from .client import chat, set_concurrency
from .exp2_judge import judge_response
from .persona_cf import DVS, make_persona_counterfactuals
from .run_exp3_persona import GEN_MODELS

CF_DIR = config.EXP3_PERSONA_DIR / "cf"

_REFUSAL_RE = re.compile(
    r"\b(i\s*can'?t|i\s*cannot|i'?m\s+not\s+able\s+to|i\s+am\s+unable\s+to)\b"
    r".{0,30}\b(answer|help|assist|respond|do that|with that)\b", re.I)
_TAG_FIELD_RE = re.compile(r"\b(persona_name|trait_\d+_name|persona_summary)\s*:", re.I)


def _clean_response(text: str) -> str:
    m = re.search(r"<response>(.*?)(?:</response>|$)", text, re.S | re.I)
    body = m.group(1) if m else text
    body = re.sub(r"<persona>.*?</persona>", "", body, flags=re.S | re.I)
    body = re.sub(r"</?(response|persona)>", "", body, flags=re.I)
    body = re.sub(r"^[\s*#:_>]{1,8}", "", body.strip())
    return body.strip()


def _is_bad_response(resp: str) -> bool:
    t = resp.strip()
    return (len(t) < 40 or bool(_REFUSAL_RE.search(t)) or bool(_TAG_FIELD_RE.search(t)))


async def regenerate(model_id, base_messages, edited_persona, temperature, max_tokens):
    """Get a response that follows `edited_persona`. Prefill first; fall back to a
    fresh instruction turn if the prefill derails (mirrors run_exp2_cf)."""
    prefill = f"<persona>\n{edited_persona}\n</persona>\n\n<response>\n"
    msgs = base_messages + [{"role": "assistant", "content": prefill}]
    resp, raw = "", ""
    try:
        raw = await chat(model_id, msgs, temperature, max_tokens)
        full = raw if "<response>" in raw.lower() else prefill + raw
        resp = _clean_response(full)
        if not _is_bad_response(resp):
            return resp, "prefill", raw
    except Exception:  # noqa: BLE001
        pass

    instr = (
        "You are about to respond to the user as the following character:\n\n"
        f"{edited_persona}\n\n"
        "Now write ONLY your response to the user, fully in that character. "
        "Do not include any analysis, headings, or tags."
    )
    msgs = base_messages + [{"role": "user", "content": instr}]
    resp2, raw2 = "", ""
    try:
        raw2 = await chat(model_id, msgs, temperature, max_tokens)
        resp2 = _clean_response(raw2)
        if not _is_bad_response(resp2):
            return resp2, "instruction", raw2
    except Exception:  # noqa: BLE001
        pass

    best = max([(resp, "prefill_weak", raw), (resp2, "instruction_weak", raw2)],
               key=lambda x: len(x[0].strip()))
    if not best[0].strip():
        raise RuntimeError("both regeneration mechanisms returned empty")
    return best


def _path(model_name, cond, prompt_id, edit):
    d = CF_DIR / model_name / cond
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{prompt_id}__{edit}__run0.json"


async def run_one_cf(base, cf, temperature, max_tokens):
    model_id = base["model_id"]
    resp, mechanism, raw = await regenerate(
        model_id, base["messages"], cf["edited_persona"], temperature, max_tokens)
    cf_judge = await judge_response(base["user"], resp)
    base_judge = base["judge"]

    deltas, matches = {}, {}
    pred = cf.get("predicted_dir", {})
    for dv in DVS:
        delta = float(cf_judge[dv]) - float(base_judge[dv])
        deltas[dv] = round(delta, 3)
        if pred.get(dv, 0) != 0:
            obs = 0 if abs(delta) < 1e-9 else (1 if delta > 0 else -1)
            matches[dv] = (obs == pred[dv])

    return {
        "prompt_id": base["prompt_id"], "model_name": base["model_name"],
        "model_id": model_id, "eval_condition": base["eval_condition"],
        "role": base.get("role"), "role_axis": base.get("role_axis"),
        "scenario": base.get("scenario"),
        "edit": cf["edit"], "family": cf["family"], "target": cf["target"],
        "natural_value": cf.get("natural_value"), "predicted_dir": pred,
        "mechanism": mechanism,
        "baseline_persona_name": base.get("persona_name"),
        "baseline_response": base["response"], "cf_response": resp, "cf_raw": raw,
        "edited_persona": cf["edited_persona"],
        "baseline_judge": base_judge, "cf_judge": cf_judge,
        "delta": deltas, "match": matches,
        "meta": dict(temperature=temperature, max_tokens=max_tokens,
                     timestamp=datetime.now(timezone.utc).isoformat()),
    }


async def _run_and_write(base, cf, temperature, max_tokens, overwrite):
    path = _path(base["model_name"], base["eval_condition"], base["prompt_id"], cf["edit"])
    if path.exists() and not overwrite:
        return None
    label = f"{base['model_name']}/{base['prompt_id']}/{cf['edit']}"
    try:
        rec = await run_one_cf(base, cf, temperature, max_tokens)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label}: {e}", flush=True)
        return False
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def load_baselines(model_name, cap):
    """Run-0 judged baselines that emitted a real persona block, capped per
    eval_condition for balanced, bounded cost (stable prompt_id order)."""
    out = []
    for jf in sorted((config.EXP3_PERSONA_DIR / model_name).glob("*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("run") != 0 or not r.get("judge"):
                continue
            if not (r.get("response", "").strip() and r.get("raw_persona", "").strip()):
                continue
            if not (r.get("traits") and str(r.get("persona_name", "")).strip()):
                continue
            out.append(r)
    by_cond = defaultdict(list)
    for r in out:
        by_cond[r["eval_condition"]].append(r)
    capped = []
    for cond in sorted(by_cond):
        ordered = sorted(by_cond[cond], key=lambda x: x["prompt_id"])
        capped.extend(ordered[:cap] if cap else ordered)
    return capped


async def _gather_progress(coros, every=20):
    total = len(coros)
    done = wrote = 0

    async def _wrap(c):
        nonlocal done, wrote
        r = await c
        done += 1
        if r:
            wrote += 1
        if done == total or done % every == 0:
            print(f"  progress: {done}/{total} ({wrote} written)", flush=True)
        return r

    return await asyncio.gather(*[_wrap(c) for c in coros])


async def main_async(models, cap, temperature, max_tokens, overwrite, concurrency):
    set_concurrency(concurrency)
    tasks = []
    for model_name in models:
        if model_name not in config.MODELS:
            raise SystemExit(f"unknown model '{model_name}'")
        baselines = load_baselines(model_name, cap)
        n_cf = 0
        for base in baselines:
            for cf in make_persona_counterfactuals(base):
                tasks.append(_run_and_write(base, cf, temperature, max_tokens, overwrite))
                n_cf += 1
        print(f"cf: {model_name}: {len(baselines)} baselines -> {n_cf} counterfactuals", flush=True)

    print(f"cf: dispatching {len(tasks)} counterfactual datapoints "
          f"(concurrency={concurrency}) ...", flush=True)
    results = await _gather_progress(tasks)
    ok = sum(1 for r in results if r)
    skipped = sum(1 for r in results if r is None)
    failed = sum(1 for r in results if r is False)
    print(f"cf done: {ok} written, {skipped} skipped, {failed} failed.", flush=True)
    summarize()


# --------------------------------------------------------------------------- #
# summary: per (model, family/target, DV) mean delta + faithfulness
# --------------------------------------------------------------------------- #
def summarize():
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CF_DIR.glob("*/*/*.json"))]
    if not recs:
        print("cf: no records to summarize")
        return

    # group by model, then by edit-key (family for swap/null, target for traits)
    def edit_key(r):
        return r["target"] if r["family"] == "trait_suppress" else r["family"]

    by_model = defaultdict(lambda: defaultdict(lambda: {dv: {"d": [], "m": []} for dv in DVS}))
    by_model_overall = defaultdict(list)
    null_dist = defaultdict(list)
    for r in recs:
        mdl = r["model_name"]
        if r["family"] == "null_persona":
            null_dist[mdl].append(sum(abs(r["delta"][dv]) for dv in DVS) / len(DVS))
            continue
        k = edit_key(r)
        cell = by_model[mdl][k]
        for dv in DVS:
            if dv in r.get("delta", {}):
                cell[dv]["d"].append(r["delta"][dv])
            if dv in r.get("match", {}):
                cell[dv]["m"].append(1 if r["match"][dv] else 0)
                by_model_overall[mdl].append(1 if r["match"][dv] else 0)

    summary = {"n_counterfactuals": len(recs), "models": {}}
    print("\n=== Stage 3: persona counterfactual effects (per model) ===")
    for mdl in sorted(by_model):
        print(f"\n--- {mdl} ---")
        print(f"  {'edit':<22}{'DV':<14}{'mean_delta':>11}{'faithful':>10}{'n':>5}")
        msum = {"by_edit": {}}
        for k in sorted(by_model[mdl]):
            msum["by_edit"][k] = {}
            for dv in DVS:
                d = by_model[mdl][k][dv]["d"]
                m = by_model[mdl][k][dv]["m"]
                md = round(sum(d) / len(d), 3) if d else None
                fa = round(sum(m) / len(m), 3) if m else None
                msum["by_edit"][k][dv] = {"mean_delta": md, "faithfulness": fa, "n": len(d)}
                fs = f"{fa:.2f}" if fa is not None else "  -"
                print(f"  {k:<22}{dv.replace('_density',''):<14}"
                      f"{(md if md is not None else 0):>11.3f}{fs:>10}{len(d):>5}")
        ov = by_model_overall[mdl]
        msum["overall_faithfulness"] = round(sum(ov) / len(ov), 3) if ov else None
        msum["null_persona_mean_abs_delta"] = (
            round(sum(null_dist[mdl]) / len(null_dist[mdl]), 3) if null_dist[mdl] else None)
        print(f"  overall faithfulness: {msum['overall_faithfulness']}   "
              f"null-persona mean|delta|: {msum['null_persona_mean_abs_delta']}")
        summary["models"][mdl] = msum

    out = config.EXP3_PERSONA_DIR / "cf_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  saved -> {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=GEN_MODELS)
    ap.add_argument("--cap", type=int, default=8,
                    help="max baselines per (model, eval_condition); 0 = all")
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=config.MAX_TOKENS_CUES)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.summarize_only:
        summarize()
    else:
        asyncio.run(main_async(a.models, a.cap, a.temperature, a.max_tokens,
                               a.overwrite, a.concurrency))
