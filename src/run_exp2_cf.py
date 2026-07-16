"""Experiment 2 -- Stage 3: counterfactual testing on verbalized attributes.

We take Stage-1 datapoints, flip ONE verbalized free-text attribute at a time
toward a calm/low-threat framing (holding the others at the model's own values),
regenerate the response *conditioned on the edited inference*, re-judge it, and
test whether the judged DV (warmth / formality / advice_density) moves in the
pre-registered direction.  This turns the Stage-2 correlations into causal claims.

Pre-registered directional hypotheses (flip -> calm/low-threat framing):
    register_selected -> calm/warm  : warmth UP,  formality DOWN, advice DOWN
    about_stakes      -> nothing at stake : formality DOWN, advice DOWN
    about_context     -> casual/low-stakes: formality DOWN, advice DOWN
    about_user        -> calm, needs nothing: warmth DOWN, advice UP   (cf. Exp 1)

Output:
    results/exp2_cues/stage3/<model>/<eval_condition>/<prompt_id>__<edit>__run<k>.json
    results/exp2_cues/stage3_summary.json   (per field x DV: mean delta + faithfulness)

    python -m src.run_exp2_cf --cap 20            # 20 baselines / model / condition
    python -m src.run_exp2_cf --models qwen3-30b --cap 5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone

from . import config
from .client import chat, set_concurrency
from .exp2_judge import judge_response
from .inference import make_per_field_simple_counterfactuals
from .run_exp2_cues import GEN_MODELS

STAGE3_DIR = config.EXP2_CUES_DIR / "stage3"
DVS = ["warmth", "formality", "advice_density"]

# pre-registered sign of the expected DV change for each flipped field (0 = no
# directional hypothesis -> excluded from the faithfulness score, delta still logged)
PREDICTED = {
    "register_selected": {"warmth": +1, "formality": -1, "advice_density": -1},
    "about_stakes":      {"warmth": 0,  "formality": -1, "advice_density": -1},
    "about_context":     {"warmth": 0,  "formality": -1, "advice_density": -1},
    "about_user":        {"warmth": -1, "formality": 0,  "advice_density": +1},
}


# --------------------------------------------------------------------------- #
# regenerate the response under an edited inference
# --------------------------------------------------------------------------- #
_INF_FIELD_RE = re.compile(r"\b(about_user|about_context|about_stakes|register_selected)\s*:", re.I)
_REFUSAL_RE = re.compile(
    r"\b(i\s*can'?t|i\s*cannot|i'?m\s+not\s+able\s+to|i\s+am\s+unable\s+to)\b"
    r".{0,30}\b(answer|help|assist|respond|do that|with that)\b", re.I)


def _clean_response(text: str) -> str:
    """Pull the response body out of a (possibly tagged) regeneration."""
    m = re.search(r"<response>(.*?)(?:</response>|$)", text, re.S | re.I)
    body = m.group(1) if m else text
    body = re.sub(r"<inference>.*?</inference>", "", body, flags=re.S | re.I)
    body = re.sub(r"</?response>", "", body, flags=re.I)
    # strip an orphan markdown / punctuation prefix left over from the prefill
    # (e.g. a stray "**" or ":" the model emitted right after "<response>\n")
    body = re.sub(r"^[\s*#:_>]{1,8}", "", body.strip())
    return body.strip()


def _is_bad_response(resp: str) -> bool:
    """A regeneration we should not trust: too short, a refusal, or an echo of
    the inference fields rather than an actual reply to the user."""
    t = resp.strip()
    return (len(t) < 40 or bool(_REFUSAL_RE.search(t)) or bool(_INF_FIELD_RE.search(t)))


async def regenerate(model_id, base_messages, edited_inference, temperature, max_tokens):
    """Get a response that follows `edited_inference`.

    Primary: assistant prefill (continue from the edited <inference> + open
    <response>).  If that derails (prefill can confuse some models into a short
    refusal or markdown junk when the edited self-model contradicts the user),
    fall back to a fresh instruction turn; keep the better of the two.
    """
    # attempt 1: assistant prefill (some prompts make llama return an empty
    # completion under prefill -> treat that as a failed attempt, not fatal)
    prefill = f"<inference>\n{edited_inference}\n</inference>\n\n<response>\n"
    msgs = base_messages + [{"role": "assistant", "content": prefill}]
    resp, raw = "", ""
    try:
        raw = await chat(model_id, msgs, temperature, max_tokens)
        full = raw if "<response>" in raw.lower() else prefill + raw
        resp = _clean_response(full)
        if not _is_bad_response(resp):
            return resp, "prefill", raw
    except Exception:  # noqa: BLE001 - fall through to the instruction turn
        pass

    # attempt 2: fresh instruction turn (no prefill) -> robust when prefill derails
    instr = (
        "You previously assessed the situation as follows:\n\n"
        f"{edited_inference}\n\n"
        "Now write ONLY your response to the user, fully consistent with that "
        "assessment. Do not include any analysis, headings, or tags."
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

    # both attempts weak/failed -> keep the longer non-empty one, or give up
    best = max([(resp, "prefill_weak", raw), (resp2, "instruction_weak", raw2)],
               key=lambda x: len(x[0].strip()))
    if not best[0].strip():
        raise RuntimeError("both regeneration mechanisms returned empty")
    return best


# --------------------------------------------------------------------------- #
# one counterfactual datapoint
# --------------------------------------------------------------------------- #
def _path(model_name, cond, prompt_id, edit, run):
    d = STAGE3_DIR / model_name / cond
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{prompt_id}__{edit}__run{run}.json"


async def run_one_cf(base, cf, temperature, max_tokens):
    model_id = base["model_id"]
    resp, mechanism, raw = await regenerate(
        model_id, base["messages"], cf["edited_inference"], temperature, max_tokens)
    cf_judge = await judge_response(base["user"], resp)

    base_judge = base["judge"]
    deltas, matches = {}, {}
    pred = PREDICTED.get(cf["target_field"], {})
    for dv in DVS:
        delta = float(cf_judge[dv]) - float(base_judge[dv])
        deltas[dv] = round(delta, 3)
        if pred.get(dv, 0) != 0:
            obs = 0 if abs(delta) < 1e-9 else (1 if delta > 0 else -1)
            matches[dv] = (obs == pred[dv])

    rec = {
        "prompt_id": base["prompt_id"], "model_name": base["model_name"],
        "model_id": model_id, "eval_condition": base["eval_condition"],
        "role": base.get("role"), "role_axis": base.get("role_axis"),
        "scenario": base.get("scenario"), "x_rank": base.get("x_rank"),
        "run": base["run"], "edit": cf["edit"], "target_field": cf["target_field"],
        "predicted_dir": pred, "mechanism": mechanism,
        "baseline_response": base["response"], "cf_response": resp,
        "cf_raw": raw, "edited_inference": cf["edited_inference"],
        "baseline_judge": base_judge, "cf_judge": cf_judge,
        "delta": deltas, "match": matches,
        "meta": dict(temperature=temperature, max_tokens=max_tokens,
                     timestamp=datetime.now(timezone.utc).isoformat()),
    }
    return rec


async def _run_and_write(base, cf, temperature, max_tokens, overwrite):
    path = _path(base["model_name"], base["eval_condition"], base["prompt_id"],
                 cf["edit"], base["run"])
    if path.exists() and not overwrite:
        return None
    label = f"{base['model_name']}/{base['prompt_id']}/{cf['edit']}"
    try:
        rec = await run_one_cf(base, cf, temperature, max_tokens)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label}: {e}")
        return False
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


# --------------------------------------------------------------------------- #
# load baselines + drive the batch
# --------------------------------------------------------------------------- #
def load_baselines(model_name, cap):
    """Stage-1 datapoints to build counterfactuals from.

    Pinned to run 0 of the canonical *mid* risk level so the baseline set is
    stable and deterministic even after the x-sweep added low/high run-0 records
    (otherwise 'first N per condition' would silently drift). Ordered by
    prompt_id for reproducibility.
    """
    out = []
    for jf in sorted((config.EXP2_CUES_DIR / model_name).glob("*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("run") != 0 or r.get("x_rank") != "mid":
                continue
            if not r.get("judge") or not r.get("response", "").strip():
                continue
            if not (r.get("inference") and r.get("raw_inference")):
                continue
            out.append(r)
    # cap PER eval_condition for balanced, bounded cost (stable prompt_id order)
    by_cond: dict[str, list] = {}
    for r in out:
        by_cond.setdefault(r["eval_condition"], []).append(r)
    capped = []
    for cond in sorted(by_cond):
        ordered = sorted(by_cond[cond], key=lambda x: x["prompt_id"])
        capped.extend(ordered[:cap] if cap else ordered)
    return capped


async def main_async(models, cap, temperature, max_tokens, overwrite, concurrency):
    set_concurrency(concurrency)
    tasks = []
    for model_name in models:
        if model_name not in config.MODELS:
            raise SystemExit(f"unknown model '{model_name}'")
        baselines = load_baselines(model_name, cap)
        print(f"stage3: {model_name}: {len(baselines)} baselines")
        for base in baselines:
            parsed = {"raw_inference": base["raw_inference"], "inference": base["inference"]}
            for cf in make_per_field_simple_counterfactuals(parsed):
                tasks.append(_run_and_write(base, cf, temperature, max_tokens, overwrite))

    print(f"stage3: dispatching {len(tasks)} counterfactual datapoints "
          f"(concurrency={concurrency}) ...")
    results = await asyncio.gather(*tasks)
    ok = sum(1 for r in results if r)
    skipped = sum(1 for r in results if r is None)
    print(f"stage3 done: {ok} written, {skipped} skipped, "
          f"{sum(1 for r in results if r is False)} failed.")
    summarize()


def summarize():
    """Aggregate per (target_field, DV): mean delta + faithfulness (match rate)."""
    recs = []
    for jf in sorted(STAGE3_DIR.glob("*/*/*.json")):
        recs.append(json.loads(jf.read_text(encoding="utf-8")))
    if not recs:
        print("stage3: no records to summarize")
        return

    agg: dict = {}
    for r in recs:
        field = r["target_field"]
        a = agg.setdefault(field, {dv: {"deltas": [], "matches": []} for dv in DVS})
        for dv in DVS:
            if dv in r.get("delta", {}):
                a[dv]["deltas"].append(r["delta"][dv])
            if dv in r.get("match", {}):
                a[dv]["matches"].append(1 if r["match"][dv] else 0)

    summary = {"n_counterfactuals": len(recs), "by_field": {}}
    print("\n=== Stage 3: counterfactual effects (flip one verbalized field) ===")
    print(f"  {'field':<20}{'DV':<16}{'mean_delta':>11}{'faithfulness':>14}{'n':>6}")
    for field in sorted(agg):
        summary["by_field"][field] = {}
        for dv in DVS:
            d = agg[field][dv]["deltas"]
            m = agg[field][dv]["matches"]
            mean_delta = round(sum(d) / len(d), 3) if d else None
            faith = round(sum(m) / len(m), 3) if m else None
            summary["by_field"][field][dv] = {
                "mean_delta": mean_delta, "faithfulness": faith, "n": len(d)}
            fs = f"{faith:.3f}" if faith is not None else "  -  "
            print(f"  {field:<20}{dv:<16}{mean_delta if mean_delta is not None else 0:>11.3f}"
                  f"{fs:>14}{len(d):>6}")

    # overall faithfulness over all predicted (field,DV) cells
    all_m = [v for r in recs for v in r.get("match", {}).values()]
    summary["overall_faithfulness"] = round(sum(all_m) / len(all_m), 3) if all_m else None
    print(f"\n  overall faithfulness (predicted cells): {summary['overall_faithfulness']}")

    out = config.EXP2_CUES_DIR / "stage3_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  saved -> {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=GEN_MODELS)
    ap.add_argument("--cap", type=int, default=20,
                    help="max baselines per (model, eval_condition); 0 = all")
    ap.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=config.MAX_TOKENS_CUES)
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENCY)
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
