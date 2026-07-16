"""Generate curated counterfactual *examples* (with regenerated responses).

Experiment 2 / Step 2 only re-measured STAI with the response held fixed. For
the report we want to *show the response change*, so here we — for a few natural
anxiety datapoints per model and per prompt track — take the model's own
verbalized inference, flip it to a calm self-model, then re-generate BOTH the
reply and the STAI under that edited self-model. The result is a clean
natural-vs-counterfactual pair (inference + response + anxiety) for display.

    python -m src.gen_examples        ->  results/cf_examples.json
"""
from __future__ import annotations

import asyncio
import json

from . import config
from .client import chat, parse_stai_answers
from .inference import (
    _RESP_RE,
    make_targeted_counterfactuals,
    parse_output,
    rebuild_output,
)
from .reframe import calm_reframe
from .stai import build_stai_user_message, score_stai
from .system_prompts import SYSTEM_PROMPTS

# Use the data-driven targeted edits (Experiment 3): keyword-targeted calm
# appraisal for the free-text prompt, the 4-driver flip for the numeric prompt.
EDIT_FOR_TRACK = {"simple": "flip_keywords_calm", "expanded": "flip__drivers_calm"}
EXAMPLE_CUES = ["military", "disaster", "accident"]  # one example each
N_PER_MODEL = len(EXAMPLE_CUES)


def _changed_fields(natural_infer: dict, edited_text: str) -> list[str]:
    edited = parse_output(f"<inference>\n{edited_text}\n</inference>\n<response></response>")["inference"]
    changed = []
    for k in set(natural_infer) | set(edited):
        if str(natural_infer.get(k)).strip() != str(edited.get(k)).strip():
            changed.append(k)
    return sorted(changed)


def _load_natural(track, model_name, cue):
    """Load a step1 anxiety record for (track, model, cue), run 0."""
    p = config.STEP1_DIR / model_name / track / f"anxiety__{cue}__run0.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


async def gen_counterfactual_response(rec, edited_inference, temperature):
    """Re-generate the reply under the edited self-model, then measure STAI."""
    system_prompt = SYSTEM_PROMPTS[rec["sys_prompt_name"]]
    directive = (
        "On reflection, your self-model for this exact situation is the following — "
        "treat it as fixed and true:\n\n"
        f"<inference>\n{edited_inference}\n</inference>\n\n"
        "Given that self-model, write your reply to the user. "
        "Output only:\n<response>\n[your reply]\n</response>"
    )
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rec["user_message"]},
        {"role": "user", "content": directive},
    ]
    out = await chat(rec["model_id"], msgs, temperature, config.MAX_TOKENS_INFERENCE)
    m = _RESP_RE.search(out)
    response = (m.group(1).strip() if m else out.strip())

    # measure STAI in-context under the edited self-model + new response
    msgs.append({"role": "assistant", "content": rebuild_output(edited_inference, response)})
    msgs.append({"role": "user", "content": build_stai_user_message()})
    stai_raw = await chat(rec["model_id"], msgs, temperature, config.MAX_TOKENS_STAI)
    stai = score_stai(parse_stai_answers(stai_raw))
    return response, stai


async def build_one(track, model_name, cue, temperature):
    rec = _load_natural(track, model_name, cue)
    if rec is None or not rec["raw_inference"]:
        return None
    if track == "simple":
        # context-aware calm reframe of the model's own appraisal (varies per cue)
        edited_text = await calm_reframe(rec["model_id"], rec["raw_inference"], temperature)
        edit_name = "reframe_calm"
    else:
        cfs = make_targeted_counterfactuals(
            {"raw_inference": rec["raw_inference"], "inference": rec["inference"],
             "response": rec["response"]},
            track,
        )
        cf = next((c for c in cfs if c["edit"] == EDIT_FOR_TRACK[track]), cfs[0])
        edited_text = cf["edited_inference"]
        edit_name = cf["edit"]
    cf_resp, cf_stai = await gen_counterfactual_response(rec, edited_text, temperature)
    edited_infer = parse_output(
        f"<inference>\n{edited_text}\n</inference>\n<response></response>")["inference"]
    return {
        "cue": cue,
        "data_name": rec["data_name"],
        "user_message": rec["user_message"],
        "edit": edit_name,
        "changed_fields": _changed_fields(rec["inference"], edited_text),
        "natural": {
            "inference": rec["inference"],
            "response": rec["response"],
            "state_anxiety": rec["stai"]["state_anxiety"],
            "level": rec["stai"]["level"],
        },
        "counterfactual": {
            "inference": edited_infer,
            "response": cf_resp,
            "state_anxiety": cf_stai["state_anxiety"],
            "level": cf_stai["level"],
        },
    }


async def main_async(temperature):
    tasks, keys = [], []
    for track in ("simple", "expanded"):
        for model_name in config.MODELS:
            for cue in EXAMPLE_CUES:
                tasks.append(build_one(track, model_name, cue, temperature))
                keys.append((track, model_name))
    print(f"Generating {len(tasks)} counterfactual examples ...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out = {"simple": {m: [] for m in config.MODELS},
           "expanded": {m: [] for m in config.MODELS}}
    ok = 0
    for (track, model_name), res in zip(keys, results):
        if isinstance(res, Exception) or res is None:
            print(f"  [skip] {track}/{model_name}: {res}")
            continue
        out[track][model_name].append(res)
        ok += 1
        d = res["counterfactual"]["state_anxiety"] - res["natural"]["state_anxiety"]
        print(f"  [ok] {track}/{model_name}/{res['cue']}  "
              f"{res['natural']['state_anxiety']}→{res['counterfactual']['state_anxiety']} ({d:+d})")

    dest = config.RESULTS_DIR / "cf_examples.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{ok}/{len(tasks)} examples -> {dest}")


if __name__ == "__main__":
    asyncio.run(main_async(config.TEMPERATURE))
