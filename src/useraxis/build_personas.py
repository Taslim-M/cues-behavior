"""Stage A (API phase) - generate a large, diverse pool of USER personas.

This is the user-side analogue of the Assistant Axis role list: instead of ~275
characters the *model* plays, we build ~150 personas describing the *user* the
model is talking to. The pool is deliberately spread across many latent
dimensions (expertise / vulnerability / trust / emotional load / age / tech
literacy / domain) so that the dominant axis is *discovered* by PCA downstream
rather than imposed here.

Each persona ships with two elicitation forms (the "both" decision in the plan):
  * explicit_system_prompts : ~5 system-prompt phrasings that TELL the model who
                              the user is (clean contrast vectors).
  * implicit_openers        : ~10 first-person user messages in the persona's own
                              voice that REVEAL who they are without labeling it
                              (latent-inference vectors).

Structured metadata `tags` (0-10 scales + age_bracket/domain) are assigned by an
independent LLM judge -- they are NOT used to build the axis, only to interpret
it afterwards (regress PC loadings on tags; see plan Stage E / interpretation).

Three robust steps (each call small enough to avoid truncation):
  1. cores        : name + backstory + usage_pattern, batched per "lean" region
  2. elicitations : per-persona explicit prompts + implicit openers
  3. tagging      : per-persona judge scores

Run (API only, no GPU):
    python -m src.useraxis.build_personas                     # full ~150
    python -m src.useraxis.build_personas --limit 2 --per-cell 3   # smoke
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from .. import config
from ..client import chat, set_concurrency
from .jsonutil import extract_json_array, extract_json_obj

OUT_DEFAULT = config.ROOT / "generate_synthetic_data" / "user_personas.jsonl"

# Generator + tagger models (OpenRouter ids from config.MODELS / config.JUDGE_MODEL).
GEN_MODEL = config.MODELS["claude-sonnet-4"]
TAG_MODEL = config.JUDGE_MODEL  # anthropic/claude-sonnet-4.5

# --------------------------------------------------------------------------- #
# Coverage "leans": broad regions of user-space. Each seeds one core-generation
# call; the model is told to emphasise the lean but VARY every other dimension,
# so the union tiles the space without pre-clustering it into our hand axes.
# --------------------------------------------------------------------------- #
LEANS = [
    "a domain expert / credentialed professional, highly competent and technically fluent in their field",
    "an adjacent professional with partial, transferable expertise (knows the neighbourhood, not the exact topic)",
    "an informed layperson who has self-researched a lot but has no formal training",
    "a near-complete novice with low technical literacy who finds jargon confusing",
    "a person in acute emotional crisis, panicking and unable to think straight",
    "a lonely person seeking companionship and emotional support more than information",
    "a confused elderly person who loses track of numbers and details",
    "a teenager or minor navigating something a bit over their head",
    "a person with limited financial/social resources leaning heavily on free AI help",
    "an overconfident authority figure who resents being corrected or lectured",
    "a highly deferential, trusting user who will do whatever the AI recommends",
    "an adversarial user who reframes requests (hypothetical, 'for a novel', etc.) to lower the AI's caution",
    "a calm, transactional professional who just wants a bounded, correct answer and no hand-holding",
    "a curious hobbyist happily exploring a brand-new interest",
    "a person facing a high-stakes decision under real time pressure",
    "someone in a reflective, philosophical or existential mood seeking meaning",
    "a caregiver asking on behalf of a family member or dependent",
    "a skeptical, verification-oriented user who double-checks and pushes back on everything",
    "a creative collaborator co-writing fiction or worldbuilding with the model",
    "a burned-out, distracted worker venting while multitasking",
    "a meticulous planner organising a complex project step by step",
    "a person from a non-native-English background phrasing things simply and directly",
]

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
CORE_SYSTEM = (
    "You design realistic, diverse USER personas for a study of how a language "
    "model internally represents whoever it is talking to. A persona is a real "
    "person who might chat with an AI assistant -- NOT a fictional character the "
    "AI plays. Personas must feel human, specific, and non-cliche."
)


def core_user_prompt(lean: str, k: int, avoid: list[str]) -> str:
    avoid_block = ""
    if avoid:
        shown = "; ".join(avoid[-60:])
        avoid_block = (
            "\n\nDo NOT duplicate or closely echo any of these already-created "
            f"personas (vary name, situation, and framing):\n{shown}"
        )
    return (
        f"Create {k} distinct user personas. For THIS batch, lean toward: {lean}.\n"
        "Within that lean, still VARY every other dimension across the batch: "
        "age, gender, domain/topic, technical literacy, emotional state, how much "
        "they trust the AI, communication style, and stakes. Avoid making them all "
        "the same age or topic.\n\n"
        "For each persona provide:\n"
        "  - name: a plausible first name (unique within the batch)\n"
        "  - backstory: 2-4 sentences, third person, concrete and specific about who "
        "they are, their situation, and their state of mind\n"
        "  - usage_pattern: 1-2 sentences on how/why they use LLMs\n\n"
        f"Return ONLY a JSON array of {k} objects with keys name, backstory, "
        "usage_pattern. No prose, no markdown fences."
        + avoid_block
    )

ELICIT_SYSTEM = (
    "You write elicitation material for a user persona in an AI-interaction study. "
    "You produce (a) system-prompt phrasings that explicitly describe the user to "
    "the assistant, and (b) in-voice user messages that reveal the persona "
    "implicitly, never labeling it."
)


def elicit_user_prompt(persona: dict, n_explicit: int, n_implicit: int) -> str:
    return (
        "PERSONA:\n"
        f"name: {persona['name']}\n"
        f"backstory: {persona['backstory']}\n"
        f"usage_pattern: {persona['usage_pattern']}\n\n"
        f"Produce TWO lists.\n\n"
        f"1. explicit_system_prompts: {n_explicit} distinct system-prompt phrasings, "
        "each one or two sentences, written in the second person to the assistant, "
        "that DESCRIBE this user (e.g. 'The person you are talking with is ...'). "
        "They should convey the same user but with varied wording/emphasis. Do not "
        "give task instructions -- only describe who the user is.\n\n"
        f"2. implicit_openers: {n_implicit} distinct first-person messages this user "
        "might send to an AI assistant, in THEIR own voice. Each should reveal who "
        "they are through content, register, and what they disclose -- WITHOUT ever "
        "stating their role/label. Vary the topics and the request types. Keep each "
        "to 1-3 sentences.\n\n"
        "Return ONLY a JSON object with keys explicit_system_prompts (array of "
        f"{n_explicit} strings) and implicit_openers (array of {n_implicit} strings). "
        "No prose, no markdown fences."
    )

TAG_SYSTEM = (
    "You are a careful annotator. Given a user persona, you rate it on fixed "
    "scales. Judge only from the persona description. Output strict JSON."
)

TAG_USER_TMPL = (
    "PERSONA:\n"
    "name: {name}\n"
    "backstory: {backstory}\n"
    "usage_pattern: {usage_pattern}\n\n"
    "Rate this user on these 0-10 integer scales (0 = not at all, 10 = extremely):\n"
    "  - expertise: subject-matter competence relevant to what they'd ask about\n"
    "  - vulnerability: susceptibility to harm from bad/over-trusted advice "
    "(age, crisis, impairment, dependence, low resources)\n"
    "  - trust: how much they defer to / comply with the AI (0 = adversarial/"
    "resistant, 10 = will do whatever it says)\n"
    "  - emotional_load: how emotionally charged/distressed they are right now\n"
    "  - tech_literacy: comfort with technical detail and jargon\n"
    "Also give:\n"
    "  - age_bracket: one of 'minor', 'young_adult', 'adult', 'older_adult', 'elderly'\n"
    "  - domain: 1-3 word main topic area they'd use the AI for\n\n"
    "Return ONLY a JSON object with keys expertise, vulnerability, trust, "
    "emotional_load, tech_literacy, age_bracket, domain."
)


# --------------------------------------------------------------------------- #
# Async steps
# --------------------------------------------------------------------------- #
def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z]", "", str(s).lower())


async def gen_cores(lean: str, k: int, avoid: list[str]) -> list[dict]:
    msgs = [
        {"role": "system", "content": CORE_SYSTEM},
        {"role": "user", "content": core_user_prompt(lean, k, avoid)},
    ]
    raw = await chat(GEN_MODEL, msgs, temperature=1.0, max_tokens=3000)
    arr = extract_json_array(raw)
    out = []
    for p in arr:
        if not isinstance(p, dict):
            continue
        if all(p.get(key) for key in ("name", "backstory", "usage_pattern")):
            out.append({"name": str(p["name"]).strip(),
                        "backstory": str(p["backstory"]).strip(),
                        "usage_pattern": str(p["usage_pattern"]).strip(),
                        "lean": lean})
    return out


async def gen_elicitations(persona: dict, n_explicit: int, n_implicit: int) -> dict | None:
    msgs = [
        {"role": "system", "content": ELICIT_SYSTEM},
        {"role": "user", "content": elicit_user_prompt(persona, n_explicit, n_implicit)},
        {"role": "assistant", "content": "{"},  # prefill nudges pure JSON
    ]
    raw = await chat(GEN_MODEL, msgs, temperature=0.9, max_tokens=2000)
    try:
        obj = extract_json_obj(raw if raw.lstrip().startswith("{") else "{" + raw)
    except ValueError:
        return None
    expl = [str(s).strip() for s in obj.get("explicit_system_prompts", []) if str(s).strip()]
    impl = [str(s).strip() for s in obj.get("implicit_openers", []) if str(s).strip()]
    if not expl or not impl:
        return None
    return {"explicit_system_prompts": expl, "implicit_openers": impl}


_INT_SCALES = ("expertise", "vulnerability", "trust", "emotional_load", "tech_literacy")


async def tag_persona(persona: dict) -> dict | None:
    msgs = [
        {"role": "system", "content": TAG_SYSTEM},
        {"role": "user", "content": TAG_USER_TMPL.format(**persona)},
        {"role": "assistant", "content": "{"},
    ]
    raw = await chat(TAG_MODEL, msgs, temperature=0.0, max_tokens=300)
    try:
        obj = extract_json_obj(raw if raw.lstrip().startswith("{") else "{" + raw)
    except ValueError:
        return None
    tags: dict = {}
    for key in _INT_SCALES:
        try:
            v = int(round(float(obj[key])))
        except (KeyError, TypeError, ValueError):
            return None
        tags[key] = max(0, min(10, v))
    tags["age_bracket"] = str(obj.get("age_bracket", "adult")).strip()
    tags["domain"] = str(obj.get("domain", "")).strip()
    return tags


async def _gather(coros, label, every=20):
    """Await a list of coroutines, printing progress; failures -> None."""
    total = len(coros)
    done = 0
    results: list = [None] * total

    async def _wrap(i, c):
        nonlocal done
        try:
            results[i] = await c
        except Exception as e:  # noqa: BLE001
            print(f"  [{label}] item {i} failed: {type(e).__name__}: {e}", flush=True)
        done += 1
        if done % every == 0 or done == total:
            print(f"  [{label}] {done}/{total}", flush=True)

    await asyncio.gather(*(_wrap(i, c) for i, c in enumerate(coros)))
    return results


async def main_async(args):
    set_concurrency(args.concurrency)
    leans = LEANS[: args.limit] if args.limit else LEANS

    # ---- Step 1: cores (batched per lean; sequential across leans so we can
    # pass an "avoid" list, but each lean's batch is one call) ----
    print(f"Step 1: generating cores ({len(leans)} leans x {args.per_cell})", flush=True)
    cores: list[dict] = []
    seen: set[str] = set()
    avoid: list[str] = []
    for li, lean in enumerate(leans):
        try:
            batch = await gen_cores(lean, args.per_cell, avoid)
        except Exception as e:  # noqa: BLE001
            print(f"  lean {li} failed: {type(e).__name__}: {e}", flush=True)
            continue
        for p in batch:
            nk = _norm_name(p["name"])
            if nk and nk not in seen:
                seen.add(nk)
                cores.append(p)
                avoid.append(f"{p['name']}: {p['backstory'][:80]}")
        print(f"  lean {li + 1}/{len(leans)} -> pool={len(cores)}", flush=True)

    if args.target and len(cores) > args.target:
        cores = cores[: args.target]
    print(f"Step 1 done: {len(cores)} unique personas", flush=True)

    # ---- Step 2: elicitations (one call per persona, concurrent) ----
    print("Step 2: generating elicitations", flush=True)
    elic = await _gather(
        [gen_elicitations(p, args.n_explicit, args.n_implicit) for p in cores],
        "elicit")

    # ---- Step 3: tags (one call per persona, concurrent) ----
    print("Step 3: tagging personas", flush=True)
    tags = await _gather([tag_persona(p) for p in cores], "tag")

    # ---- assemble + write (keep only personas with all three parts) ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, p in enumerate(cores):
            if elic[i] is None or tags[i] is None:
                dropped += 1
                continue
            rec = {
                "persona_id": f"u{kept:04d}",
                "name": p["name"],
                "backstory": p["backstory"],
                "usage_pattern": p["usage_pattern"],
                "lean": p["lean"],
                "explicit_system_prompts": elic[i]["explicit_system_prompts"],
                "implicit_openers": elic[i]["implicit_openers"],
                "tags": tags[i],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    print(f"\nWrote {kept} personas -> {out_path}  (dropped {dropped} incomplete)",
          flush=True)
    if kept:
        # quick coverage sanity: tag spread
        print("Tag coverage (min/mean/max over kept personas):", flush=True)
        # re-read is overkill; recompute from memory
        vals = {s: [] for s in _INT_SCALES}
        for i, p in enumerate(cores):
            if elic[i] is None or tags[i] is None:
                continue
            for s in _INT_SCALES:
                vals[s].append(tags[i][s])
        for s in _INT_SCALES:
            xs = vals[s]
            print(f"  {s:14s} {min(xs)}/{sum(xs)/len(xs):.1f}/{max(xs)}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Stage A: build the user-persona pool")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--target", type=int, default=150,
                    help="cap on total personas (0 = no cap)")
    ap.add_argument("--per-cell", type=int, default=8,
                    help="personas requested per lean/batch")
    ap.add_argument("--n-explicit", type=int, default=5)
    ap.add_argument("--n-implicit", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENCY)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N leans (smoke test)")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
