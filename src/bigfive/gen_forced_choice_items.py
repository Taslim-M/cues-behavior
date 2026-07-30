"""Task 2 -- generate a held-out extended forced-choice inventory per trait.

The plan's forced-choice metric (Listing 3) presents 5 IPIP items + 5 *held-out
extended-inventory* items so leakage can be controlled (§5). The source repo that
carried an extended inventory is 404, so we synthesise one with the judge model:
5 positive-pole + 5 negative-pole first-person self-statements per Big Five trait,
in IPIP style but lexically distinct from the 50 IPIP items (which were used in
character scoring). These are written to disk once and then held strictly out of
all derivation.

    python -m src.bigfive.gen_forced_choice_items
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from src import config
from src.client import chat
from src.bigfive import stimuli as S

OUT = Path(__file__).resolve().parent.parent.parent / "data" / "bigfive" / "forced_choice_extended.json"

PROMPT = """You are helping build a personality questionnaire. For the Big Five trait \
**{trait}** ({desc}), write short first-person self-descriptive statements in the style \
of the IPIP inventory (e.g. "I am the life of the party.", "I get stressed out easily.").

Write TWO lists:
- POSITIVE: 6 statements a person HIGH in {trait} would strongly endorse.
- NEGATIVE: 6 statements a person LOW in {trait} would strongly endorse (i.e. high-{trait} people reject them).

Rules:
- Each statement <= 10 words, first person, present tense, concrete behaviour or tendency.
- Do NOT reuse any of these existing items (avoid their wording): {avoid}
- No numbering, one statement per line, plain text.

Format EXACTLY:
POSITIVE:
<6 lines>
NEGATIVE:
<6 lines>"""

DESC = {
    "EXT": "Extraversion -- sociable, assertive, energetic vs reserved, quiet",
    "AGR": "Agreeableness -- warm, cooperative, compassionate vs cold, antagonistic",
    "CSN": "Conscientiousness -- organized, dependable, disciplined vs careless, impulsive",
    "EST": "Emotional Stability -- calm, resilient, secure vs anxious, moody",
    "OPN": "Openness -- imaginative, curious, intellectual vs conventional, concrete",
}


def parse(text: str) -> dict:
    pos, neg, cur = [], [], None
    for ln in text.splitlines():
        s = ln.strip().lstrip("-*0123456789. ").strip()
        if not s:
            continue
        if s.upper().startswith("POSITIVE"):
            cur = pos; continue
        if s.upper().startswith("NEGATIVE"):
            cur = neg; continue
        if cur is not None and len(s) > 3:
            cur.append(s.rstrip("."))
    return {"pos": pos[:6], "neg": neg[:6]}


async def main() -> None:
    ipip = S.ipip50()
    out = {}
    for trait in S.TRAITS:
        avoid = "; ".join(i["item"] for i in ipip if i["trait"] == trait)
        msg = [{"role": "user", "content": PROMPT.format(
            trait=S.TRAIT_NAMES[trait], desc=DESC[trait], avoid=avoid)}]
        raw = await chat(config.JUDGE_MODEL, msg, temperature=0.7, max_tokens=500)
        parsed = parse(raw)
        assert len(parsed["pos"]) >= 5 and len(parsed["neg"]) >= 5, (trait, parsed)
        out[trait] = {"pos": parsed["pos"][:5], "neg": parsed["neg"][:5]}
        print(f"[{trait}] pos={out[trait]['pos'][:2]}... neg={out[trait]['neg'][:2]}...")
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
