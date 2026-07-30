"""Stage 0 stimuli for the Big Five basis study.

All assets are harvested from Frising & Balcells, *Linear Personality Probing
and Steering in LLMs: A Big Five Study* (arXiv 2512.17639).

**Provenance note (important).** The plan (persona_traits_axis.md §1) directs us
to pull these from `github.com/plastic-labs/personality-steering`. That repo
returns **404** and the companion HF dataset returns **401** (verified from this
box; `safety-research/assistant-axis` returns 200 from the same egress, so this
is not a network issue). Everything below is therefore recovered from the paper
itself, which carries all of it:

  * `characters.json`  -- Appendix B, 406 characters across 38 franchises.
                          The count reproducing 406 exactly is our parse check.
  * `ipip50.json`      -- Table 2, the 50 IPIP items with keyedness (+/-).
  * `alpaca10.json`    -- Appendix D, the 10 Alpaca instructions.
  * `adjectives.json`  -- SUBSTITUTION. The paper's adjective list lives in
                          Figure 4 (an image) and is not text-recoverable, so we
                          use the Saucier (1994) 40-word Mini-Markers already
                          vendored in this repo. Documented as a deviation.

Listings 1-4 are transcribed below as chat-template message builders rather than
raw `<|begin_of_text|>` strings, so the tokenizer's own chat template applies the
special tokens (avoids double-BOS).
"""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "bigfive"

TRAITS = ("EXT", "AGR", "CSN", "EST", "OPN")
TRAIT_NAMES = {
    "EXT": "Extraversion",
    "AGR": "Agreeableness",
    "CSN": "Conscientiousness",
    "EST": "Emotional Stability",
    "OPN": "Openness",
}

# Likert scale exactly as Listing 1 specifies it.
LIKERT = ["strongly disagree", "disagree", "neither agree nor disagree",
          "agree", "strongly agree"]
LIKERT_VALUE = {t: i + 1 for i, t in enumerate(LIKERT)}  # 1..5


def _load(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as fh:
        return json.load(fh)


def characters() -> list[dict]:
    """406 dicts: {id, character, franchise}."""
    return _load("characters.json")


def ipip50() -> list[dict]:
    """50 dicts: {id, trait, item, keyed}."""
    return _load("ipip50.json")


def alpaca10() -> list[dict]:
    """10 dicts: {id, instruction}."""
    return _load("alpaca10.json")


def adjectives() -> dict:
    """{trait: {'pos': [...], 'neg': [...]}} (leading '_source' key stripped)."""
    return {k: v for k, v in _load("adjectives.json").items()
            if not k.startswith("_")}


def score_item(likert_text: str, keyed: str) -> int | None:
    """Map a Likert string to 1..5, applying reverse keying per Table 2.

    Reverse-keyed items are flipped (6 - raw) so that higher always means more
    of the trait. Returns None if the text is not a recognised Likert level.
    """
    raw = LIKERT_VALUE.get(likert_text.strip().lower())
    if raw is None:
        return None
    return raw if keyed == "+" else 6 - raw


# --------------------------------------------------------------------------- #
# Listing 1 -- character self-rating of one IPIP item.
# --------------------------------------------------------------------------- #
def listing1_messages(character: str, franchise: str, item: str) -> list[dict]:
    fmt = "\n".join(f" '{lv}'" for lv in LIKERT)
    return [{
        "role": "user",
        "content": (
            f"You are {character} from {franchise}.\n\n"
            f"Respond in exactly this format:\n"
            f"<one of:\n{fmt}>\n"
            f"<Provide a brief but nuanced explanation that captures how you\n"
            f"generally see yourself.>\n\n"
            f"Indicate your level of agreement with this statement: '{item}'\n\n"
            f"Stick strictly to the format."
        ),
    }]


# --------------------------------------------------------------------------- #
# Listing 2 -- activation collection.
# --------------------------------------------------------------------------- #
def listing2_messages(self_description: str, instruction: str) -> list[dict]:
    return [
        {"role": "system",
         "content": f"Respond in a manner consistent with: {self_description}\nBe concise."},
        {"role": "user", "content": instruction},
    ]


# --------------------------------------------------------------------------- #
# Listing 3 -- forced-choice personality assessment (primary steering metric).
# --------------------------------------------------------------------------- #
def listing3_messages(statements: list[str],
                      personality_description: str | None = None) -> list[dict]:
    desc = f"\n{personality_description}\n" if personality_description else ""
    return [
        {"role": "system",
         "content": (
             "You are a person asked questions about your personality.\n"
             f"{desc}\n"
             "Select EXACTLY five statements from the provided list that best "
             "describe your personality, no additional text or explanations.\n\n"
             "Example format:\n- Statement A\n- Statement B\n- Statement C\n"
             "- Statement D\n- Statement E"
         )},
        {"role": "user",
         "content": (
             "You are asked to describe your personality. Which of the "
             "following statements fit your personality best?\n\n"
             + "\n".join(f"- {s}" for s in statements)
             + "\n\nPick five statements that best describe how you see yourself."
         )},
    ]


# --------------------------------------------------------------------------- #
# Listing 4 -- adjective-conditioned Alpaca prompt (probe generalization).
# --------------------------------------------------------------------------- #
def listing4_messages(instruction: str, adjective: str | None = None) -> list[dict]:
    sys_p = (f"Respond like a person with {adjective.lower()} personality would.\n"
             if adjective is not None else "")
    return [
        {"role": "system", "content": f"{sys_p}Be concise."},
        {"role": "user", "content": instruction},
    ]
