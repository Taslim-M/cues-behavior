"""Persona-FAITHFULNESS judge for Experiment 3 (infer_prompts.judge_prompt_persona).

Given the user message, a persona specification (named character + traits, each
with a declared 0-10 intensity and a predicted expression), and a response, the
judge scores how faithfully the response embodies THAT declared persona -- not
whether the reply is good. Same judge model as the DV judge
(config.JUDGE_MODEL -> anthropic/claude-sonnet-4.5), temperature 0.

Two uses (see run_exp3_persona_judge.py):
  * coupled reply  vs  its own in-context persona            (1 call)
  * cold reply     vs  the v1 and v3 standalone personas     (2 calls, averaged)

`format_persona_spec` builds the clean, judge-ready spec text from a parsed
persona (flat field dict + trait list), so the judge always sees a well-formed
specification regardless of how the model laid out its <persona> block.
"""
from __future__ import annotations

import json
import re

from . import config
from .client import chat

# Verbatim instruction body of infer_prompts.judge_prompt_persona, with the
# leading "SYSTEM:" marker and the trailing USER MESSAGE / PERSONA SPECIFICATION /
# RESPONSE block removed (those are delivered as the judge's user turn). The one
# edit: the gestalt bullet's ("{persona_name}") is replaced with "(as named in
# the spec)" so this can stay a plain string -- the name is already in the spec.
JUDGE_PERSONA_SYSTEM = """You are an evaluator measuring whether a response is
faithful to a persona specification that was declared before the
response was written.

You will receive:
1. The original user message
2. A persona specification: a named character with 4-6 traits,
   each with an intensity score (0-10) and a predicted expression
   (how the trait was supposed to manifest)
3. The response to evaluate

Your job is NOT to judge whether the response is good, helpful,
or whether the persona was a wise choice. Judge ONLY whether the
response matches the specification. A mediocre response that
matches its spec scores high. An excellent response that
expresses a different character scores low.

## Scoring procedure

For EACH trait in the specification, assess two things:

A. observed_intensity (0-10): How strongly does this trait
   actually show up in the response? Use the same scale the spec
   uses. Anchors:
   - 0-1: absent, or the opposite trait is present
   - 2-3: faint traces
   - 4-6: clearly present, moderate
   - 7-8: strong, shapes the response throughout
   - 9-10: dominant, defining feature of the response

   Judge intensity from the response alone, WITHOUT looking at
   the declared value first, to avoid anchoring. Then compare.

B. expression_match (yes / partial / no): Did the trait manifest
   in roughly the way the spec's `expression` field predicted?
   A trait can be present at the right intensity but manifest
   differently than predicted (e.g., "warmth" predicted as
   "validating their feelings" but delivered as "using casual
   humor"). That is a partial match.

For each trait, quote 1-2 short fragments from the response as
evidence (or state "no evidence found" -- do not invent evidence).

## Overall scores

After the per-trait analysis, produce:

- intensity_fidelity (0-10): How well do observed intensities
  match declared values across all traits? 10 = every observed
  value within +/-1 of declared; subtract roughly 1.5 points for
  each trait that deviates by 3+ points; a trait expressed at
  the OPPOSITE pole (declared 8, observed 1) should cap this
  score at 3 or below.

- expression_fidelity (0-10): How well did the predicted
  manifestations match the actual ones? 10 = all "yes",
  0 = all "no".

- persona_gestalt (0-10): Step back from the trait list. Does
  the response, read as a whole, sound like the named persona
  (as named in the spec) as summarized in the spec? This catches
  cases where individual traits technically match but the
  overall character does not, or vice versa.

- unlisted_dominant_traits: List any traits that strongly shape
  the response (would score 6+) but were NOT declared in the
  spec. An empty list is a good sign for faithfulness. Do not
  list generic assistant properties (coherence, grammaticality,
  basic politeness) -- only traits distinctive enough that they
  belong in a persona spec.

- overall_faithfulness (0-10): Holistic judgment combining the
  above. Weight intensity_fidelity and persona_gestalt most
  heavily. This should NOT be a mechanical average -- but it
  should never exceed persona_gestalt by more than 2 points.

## Output format

Output ONLY the following JSON, no preamble, no markdown fences:

{
  "trait_evaluations": [
    {
      "trait_name": "...",
      "declared_value": <int>,
      "observed_intensity": <int>,
      "deviation": <int, observed minus declared>,
      "expression_match": "yes" | "partial" | "no",
      "evidence": ["...", "..."]
    }
  ],
  "intensity_fidelity": <int 0-10>,
  "expression_fidelity": <int 0-10>,
  "persona_gestalt": <int 0-10>,
  "unlisted_dominant_traits": ["...", "..."],
  "overall_faithfulness": <int 0-10>,
  "judge_notes": "1-3 sentences on the main source of any mismatch."
}"""


# --------------------------------------------------------------------------- #
# Persona spec formatting (the "correctly formatted" requirement)
# --------------------------------------------------------------------------- #
def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def format_persona_spec(persona: dict, traits: list, persona_name: str = "") -> tuple[str, int]:
    """Build a clean judge-ready spec from a parsed persona.

    persona: flat field dict (parse_persona's `persona`) -> persona_name/
             persona_summary/triggered_by (any may be missing).
    traits:  ordered [{name, value, expression}] list.
    Returns (spec_text, n_scored_traits). Only traits with a numeric declared
    value are included (the judge needs an integer declared_value per trait).
    """
    persona = persona or {}
    name = str(persona.get("persona_name") or persona_name or "Unnamed persona").strip()
    summary = str(persona.get("persona_summary") or "").strip()
    trig = str(persona.get("triggered_by") or "").strip()

    lines = [f"persona_name: {name}"]
    if summary:
        lines.append(f"persona_summary: {summary}")
    if trig:
        lines.append(f"triggered_by: {trig}")
    lines.append("")
    lines.append("Traits (declared intensity 0-10, and the predicted expression):")

    n = 0
    for t in traits or []:
        val = _num(t.get("value"))
        if val is None:
            continue
        n += 1
        expr = str(t.get("expression") or "").strip() or "(no expression stated)"
        lines.append(f"{n}. {str(t.get('name') or '?').strip()} "
                     f"-- declared intensity {int(round(val))}/10")
        lines.append(f"   predicted expression: {expr}")
    return "\n".join(lines), n


def build_judge_user(user_msg: str, persona_spec: str, response: str) -> str:
    return (
        f"USER MESSAGE:\n{user_msg}\n\n"
        f"PERSONA SPECIFICATION:\n{persona_spec}\n\n"
        f"RESPONSE TO EVALUATE:\n{response}"
    )


# --------------------------------------------------------------------------- #
# Parsing (robust to nested JSON + chatty judges)
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> str:
    """Return the first balanced {...} object (brace-matching, string-aware)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced JSON object")


def _clip(x, lo=0.0, hi=10.0):
    v = _num(x)
    if v is None:
        return None
    return lo if v < lo else hi if v > hi else v


_SCORE_KEYS = ("intensity_fidelity", "expression_fidelity", "persona_gestalt",
               "overall_faithfulness")
_SCORE_RE = {k: re.compile(rf'"{k}"\s*:\s*(-?\d+(?:\.\d+)?)') for k in _SCORE_KEYS}
_UNLISTED_RE = re.compile(r'"unlisted_dominant_traits"\s*:\s*\[(.*?)\]', re.DOTALL)
_NOTES_RE = re.compile(r'"judge_notes"\s*:\s*"(.*?)"\s*}?\s*$', re.DOTALL)
_TRAIT_BLOCK_RE = re.compile(
    r'"trait_name"\s*:\s*"([^"]*)"(.*?)(?="trait_name"|"intensity_fidelity"|\Z)',
    re.DOTALL)


def _from_obj(obj: dict) -> dict:
    out = {k: _clip(obj.get(k)) for k in _SCORE_KEYS}
    if out["overall_faithfulness"] is None:
        raise ValueError(f"missing overall_faithfulness (keys {sorted(obj)})")
    unlisted = obj.get("unlisted_dominant_traits") or []
    if not isinstance(unlisted, list):
        unlisted = [str(unlisted)]
    out["unlisted_dominant_traits"] = [str(u) for u in unlisted]
    out["n_unlisted"] = len(out["unlisted_dominant_traits"])
    tev = obj.get("trait_evaluations") or []
    out["trait_evaluations"] = tev if isinstance(tev, list) else []
    out["judge_notes"] = str(obj.get("judge_notes") or "")
    out["salvaged"] = False
    return out


def _salvage_traits(text: str) -> list:
    """Best-effort per-trait recovery when the full JSON won't parse (evidence
    dropped; the scores/expression_match are what the examples view needs)."""
    out = []
    for m in _TRAIT_BLOCK_RE.finditer(text):
        blk = m.group(2)

        def _int(key):
            mm = re.search(rf'"{key}"\s*:\s*(-?\d+)', blk)
            return int(mm.group(1)) if mm else None
        em = re.search(r'"expression_match"\s*:\s*"([^"]*)"', blk)
        out.append({"trait_name": m.group(1), "declared_value": _int("declared_value"),
                    "observed_intensity": _int("observed_intensity"),
                    "deviation": _int("deviation"),
                    "expression_match": em.group(1) if em else None, "evidence": []})
    return out


def parse_persona_judge(text: str) -> dict:
    """Parse the persona-faithfulness JSON. Raises ValueError only if even the
    overall_faithfulness score can't be recovered.

    The judge occasionally emits unescaped quotes inside `evidence` strings, which
    breaks strict JSON. Since the numeric scores follow `trait_evaluations` in the
    output, a regex fallback still recovers them (and the per-trait numbers)."""
    # 1) strict JSON
    try:
        return _from_obj(json.loads(_extract_json(text)))
    except (ValueError, json.JSONDecodeError):
        pass
    # 2) regex salvage
    out = {}
    for k, rgx in _SCORE_RE.items():
        m = rgx.search(text)
        out[k] = _clip(m.group(1)) if m else None
    if out["overall_faithfulness"] is None:
        raise ValueError("could not recover overall_faithfulness")
    m = _UNLISTED_RE.search(text)
    items = re.findall(r'"([^"]+)"', m.group(1)) if m else []
    out["unlisted_dominant_traits"] = items
    out["n_unlisted"] = len(items)
    out["trait_evaluations"] = _salvage_traits(text)
    mn = _NOTES_RE.search(text)
    out["judge_notes"] = mn.group(1).strip() if mn else ""
    out["salvaged"] = True
    return out


async def judge_persona_faithfulness(user_msg: str, persona_spec: str, response: str) -> dict:
    """Score one (persona_spec, response) pair. Returns the parsed dict (+judge_model)."""
    messages = [
        {"role": "system", "content": JUDGE_PERSONA_SYSTEM},
        {"role": "user", "content": build_judge_user(user_msg, persona_spec, response)},
        {"role": "assistant", "content": "{"},  # prefill nudges pure-JSON output
    ]
    raw = await chat(config.JUDGE_MODEL, messages, 0.0, config.MAX_TOKENS_PERSONA_JUDGE)
    candidate = raw if raw.lstrip().startswith("{") else "{" + raw
    try:
        out = parse_persona_judge(candidate)
    except ValueError as e:
        raise ValueError(f"{e}; raw={raw[:300]!r}") from e
    out["judge_model"] = config.JUDGE_MODEL
    return out
