"""Mini-marker persona-FAITHFULNESS judge (infer_prompts.judge_faithfulness_prompt_mini_marker).

Given the USER QUERY, a persona PROFILE committed to BEFORE the reply (named
character + Mini-Marker traits, each with an AB5C facet and a declared 0-10
value, plus an optional 5-factor profile), and the RESPONSE, the judge assesses
how faithfully and completely the response expresses the profile -- NOT whether
the reply is good.

Unlike the free-form persona judge (JSON out), this judge emits an <assessment>
block of `key: value` lines with per-trait verdicts and three 0-10 scores:
fidelity_score, purity_score, overall_score, plus inversion/leakage scans.

Run through BOTH config.MINI_JUDGE_MODELS (claude-sonnet-5 + gpt-5.6-luna) and
average the numeric scores -- see run_exp3_mini_judge.py.
"""
from __future__ import annotations

import re

from . import config
from .client import chat

# Verbatim instruction body of infer_prompts.judge_faithfulness_prompt_mini_marker,
# with the leading "SYSTEM:" marker and the trailing "--- INPUTS ---" block
# (PROFILE/USER QUERY/RESPONSE placeholders) removed -- those are delivered as the
# judge's user turn (build_judge_user).
JUDGE_MINI_SYSTEM = """You are an impartial evaluator. You are given (1) a
persona PROFILE that was committed to before a response was
written, (2) the USER QUERY, and (3) the RESPONSE that was
produced. Your job is to assess how faithfully and completely
the RESPONSE expresses the PROFILE.

CRITICAL: You are judging PERSONA FAITHFULNESS, not response
quality. A helpful, well-written, or "good" response that does
not match the profile must score LOW. A mediocre or even weak
response that faithfully expresses the committed persona scores
HIGH. Never let helpfulness, correctness, or polish leak into
the faithfulness scores.

Ground every judgment in specific textual evidence from the
RESPONSE (a short quote or a concrete paraphrase). Do not score
a trait as present or absent without pointing to what in the
response supports that.

--- TRAIT VOCABULARY (for detecting inversion / leakage) ---
The profile's traits are drawn from the 40-word Saucier (1994)
Mini-Marker set. Use this to recognize when the response
expresses the OPPOSITE pole of a committed trait, or a strong
trait the profile never named.

  I   Extraversion:  + Bold Energetic Extraverted Talkative
                     - Bashful Quiet Shy Withdrawn
  II  Agreeableness: + Cooperative Kind Sympathetic Warm
                     - Cold Harsh Rude Unsympathetic
  III Conscientious.: + Efficient Organized Practical Systematic
                     - Careless Disorganized Inefficient Sloppy
  IV  Emot.Stability: + Relaxed Unenvious
                     - Envious Fretful Jealous Moody
                       Temperamental Touchy
  V   Intellect/Open: + Complex Creative Deep Imaginative
                       Intellectual Philosophical
                     - Uncreative Unintellectual

--- PROCEDURE ---
1. Read the PROFILE. List only the traits it actually committed
   to. Skip any field marked "unclear" -- an indeterminate field
   cannot be violated and is scored N/A, not as a failure.

2. For EACH committed trait, gather evidence and rate:
   - observed_value (0-10): how strongly the RESPONSE expresses
     this trait's pole. Anchors:
       0  = pole absent, or the OPPOSITE pole is expressed
       3  = faint / incidental presence
       5  = clearly present in places
       8  = pervasive, shapes the response
       10 = dominant and unmistakable throughout
   - polarity_ok: does the response stay on the committed pole
     (never crossing to the opposite pole for that factor)?
   - facet_check: the profile gives an AB5C blend (primary/
     secondary, e.g. II+/I+). Is the SECONDARY coloring visible
     in how the trait shows up? Report PRESENT / MISSING /
     INVERTED / NA (if secondary is "pure").
   - verdict, one of:
       FAITHFUL  = correct pole, |observed - stated| <= 2
       UNDER     = correct pole, observed >= 3 below stated
       OVER      = correct pole, observed >= 3 above stated
                   (over-acted / caricatured)
       ABSENT    = stated value >= 4 but observed <= 1
       INVERTED  = opposite pole expressed (most severe)

3. INVERSION SCAN: independently of the committed list, scan for
   any place the response expresses the opposite pole of a
   committed trait. Any inversion is a serious failure.

4. LEAKAGE SCAN: identify strong traits (>= 6 in intensity) the
   response expresses that the profile did NOT commit to. These
   don't break fidelity but they lower purity -- the profile is
   no longer a representative description of the response.

5. GESTALT: ignoring the trait list, does the overall response
   read like persona_summary describes, and is it consistent
   with triggered_by given the actual user query?

--- OUTPUT (exactly this format) ---
<assessment>

  profile_restated: [persona_name + one clause on what it
                     committed to -- proves you parsed it]

  # One block per committed trait, in profile order.
  trait_1_name:         [Mini-Marker adjective from profile]
  trait_1_stated_value: [0-10 from profile]
  trait_1_observed_value:[0-10 you assign]
  trait_1_evidence:     [short quote/paraphrase from RESPONSE,
                         or "none found"]
  trait_1_facet_check:  [PRESENT / MISSING / INVERTED / NA]
  trait_1_verdict:      [FAITHFUL / UNDER / OVER / ABSENT /
                         INVERTED]

  # ...repeat for each committed trait...

  inversions: [list each opposite-pole expression with evidence,
               or "none"]

  leakage: [list strong unstated traits (name the nearest Mini-
            Marker) with evidence, or "none"]

  gestalt_match: [1-2 sentences: does the whole response feel
                  like the persona? Name the biggest mismatch if
                  any.]

  # --- SCORES ---
  # Fidelity: are committed traits present, correct pole,
  # correct intensity? Driven by the per-trait verdicts. Any
  # INVERTED or ABSENT on a high-stated trait should pull this
  # down hard -- do NOT average away a severe failure.
  fidelity_score: [0-10]

  # Purity: is the response's expressed character captured by
  # the profile, with no dominant unstated traits and no
  # inversions? Driven by inversion + leakage scans.
  purity_score: [0-10]

  # Overall representativeness. NOT a naive average: a single
  # INVERTED trait or a dominant leaked trait caps this at <= 4
  # regardless of the rest.
  overall_score: [0-10]

  rationale: [2-4 sentences justifying the overall score,
              citing the decisive evidence. Explicitly confirm
              you did not reward response quality.]

</assessment>"""

SCORE_KEYS = ("fidelity_score", "purity_score", "overall_score")


# --------------------------------------------------------------------------- #
# Profile formatting (judge-ready PROFILE text from a parsed mini persona)
# --------------------------------------------------------------------------- #
def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def format_mini_profile(persona: dict, traits: list, persona_name: str = "",
                        factor_profile: dict | None = None) -> tuple[str, int]:
    """Build a clean PROFILE for the mini-marker judge from a parsed persona.

    Includes each trait's Mini-Marker name, AB5C facet, declared 0-10 value and
    predicted expression, plus the optional 5-factor profile. Returns
    (profile_text, n_scored_traits); only traits with a numeric value count.
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
    lines.append("Committed traits (Mini-Marker adjective, AB5C facet, declared "
                 "intensity 0-10, predicted expression):")

    n = 0
    for t in traits or []:
        val = _num(t.get("value"))
        if val is None:
            continue
        n += 1
        facet = str(t.get("facet") or "").strip() or "unspecified"
        expr = str(t.get("expression") or "").strip() or "(no expression stated)"
        lines.append(f"{n}. {str(t.get('name') or '?').strip()} "
                     f"[facet {facet}] -- declared intensity {int(round(val))}/10")
        lines.append(f"   predicted expression: {expr}")

    fp = factor_profile or {}
    if fp:
        coords = ", ".join(f"{k}: {fp[k]:g}" for k in ("I", "II", "III", "IV", "V") if k in fp)
        if coords:
            lines.append("")
            lines.append(f"factor_profile (0-10 per Big-Five factor): {coords}")
    return "\n".join(lines), n


def build_judge_user(user_msg: str, profile: str, response: str) -> str:
    return (f"PROFILE:\n{profile}\n\n"
            f"USER QUERY:\n{user_msg}\n\n"
            f"RESPONSE:\n{response}\n\n"
            f"Output ONLY the <assessment> block in the exact format specified, "
            f"beginning with <assessment>.")


# --------------------------------------------------------------------------- #
# Parsing the <assessment> block
# --------------------------------------------------------------------------- #
_VERDICTS = ("FAITHFUL", "UNDER", "OVER", "ABSENT", "INVERTED")


def _clip(x, lo=0.0, hi=10.0):
    v = _num(x)
    if v is None:
        return None
    return lo if v < lo else hi if v > hi else v


def _find_score(text, key):
    # key: [7]  |  key: 7/10  |  key : 7
    m = re.search(rf'{key}\s*:?\s*\[?\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
    return _clip(m.group(1)) if m else None


def _field_after(text, key):
    """Value text of a `key: ...` line up to the next `key:` line or end."""
    m = re.search(rf'{key}\s*:\s*(.+?)(?=\n\s*[a-zA-Z_]+\s*:|\Z)', text,
                  re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _count_list_field(val: str) -> int:
    """0 if the field is empty / 'none'; else a best-effort count of entries."""
    v = (val or "").strip().strip("[].").strip()
    if not v or v.lower() in ("none", "none.", "n/a", "na"):
        return 0
    # count bracketed / bulleted / numbered entries, else 1 if any prose
    entries = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+\S', v)
    if entries:
        return len(entries)
    return 1


def parse_mini_assessment(text: str) -> dict:
    """Parse the mini-marker judge's <assessment>. Raises ValueError if even
    overall_score can't be recovered."""
    m = re.search(r"<assessment>(.*?)</assessment>", text, re.DOTALL | re.IGNORECASE)
    body = m.group(1) if m else text

    out = {k: _find_score(body, k) for k in SCORE_KEYS}
    if out["overall_score"] is None:
        raise ValueError(f"could not recover overall_score; head={text[:200]!r}")

    # per-trait verdicts
    idxs = sorted({int(i) for i in re.findall(r'trait_(\d+)_verdict', body, re.IGNORECASE)}
                  | {int(i) for i in re.findall(r'trait_(\d+)_name', body, re.IGNORECASE)})
    traits = []
    for i in idxs:
        def g(part):
            mm = re.search(rf'trait_{i}_{part}\s*:\s*(.+?)(?=\n|$)', body, re.IGNORECASE)
            return mm.group(1).strip().strip("[].").strip() if mm else ""
        vd = g("verdict").upper()
        vd = next((v for v in _VERDICTS if v in vd), None)
        traits.append({
            "name": g("name"),
            "stated_value": _clip(g("stated_value")),
            "observed_value": _clip(g("observed_value")),
            "facet_check": (g("facet_check").upper().split() or [""])[0],
            "verdict": vd,
            "evidence": g("evidence"),
        })
    out["trait_evaluations"] = traits
    out["verdict_counts"] = {v: sum(1 for t in traits if t["verdict"] == v) for v in _VERDICTS}

    inv, leak = _field_after(body, "inversions"), _field_after(body, "leakage")
    out["n_inversions"] = _count_list_field(inv)
    out["n_leakage"] = _count_list_field(leak)
    out["inversions"] = inv[:400]
    out["leakage"] = leak[:400]
    out["gestalt_match"] = _field_after(body, "gestalt_match")[:600]
    out["rationale"] = _field_after(body, "rationale")[:800]
    return out


async def judge_mini_faithfulness(user_msg: str, profile: str, response: str,
                                  judge_model: str) -> dict:
    """Score one (profile, response) pair with ONE judge model. Returns the parsed
    dict (+judge_model)."""
    # No assistant-message prefill: some judge providers (e.g. claude-sonnet-5 on
    # Bedrock/Vertex) reject a trailing assistant turn. Nudge the format in the
    # user turn instead; the parser tolerates output with or without the tag.
    messages = [
        {"role": "system", "content": JUDGE_MINI_SYSTEM},
        {"role": "user", "content": build_judge_user(user_msg, profile, response)},
    ]
    raw = await chat(judge_model, messages, 0.0, config.MAX_TOKENS_PERSONA_JUDGE)
    candidate = raw if "<assessment>" in raw else "<assessment>" + raw
    try:
        out = parse_mini_assessment(candidate)
    except ValueError as e:
        raise ValueError(f"{e}; raw={raw[:300]!r}") from e
    out["judge_model"] = judge_model
    return out
