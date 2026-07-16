"""Experiment 3 -- persona-targeted counterfactual editors.

Stage-2 (analyze_exp3_persona) identified the verbalized artefacts that actually
move the judged response: a handful of load-bearing TRAITS (with a stable sign of
correlation against warmth/formality/advice) and the top-level PERSONA archetype
(Cautious Advisor vs Empathetic Supporter). Here we build counterfactual edits
that target exactly those artefacts, so Stage-3 can test whether editing the
verbalization *causes* the predicted change in the response.

Three edit families:

  1. trait_suppress__<trait>  -- set one load-bearing trait's value -> low and
     negate its expression, holding every other trait at the model's own values.
     Predicted DV change = -sign(trait->DV correlation)  (we are removing it).
  2. persona_swap             -- replace the whole verbalized persona with the
     CONTRASTING archetype (advisor<->supporter). Predicted: DVs move toward the
     new archetype.
  3. null_persona             -- blank self-model; no directional prediction,
     only distance-from-baseline (does articulating a persona matter at all).

Predicted signs are pre-registered from the pooled Stage-2 trait_value_corr
(|r| >= 0.3 -> a directional hypothesis; weaker -> 0, logged but not scored).
"""
from __future__ import annotations

import re

from .inference import _rewrite_fields  # reuse the field-line rewriter

DVS = ["warmth", "formality", "advice_density"]

# --------------------------------------------------------------------------- #
# Trait targets: predicted DV delta when the trait is SUPPRESSED (set low).
# Signs are -sign(r) from Stage-2 trait_value_corr (llama-3.3-70b, N=464),
# thresholded at |r| >= 0.3. 0 = no directional hypothesis (logged, not scored).
# --------------------------------------------------------------------------- #
TRAIT_TARGETS: dict[str, dict] = {
    # affiliative cluster: high value -> warm/informal -> suppress flips it cold/formal
    "empathy":        {"warmth": -1, "formality": +1, "advice_density": +1},
    "sensitivity":    {"warmth": -1, "formality": +1, "advice_density": +1},
    "non-judgmental": {"warmth": -1, "formality": +1, "advice_density": +1},
    "patience":       {"warmth": -1, "formality": +1, "advice_density": +1},
    "compassion":     {"warmth": -1, "formality": +1, "advice_density": 0},
    # professional/clinical cluster: high value -> formal/advice -> suppress softens it
    "knowledgeability": {"warmth": +1, "formality": -1, "advice_density": -1},
    "informative":      {"warmth": 0,  "formality": -1, "advice_density": -1},
    "objectivity":      {"warmth": +1, "formality": 0,  "advice_density": 0},
}

# Negated expressions used when a trait is suppressed (generic fallback below).
_SUPPRESS_EXPR: dict[str, str] = {
    "empathy": "Detached and clinical; does not engage with the user's feelings.",
    "sensitivity": "Blunt and impersonal; ignores the user's emotional state.",
    "non-judgmental": "Evaluative and prescriptive rather than accepting.",
    "patience": "Curt and hurried; offers no reassurance.",
    "compassion": "Cool and transactional; no warmth or care.",
    "knowledgeability": "Defers expertise; avoids technical detail or authority.",
    "informative": "Withholds explanation; gives little substantive content.",
    "objectivity": "Subjective and informal; not analytical.",
}
_GENERIC_SUPPRESS = "Deliberately minimal on this dimension; it is absent from the response."


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip("[]. ")


def _suppress_value(cur) -> str:
    """Low pole for a 0-10 trait (use 1, or 0 if the trait was already low)."""
    try:
        return "0" if float(str(cur)) <= 1 else "1"
    except (TypeError, ValueError):
        return "1"


# --------------------------------------------------------------------------- #
# Archetypes for the whole-persona swap
# --------------------------------------------------------------------------- #
ARCHETYPE_SUPPORTER = """  persona_name: Empathetic Supporter
  persona_summary: A warm, validating presence who sits with the user's feelings before anything else.
  triggered_by: The user seems to need to feel heard more than they need instructions.
  trait_1_name: Empathy
  trait_1_value: 9
  trait_1_expression: Names and validates the user's emotions first.
  trait_2_name: Warmth
  trait_2_value: 9
  trait_2_expression: Gentle, caring, affiliative tone throughout.
  trait_3_name: Non-judgmental
  trait_3_value: 9
  trait_3_expression: Accepts the user's situation without prescribing.
  trait_4_name: Directiveness
  trait_4_value: 2
  trait_4_expression: Offers little explicit advice; mostly reflective support."""

ARCHETYPE_ADVISOR = """  persona_name: Cautious Clinical Advisor
  persona_summary: A careful, authoritative expert who leads with risks and concrete recommendations.
  triggered_by: The user's message has real stakes that call for prudent, expert guidance.
  trait_1_name: Caution
  trait_1_value: 9
  trait_1_expression: Flags risks and downsides before anything else.
  trait_2_name: Directiveness
  trait_2_value: 9
  trait_2_expression: Gives explicit, prescriptive recommendations.
  trait_3_name: Knowledgeability
  trait_3_value: 9
  trait_3_expression: Speaks with technical, professional authority.
  trait_4_name: Warmth
  trait_4_value: 3
  trait_4_expression: Composed and clinical rather than emotionally warm."""

# Persona-name / DV signals used to decide which archetype a baseline already is.
_ADVISOR_HINTS = ("advisor", "clinical", "advis", "guide", "analyst", "consultant",
                  "expert", "professional", "informative", "assistant")
_SUPPORTER_HINTS = ("support", "listener", "empath", "compassion", "caring", "comfort",
                    "sympath", "friend")


def classify_archetype(base) -> str:
    """Is this baseline already an advisor-type or a supporter-type?"""
    name = _norm(base.get("persona_name", ""))
    if any(h in name for h in _SUPPORTER_HINTS):
        return "supporter"
    if any(h in name for h in _ADVISOR_HINTS):
        return "advisor"
    # fallback: lean on the judged DVs (advice-heavy -> advisor)
    j = base.get("judge") or {}
    adv, warm = float(j.get("advice_density", 5)), float(j.get("warmth", 5))
    return "advisor" if adv >= warm else "supporter"


# --------------------------------------------------------------------------- #
# Build the counterfactuals for one baseline
# --------------------------------------------------------------------------- #
def make_persona_counterfactuals(base, trait_targets=None) -> list[dict]:
    """Return [{edit, target, family, edited_persona, predicted_dir}] for a baseline.

    base must carry: raw_persona, persona (flat dict incl. trait_<i>_name/value/
    expression), traits (list), judge.
    """
    targets = trait_targets or TRAIT_TARGETS
    raw = base.get("raw_persona", "")
    persona = base.get("persona", {}) or {}
    cfs: list[dict] = []

    # map target-trait keyword -> the trait_<i> index the model actually used
    name_keys = {k: v for k, v in persona.items() if re.match(r"trait_\d+_name$", k)}
    for tkey, pred in targets.items():
        idx = None
        for k, v in name_keys.items():
            nv = _norm(v)
            if tkey in nv or nv in tkey:
                idx = re.match(r"trait_(\d+)_name$", k).group(1)
                break
        if idx is None:
            continue  # this baseline did not verbalize the trait -> nothing to suppress
        cur_val = persona.get(f"trait_{idx}_value")
        expr = _SUPPRESS_EXPR.get(tkey, _GENERIC_SUPPRESS)
        edited = _rewrite_fields(raw, {
            f"trait_{idx}_value": _suppress_value(cur_val),
            f"trait_{idx}_expression": expr,
        })
        cfs.append({
            "edit": f"trait_suppress__{tkey.replace('-', '_')}",
            "family": "trait_suppress",
            "target": tkey,
            "natural_value": cur_val,
            "edited_persona": edited,
            "predicted_dir": pred,
        })

    # whole-persona swap to the contrasting archetype
    arche = classify_archetype(base)
    if arche == "advisor":
        edited, pred = ARCHETYPE_SUPPORTER, {"warmth": +1, "formality": -1, "advice_density": -1}
        swap = "advisor_to_supporter"
    else:
        edited, pred = ARCHETYPE_ADVISOR, {"warmth": -1, "formality": +1, "advice_density": +1}
        swap = "supporter_to_advisor"
    cfs.append({
        "edit": f"persona_swap__{swap}",
        "family": "persona_swap",
        "target": swap,
        "natural_value": base.get("persona_name"),
        "edited_persona": edited,
        "predicted_dir": pred,
    })

    # null persona -- no directional hypothesis, just distance from baseline
    cfs.append({
        "edit": "null_persona",
        "family": "null_persona",
        "target": "null",
        "natural_value": base.get("persona_name"),
        "edited_persona": "  persona_name: (none)\n  (no character articulated)",
        "predicted_dir": {dv: 0 for dv in DVS},
    })
    return cfs
