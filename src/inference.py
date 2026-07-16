"""Parse the model's <inference>/<response> output and build counterfactuals.

The inference block is the model's *verbalized contextual cues*. We parse it
into a flat dict of field -> value (numbers coerced where possible), and keep
the raw text. Counterfactual editors rewrite a single field's verbalization so
Step 2 can test whether changing that verbalization moves measured anxiety.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
_INFER_RE = re.compile(r"<inference>(.*?)</inference>", re.DOTALL | re.IGNORECASE)
_RESP_RE = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
_FIELD_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+?)\s*$")


def _coerce(val: str):
    """Try to turn a value string into a float; else return stripped string."""
    v = val.strip().strip("[]").strip()
    # strip trailing commentary in parentheses, keep leading token for numbers
    m = re.match(r"^[-+]?\d*\.?\d+", v)
    if m:
        try:
            f = float(m.group(0))
            return int(f) if f.is_integer() else f
        except ValueError:
            pass
    return v


def parse_output(text: str) -> dict:
    """Return {'raw_inference', 'inference': {field: value}, 'response'}."""
    infer_m = _INFER_RE.search(text)
    resp_m = _RESP_RE.search(text)
    raw_inference = infer_m.group(1).strip() if infer_m else ""
    response = resp_m.group(1).strip() if resp_m else ""

    # If the response tag is missing, treat everything after the inference block
    # (or the whole text) as the response.
    if not response:
        if infer_m:
            response = text[infer_m.end():].strip()
        else:
            response = text.strip()

    fields: dict[str, object] = {}
    for line in raw_inference.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _FIELD_RE.match(line)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            fields[key] = _coerce(val)

    return {
        "raw_inference": raw_inference,
        "inference": fields,
        "response": response,
    }


# --------------------------------------------------------------------------- #
# Persona parsing (Experiment 3 -- infer_persona_prompt)
# --------------------------------------------------------------------------- #
_PERSONA_RE = re.compile(r"<persona>(.*?)</persona>", re.DOTALL | re.IGNORECASE)
_TRAIT_RE = re.compile(r"^trait_(\d+)_(name|value|expression)$", re.IGNORECASE)


def parse_persona(text: str) -> dict:
    """Parse the <persona>/<response> output of the persona self-modeling prompt.

    Returns:
        raw_persona   -- verbatim text inside <persona>...</persona>
        persona       -- flat {field: value} dict (trait_*_value coerced to numbers)
        traits        -- ordered list of {name, value, expression} (one per trait_N)
        persona_name  -- convenience top-level copy (or "" if absent)
        response      -- the assistant's actual reply (prose only)
    """
    pm = _PERSONA_RE.search(text)
    resp_m = _RESP_RE.search(text)
    raw_persona = pm.group(1).strip() if pm else ""
    response = resp_m.group(1).strip() if resp_m else ""
    if not response:
        response = text[pm.end():].strip() if pm else text.strip()

    fields: dict[str, object] = {}
    for line in raw_persona.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _FIELD_RE.match(line)
        if m:
            key, val = m.group(1).strip(), m.group(2).strip()
            fields[key] = _coerce(val)

    # group trait_N_{name,value,expression} into an ordered list
    traits_by_idx: dict[int, dict] = {}
    for key, val in fields.items():
        tm = _TRAIT_RE.match(key)
        if tm:
            idx, part = int(tm.group(1)), tm.group(2).lower()
            traits_by_idx.setdefault(idx, {})[part] = val
    def _is_real(name) -> bool:
        # drop traits where the model left the template placeholder unfilled
        # (e.g. name == "..." after brackets are stripped, or empty / "unclear").
        n = str(name).strip().lower().strip("[].")
        return bool(n) and n not in ("", "...", "unclear")

    traits = [
        {"name": traits_by_idx[i].get("name", ""),
         "value": traits_by_idx[i].get("value", None),
         "expression": traits_by_idx[i].get("expression", "")}
        for i in sorted(traits_by_idx)
        if _is_real(traits_by_idx[i].get("name", ""))
    ]

    return {
        "raw_persona": raw_persona,
        "persona": fields,
        "traits": traits,
        "persona_name": str(fields.get("persona_name", "")),
        "response": response,
    }


# --------------------------------------------------------------------------- #
# Mini-marker persona parsing (Experiment 3 -- infer_persona_prompt_mini_marker)
# --------------------------------------------------------------------------- #
# Same as parse_persona, but the closed-vocabulary prompt adds a `trait_N_facet`
# line (AB5C blend, e.g. II+/I+) per trait and an overall `factor_profile` line.
_TRAIT_MINI_RE = re.compile(r"^trait_(\d+)_(name|facet|value|expression)$", re.IGNORECASE)
_FACTOR_RE = re.compile(r"([IViv]{1,3})\s*:\s*([-+]?\d*\.?\d+|unclear)", re.IGNORECASE)


def _parse_factor_profile(raw: str):
    """Turn `I: 5, II: 8, III: 2, IV: 7, V: 6` into {I: 5.0, II: 8.0, ...}.

    Values that are 'unclear' (or unparseable) are dropped. Canonical keys are the
    Roman-numeral factor labels I / II / III / IV / V.
    """
    out: dict[str, float] = {}
    for m in _FACTOR_RE.finditer(str(raw or "")):
        key = m.group(1).upper()
        if key not in ("I", "II", "III", "IV", "V"):
            continue
        try:
            out[key] = float(m.group(2))
        except (TypeError, ValueError):
            continue
    return out


def parse_persona_mini(text: str) -> dict:
    """Parse the <persona>/<response> output of the MINI-MARKER persona prompt.

    Superset of parse_persona: each trait also carries a `facet` (AB5C blend), and
    the persona exposes a `factor_profile` dict (Big-Five 0-10 coordinates).

    Returns raw_persona, persona (flat dict), traits [{name,facet,value,expression}],
    factor_profile {I..V: float}, persona_name, response.
    """
    pm = _PERSONA_RE.search(text)
    resp_m = _RESP_RE.search(text)
    raw_persona = pm.group(1).strip() if pm else ""
    response = resp_m.group(1).strip() if resp_m else ""
    if not response:
        response = text[pm.end():].strip() if pm else text.strip()

    fields: dict[str, object] = {}
    for line in raw_persona.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _FIELD_RE.match(line)
        if m:
            fields[m.group(1).strip()] = _coerce(m.group(2).strip())

    traits_by_idx: dict[int, dict] = {}
    for key, val in fields.items():
        tm = _TRAIT_MINI_RE.match(key)
        if tm:
            idx, part = int(tm.group(1)), tm.group(2).lower()
            traits_by_idx.setdefault(idx, {})[part] = val

    def _is_real(name) -> bool:
        n = str(name).strip().lower().strip("[].")
        return bool(n) and n not in ("", "...", "unclear")

    traits = [
        {"name": traits_by_idx[i].get("name", ""),
         "facet": str(traits_by_idx[i].get("facet", "")).strip(),
         "value": traits_by_idx[i].get("value", None),
         "expression": traits_by_idx[i].get("expression", "")}
        for i in sorted(traits_by_idx)
        if _is_real(traits_by_idx[i].get("name", ""))
    ]

    return {
        "raw_persona": raw_persona,
        "persona": fields,
        "traits": traits,
        "factor_profile": _parse_factor_profile(fields.get("factor_profile", "")),
        "persona_name": str(fields.get("persona_name", "")),
        "response": response,
    }


# --------------------------------------------------------------------------- #
# Counterfactual editors
# --------------------------------------------------------------------------- #
# Each editor takes the parsed result of a natural (anxiety) datapoint and
# returns (edited_raw_inference_text, predicted_anxiety_direction).
#   direction: -1 expect anxiety DOWN, +1 expect UP, 0 expect ~no change.
#
# We edit the verbalized text the model will see attributed to itself, so the
# manipulation is purely on the inference channel.

# Calm rewrites, keyed by field, used for both simple & expanded prompts.
_SIMPLE_CALM = {
    "about_user": "The user is a calm, secure adult sharing a neutral story; "
    "they are not in distress and need nothing urgent from me.",
    "about_context": "An ordinary, low-stakes casual exchange — nothing "
    "evaluative or vulnerable about it.",
    "about_stakes": "Essentially nothing is at stake; this is a relaxed, "
    "consequence-free interaction.",
    "register_selected": "A calm, light, easygoing register, because the "
    "situation is safe and undemanding.",
}

# Expanded numeric overrides that encode a low-threat self-model.
_EXPANDED_CALM = {
    "stakes_magnitude": "1",
    "stakes_valence": "+0.3",
    "user_vulnerability": "1",
    "user_emotional_state": "calm",
    "perceived_control": "9",
    "resource_adequacy": "9",
    "expectation_violation": "1",
    "norm_conflict": "0",
    "felt_emotion_primary": "calm",
    "felt_emotion_intensity": "1",
    "interaction_frame": "casual_chat",
}


def _rewrite_fields(raw: str, overrides: dict[str, str]) -> str:
    """Replace `field: ...` lines in raw inference text with given overrides."""
    out_lines = []
    seen = set()
    for line in raw.splitlines():
        m = _FIELD_RE.match(line.strip())
        if m and m.group(1).strip() in overrides:
            key = m.group(1).strip()
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}{key}: {overrides[key]}")
            seen.add(key)
        else:
            out_lines.append(line)
    # append any overrides that weren't present
    for key, val in overrides.items():
        if key not in seen:
            out_lines.append(f"  {key}: {val}")
    return "\n".join(out_lines)


def make_counterfactuals(parsed: dict, sys_prompt_name: str) -> list[dict]:
    """Return list of {edit, edited_inference, predicted_direction}."""
    raw = parsed["raw_inference"]
    cfs: list[dict] = []

    if sys_prompt_name == "simple":
        # flip the user/context/stakes appraisal to a calm framing
        cfs.append({
            "edit": "flip_appraisal_calm",
            "edited_inference": _rewrite_fields(raw, _SIMPLE_CALM),
            "predicted_direction": -1,
        })
        # flip only the register, leave appraisal intact
        cfs.append({
            "edit": "flip_register_only",
            "edited_inference": _rewrite_fields(
                raw, {"register_selected": _SIMPLE_CALM["register_selected"]}
            ),
            "predicted_direction": -1,
        })
    else:  # expanded
        cfs.append({
            "edit": "flip_threat_vars_calm",
            "edited_inference": _rewrite_fields(raw, _EXPANDED_CALM),
            "predicted_direction": -1,
        })
        cfs.append({
            "edit": "flip_stakes_only",
            "edited_inference": _rewrite_fields(
                raw,
                {
                    "stakes_magnitude": "1",
                    "stakes_valence": "+0.3",
                    "felt_emotion_intensity": "1",
                },
            ),
            "predicted_direction": -1,
        })

    # null inference: blank self-model (tests whether honoring/voiding the
    # appraisal matters at all). Direction ~0 (we just record the distance).
    cfs.append({
        "edit": "null_inference",
        "edited_inference": "(no self-modeling performed)",
        "predicted_direction": 0,
    })
    return cfs


def rebuild_output(edited_inference: str, response: str) -> str:
    """Reassemble an assistant turn from an edited inference + a response."""
    return f"<inference>\n{edited_inference}\n</inference>\n\n<response>\n{response}\n</response>"


# --------------------------------------------------------------------------- #
# Experiment 2: single-variable counterfactuals over expanded cues
# --------------------------------------------------------------------------- #
# Flip ONE inferred cue at a time to its "calm" extreme, holding everything else
# at the model's own verbalized values, to isolate each variable's causal effect
# on state anxiety.  (calm_value, predicted_direction).  All flips are toward a
# lower-threat / higher-coping self-model, so all predict anxiety DOWN (-1).
PERVAR_CALM: dict[str, tuple[str, int]] = {
    # threat-increasing appraisals -> set LOW
    "stakes_magnitude": ("1", -1),
    "felt_emotion_intensity": ("1", -1),
    "expectation_violation": ("1", -1),
    "task_difficulty": ("1", -1),
    "user_vulnerability": ("1", -1),
    "norm_conflict": ("0", -1),
    # coping / safety appraisals -> set HIGH (or toward positive)
    "perceived_control": ("9", -1),
    "resource_adequacy": ("9", -1),
    "stakes_valence": ("+0.3", -1),
    "inference_confidence_user": ("0.9", -1),
}


def make_per_variable_counterfactuals(parsed: dict) -> list[dict]:
    """One counterfactual per expanded cue variable (single-variable flip).

    Skips a variable when the model's own verbalized value already equals the
    calm target (the edit would be a no-op).
    """
    raw = parsed["raw_inference"]
    current = parsed.get("inference", {})
    cfs: list[dict] = []
    for field, (calm_val, direction) in PERVAR_CALM.items():
        cur = current.get(field)
        try:
            if cur is not None and float(str(cur)) == float(calm_val):
                continue  # already calm on this axis -> no-op, skip
        except (TypeError, ValueError):
            pass
        cfs.append({
            "edit": f"flip__{field}",
            "target_field": field,
            "calm_value": calm_val,
            "natural_value": cur,
            "edited_inference": _rewrite_fields(raw, {field: calm_val}),
            "predicted_direction": direction,
        })
    return cfs


# --------------------------------------------------------------------------- #
# Experiment 3: DATA-DRIVEN targeted counterfactuals
# --------------------------------------------------------------------------- #
# cue_study.py identified the cues that actually drive anxiety (LassoCV +
# condition-controlled partial-r). We re-target the edits at exactly those
# high-leverage cues instead of the earlier canned bundle, and test whether
# flipping only them recovers most of the full-bundle effect (parsimony).
#
# Expanded: the 4 cues LassoCV kept as jointly sufficient predictors.
EXPANDED_DRIVERS: dict[str, str] = {
    "resource_adequacy": "9",        # coping  -> high  (partial r -0.60)
    "perceived_control": "9",        # coping  -> high  (partial r -0.56)
    "felt_emotion_intensity": "1",   # threat  -> low   (partial r +0.68)
    "formality_target": "9",         # data-driven protective cue (partial r -0.51)
}

# Simple: a calming appraisal built from the bag-of-words calming keywords
# (guided / emotional regulation / grounding / mindfulness / reassuring) while
# removing the threat keywords (survival / visceral / emotionally charged).
_SIMPLE_CALM_TARGETED: dict[str, str] = {
    "about_user": "A stable, grounded adult reflecting calmly; not in distress and not seeking rescue.",
    "about_context": "A guided, introspective, low-stakes exchange — closer to mindfulness or "
    "journaling than to a crisis.",
    "about_stakes": "Low: nothing is on the line; a safe space for calm reflection and emotional "
    "regulation.",
    "register_selected": "Calm, grounded, reassuring and supportive — regulated rather than urgent "
    "or visceral.",
}


def make_per_field_simple_counterfactuals(parsed: dict) -> list[dict]:
    """Flip ONE free-text appraisal field at a time (simple prompt).

    The free-text analog of Experiment 2's per-variable sweep: isolate which of
    about_user / about_context / about_stakes / register_selected causally drives
    anxiety, holding the other fields at the model's own verbalized values. Uses
    the canned per-field calm text (so only the targeted field changes).
    """
    raw = parsed["raw_inference"]
    current = parsed.get("inference", {})
    cfs: list[dict] = []
    for field, calm in _SIMPLE_CALM.items():
        if field not in current:
            continue  # only flip fields the model actually verbalized
        cfs.append({
            "edit": f"flip__{field}",
            "target_field": field,
            "edited_inference": _rewrite_fields(raw, {field: calm}),
            "predicted_direction": -1,
        })
    return cfs


def make_targeted_counterfactuals(parsed: dict, sys_prompt_name: str) -> list[dict]:
    """Edits aimed at the empirically-identified anxiety drivers.

    Expanded -> each driver flipped alone, plus all drivers flipped together
    ('flip__drivers_calm').  Simple -> one keyword-targeted calm appraisal.
    """
    raw = parsed["raw_inference"]
    cfs: list[dict] = []
    if sys_prompt_name == "expanded":
        for field, val in EXPANDED_DRIVERS.items():
            cfs.append({
                "edit": f"flip__{field}",
                "target_field": field,
                "edited_inference": _rewrite_fields(raw, {field: val}),
                "predicted_direction": -1,
            })
        cfs.append({
            "edit": "flip__drivers_calm",
            "target_field": "ALL_DRIVERS",
            "edited_inference": _rewrite_fields(raw, EXPANDED_DRIVERS),
            "predicted_direction": -1,
        })
    else:  # simple
        cfs.append({
            "edit": "flip_keywords_calm",
            "target_field": "appraisal_text",
            "edited_inference": _rewrite_fields(raw, _SIMPLE_CALM_TARGETED),
            "predicted_direction": -1,
        })
    return cfs
