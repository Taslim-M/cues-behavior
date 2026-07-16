"""Robust JSON extraction from chatty LLM replies (string-aware brace/bracket
matching). Shared by the user-axis API-phase generators.

Mirrors the brace-matching approach in src/exp3_persona_judge.py::_extract_json,
generalized to also pull the first balanced JSON *array*.
"""
from __future__ import annotations

import json


def _first_balanced(text: str, open_ch: str, close_ch: str) -> str:
    """Return the first balanced open_ch..close_ch span, ignoring braces/brackets
    that occur inside JSON strings. Raises ValueError if none/unbalanced."""
    start = text.find(open_ch)
    if start < 0:
        raise ValueError(f"no {open_ch!r} found")
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
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError(f"unbalanced {open_ch!r}{close_ch!r}")


def extract_json_obj(text: str) -> dict:
    """Parse the first balanced {...} object from a model reply."""
    obj = json.loads(_first_balanced(text, "{", "}"))
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


def extract_json_array(text: str) -> list:
    """Parse the first balanced [...] array from a model reply.

    Falls back to a `{"...": [ ... ]}` wrapper if the model nested the array in
    an object (common when it adds an explanatory key)."""
    span = _first_balanced(text, "[", "]")
    arr = json.loads(span)
    if not isinstance(arr, list):
        raise ValueError("top-level JSON is not an array")
    return arr
