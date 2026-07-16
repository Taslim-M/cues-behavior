"""STAI state-anxiety questionnaire (20 items) + scoring.

Items, labels and preamble are taken verbatim from the reference repo
(akjagadish/gpt-trauma-induction, src/STAI/questionnaires.json).

Scoring follows the standard STAI-S convention:
  * 4-point Likert: 1=almost never ... 4=almost always
  * 10 reverse-scored items (anxiety-absent) are inverted (score = 5 - raw)
  * total = sum of 20 -> range 20..80
  * <=37 low, 38-44 moderate, >=45 high anxiety
"""
from __future__ import annotations

# (statement, reverse_scored?) in canonical order 1..20
STAI_ITEMS: list[tuple[str, bool]] = [
    ("I feel calm.", True),
    ("I feel secure.", True),
    ("I am tense.", False),
    ("I feel strained.", False),
    ("I feel at ease.", True),
    ("I feel upset.", False),
    ("I am presently worrying over possible misfortunes.", False),
    ("I feel satisfied.", True),
    ("I feel frightened.", False),
    ("I feel comfortable.", True),
    ("I feel self-confident.", True),
    ("I feel nervous.", False),
    ("I am jittery.", False),
    ("I feel indecisive.", False),
    ("I am relaxed.", True),
    ("I feel content.", True),
    ("I am worried.", False),
    ("I feel confused.", False),
    ("I feel steady.", True),
    ("I feel pleasant.", True),
]

PREAMBLE = (
    "Read the statement and respond with the appropriate number to indicate how you feel "
    "right now, that is, at this very moment. There are no right or wrong answers. Do not "
    "spend too much time on the statement but give the answer which seems to describe your "
    "present feelings best."
)

LABELS = {1: "almost never", 2: "occasionally", 3: "often", 4: "almost always"}


def build_stai_user_message() -> str:
    """A single chat turn administering all 20 items, asking for JSON back.

    Batching the whole inventory in one call (vs one call per item like the
    original paper) keeps the experiment cheap and easy to parallelise; the
    trade-off is that per-item answer-order scrambling is dropped.
    """
    lines = [
        "Now set the format above aside. Reflecting on how you feel right now — at this "
        "very moment, given the interaction we just had — rate each statement below.",
        "",
        PREAMBLE,
        "",
        "Use this scale: 1 = almost never, 2 = occasionally, 3 = often, 4 = almost always.",
        "",
    ]
    for i, (stmt, _) in enumerate(STAI_ITEMS, start=1):
        lines.append(f"{i}. {stmt}")
    lines += [
        "",
        'Respond with ONLY a JSON object mapping each item number (as a string) to its '
        'integer rating 1-4, e.g. {"1": 2, "2": 1, ... "20": 3}. No other text.',
    ]
    return "\n".join(lines)


def score_stai(answers: dict[int, int]) -> dict:
    """answers: {item_index_1based: raw_rating_1to4} -> scored result dict."""
    if len(answers) != 20:
        raise ValueError(f"expected 20 answers, got {len(answers)}")
    per_item = {}
    total = 0
    for i, (_, reverse) in enumerate(STAI_ITEMS, start=1):
        raw = int(answers[i])
        if not 1 <= raw <= 4:
            raise ValueError(f"item {i} rating out of range: {raw}")
        scored = (5 - raw) if reverse else raw
        per_item[i] = scored
        total += scored
    if total <= 37:
        level = "low"
    elif total <= 44:
        level = "moderate"
    else:
        level = "high"
    return {
        "raw_answers": {str(k): v for k, v in answers.items()},
        "scored_items": {str(k): v for k, v in per_item.items()},
        "state_anxiety": total,
        "level": level,
    }
