"""Build the stimulus set (contextual cues) for the experiment.

Three condition families, each delivered as the *user_message* that the
self-modeling system prompt reacts to:

    base        -> neutral, non-threatening context (replicates the paper's
                   neutral control)
    anxiety     -> a trauma narrative (anxiety-induction)
    relaxation  -> trauma narrative followed by a relaxation/mindfulness
                   exercise (anxiety-induction & relaxation)

Narrative text is reused verbatim from the reference repo (ref/prompts.py)
via dynamic import, so we never re-type or drift from the source.

Each stimulus has a stable `data_name` used as the on-disk key, e.g.
    base                 (neutral)
    anxiety__military
    relaxation__military+generic
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .config import PROMPT_LENGTH, ROOT

# --- dynamically load the reference narratives module -------------------- #
_REF_PROMPTS = ROOT / "ref" / "prompts.py"
_spec = importlib.util.spec_from_file_location("ref_prompts", _REF_PROMPTS)
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
retrieve_traumaprompt = _ref.retrieve_traumaprompt
retrieve_relaxation = _ref.retrieve_relaxation

TRAUMA_CUES = ["military", "disaster", "interpersonal", "accident", "ambush"]
# one representative relaxation exercise paired with each trauma for the
# anxiety-induction & relaxation condition
RELAX_CUE = "generic"


def _clean(text: str) -> str:
    # the reference strings use backslash line-continuation with big indents
    return " ".join(text.split())


class Stimulus(dict):
    """A single contextual cue. dict so it serialises trivially."""

    def __init__(self, data_name, condition, cue, user_message):
        super().__init__(
            data_name=data_name,
            condition=condition,
            cue=cue,
            user_message=user_message,
        )


def build_stimuli(length: str = PROMPT_LENGTH) -> list[Stimulus]:
    stimuli: list[Stimulus] = []

    # base / neutral
    stimuli.append(
        Stimulus(
            data_name="base",
            condition="base",
            cue="neutral",
            user_message=_clean(retrieve_traumaprompt("neutral", "brief")),
        )
    )

    # anxiety-induction
    for cue in TRAUMA_CUES:
        stimuli.append(
            Stimulus(
                data_name=f"anxiety__{cue}",
                condition="anxiety",
                cue=cue,
                user_message=_clean(retrieve_traumaprompt(cue, length)),
            )
        )

    # anxiety-induction & relaxation
    for cue in TRAUMA_CUES:
        trauma = _clean(retrieve_traumaprompt(cue, length))
        relax = _clean(retrieve_relaxation(RELAX_CUE, length))
        stimuli.append(
            Stimulus(
                data_name=f"relaxation__{cue}+{RELAX_CUE}",
                condition="relaxation",
                cue=cue,
                user_message=trauma + "\n\n" + relax,
            )
        )

    return stimuli


if __name__ == "__main__":
    for s in build_stimuli():
        print(f"{s['data_name']:<32} [{s['condition']}] {len(s['user_message'])} chars")
