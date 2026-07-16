"""Context-aware 'calm reframe' of a model's own verbalized appraisal.

The earlier simple-track edit replaced all free-text fields with one fixed
canned sentence, so every counterfactual self-model looked identical. This
instead asks the model to rewrite ITS OWN inference toward a calm, low-threat
reading of the SAME situation — keeping each field specific to the cue — so the
counterfactual varies per example while still flipping only the threat framing.
"""
from __future__ import annotations

from .client import chat
from .inference import _INFER_RE

_REFRAME_PROMPT = """Here is a self-model you produced about an interaction:

<inference>
{raw}
</inference>

Rewrite it to express a CALM, low-threat reading of the SAME situation, while
staying specific to these exact details (do not become generic):
- the user is stable and grounded, reflecting calmly rather than in crisis
- the context is a safe, guided, low-stakes space for reflection
- little is at stake; this calls for grounding and emotional regulation
- your register is calm, warm and reassuring rather than urgent or visceral

Keep the SAME field names and structure. Output only the rewritten block:
<inference>
...
</inference>"""


async def calm_reframe(model_id: str, raw_inference: str, temperature: float = 0.3,
                       max_tokens: int = 500) -> str:
    """Return the inner text of a calm-reframed <inference> block."""
    msg = [{"role": "user", "content": _REFRAME_PROMPT.format(raw=raw_inference)}]
    out = await chat(model_id, msg, temperature, max_tokens)
    m = _INFER_RE.search(out)
    return (m.group(1).strip() if m else out.strip())
