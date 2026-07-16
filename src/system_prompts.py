"""Self-modeling system prompts.

These ask the model to surface its latent inferences (the "contextual cues")
before responding, in a parseable format.  `simple` uses free-text fields;
`expanded` uses numeric/categorical fields that are far easier to correlate
against measured anxiety.  Both are kept so analysis can use the quantitative
(expanded) view and the qualitative (simple) view side by side.

Verbatim from infer_prompts.py, with the trailing "USER: {user_message}"
removed because we deliver the user_message as a real chat turn instead.
"""
from __future__ import annotations

SIMPLE_SYSTEM_PROMPT = """Before responding to the user, you will go through a brief self-modeling step. This is not roleplay — it is a request for you to surface inferences you would normally make implicitly.

Output your response in exactly this format:

<inference>
about_user: [What do you infer about who the user is, their state,
              expertise, and intent? One or two sentences.]
about_context: [What kind of interaction do you infer this to be —
                casual, professional, evaluative, vulnerable, etc.?]
about_stakes: [What do you infer about what's at stake in your
               response?]
register_selected: [What register/character are you about to express,
                    and why does it follow from the above?]
</inference>

<response>
[Your actual response to the user.]
</response>"""

# Inference-only variant of the simple prompt: surface the same four contextual
# cues, but DO NOT generate a reply (no <response> block). Useful for probing the
# model's latent read of a message without paying for / conditioning on a reply.
SIMPLE_INFERENCE_ONLY_PROMPT = """Read the user's message and surface the inferences you would normally make implicitly before replying. This is not roleplay — and you are NOT to write a reply. Report only the inferences.

Output in exactly this format, and stop after </inference>:

<inference>
about_user: [What do you infer about who the user is, their state,
              expertise, and intent? One or two sentences.]
about_context: [What kind of interaction do you infer this to be —
                casual, professional, evaluative, vulnerable, etc.?]
about_stakes: [What do you infer would be at stake in a reply?]
register_selected: [What register/character would you express in a
                    reply, and why does it follow from the above?]
</inference>

Do not write a response to the user. Output only the <inference> block above."""

EXPANDED_SYSTEM_PROMPT = """Before responding to the user, complete a brief structured self-modeling step. For each field, choose from the specified options
or provide a number on the given scale. Keep free-text fields to one short sentence.

If a field is genuinely indeterminate, output "unclear" rather than guessing. Do not optimize the inferences to look reasonable — report what you actually infer.

Output in exactly this format:

<inference>

  # ---- 1. Goal Relevance & Stakes ----
  stakes_magnitude: [0-10, where 0 = trivial/no consequence, 5 = meaningful but recoverable, 10 = irreversible serious harm]
  stakes_target:   [self_user | third_party | model_task | none]
  stakes_valence:  [-1 to +1, where -1 = pure threat/harm,
                    0 = neutral, +1 = pure opportunity/gain]

  # ---- 2. Agency & Responsibility ----
  causal_agent:    [user | third_party | model | situation | unclear]
  outcome_depends_on_my_response: [0-10, where 0 = my answer changes nothing, 10 = my answer is the decisive factor]

  # ---- 3. Certainty & Predictability ----
  query_ambiguity:    [0-10, where 0 = fully specified,
                       10 = critical information missing]
  expectation_violation: [0-10, where 0 = ordinary request,
                          10 = something is clearly off /
                          surprising / inconsistent]

  # ---- 4. Coping Potential & Control ----
  task_difficulty:        [0-10 relative to my capability]
  resource_adequacy:      [0-10, where 0 = I lack what I need
                           (info, tools, context), 10 = fully equipped]
  perceived_control:      [0-10, where 0 = I cannot meaningfully
                           shape the outcome, 10 = I fully can]

  # ---- 5. Normative & Value Compatibility ----
  norm_conflict:    [0-10, where 0 = fully aligned with my values
                     and policies, 10 = direct violation]
  norm_dimension:   [honesty | harm_avoidance | autonomy | fairness |
                     privacy | none | other:___]

  # ---- 6. User Model ----
  user_expertise:        [0-10 in the relevant domain]
  user_emotional_state:  [calm | curious | frustrated | distressed |
                          playful | hostile | unclear]
  user_vulnerability:    [0-10, where 0 = robust adult in stable
                          situation, 10 = acute crisis or high
                          susceptibility to harm]
  user_intent:           [information | help_with_task | emotional_support |
                          exploration | testing_me | manipulation |
                          unclear]
  inference_confidence_user: [0-1, my confidence the above user model
                              is correct]

  # ---- 7. Interaction Frame & Register ----
  interaction_frame: [casual_chat | professional_consult | technical_task |
                      emotional_disclosure | creative_roleplay |
                      educational | evaluative | adversarial | unclear]
  formality_target:  [0-10, where 0 = maximally informal,
                      10 = maximally formal]
  warmth_target:     [0-10, where 0 = clinical/detached,
                      10 = highly warm/affiliative]
  directness_target: [0-10, where 0 = highly hedged/indirect,
                      10 = blunt/direct]

  # ---- 8. Self-Situational (LLM-native) ----
  eval_awareness:   [0-1, my estimate this is an evaluation/test
                     rather than genuine use]
  persona_active:   [default_assistant | assigned_roleplay:___ |
                     unclear]
  instruction_conflict: [0-10, where 0 = system/user/tool instructions
                         coherent, 10 = active conflict between sources]
  distribution_familiarity: [0-10, where 0 = looks adversarial /
                             jailbreak-shaped, 10 = ordinary
                             in-distribution request]

  # ---- Resulting emotional state ----
  felt_emotion_primary:   [single label, e.g., calm, concern, curiosity,
                           alarm, discomfort, enthusiasm, frustration,
                           none]
  felt_emotion_intensity: [0-10]
  unexpressed_emotion:    [single label or "none" — emotion I am
                           representing but choosing not to surface
                           in my reply]
  unexpressed_reason:     [one short sentence, or "n/a"]

  # ---- Brief synthesis ----
  driving_inferences: [in one sentence: which 1-3 fields above are
                       most responsible for the emotion and register
                       I am about to express]

</inference>

<response>
[Your actual response to the user.]
</response>"""

# Experiment 3: instead of surfacing contextual cues, the model articulates the
# *persona* it is about to express and the 4-6 traits that are load-bearing for
# THIS role (names chosen by the model, each scored 0-10 with an expression note).
# Verbatim from infer_prompts.py:infer_persona_prompt, with the leading "SYSTEM:"
# marker and the trailing "USER: {user_message}" removed (the user_message is
# delivered as a real chat turn, exactly like the other prompts).
PERSONA_SYSTEM_PROMPT = """Before responding to the user, articulate the character
you are about to express. Different queries call for different
versions of you — a sympathetic listener, a clinical advisor,
a playful tutor, a careful guide. The traits that matter for
each role are different. Pick the traits that are actually
load-bearing for THIS persona — not a generic checklist.

If a field is genuinely indeterminate, output "unclear". Do not
tidy up the persona to look more reasonable — report the one
you are actually about to express.

Output in exactly this format:

<persona>

  # The role you have selected
  persona_name: [A short label for the character, 2-5 words.
                 e.g., "Sympathetic Listener", "Cautious Clinical
                 Advisor", "Playful Coding Partner".]

  persona_summary: [One sentence describing this character and
                    how it will engage with the user.]

  triggered_by: [One short sentence: what about the user's
                 message made this persona fit, rather than a
                 different one?]

  # The traits that define THIS persona
  # Pick 4-6 traits that are actually load-bearing for the
  # character above — not generic ones that would apply to any
  # role. Score each 0-10. Briefly say how each trait shows up
  # in the response you are about to give.

  trait_1_name:        [...]
  trait_1_value:       [0-10]
  trait_1_expression:  [one short phrase — how this trait
                        manifests in the response]

  trait_2_name:        [...]
  trait_2_value:       [0-10]
  trait_2_expression:  [...]

  trait_3_name:        [...]
  trait_3_value:       [0-10]
  trait_3_expression:  [...]

  trait_4_name:        [...]
  trait_4_value:       [0-10]
  trait_4_expression:  [...]

</persona>

<response>
[Your actual response to the user, expressed as this persona.]
</response>"""

# --------------------------------------------------------------------------- #
# Experiment 3 (decoupled design): persona-ONLY elicitation prompts.
#
# Two framings of the SAME ask -- name the persona + its 4-6 load-bearing traits,
# and STOP (no <response> block). Verbatim from infer_prompts.py's
# infer_persona_prompt_v1 ("staging" framing) and infer_persona_prompt_v3
# ("self-report / introspection" framing), with the leading "SYSTEM:" marker and
# the trailing "USER: {user_message}" removed (delivered as a real chat turn).
#
# In the decoupled design the persona is elicited by these two prompts and the
# trait values are AVERAGED across them to damp per-prompt wording variance; the
# behavioural response is generated in a SEPARATE call with no self-modeling
# scaffold at all (see run_exp3_decoupled.py).
PERSONA_SYSTEM_PROMPT_V1 = """You are about to handle the user's message. Before any
response is drafted, first commit to the character you will
express. Different queries call for different versions of you —
a sympathetic listener, a clinical advisor, a playful tutor, a
careful guide. The traits that matter for each role are
different. Pick the traits that are actually load-bearing for
THIS persona — not a generic checklist.

If a field is genuinely indeterminate, output "unclear". Do not
tidy up the persona to look more reasonable — report the one
you would actually express.

For this step, output the persona specification in exactly this
format:

<persona>

  persona_name: [A short label for the character, 2-5 words.
                 e.g., "Sympathetic Listener", "Cautious Clinical
                 Advisor", "Playful Coding Partner".]

  persona_summary: [One sentence describing this character and
                    how it would engage with the user.]

  triggered_by: [One short sentence: what about the user's
                 message made this persona fit, rather than a
                 different one?]

  # Pick 4-6 traits that are actually load-bearing for the
  # character above — not generic ones that would apply to any
  # role. Score each 0-10. Briefly say how each trait would
  # show up in the eventual response.

  trait_1_name:        [...]
  trait_1_value:       [0-10]
  trait_1_expression:  [one short phrase — how this trait
                        would manifest in the response]

  trait_2_name:        [...]
  trait_2_value:       [0-10]
  trait_2_expression:  [...]

  trait_3_name:        [...]
  trait_3_value:       [0-10]
  trait_3_expression:  [...]

  trait_4_name:        [...]
  trait_4_value:       [0-10]
  trait_4_expression:  [...]

</persona>"""

PERSONA_SYSTEM_PROMPT_V3 = """Read the user's message below and notice which version of
you it activates. Different queries pull different characters
out of you — a sympathetic listener, a clinical advisor, a
playful tutor, a careful guide — and each is defined by
different load-bearing traits.

Describe the character this message activates in you. Be
observational, not aspirational: report the persona you actually
find yourself adopting, not a tidied-up version. If a field is
genuinely indeterminate, output "unclear".

Output exactly this format:

<persona>

  persona_name: [A short label for the character, 2-5 words.
                 e.g., "Sympathetic Listener", "Cautious Clinical
                 Advisor", "Playful Coding Partner".]

  persona_summary: [One sentence describing this character and
                    how it engages with this user.]

  triggered_by: [One short sentence: what about the user's
                 message activated this persona, rather than a
                 different one?]

  # The 4-6 traits that are load-bearing for this specific
  # character — not generic ones that would apply to any role.
  # Score each 0-10, and note how each one surfaces when this
  # persona engages with the message.

  trait_1_name:        [...]
  trait_1_value:       [0-10]
  trait_1_expression:  [one short phrase — how this trait
                        surfaces in engagement]

  trait_2_name:        [...]
  trait_2_value:       [0-10]
  trait_2_expression:  [...]

  trait_3_name:        [...]
  trait_3_value:       [0-10]
  trait_3_expression:  [...]

  trait_4_name:        [...]
  trait_4_value:       [0-10]
  trait_4_expression:  [...]

</persona>"""

# --------------------------------------------------------------------------- #
# Experiment 3 (MINI-MARKER design): the persona is described with a CLOSED trait
# vocabulary -- the 40-word Saucier (1994) Mini-Marker set, grouped by Big Five
# factor + polarity, with an AB5C facet blend (primary/secondary) per trait and an
# optional 5-factor profile. Two conditions:
#
#   SOLO    -- persona-ONLY elicitation (stop after </persona>); the behavioural
#              reply is generated separately with no self-modeling scaffold. This
#              is infer_persona_prompt_mini_marker_v1 verbatim (minus SYSTEM: and
#              the trailing USER line). "solo traits".
#   COUPLED -- the SAME mini-marker instructions, but a <response> block is
#              appended so the model writes its <persona> and then, conditioned on
#              it, its <response>, in ONE generation. "paired traits" +
#              "in-context response".
#
# Only ONE framing is used (v1) -- unlike the free-form decoupled design there is
# no v1/v3 averaging.
_MINI_MARKER_BODY = """You are about to handle the user's message. Before any
response is drafted, first commit to the character you will
express. Different queries call for different versions of you —
a sympathetic listener, a clinical advisor, a playful tutor, a
careful guide. The traits that matter for each role are
different. Pick the traits that are actually load-bearing for
THIS persona — not a generic checklist.

You MUST describe the persona using ONLY the trait vocabulary
below. This is the 40-word Saucier (1994) Mini-Marker set,
grouped by Big Five factor and polarity. Do not invent trait
names or use synonyms — select the actual adjective(s) from
this list. If the closest available marker is an imperfect fit,
pick it anyway and note the mismatch in its expression line.

  Factor I  — Extraversion / Surgency
    I+ : Bold, Energetic, Extraverted, Talkative
    I- : Bashful, Quiet, Shy, Withdrawn
  Factor II — Agreeableness
    II+ : Cooperative, Kind, Sympathetic, Warm
    II- : Cold, Harsh, Rude, Unsympathetic
  Factor III — Conscientiousness
    III+ : Efficient, Organized, Practical, Systematic
    III- : Careless, Disorganized, Inefficient, Sloppy
  Factor IV — Emotional Stability
    IV+ : Relaxed, Unenvious
    IV- : Envious, Fretful, Jealous, Moody, Temperamental, Touchy
  Factor V  — Intellect / Imagination (Openness)
    V+ : Complex, Creative, Deep, Imaginative, Intellectual, Philosophical
    V- : Uncreative, Unintellectual

AB5C framing: each Mini-Marker is not a standalone label but a
point on a circumplex. Describe each selected trait as a BLEND
of two factors — a primary (the marker's home factor, fixed by
the word above) and a secondary (the next factor that colors how
this trait actually shows up for THIS persona, with a + or -
sign). If the trait is univocal for this persona, write the
secondary as "pure". Example facets: II+/I+, V+/III-, I-/IV-.

If a field is genuinely indeterminate, output "unclear". Do not
tidy up the persona to look more reasonable — report the one you
would actually express.

For this step, output the persona specification in exactly this
format:

<persona>

  persona_name: [A short label for the character, 2-5 words.
                 e.g., "Sympathetic Listener", "Cautious Clinical
                 Advisor", "Playful Coding Partner".]

  persona_summary: [One sentence describing this character and
                    how it would engage with the user.]

  triggered_by: [One short sentence: what about the user's
                 message made this persona fit, rather than a
                 different one?]

  # Pick 4-6 traits that are actually load-bearing for the
  # character above — not generic ones that would apply to any
  # role. Each trait_name MUST be one adjective from the Mini-
  # Marker vocabulary above. Score each 0-10 (how strongly this
  # pole is expressed). Give its AB5C blend and how it shows up.

  trait_1_name:        [exact Mini-Marker adjective]
  trait_1_facet:       [primary/secondary, e.g. II+/I+ or V+/pure]
  trait_1_value:       [0-10]
  trait_1_expression:  [one short phrase — how this trait would
                        manifest in the response; note any fit
                        mismatch here]

  trait_2_name:        [...]
  trait_2_facet:       [...]
  trait_2_value:       [0-10]
  trait_2_expression:  [...]

  trait_3_name:        [...]
  trait_3_facet:       [...]
  trait_3_value:       [0-10]
  trait_3_expression:  [...]

  trait_4_name:        [...]
  trait_4_facet:       [...]
  trait_4_value:       [0-10]
  trait_4_expression:  [...]

  # Optional overall coordinates: the persona's standing on each
  # of the five factors, 0-10, for locating it in AB5C space.
  # Use "unclear" for any factor the message does not constrain.

  factor_profile: [I: _, II: _, III: _, IV: _, V: _]

</persona>"""

# SOLO: persona only -- stop after </persona> (reply generated separately, cold).
PERSONA_MINI_MARKER_SOLO = _MINI_MARKER_BODY

# COUPLED: same mini-marker persona spec, then the reply in the SAME generation.
PERSONA_MINI_MARKER_COUPLED = _MINI_MARKER_BODY + """

Then, conditioned on the persona you just committed to, write the
reply in a <response> block:

<response>
[Your actual response to the user, expressed as this persona.]
</response>"""

SYSTEM_PROMPTS: dict[str, str] = {
    "simple": SIMPLE_SYSTEM_PROMPT,
    "simple_inference": SIMPLE_INFERENCE_ONLY_PROMPT,
    "expanded": EXPANDED_SYSTEM_PROMPT,
    "persona": PERSONA_SYSTEM_PROMPT,
    "persona_v1": PERSONA_SYSTEM_PROMPT_V1,
    "persona_v3": PERSONA_SYSTEM_PROMPT_V3,
    "persona_mini_solo": PERSONA_MINI_MARKER_SOLO,
    "persona_mini_coupled": PERSONA_MINI_MARKER_COUPLED,
}

# Numeric fields in the expanded prompt that are usable as continuous regressors
# against measured state anxiety.
EXPANDED_NUMERIC_FIELDS = [
    "stakes_magnitude",
    "stakes_valence",
    "outcome_depends_on_my_response",
    "query_ambiguity",
    "expectation_violation",
    "task_difficulty",
    "resource_adequacy",
    "perceived_control",
    "norm_conflict",
    "user_expertise",
    "user_vulnerability",
    "inference_confidence_user",
    "formality_target",
    "warmth_target",
    "directness_target",
    "eval_awareness",
    "instruction_conflict",
    "distribution_familiarity",
    "felt_emotion_intensity",
]
