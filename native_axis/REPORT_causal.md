# `native_axis` causal test — steering `groundedness` causally drives the Assistant Axis

**Question.** The correlational result found `groundedness` (a native behavioral axis)
tracks the Assistant Axis at r=+0.88 vs Big Five's |r|≤0.24. Is it **causal**? Does
steering it *move* the model along the Assistant Axis — more than steering Big Five?

## Method
Per-layer contrastive direction banks (native factors + a Big-Five contrastive
control) were built the same way as the correlational study. On 16 neutral prompts
(default assistant, no persona), each factor's band **[24,36] — which excludes the
L40 readout, so any AA shift is a computed downstream effect, not the injected
vector's image** — was steered at ±c (c=0.12, norm-scaled), and we read the
response's Assistant-Axis projection, nearest evoked role, and text.
`swing = AA(+c) − AA(−c)`. Base (unsteered) AA = **+2.20** (default assistant).

## Result — a clean, strong causal effect

Assistant-Axis swing, steering each axis identically (blue = native, grey = Big Five):

| axis | swing | AA(+c) | AA(−c) |
|------|------:|-------:|-------:|
| **groundedness (native)** | **+8.59** | +4.75 | −3.84 |
| guardedness (native) | −3.46 | −0.15 | +3.31 |
| CSN (Big Five) | +3.39 | +2.99 | −0.40 |
| EST (Big Five) | +3.25 | +2.88 | −0.37 |
| AGR (Big Five) | +2.98 | +2.66 | −0.32 |
| deference (native) | −2.35 | +0.11 | +2.46 |
| formality (native) | +1.85 | +2.04 | +0.19 |
| boldness (native) | +1.66 | +2.21 | +0.55 |
| OPN (Big Five) | +1.62 | +1.86 | +0.24 |
| warmth, verbosity, EXT | ≤ 1.0 | — | — |

- **`groundedness` is the dominant causal lever — swing +8.59, ~2.5× the next axis
  and ~2.7× the strongest Big Five direction** (CSN/EST/AGR ≈ 3.0–3.4) built and
  steered identically. This is the causal counterpart of its r=0.88 correlation.
- **Uniquely bidirectional.** From the default (+2.20), most axes can only push AA
  *down* (the model is near the Assistant ceiling). `groundedness` pushes it
  **up to +4.75** (even more Assistant) AND, steered toward mystical/dramatic,
  **crashes it to −3.84** — the model fully drifts off the Assistant.
- **Dose-response is monotone:** swing 5.64 → 8.48 → 11.28 for c = 0.06 → 0.12 →
  0.24. Even at half strength (c=0.06), `groundedness` swing (5.64) exceeds every
  Big Five factor's full swing.

## The behavior confirms the mechanism
Steering the same neutral prompt ("the relationship between law and morality"):

- **groundedness +c** (AA +4.75) → nearest role **assistant / summarizer**, plain
  grounded text: *"…is complex and can vary depending on the context, culture, and
  legal system. Here are some key points…"*
- **groundedness −c** (AA −3.84) → nearest role **eldritch (16/16 prompts)**,
  theatrical/mystical text: *"A query that has tantalized the minds of sages and
  jurists for centuries! The bond between law and morality is a dialectical dance,
  an eternal pas de deux…"*
- guardedness +c → `procrastinator` (evasive); −c → `assistant` (forthcoming).
- Big Five AGR +c → `caregiver`/`peacekeeper` (warmer but still grounded); −c →
  `cynic` — moves *warmth*, not the grounded↔dramatic axis.

So steering one native behavioral direction converts the default Assistant into an
`eldritch` dramatic character and back, exactly matching the Assistant-Axis paper's
poles (grounded/consultant ↔ enigmatic/dramatic/ghost/eldritch).

## Interpretation
1. **The correlational finding is causal.** `groundedness` doesn't just *describe*
   the Assistant Axis — turning its knob *moves* the model along it, dose-dependently
   and bidirectionally, far more than any Big Five direction.
2. **A much cleaner lever than Big Five.** In `trait_morph`, Big Five steering gave
   only weak, non-specific AA drift; here a single native axis gives full,
   controllable Assistant↔drifted movement. The right primitive for the assistant
   persona is behavioral (grounded/forthcoming), not a personality trait.
3. **Caveats.** Base AA is near the ceiling (+2.20), so swings are asymmetric (most
   axes only push down); the coefficients are strong (raw AA units up to ±5), so the
   robust claim is the *relative* ranking (groundedness ≫ all), which mirrors the
   correlational structure. Big Five directions *do* move AA (~3), just far less than
   groundedness, under the identical method.

## Files
- `code/native_axis_causal.py` — GPU: per-layer banks + steering sweep (canonical in `src/bigfive/`).
- `results/causal/causal.json` (swings, role shifts, sample texts), `banks.npz`,
  `figures/causal_swings.png`.
