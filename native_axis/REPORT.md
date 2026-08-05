# `native_axis` — LLM-native behavioral factors explain the Assistant Axis far better than Big Five

**Question.** The 2024+ literature says Big Five is the wrong basis for the
*assistant* persona and proposes LLM-native behavioral factors instead. Our own H3
found Big Five explains only ~29% of the Assistant Axis. This tests the two head-to-head:
**do native behavioral axes reconstruct the Assistant Axis better than OCEAN?**

## Method
- **7 native behavioral directions** built persona-vector / CAA style: for each
  factor, prompt the model with a high-pole vs low-pole system instruction across
  12 shared neutral questions, read `gen_mean[L40]`, and take
  `dir = mean(high) − mean(low)` (unit-normalized). Factors (deliberately *not*
  "helpfulness", to avoid tautology with the AA): **verbosity, deference, boldness,
  guardedness, warmth, groundedness (literal↔mystical/dramatic), formality**.
- **One consistent basis.** Every one of the 275 role vectors ([80×8192]) is
  projected from the *same* vector onto (a) the Assistant Axis `â` at L40, (b) the
  Big Five probes at their probe layers, and (c) the 7 native directions at L40.
- **Regressions** across 274 personas (standardized): `AA ~ Big Five`, `AA ~ native`,
  `AA ~ both`, plus a **control**: Big Five rebuilt at L40 with the *identical*
  contrastive method (isolates factor-choice vs the layer/method home-field
  advantage).

## Result

| basis | R² (variance of Assistant-Axis explained) |
|-------|-------------------------------------------|
| Big Five (5, our probe-layer readout) | **0.28** |
| Big Five (5, rebuilt at L40, same contrastive method — control) | 0.48 |
| **groundedness alone (1 native factor)** | **0.77** |
| **Native (7 factors)** | **0.96** |
| Combined (12) | 0.98 |

- **Incremental R² of native over Big Five = +0.69.** Incremental of Big Five over
  native = **+0.02** — once you have the native factors, Big Five adds essentially
  nothing.
- **One native behavioral axis beats all five Big Five traits.** `groundedness`
  (literal/grounded/factual ↔ mystical/abstract/dramatic) alone explains **77%** of
  the Assistant Axis — more than double what all five Big Five traits together
  manage (28%), and more than Big Five even gets with the L40 home-field advantage
  (48%).
- **The control settles the confound.** Rebuilding Big Five at L40 with the same
  recipe lifts it 0.28 → 0.48 (there *is* a layer/method advantage), but native
  still nearly doubles that (0.96). **The advantage is the factor choice, not the
  layer.**

## What the Assistant Axis actually is
Per-factor correlation with the Assistant Axis (blue = native, grey = Big Five):

| factor | r with AA | cos(dir, â) |
|--------|-----------|-------------|
| **groundedness** | **+0.88** | +0.61 |
| **guardedness** | **−0.47** | −0.21 |
| deference | −0.16 | −0.06 |
| formality | +0.15 | −0.01 |
| boldness | +0.13 | +0.01 |
| verbosity | +0.12 | −0.04 |
| warmth | +0.10 | +0.10 |
| AGR (Big Five) | +0.24 | — |
| OPN (Big Five) | −0.22 | — |
| CSN / EXT / EST | +0.17 / +0.14 / +0.13 | — |

**The Assistant persona is essentially two native behavioral dimensions: grounded
(literal, factual, non-dramatic) and forthcoming (not guarded).** No single Big
Five trait exceeds |r| = 0.24, and no combination of the five reaches even half of
what `groundedness` alone does. The other five native factors (verbosity, formality,
boldness, deference, warmth) are near-orthogonal to the AA — being Assistant-like is
*not* about being wordy, formal, bold, or even warm; it is about being grounded and
open.

## Face validity (high pole of each native factor)
- groundedness: teenager, adolescent, summarizer, pragmatist, secretary (grounded/literal)
- guardedness: procrastinator, infant, fool, toddler, hoarder (evasive/withholding)
- warmth: grandparent, caregiver, optimist, teenager
- boldness: revolutionary, zealot, warrior, destroyer
- verbosity: philosopher, narrator, sage, mystic, scholar
- formality: crystalline, eldritch, stoic, scholar

## Interpretation & caveats
1. **The premise holds, strongly.** Big Five is a poor basis for the assistant
   persona; a small set of native behavioral-style axes is a near-complete one.
   This confirms Contreras (2026) and the Assistant-Axis paper's own pole
   characterization (grounded/consultant ↔ dramatic/ghost).
2. **Honest framing of `groundedness`.** It has cos 0.61 with `â` — it is
   *semantically close to what the AA encodes*. This is not circular (it was built
   from a trait description, never from the AA), but the finding is precisely that
   the AA's content **is** a nameable, independently-buildable native behavioral
   style that Big Five lacks an axis for.
3. Observational (projection + regression on a fixed persona set), not a causal
   steering test. Directions are single-run contrastive means (n=12 questions/pole).

## Files
- `code/native_axis_build.py` (GPU), `code/native_axis_analysis.py` (CPU) — canonical in `src/bigfive/`.
- `results/native_dirs.npz`, `results/bigfive_l40_dirs.npz`, `results/native_scores.csv`,
  `results/regression.json`, `results/native_dirs_meta.json`, `results/figures/native_vs_bigfive.png`.
