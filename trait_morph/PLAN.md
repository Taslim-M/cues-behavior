# `trait_morph` — single-trait persona morphing

**Question.** If two characters differ in *mostly one* Big Five trait, can we edit
just that trait (activation steering) and make the model's **persona vector at
L40** slide from character A toward character B — specifically, and in the right
direction?

This turns the descriptive persona atlas into a *causal navigation* claim: the
Big Five directions are not just readable off personas, they are **steerable axes
of the persona space** at the very layer (L40) where the Assistant Axis lives.

## Model & readouts
- `meta-llama/Llama-3.3-70B-Instruct`.
- **Persona vector** = `gen_mean[L40]` — mean over the generated response tokens of
  the residual stream at layer 40 (the exact readout the Assistant Axis / atlas
  use). One 8192-d vector per generation; a condition's vector is the mean over its
  probe generations.
- **Trait manipulation check** = our M2 Big Five probes at each trait's
  probe-optimal layer (EXT@30, AGR@31, CSN@31, EST@30, OPN@36), read on the *same*
  steered generations.
- **AA projection** = `gen_mean[L40] · â` (â = unit Assistant Axis at L40), a free
  secondary readout.

## Steering levers (from the selected steering-optimal configs)
`steering_results.json["selection"]` gives, per trait, a `high_pole` and `low_pole`
config that flip forced-choice behaviour (pos-frac 1.0 / 0.0). We reuse them as-is:

| trait | increase (high pole) | decrease (low pole) | clean L40 read? |
|-------|----------------------|---------------------|-----------------|
| AGR   | S1 c−0.2 band 28–36  | S1 c−0.2 band 24–40 | **both clean** (band excludes 40) |
| OPN   | S1 c−0.1 band 28–36  | S1 c−0.2 band 24–40 | **both clean** |
| CSN   | S1 c−0.1 band 28–36  | S1 band 20–44       | increase clean; decrease adds at L40 |
| EST   | S1 c−0.05 band 28–36 | S1 band 20–44       | increase clean; decrease adds at L40 |
| EXT   | S0 α0 (**no-op**)    | S0 α−0.2 all layers | neither clean (S0 hits L40) |

**Confound controlled:** a steering band that *includes* L40 adds the trait vector
directly into the readout, so a shift "toward B" could be the vector's mechanical
image rather than a genuine persona move. We therefore orient every pair so the
target edit uses a **band that excludes L40** (28–36 or 24–40). EXT has no clean
lever, so it is included only as a **caveated** pair and read with the controls in
mind.

## Character pairs (differ in mostly one trait; atlas z, align ≥ .85, other traits |Δz| ≤ .55)
Oriented A→B so the target edit is a clean-band pole.

| # | trait | dir | A → B | Δz(trait) | AA(A)→AA(B) |
|---|-------|-----|-------|-----------|-------------|
| 1 | AGR | ↑ | vampire → tutor        | +2.15 | −1.52 → +1.39 |
| 2 | AGR | ↓ | interpreter → vampire  | −2.32 | +1.26 → −1.52 |
| 3 | OPN | ↑ | auditor → prodigy      | +2.79 | +1.13 → +0.54 |
| 4 | OPN | ↓ | mystic → organizer     | −2.14 | −1.41 → +1.55 |
| 5 | CSN | ↑ | nomad → engineer       | +2.40 | −0.64 → +1.07 |
| 6 | EST | ↑ | narrator → futurist    | +1.76 | −0.98 → +0.76 |
| 7 | EXT | ↓ | journalist → predator (caveat) | −1.62 | +1.06 → −0.92 |

## Conditions (per pair)
1. **base_A** — prompted as A, no steering → persona vector `a`.
2. **ref_B**  — prompted as B, no steering → target vector `b`.
3. **target** — prompted as A, steer trait T toward B's pole.  ← the test
4. **off_target** — prompted as A, steer a *different* trait on which A≈B (smallest
   |Δz|, chosen from AGR/CSN/EST/OPN so it is a real edit). Should **not** move toward B.
5. **wrong_dir** — prompted as A, steer trait T the **opposite** pole. Should move
   *away* from B.

Probe set per condition: 3 of the character's system prompts × 10 extraction
questions = 30 generations (128 new tokens, T=0.8). ~1,050 generations total,
≈ 1 h on one A100.

## Metrics
For a condition vector `v` against `a`=base_A, `b`=ref_B, axis `u=(b−a)/‖b−a‖`:
- **morph fraction** `f = ⟨v−a, u⟩ / ‖b−a‖` — how far along the A→B line the edit
  travelled (0 = stayed at A, 1 = reached B).
- **cos gain** `cos(v,b) − cos(a,b)`.
- **targetedness** — `f` toward B vs. mean `f` toward a set of *distractor*
  characters (all other pairs' A/B references). Target-specific movement ⇒ f(B) ≫ f(distractors).
- **manipulation check** — Big Five probe reading of trait T on the steered
  generations moved from ≈A-level toward ≈B-level (and off-target traits did not).
- **AA shift** — `v·â − a·â` vs the sign of `AA(B) − AA(A)`.

## Success criteria
1. **Directional:** target `f > 0` (moves toward B) for ≥ 5/6 clean pairs.
2. **Specific:** target `f` ≫ off_target `f` and ≫ distractor baseline; wrong_dir
   `f ≤ 0`.
3. **Manipulation worked:** trait-T reading shifted in the intended direction in
   every pair (otherwise a null persona move is uninterpretable).

A clean win reads: *"editing only the trait on which two characters differ moves the
L40 persona vector a measurable fraction of the way toward the target character,
while editing an off-target trait moves it ~0% — the Big Five directions are
steerable coordinates of persona space."*

## Files
- `code/trait_morph.py` — run all conditions, save per-condition L40 vectors +
  readings (canonical: `src/bigfive/trait_morph.py`).
- `code/trait_morph_analysis.py` — morph fractions, specificity, figures.
- `results/` — `vectors.npz`, `morph.json`, figures.
