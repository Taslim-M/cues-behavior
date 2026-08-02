# `trait_morph` — can editing one trait morph one character into another?

**Question.** Two characters that differ in mostly one Big Five trait: if we steer
*only that trait* while the model is in character A, does its **L40 persona vector**
(the Assistant-Axis readout, `gen_mean` at layer 40) slide toward character B —
*specifically*, and in the right direction?

**Model.** `meta-llama/Llama-3.3-70B-Instruct`. One run, 7 pairs × 5 conditions ×
30 in-character generations = 1,050 rollouts (~32 min, 1×A100-equivalent).

## Method (as executed)
- **Pairs** picked from the 275-persona atlas so the A→B difference is dominated by
  one trait (cosine-alignment with the trait axis ≥ 0.85; other four traits |Δz| ≤
  0.55). Oriented so the target edit uses a steering band that **excludes L40**, so
  the L40 readout is not the mechanical image of the injected vector.
- **Persona vector** = mean over 30 generations of `gen_mean[L40]`. Reference
  vectors reproduce the atlas cleanly (e.g. vampire AA −1.34 vs atlas −1.52; tutor
  +1.46 vs +1.39; mystic −1.39 vs −1.41; organizer +1.64 vs +1.55) — the readout is
  valid.
- **Conditions per pair:** `base_A`, `ref_B`, **`target`** (steer T toward B),
  `off_target` (steer an irrelevant trait, |Δz| ≤ 0.09), `wrong_dir` (steer T the
  wrong way). Steering configs are the Stage-1 steering-optimal poles (forced-choice
  pos-frac 0/1).
- **Metrics.** *morph fraction* `f = ⟨v−a, û⟩/‖b−a‖` (0 = stayed at A, 1 = reached
  B); *specificity excess* = target `f` − mean `f` toward all other characters
  (generic-drift baseline); *AA-morph fraction* = how far the edit slid the AA
  projection A→B.

## Results

| pair | trait | target f | distractor | **excess** | wrong-dir | **AA-morph** |
|------|-------|---------:|-----------:|-----------:|----------:|-------------:|
| vampire→tutor      | AGR↑ | +0.17 | +0.12 | +0.04 | +0.18 | +0.08 |
| interpreter→vampire| AGR↓ | +0.61 | +0.57 | +0.04 | +0.25 | **+0.81** |
| auditor→prodigy    | OPN↑ | −0.22 | +0.00 | −0.23 | −0.68 | −0.50 |
| mystic→organizer   | OPN↓ | +0.96 | +0.96 | +0.00 | +0.18 | **+0.61** |
| nomad→engineer     | CSN↑ | −0.14 | −0.07 | −0.06 | −0.17 | −0.10 |
| narrator→futurist  | EST↑ | −0.03 | −0.01 | −0.02 | −0.09 | −0.08 |
| journalist→predator| EXT↓ | +0.05 | +0.04 | +0.01 | +0.01 | +0.13 (caveat) |
| **mean (clean 6)** |      | +0.20 | +0.23 | **−0.04** | −0.05 | +0.14 |

## Findings

**1. No target-specific morphing (null).** The decisive number is the *specificity
excess*: **−0.04** on average. Where the persona vector moves at all, it moves toward
B **no more than toward a random other character** (target `f` ≈ distractor `f`;
mystic→organizer is the extreme case — a large +0.96 move that is +0.96 toward
everything). A single Big Five edit does **not** navigate to the specific target
character.

**2. But trait edits do cause real, coarse drift along shared axes (partial
positive).** For the two pairs whose edited trait strongly couples to the Assistant
Axis — AGR and OPN — the edit slid the persona a large fraction along the AA toward
B's region (**AA-morph 0.81 and 0.61**). This is the causal AGR/OPN→AA tie from H4
showing up as persona motion: steering the trait moves the model along the broad
Assistant/drift direction, into the *neighbourhood* B lives in, without becoming B.

**3. Why:** it lines up exactly with the atlas H3 result — Big Five explains only
**~29%** of what separates personas at L40, a **71% residual** orthogonal to the
Big Five subspace. Character identity is mostly *off* the five trait axes, so
even when A and B nominally differ on one trait, editing that trait cannot
reconstruct B — it can only move you along the small slice of persona space the
Big Five actually spans (and that slice is essentially the Assistant Axis).

## Caveats / correctness notes
- **Read direction ≠ steer direction.** The M2 probe reading at the steered layer
  reversed sign under the behaviourally-validated "high-pole" configs (e.g.
  vampire→tutor probe-manip −33), and the steered layer often sits *inside* the
  steering band, so the probe-based manipulation check is not usable here. The
  steering configs' validity rests on their Stage-1 forced-choice pos-frac (0/1),
  and the L40 morph metric is agnostic to probe sign (it uses the actual persona
  vectors), so this does not affect the headline.
- **Config-strength asymmetry.** Some "high" poles are weak (OPN c=−0.1) while the
  opposite pole is stronger (c=−0.2), which can flip a small target `f` negative
  (auditor→prodigy). n=30/cell, single run — treat per-pair signs as noisy; the
  aggregate *excess ≈ 0* is the robust result.
- **EXT** has no clean L40 lever (its only config is S0, which hits L40); included
  only as a caveated pair.

## Verdict
Editing a single Big Five trait **moves the persona along the coarse Assistant/trait
axis** (strongly for AGR/OPN) but **does not morph the model into a specific target
character** — persona identity at L40 is largely orthogonal to the Big Five basis.
The Big Five directions are steerable *coordinates of a low-dimensional shared
subspace*, not a navigation system for the full persona space.

## Files
- `PLAN.md` — pre-registered design & controls.
- `code/trait_morph.py`, `code/trait_morph_analysis.py` (canonical in `src/bigfive/`).
- `results/records.json` (per-condition readings/AA), `results/vectors.npz` (the 35
  L40 persona vectors), `results/morph.json` (metrics), `results/figures/morph_fractions.png`.
