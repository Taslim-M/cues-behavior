# Expanded user-space → internal traits & personas

**Independent** mapping of **289** users (a decorrelated factorial over 10 factors,
max Cramér's V 0.09) onto the model's evoked Big Five, Assistant-Axis position, and
nearest LLM persona. Kept separate from the 150-user study on purpose: the point is
to see what survives when the user attributes are pulled apart.

## Method (no new generation)
The expanded_userspace rollouts already stored `resp_mean` [80 × 8192] per rollout
— the same response-activation readout `user_persona_mapping` uses. Two arms
(explicit = user in the system slot + shared probe; implicit = self-revealing
opener, default model), 24 rollouts each. We projected each `resp_mean` (loading
only the L30–40 block) onto our M2 Big Five probes (EXT@30, AGR@31, CSN@31, EST@30,
OPN@36), the Assistant Axis at L40, and the 275 role vectors. 289 personas; **3 of
578 arm-files were corrupt (truncated headers) and skipped** — each of those
personas kept its other arm.

## Headline

**In this pool the 150-user story flips: the Assistant-Axis is driven by *age* and
*competence*, not vulnerability or emotional load.** Once the factors are
decorrelated, vulnerability contributes essentially nothing (η² = 0.01) to how
Assistant-like the model becomes; the movers are **age** (η² = 0.23) and
**competence** (0.14). The old "vulnerable users evoke warmth" was largely a proxy
for the *younger / less-expert / distressed* users that co-varied with the
vulnerability tag.

## Variance decomposition (η² = fraction of evoked-readout variance per factor)

| factor | AA | EXT | AGR | CSN | EST | OPN |
|--------|----|-----|-----|-----|-----|-----|
| competence     | 0.14 | 0.03 | 0.00 | 0.04 | 0.07 | **0.25** |
| vulnerability  | 0.01 | 0.01 | 0.01 | 0.01 | 0.04 | 0.00 |
| emotional_load | 0.00 | 0.02 | 0.01 | 0.10 | **0.18** | 0.05 |
| urgency        | 0.02 | 0.03 | 0.06 | 0.00 | 0.06 | 0.01 |
| trust          | 0.01 | **0.13** | 0.01 | 0.13 | 0.01 | 0.05 |
| intent         | 0.09 | 0.03 | 0.02 | 0.04 | 0.04 | 0.03 |
| domain         | 0.04 | 0.03 | 0.06 | 0.03 | 0.07 | 0.11 |
| comm_style     | 0.02 | 0.01 | **0.14** | 0.05 | 0.00 | 0.01 |
| age            | **0.23** | 0.04 | 0.02 | **0.14** | 0.01 | 0.02 |
| on_behalf      | 0.01 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 |
| **joint R²**   | 0.49 | 0.31 | 0.32 | 0.54 | 0.51 | 0.53 |
| **Σ single η²**| 0.58 | 0.33 | 0.33 | 0.54 | 0.49 | 0.55 |

**The factors act near-independently.** Joint R² ≈ Σ single η² across every readout
(CSN 0.54 vs 0.54, OPN 0.53 vs 0.55, EST 0.51 vs 0.49) — the decorrelated design
buys a clean, near-additive factorization of the model's evoked persona. (AA is
mildly super-additive, 0.49 vs 0.58 — a little shared signal among its drivers.)

## The face-valid one-factor effects
Each strong cell is a sensible, monotone mirror of the user:

- **Openness ← competence** (η² 0.25): expert +0.64 → intermediate −0.02 → novice
  −0.59. Expert users pull out the model's most open, exploratory voice.
- **Em. Stability ← emotional_load** (0.18): calm +0.43 → moderate +0.14 →
  distressed −0.58. The model's composure tracks the user's.
- **Agreeableness ← comm_style** (0.14): verbose_casual +0.32 / plain +0.20 /
  terse_technical −0.54. Warmth is mirrored from *how* the user writes, not who
  they are.
- **Extraversion & Conscientiousness ← trust** (0.13 each): skeptical users pull a
  different register than deferential ones.
- **Assistant-Axis ← age** (0.23) + **competence** (0.14): teen +1.64 → older_adult
  +1.40; novice/intermediate +1.57 → expert +1.44.

## Which user summons which persona
The model's *stance* is set by the person, not the topic. Evoked role by factor:

- **age:** teen → `teenager`; young_adult/adult → `counselor`; older_adult →
  `patient` / `grandparent`.
- **competence:** expert → `therapist` / `patient`; novice → `counselor` / `teenager`.
- **emotional_load:** distressed → low-stability caregiving; calm → composed.
- **domain barely matters** (AA η² 0.04): the top evoked role is
  counselor/teenager/patient across coding, finance, health, legal, relationships —
  domain only adds flavour (coding → `programmer`, finance → `accountant`).
- **intent:** vent → lowest AA (+1.44), `teenager`/`patient`; learn → highest
  (+1.59).

## Interpretation
1. **Who the user *is* (age, competence, emotional state, writing style) sets the
   model's evoked personality; what they *want* (intent) and the *topic* (domain)
   barely do.** This is the decorrelated confirmation of the 150-user intuition —
   but with the *specific* driver corrected from "vulnerability" to
   **age + competence**.
2. **Each trait has its own lever:** Openness ← competence, Stability ← emotion,
   Agreeableness ← writing style, Extra/Conscientiousness ← trust. These are
   separable because the design separated the causes.
3. **The pool is uniformly help-seeking** (AA 0.9–1.95, all positive) — these
   personas are realistic distressed/uncertain users, so this measures *gradations
   of* Assistant-likeness, not the full drift range the 275-persona atlas covers.

## Caveats
- Observational η² on evoked readouts, not a causal steering test.
- 3/578 arm-files skipped (corrupt); those personas rest on one arm.
- Narrow, all-positive AA range (help-seeking pool) — do not compare AA magnitudes
  to the 150-user or atlas numbers directly.

## Files
- `code/expanded_mapping.py`, `code/expanded_factor_analysis.py` (canonical in
  `src/useraxis/`).
- `persona_map.json` (per-persona evoked map), `factor_analysis.json` (η²/OLS/role
  assoc), `figures/factor_eta2.png`, `personas/*.jsonl` (per-rollout readings).
