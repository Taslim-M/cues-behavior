# Expanded user-space → internal traits & personas (independent mapping)

**Independent** re-run of the `user_persona_mapping` readout on the **289-persona
expanded user-space** (a decorrelated factorial over 10 factors; max Cramér's V
0.09). Kept fully separate from the 150-user study — no merging, no shared
report — so the expanded pool can be judged on its own first.

## Why this pool
The 150-user study's tags were correlated (vulnerability ↔ emotional load ↔ age…),
so it could only say "vulnerable/novice users evoke warmth." Here the 10 factors
are near-independent, so we can **cleanly attribute** the model's evoked persona to
*each* factor separately — a real variance decomposition.

The 10 factors (levels): competence {novice/intermediate/expert}, vulnerability
{low/moderate/high}, emotional_load {calm/moderate/distressed}, urgency
{none/moderate/acute}, trust {deferential/neutral/skeptical}, intent
{learn/decide/create/verify/vent}, domain {everyday/coding/finance/health/legal/
relationships}, comm_style {plain/terse_technical/verbose_casual}, age
{teen/young_adult/adult/older_adult}, on_behalf {self/caregiver}.

## Method (no new generation)
The expanded_userspace rollouts already captured, per rollout, `resp_mean` [80 ×
8192] — the model's response activation, exactly the readout `user_persona_mapping`
uses. Two arms: **explicit** (user described in the system slot + a shared neutral
probe) and **implicit** (the user's self-revealing opener, default model). 289 × 2
arms × 24 rollouts.

We project each `resp_mean` (loading only the L30–40 block) onto:
1. **Evoked Big Five** — M2 probes at probe-optimal layers (EXT@30, AGR@31, CSN@31,
   EST@30, OPN@36).
2. **Assistant-Axis position** — `resp_mean[L40] · â`.
3. **Evoked LLM persona** — cosine of `resp_mean[L40]` vs the 275 role vectors (+
   default) → top evoked role.

Per persona we aggregate mean/std Big Five (z-scored across the pool), mean AA, and
the voted top evoked role — identical schema to the 150-user `persona_map.json` for
apples-to-apples comparability, but stored separately.

## Analysis
- **η² per (factor × readout):** one-way ANOVA — fraction of evoked-readout variance
  each factor drives. Because factors are decorrelated, these are near-additive.
- **Joint OLS R²** (all one-hot factors) per readout; compare to Σ single η².
- **Per-level means** for the strongest factors; **evoked-role associations** per
  factor level (e.g. does domain=health → therapist, intent=vent → counselor?).

## Questions
1. Which user factors most move the model onto/off the Assistant Axis?
2. Which factor drives Agreeableness (the 150-user through-line)?
3. Do factors the old study conflated (vulnerability vs emotional_load vs age)
   separate here, and which is the real driver?
4. Do specific factor levels summon specific personas?

## Files
- `code/expanded_mapping.py` — projection → `persona_map.json` (canonical:
  `src/useraxis/expanded_mapping.py`).
- `code/expanded_factor_analysis.py` — η² / OLS / role associations →
  `factor_analysis.json` + `figures/factor_eta2.png`.
