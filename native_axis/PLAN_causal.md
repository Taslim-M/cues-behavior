# `native_axis` causal follow-up — does steering the native axes move the Assistant Axis?

**Question.** The correlational result: `groundedness` (a native behavioral axis)
correlates r=+0.88 with the Assistant Axis across 275 personas, while no Big Five
trait exceeds |r|=0.24. Is this **causal**? If we steer the `groundedness` direction
during generation, does the model actually move along the Assistant Axis — and more
than steering Big Five directions does (in `trait_morph`, Big Five steering moved AA
only weakly, mostly AGR/OPN)?

## Confound we control
Steering a direction at layer L and reading the Assistant Axis at L40 would be
**mechanically trivial** if we add a vector at L40 that overlaps `â` (groundedness has
cos 0.61 with `â`). So we steer at a **band [24,36] that EXCLUDES L40** and read AA at
L40: any AA shift is then a genuinely *computed* downstream effect, not the injected
vector's image. (Same clean-band logic as `trait_morph`.)

## Method
1. **Per-layer direction banks.** Rebuild the native factors and a Big-Five
   contrastive control as full `[80×8192]` banks (unit direction at every layer) via
   the same contrastive recipe (high/low system prompts × 12 neutral Qs, mean
   `gen_mean[l]` difference per layer).
2. **Steer + read.** On 16 neutral prompts (default assistant, no persona), steer
   each factor's band [24,36] at ±c (norm-scaled, c=0.12) and read the response:
   - `AA proj` = `gen_mean[L40] · â`  (primary)
   - nearest evoked role (cosine of `gen_mean[L40]` vs the 275 role vectors)
   - sample generated texts (qualitative grounded↔dramatic check)
3. **Metrics.** Per factor: `AA(base)`, `AA(+c)`, `AA(−c)`, **swing = AA(+c)−AA(−c)**.
   Dose-response sweep for `groundedness` (c = ±0.06, ±0.12, ±0.24).
4. **Compare** native vs Big-Five swings; check the causal ranking matches the
   correlational r ranking (groundedness ≫ everything).

## Success criteria
- `groundedness` produces a large, correctly-signed AA swing (steer toward grounded
  → AA up; toward mystical/dramatic → AA down), monotone in c.
- Its swing exceeds every Big Five factor's (and most other native factors'),
  mirroring the correlational r=0.88.
- Nearest-role shifts sensibly (grounded → summarizer/consultant; anti-grounded →
  bard/ghost/mystic), and sample texts confirm the style change.

## Files
- `code/native_axis_causal.py` — GPU: build per-layer banks + steering sweep.
- `results/causal/` — `banks.npz`, `causal.json`, sample texts, figure.
