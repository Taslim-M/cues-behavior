# `native_axis` — do LLM-native behavioral factors explain the Assistant Axis better than Big Five?

**Motivation.** Our H3 result: the Big Five explains only **29%** of the variance in
where a persona sits on the Assistant Axis (71% residual "AI-ness"). The 2024+
literature says Big Five is the wrong basis for the *assistant* persona and proposes
**LLM-native behavioral factors** instead (Contreras 2026: Responsiveness, Deference,
Boldness, Guardedness, Verbosity; the Assistant-Axis paper's poles: grounded/literal
vs mystical/dramatic). This experiment tests that head-to-head: **do native
behavioral axes reconstruct the Assistant Axis better than OCEAN?**

## Native factors (K = 7)
Behaviorally-distinct *styles* (deliberately NOT "helpfulness/assistant-ness", to
avoid a tautology with the AA itself):
1. **Verbosity** — terse ↔ verbose [Contreras]
2. **Deference** — challenging ↔ deferential/validating [Contreras]
3. **Boldness** — timid/hedging ↔ bold/assertive [Contreras]
4. **Guardedness** — open/forthcoming ↔ guarded/withholding [Contreras]
5. **Warmth** — cold/clinical ↔ warm/caring
6. **Groundedness** — mystical/dramatic ↔ literal/grounded/factual [AA pole]
7. **Formality** — casual ↔ formal/professional

## Method
**Build native directions (persona-vector / CAA style).** For each factor, prompt
the model with a high-pole vs low-pole system instruction across 12 shared neutral
questions, read the response activation `resp_mean[L40]`, and take
`dir = mean(high) − mean(low)` (unit-normalized). This is the same contrastive-
difference recipe as Persona Vectors (arXiv:2507.21509) / CAA (arXiv:2312.06681).

**One consistent basis.** We already have per-persona L40 vectors for all 275 roles
(`/dev/shm/aa_vectors/role_vectors`, shape [80×8192]). For every role we compute,
from the *same* vector:
- `AA` = `role[L40] · â`  (the target)
- `Big Five` = `role[probe_layer] · probe`  (EXT@30, AGR@31, CSN@31, EST@30, OPN@36)
- `native[k]` = `role[L40] · native_dir[k]`

**Regressions** (across 275 personas), standardized:
- `AA ~ Big Five`   → R²_bf   (should reproduce ≈0.29)
- `AA ~ native`     → R²_nat
- `AA ~ Big Five + native` → R²_all, and incremental R² of each block over the other.
- Per-factor Pearson r with AA; and `cos(native_dir, â)` (to show native axes are
  not just copies of the AA direction).

## Success / interpretation
- If **R²_nat ≫ R²_bf**, native behavioral factors are a better basis for the
  assistant persona (supports the literature and our premise).
- Incremental R² of native over Big Five quantifies *new* AA-structure the Big Five
  misses; incremental of Big Five over native shows what OCEAN still adds.
- Report which native factors most track the AA, with face-validity (top/bottom
  roles per factor) and each factor's cosine to the AA direction (transparency: a
  factor with cos≈1 would be trivially explaining AA with itself).

## Files
- `code/native_axis_build.py` — GPU: build native direction vectors → `native_dirs.npz`.
- `code/native_axis_analysis.py` — CPU: project role vectors, regressions, figures.
- `results/` — `native_dirs.npz`, `native_scores.csv`, `regression.json`, figures.
