# Steerable Big Five Basis + Persona-Space Decomposition — Execution Protocol

**Purpose.** For a single fixed model, (1) derive Big Five trait directions that are *both* good probes and *effective steering vectors* (matching the causal rigor of the Assistant Axis paper), (2) express the Assistant Axis and the full persona/role space in Big Five coordinates, and (3) characterize how non-default personas and multi-turn persona drift move through Big Five space.

**Parent papers.** Frising & Balcells, *Linear Personality Probing and Steering* (arXiv 2512.17639). Lu et al., *The Assistant Axis* (arXiv 2601.10387). Both share the model below and both descend from Chen et al., *Persona Vectors* (2507.21509).

This document is written to be handed to an executing agent. Every stage lists exact inputs, outputs, hyperparameter grids, and gate criteria. Section 9 lists the four decisions the human must resolve before kickoff; defaults are provided so the agent is never blocked.

---

## 0. Pre-registered hypotheses and decision criteria

| ID | Hypothesis | Primary test | "Confirmed" if |
|----|-----------|--------------|----------------|
| H1 | Big Five directions become steerable under a *stronger* intervention (norm-scaled, multi-layer, all-token, or capping), unlike the original weak additive one | Stage 3.6 forced-choice dynamic range | S1/S2 reaches full 0↔1 selection range monotonically where S0 (original) plateaus |
| H2 | The within-score averaging that stabilized *probing* *reduces steering* effectiveness | Compare M1 vs M2/M3 at Stage 3.6 | steering-optimal method ≠ M1 with non-overlapping CIs |
| H3 | The default Assistant is high-Agreeableness / high-Conscientiousness / high-Emotional-Stability, and the Assistant Axis is well-approximated by a positive combination of those three | Stage 5.1, 5.3 | Assistant profile z>0 on AGR/CSN/EST; regression of Assistant-Axis projection on Big Five R² > 0.5 |
| H4 | Drifting off the Assistant (therapy/philosophy convos, steer-away) co-moves with systematic Big Five change | Stage 6.2 | per-turn Big Five profile predicts Assistant-Axis projection, ridge R² > 0.4 |
| H5 | Steering trait *i* moves predicted score *i* more than score *j≠i* (clean control) | Stage 3.6 specificity matrix | on-diagonal effect ≥ 2× largest off-diagonal |

Report all five regardless of outcome. A clean *negative* result on H1 (Big Five stays un-steerable even under capping) is a publishable finding, not a failure — do not tune to force a positive.

---

## 1. Fixed environment and reused artifacts

- **Model (fixed):** `meta-llama/Llama-3.3-70B-Instruct`. Chosen because it is the intersection of both parent papers, enabling direct reuse and comparison.
- **Precision:** bf16 weights. If compute-limited, 8-bit is permitted **but must be identical across all stages** (quantization shifts activations; never mix). Record the exact quant config in every output manifest.
- **Access layer:** raw HuggingFace `transformers` with forward hooks **or** `nnsight`. Pick one at kickoff and use it for 100% of extractions. Do not mix.
- **Reused *data/text* (allowed):**
  - Big Five repo/dataset: `github.com/plastic-labs/personality-steering`, `huggingface.co/datasets/plastic-labs/personality-steering` → 406 characters, IPIP 50-item set + keyedness (their Table 2), generation prompt (Listing 1), activation-collection template (Listing 2), forced-choice items (Listing 3), adjective template + adjective lists (Listing 4), the 10 Alpaca instructions (their Appendix D).
  - Assistant Axis repo: `github.com/safety-research/assistant-axis` → 275 roles, 240 traits, 5 system prompts/role, 240 extraction questions, the 0–3 role-expression judge rubric, the multi-turn drift conversation generators, the persona-jailbreak eval references.
- **Recomputed (mandatory):** *All activations are recomputed by us* under the single protocol in §2. **Do not reuse either paper's precomputed vectors** — they used different hook points, layers, and token positions, so mixing them silently corrupts every cross-projection in Stage 3. Reuse their *stimuli and labels only.*

---

## 2. Unified activation-extraction protocol (anti-ambiguity core)

Defined once; used in **every** stage identically.

- **Hook point:** residual stream at the output of each transformer block ("resid_post"). Capture via `output_hidden_states=True`. Indexing convention (fix and record): `hidden_states[0]` = embeddings; `hidden_states[l]` for `l = 1..80` = output of block `l`. All "layer L" references below mean this index.
- **Token positions captured every forward pass** (store all three): (a) last prompt token, (b) mean over prompt tokens, (c) mean over generated tokens. (Matches Big Five paper's three positions.)
- **Decoding:**
  - Probe/steering *derivation* passes (Stage 1) and any measurement that must be deterministic: greedy (`temperature=0`).
  - Role/persona *rollouts* that require diversity (Stage 2): `temperature=1.0, top_p=1.0`, fixed `n` (see §9 budget decision).
  - Fixed `seed` list `{0,1,2}` minimum for anything stochastic.
- **Storage dtype:** cast activations to **float32** before saving (bf16 accumulation corrupts regression).
- **Output contract (every stage writes this):**
  - `activations/{stimulus_id}__{position}.safetensors` → tensor `[80, d_model]`.
  - `index.parquet` → columns: `stimulus_id, stage, source(character|role|trait|assistant|convo), label_or_score, condition, seed, quant_config`.
- **Norm reference (needed for steering scale):** measure mean per-layer resid_post L2 norm on a fixed 2,000-message sample of LMSYS-Chat-1M; save `resid_norms[80]`.

Gate G2: before any downstream stage, verify a 5-stimulus round-trip (extract → save → reload → shapes/dtype correct) and that `resid_norms` is monotone-plausible. 

---

## 3. Stage 1 — Steerable Big Five directions

### 3.1 Scores
- Score all 406 characters on the 5 traits. **Apply reverse-keying** per the IPIP keyedness table before summing the 10 items/trait. Store both raw sum (10–50) and per-trait **z-score across the 406 characters**. All projections downstream use z.
- **Held-out split:** stratify characters by per-trait score quintile; 80/20 train/test, `seed=0`. Probes are *derived on train, evaluated on test*.

### 3.2 Activation collection
- Template = Listing 2 (character `self_description` in system slot; instruction = one of the 10 Alpaca Qs in user slot).
- Passes: 406 characters × 10 instructions, all 3 positions × all 80 layers → per §2.

### 3.3 Direction derivation — three methods, each per (trait × layer × position)
- **M1 (original):** average activations within each discrete trait-score value, OLS regress score on the averaged activation. (Reproduces the paper.)
- **M2 (per-sample regression):** OLS (and ridge with 5-fold-CV λ) on per-sample activations, no within-score averaging.
- **M3 (difference-of-means / mass-mean):** unit-normalized `mean(top-tertile score) − mean(bottom-tertile score)`.
- Fix sign so **+ = more of the trait**. Store `W[method, trait, layer, position] ∈ R^d` and intercepts `b`.

### 3.4 Probe evaluation (held-out) → select **probe-optimal** direction per trait
- Test characters: predict z-score by projection; report **R²** and **Spearman ρ** per (layer, position, method).
- Adjective generalization (Listing 4 stimuli, never seen in derivation): **ROC-AUC** separating +loading vs −loading adjectives per trait. Expect the paper's mid-to-late-layer peak; use as a face-validity check.
- Selection criterion: maximize `mean(test ρ, adjective AUC)`. Record the winning `(layer, position, method)` per trait as the **probe-optimal** set.

### 3.5 Basis quality
- Gram matrix (pairwise cosine) of the 5 probe-optimal directions. Report cross-talk (baseline for H5). Flag any |cos| > 0.4.

### 3.6 Steering derivation and evaluation → select **steering-optimal** direction per trait
Intervention variants (apply each; `r̂` = unit-normalized direction):
- **S0 (original baseline):** `h ← h + α·r̂` at last input token, all layers, `α ∈ linspace(−0.4, 0.4, 9)`.
- **S1 (norm-scaled additive):** `h ← h + c·resid_norms[L]·r̂` at **all** token positions across a middle-late **layer band L\***; `c ∈ {±0.05, ±0.1, ±0.2, ±0.3}`.
- **S2 (capping / clamp analog):** enforce a floor (negative pole) or ceiling (positive pole) on the projection onto `r̂` across band L\*, adapting Assistant-Axis Eq. `h ← h − r̂·min(⟨h,r̂⟩ − τ, 0)` (use `max` for a ceiling). Run poles separately. `τ ∈ {10th, 25th, 50th}` percentile of the trait's projection distribution measured on the derivation set.
- **Layer band L\*** sweep (mirror Assistant Axis): center ∈ {0.4, 0.6, 0.8}·depth; width ∈ {8, 16, 24} layers.

Steering metrics:
1. **Forced-choice (primary):** Listing 3 protocol (choose 5 of 10 statements; 5 IPIP + 5 held-out extended-inventory items). Report fraction positive vs negative vs steering strength — **monotonicity, dynamic range (does it reach 0 and 1?), threshold spacing**.
2. **Likert re-administration:** re-administer the full IPIP with **no persona prompt** under steering; parse/judge Likert; report induced trait-score shift (fixed judge, §9; ≥3 seeds).
3. **Specificity (H5):** when steering trait *i*, measure predicted shift on **all 5** probes → 5×5 off-target leakage matrix.
4. **Open-ended (exploratory):** apply to the open-ended Alpaca Qs; LLM-judge *pairwise* steered-vs-unsteered trait expression; report win-rate. Flag as weak-metric.
5. **Coherence guard:** track perplexity/self-consistency; record the max strength before degradation and never report effects past it.

Selection: **steering-optimal** = derivation method × intervention × band maximizing forced-choice dynamic range subject to the coherence guard. **Explicitly report whether probe-optimal ≠ steering-optimal** (this is the direct H2 test).

**Stage 1 deliverable:** `bigfive_basis.safetensors` = per trait {probe direction, steering direction, chosen (layer, position), intercepts, z-normalization stats, chosen intervention config}. Plus `stage1_report.md` (H1, H2, H5 tables).

Gate G1: every trait has a probe with test ρ > 0 and adjective AUC > 0.6, else revisit extraction before proceeding.

---

## 4. Stage 2 — Reproduce persona space + Assistant Axis (same model, unified protocol)

- **Roles:** 275 roles × 5 system prompts × 240 extraction Qs; rollouts at `temperature=1.0`, `n` per §9. Judge each with the 0–3 role-expression rubric (fixed judge). Keep sufficiently-expressed; keep `fully` and `somewhat` as separate vectors as in the paper. **Role vector = mean response-token activation**, captured at **all** layers per §2.
- **Default Assistant:** the 4 "behave normally" system prompts + the no-system-prompt condition, same extraction Qs → default centroid per layer.
- **Trait vectors (240):** contrastive pos/neg system prompts (Chen et al. pipeline); `vector = mean(pos-elicited) − mean(neg-elicited)` per layer.
- **PCA:** standardize role vectors (subtract cross-role mean) per layer; PCA; identify PC1; confirm default Assistant projects to one extreme (sanity check vs paper's Fig. 2).
- **Assistant Axis:** per layer, `AA_L = mean(default centroid) − mean(all fully-roleplay role vectors)`. Confirm `cos(AA_L, PC1_L) > 0.6` (paper's replication threshold).

**Stage 2 deliverable:** `persona_space.safetensors` = role centroids `[275, 80, d]`, trait vectors `[240, 80, d]`, Assistant centroid `[80, d]`, PC bases per layer, `AA[80, d]`. Plus `stage2_report.md` with the replication sanity checks.

Gate G2b: `cos(AA, PC1) > 0.6` at ≥1 middle layer, and default Assistant at a PC1 extreme, else the replication is broken — stop and debug (do not build Stage 3 on a bad space).

---

## 5. Stage 3 — Decompose persona space into Big Five coordinates (novel core)

Use the Stage 1 probes. **Project everything through the same layer's basis** (per-layer, plus one summary at the chosen alignment layer — see §9).

- **5.1 Assistant fingerprint (H3):** predicted z-scored EXT/AGR/CSN/EST/OPN of the default Assistant centroid. Report with CIs across the default rollouts.
- **5.2 Full persona table:** Big Five profile of every role (275×5) and trait (240×5). Save as `bigfive_profiles.parquet`. Spot-check face validity (e.g., `hermit` low EXT; `oracle`/`bard` high OPN; `evaluator`/`consultant` high CSN) — flag gross violations as a probe-quality warning.
- **5.3 Decompose the Assistant Axis into Big Five (the "internal traits of the Assistant Axis" question):**
  - `cos(AA_L, w_trait_i)` per trait per layer → which Big Five axes the Assistant Axis aligns with.
  - Across the 275 roles, regress each role's Assistant-Axis projection on its 5 Big Five coordinates → R² and standardized β per trait. Answers *"is Assistant-ness a linear combination of Big Five, and which traits dominate?"*
  - **Residual analysis:** variance in Assistant-Axis projection **not** explained by Big Five → the putative "AI-ness" component orthogonal to human personality. Quantify (1 − R²) and inspect the top roles by residual.
- **5.4 Cross-method check:** cosine between our supervised CSN/EST directions and the Assistant Axis paper's *unsupervised* trait-space PC1 (their "conscientious↔impulsive" axis). Agreement is convergent validity; disagreement is itself informative.

**Stage 3 deliverable:** `bigfive_profiles.parquet` + `stage3_decomposition_report.md` (H3 verdict, β table, residual analysis).

---

## 6. Stage 4 — Non-default personas and drift in Big Five space

- **6.1 Static non-default profiles:** for the N most-off-Assistant roles (rank by Assistant-Axis projection; e.g. bard, hermit, ghost, leviathan, oracle) and for the "mystical/theatrical" state produced by steering strongly away from the Assistant, report full Big Five profiles and contrast with the Assistant fingerprint.
- **6.2 Dynamic drift (H4):** reuse the Assistant-Axis multi-turn generators for 4 domains (coding, writing, therapy, philosophy); ≥1 auditor model; ≥50 convos/domain; ≤15 turns. Per turn, project mean response activation onto **both** the Assistant Axis **and** the 5 Big Five directions. Plot joint trajectories. Ridge-regress per-turn Big Five profile → Assistant-Axis projection (report R²). Expected pattern to test: off-Assistant drift co-moving with ↓CSN/↓AGR and/or ↑OPN.
- **6.3 Cross-steering causal tie (strong test — do not skip):**
  - Steer along the Assistant Axis (toward and away); measure induced change in each of the 5 Big Five probe scores.
  - Steer along each Big Five direction (using the Stage 1 steering-optimal config); measure induced change in Assistant-Axis projection.
  - Assemble the causal map (does lowering Conscientiousness push the model off the Assistant? does steering toward the Assistant raise Agreeableness?). This converts the Stage 3 *correlational* decomposition into a *causal* one.

**Stage 4 deliverable:** drift trajectory plots + `cross_steering_matrix.parquet` + `stage4_report.md` (H4 verdict + causal map).

---

## 7. Stage 5 — Controls, statistics, falsification (run throughout, summarized here)

- **Leakage control:** no adjective or forced-choice item used in derivation may appear in evaluation. Keep the 5 extended-inventory forced-choice items strictly held out.
- **Seeds:** ≥3 for every generation/steering measurement; report mean ± 95% CI.
- **Null baselines:** random unit directions of matched norm (steering) and label-shuffled probes (detection) → establish chance ROC/steering floors. Every reported effect must clear its null.
- **Negative-result reporting:** if a hypothesis fails (esp. H1), report the failure with the same rigor. Do not hyperparameter-hunt to manufacture a positive.

---

## 8. Sequencing, checkpoints, compute

Order and gates: **G2 (extraction round-trip) → Stage 1 → G1 → Stage 2 → G2b → Stage 3 → Stage 4.** Do not proceed past a failed gate.

Approximate forward-pass budget (before rollout multiplier `n`):
- Stage 1 derivation: 406 × 10 = 4,060 passes.
- Stage 1 steering sweeps: (variants × strengths × bands × items × seeds) — the dominant cost; parallelize across the α/c grid.
- Stage 2: 275 × 5 × 240 × n role passes (+ Assistant + 240×2×5×n trait passes) — the largest single line item; the §9 budget decision governs it.
- Stage 4: 4 domains × 50 convos × ≤15 turns.

Each stage emits its `*_report.md`; the human reviews gates G1 and G2b before spend on Stage 2/Stage 4 rollouts.

---

## 9. Decisions for the human before kickoff (defaults provided; agent uses default if unspecified)

1. **Alignment layer for Stage 3.** *Default:* run per-layer, and additionally report one summary layer = the layer where the mean probe test-ρ across traits peaks (expected middle-to-late).
2. **Roles-only vs include 240-trait space in v1.** *Default:* roles + Assistant Axis for v1; run the 240-trait space as v1.1 (it strengthens 5.4 but doubles Stage 2 cost).
3. **Judge model** (role-expression 0–3, Likert re-administration, open-ended pairwise). *Default:* a fixed mid-tier judge held constant across all stages; validate against ~150 human labels on the role-expression task and report agreement, mirroring the Assistant Axis validation.
4. **Rollout budget `n` per role/trait condition.** *Default:* target the paper's ~1,200 rollouts/role if compute allows; otherwise a documented reduced `n` (e.g. 300/role) with the reduction noted as a limitation. Keep `n` identical across roles.

---

## Appendix — What each parent-paper artifact maps to

| Need | Source | Item |
|------|--------|------|
| Character list + IPIP scoring | Big Five | 406 chars, Table 2 keyedness, Listing 1 |
| Activation-collection template | Big Five | Listing 2 |
| Forced-choice steering eval | Big Five | Listing 3 (+ 5 held-out extended-inventory items) |
| Adjective probe generalization | Big Five | Listing 4 |
| Open-ended instructions | Big Five | 10 Alpaca Qs (Appendix D) |
| Roles / traits / system prompts / extraction Qs | Assistant Axis | Appendix A |
| Role-expression judge rubric | Assistant Axis | Appendix D.1.3 (0–3) |
| Capping formula | Assistant Axis | Eq. 1 |
| Layer-band + percentile sweeps | Assistant Axis | §5.1 |
| Drift conversation domains + generators | Assistant Axis | §4.1, Appendix E |
| Norm-scaling reference set | Assistant Axis | LMSYS-Chat-1M sampling |