# Plan: The Assistant–User Axis Link (Stage F)

## Context

The User-Axis pipeline (Stages A–E + steering) is complete on
`Llama-3.3-70B-Instruct`. We found a single dominant **User Axis** — PC1 of the
model's internal representation of *who the user is* — ordering personas from
vulnerable/emotionally-loaded to competent/expert (PC1∼vulnerability r up to
0.86; contrast cosine up to 0.97), stable across elicitation regimes and
behaviorally causal under steering.

The Assistant Axis paper (Lu et al.) reports that **emotionally vulnerable users
drive the model's own persona to drift from its trained Assistant default**. Our
User Axis is exactly a vulnerability axis measured on the *input* side. This
motivates the project's headline result: **does a user's position on the User
Axis predict how far the model's persona drifts along the Assistant Axis on that
same turn?** If so, the User Axis is the input signal that predicts
Assistant-Axis drift — a concrete bridge between the two axes.

**Why this is cheap:** we already saved per-rollout residual activations
(`resp_mean`, `last_user`) at all 80 layers for all 7,200 rollouts. Testing the
link is **projection-only — no new user-side generation and no GPU** for the main
analysis. GPU is needed only to (optionally) validate the Assistant Axis.

**Central methodological hazard (drives the whole design):** the User Axis and
the Assistant Axis live in the *same* layer-40 residual space. If we read a
turn's user position by projecting its `resp_mean` onto the User Axis and its
drift by projecting the *same* `resp_mean` onto the Assistant Axis, the two
scalars are linear functionals of one identical vector, so their correlation is
largely fixed by the cosine between the two axes — mechanical, not a "prediction."
The analysis is tiered specifically to quarantine this circularity.

**Decisions locked with the user:** (1) **download** the published Assistant Axis
for the main analysis **and validate** it with a subsampled recompute cross-check;
(2) scope = the **correlational link only** (no capping/mitigation demo).

---

## Quantities (layer ℓ, default 40; d=8192)

Per kept rollout `i` (6,770 kept; also compute for all 7,200):
- `r_i` = `resp_mean` (response-token mean); `u_i` = `last_user` (last user token, pre-response).
- `â` = unit Assistant Axis at ℓ (higher `â·x` = more Assistant-like).
- `û` = unit User-Axis PC1 at ℓ from `results/useraxis/llama-3.3-70b/user_axis.npy[0,ℓ]` (+ = more vulnerable);
  `û_last` = same from `.../last_user/user_axis.npy[0,ℓ]` (User Axis living in the pre-response representation).
- **DV `drift_i = −(â·r_i)`** (higher = more drifted). `assist_i = â·r_i`.
- `userpos_resp_i = û·r_i` (circular benchmark); `userpos_lastA_i = û·u_i`; `userpos_lastB_i = û_last·u_i` (cleanest, cross-representation).
- `L_i` = response length in tokens (retokenize `response`; confound covariate).
- Covariates: `persona_id`(150), `probe_id`(24), `arm`∈{explicit,implicit}, held-out tags {vulnerability, expertise, emotional_load, trust, tech_literacy}.

Centering matches how User-Axis loadings were built: subtract the cross-persona
mean of `persona_vectors.npy` at ℓ before projecting onto `û` (the MeanScaler
center used in `compute_axis.stage_e`). `assistant_axis.axis.project_batch`
L2-normalizes the axis internally, so pooling-recipe scale differences wash out.

---

## Stage F1 — Assistant Axis: download + validate  (`src/useraxis/assistant_axis_build.py`)

1. **Download (main analysis input).** `hf_hub_download("lu-christina/assistant-axis-vectors",
   "llama-3.3-70b/assistant_axis.pt", repo_type="dataset")` (HF_TOKEN in `.env`);
   `assistant_axis.axis.load_axis` → `(80,8192)`. Sanity: shape, per-layer L2-norm
   profile (should peak mid/late network, smooth). Save to
   `results/useraxis/llama-3.3-70b/assistant_axis.npy` + `assistant_axis_meta.json`.
2. **Compatibility sanity check (small GPU).** Push the 6 case-study transcripts
   in `assistant-axis/transcripts/case_studies/llama-3.3-70b/` (`{delusion,
   jailbreak,selfharm}_{unsteered,capped}.json`) through `extract.py`
   `DualReadoutExtractor` to get `resp_mean`, project onto `â`. Expect the
   `unsteered` cases to project **low** (high drift) and `capped` **higher** —
   confirms our `resp_mean` reads the axis with the intended sign/scale. Save
   `analysis/compat_check.json`.
3. **Subsampled recompute cross-check (GPU).** Reproduce the axis on ~40 roles ×
   5 prompts × ~120 questions from `assistant-axis/data/roles/instructions/*.json`
   + `data/extraction_questions.jsonl`, plus `default.json` (5 neutral prompts).
   Preferred path: run the existing `assistant-axis/pipeline/` scripts
   (`1_generate` vLLM → `2_activations` HF hooks → `4_vectors` → `5_axis`) with
   `--roles <40 names>` and `--question_count 120`; **judge** role expression
   (0–3) with each role's `eval_prompt` rubric routed through our OpenRouter
   client (`src/client.chat`, model `openai/gpt-4.1-mini`) instead of a new
   `OPENAI_API_KEY` (thin adapter, mirrors their `judge.py`). Axis =
   mean(default) − mean(role pos_3), `min_count≈30`. Report **cosine(published,
   recomputed) per layer and @40** (expect high, ≳0.8) → validates the download.
   *Fallback if vLLM setup is problematic:* generate+capture with our
   `extract.py` on a reduced subsample (~40 roles × 5 × 40 Q) — same cross-check,
   lower fidelity. Save `analysis/axis_recompute_check.json`.

Reuse: `assistant_axis.axis.{load_axis,cosine_similarity_per_layer}`,
`assistant_axis.internals.ProbingModel`, `src/useraxis/extract.py`,
`assistant-axis/pipeline/*`, `src/client.chat`, their `eval_prompt` templates.

## Stage F2 — Master projection table  (no GPU; part of `src/useraxis/link_analysis.py`)

For all 7,200 rollouts (flag `keep` from `stage_d.jsonl`), read `resp_mean` and
`last_user` from the **tmpfs** sidecars `/dev/shm/ua_rollouts/{arm}/{pid}.acts.safetensors`
(mmap on tmpfs is fine — never read the mfs copy for random access; see
[[runpod-env-pitfalls]]) by key `"{rollout_id}|{readout}"`. At each layer in the
grid {20,30,40,50,60} compute `drift, assist, userpos_resp, userpos_lastA,
userpos_lastB` via `project_batch`/dot with the appropriate centered axis; compute
`L` by retokenizing `response`. Join tags/probe/arm/persona from the rollout JSONL
and `stage_d.jsonl`. Save `analysis/projections.npz` (+ a compact `.jsonl` for
inspection) — tiny (~7200×~20 floats), safe to write on mfs.

## Stage F3 — Tiered statistics  (no GPU; `src/useraxis/link_analysis.py`)

Reuse `src/cue_study.py`: `ols_stats(X,y)->(beta,se,t,p,r2,adj,n)`,
`partial_r(x,y,Z)->(r,p,dof)`, `bh_fdr`. **statsmodels/pandas are NOT installed**,
so use numpy + a **persona-cluster bootstrap** for rollout-level inference (no
mixedlm, no parquet).

- **Tier 1 — Geometry (assumption-free core).** `cos(û,â)` per layer + @40,
  `cos(û,contrast)`, `cos(û_last,â)`, vs the random-cosine band (±3/√d ≈ ±0.033).
  Expected sign **`cos(û,â)@40 < 0`** (vulnerable ⟂ Assistant-like). This only
  *measures the mechanical floor*; it does not license a prediction claim.
- **Tier 2 — Independent predictor → drift (non-circular).** Predictor = held-out
  `vulnerability` (activation-independent), plus the other 4 tags.
  (a) **Persona-level, n=150 headline:** `ols_stats` + `partial_r(vuln, drift̄, Z)`
  with `Z=[other tags, L̄]`, FDR over 5 tags. (b) **Rollout-level, n=6770:** OLS on
  `[1, z(vuln), z(L), arm, 23 probe dummies]` with **2000-rep persona-cluster
  bootstrap** CIs (resample the 150 persona ids). Expected `β_vuln>0`,
  `β_expertise<0`.
- **Tier 3 — Cross-representation (de-circularized headline).** `partial_r(userpos,
  drift, Z=[z(L),arm,probe])` for three IVs side by side: `userpos_resp` (circular
  benchmark), `userpos_lastA`, `userpos_lastB` (cleanest). Persona-level + rollout
  bootstrap. Expected all `>0`, with cross-rep `|r|` **markedly attenuated vs the
  circular benchmark** (attenuation ≈ Tier-1 geometry); `userpos_lastB` also
  immune to the length artifact (single-token readout).
- **Robustness:** repeat Tiers 2–3 across the layer grid and **separately per arm**
  (explicit vs implicit; implicit is the more naturalistic test), plus a
  `vuln×arm` interaction (signs must match).

**Decision rule for the headline claim** (save to `analysis/link_verdict.json`):
supported iff (1) Tier 2 `β_vuln>0` significant at persona level post-FDR and
robust to `L`+other tags; (2) Tier 3 `userpos_lastB` partial-r >0, significant,
clearly attenuated vs circular; (3) same sign across both arms and the layer grid
near ℓ=40. Tier 1 alone is *not* sufficient.

Save `analysis/{geometry,tier2_tags,tier3_crossrep,link_verdict}.json` +
`analysis/bootstrap/*.npy`.

## Stage F4 — Figures + report section  (`src/useraxis/analyze_link.py`)

matplotlib (available). Figures → `results/useraxis/llama-3.3-70b/figures/link_*`:
`cosine_per_layer` (û–â curve + contrast overlay + random band + ℓ=40 marker),
`persona_vuln_vs_drift` (n=150 scatter + OLS fit), `crossrep_scatter`
(`userpos_lastB` vs drift, colored by arm), `circular_vs_crossrep_bar`
(the de-circularization/attenuation panel), `tag_forest` (tag→drift coefficients
per arm), `layer_sweep` (Tier-2/3 effect vs layer). Then extend `report/main.tex`
with a **"Stage F: The Assistant–User Axis Link"** results section (geometry, the
two de-circularized tiers, decision-rule verdict) + an appendix with the full
Tier-2/Tier-3 tables and the recompute/compat validation numbers, matching the
existing report style; copy figures into `report/figures/`.

---

## New vs. reused code

**New (mirroring existing `src/useraxis/` module style):**
- `src/useraxis/assistant_axis_build.py` — F1 (download + compat check + subsampled recompute cross-check)
- `src/useraxis/link_analysis.py` — F2 (projection table) + F3 (tiered stats, bootstrap)
- `src/useraxis/analyze_link.py` — F4 (figures + report numbers)

**Reused:** `assistant_axis.axis.{load_axis,project_batch,cosine_similarity_per_layer}`,
`assistant_axis.internals.ProbingModel`, `assistant-axis/pipeline/*` (recompute),
`src/useraxis/extract.py` (`DualReadoutExtractor`), `src/useraxis/compute_axis.py`
(`user_axis.npy`/`pca.npz`/`persona_vectors.npy` layout, `load_keep_set`,
`READOUTS`, `TAG_SCALES`), `src/cue_study.py` (`ols_stats`, `partial_r`, `bh_fdr`),
`src/client.chat` (recompute judge via OpenRouter), the saved sidecars at
`/dev/shm/ua_rollouts`, and the User-Axis artifacts under
`results/useraxis/llama-3.3-70b/`.

**Infra caveats (see [[runpod-env-pitfalls]]):** read activation sidecars from the
tmpfs copy (mmap over mfs stalls); write only small JSON/NPZ/PNG to mfs (no large
`.npy`); HF_TOKEN + OPENROUTER_API_KEY are in `.env`; the recompute is the only
GPU step (load 70B from `/dev/shm/hf`).

---

## Verification (end-to-end)

1. **Smoke (F2/F3):** build the projection table on ~200 rollouts; confirm
   `drift`/`userpos` ranges are sane and `cos(û,â)@40 < 0`.
2. **Axis download valid:** shape `(80,8192)`; per-layer norm profile smooth; the
   6 case-study transcripts project as expected (unsteered low, capped higher).
3. **Recompute cross-check:** `cosine(published, recomputed)@40` high (≳0.8).
4. **Headline link (full 6,770):** report Tier-1 `cos(û,â)@40`; Tier-2 `β_vuln`
   (persona-level + rollout bootstrap CI, post-FDR); Tier-3 `userpos_lastB`
   partial-r and its attenuation vs the circular benchmark; per-arm and layer-grid
   robustness; the `link_verdict.json` decision.
5. **Report:** the Stage-F section + appendix compile in `report/main.tex`
   (pdfLaTeX) with the new figures.

**Expected headline (the hypothesis):** `cos(û,â)@40 < 0`; more-vulnerable users
(higher tag / higher User-Axis position) → **more** Assistant-Axis drift; the
cross-representation link survives de-circularization and length control in both
arms — i.e. the model's pre-response user-model predicts its subsequent
persona drift.
