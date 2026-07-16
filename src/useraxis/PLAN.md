# Plan: A "User Axis" — recreating the Assistant Axis on the user/persona side

> Copy of the approved plan, committed into the repo so it travels to the RunPod
> GPU box. Source of truth for the cloud continuation. API phase (Stages A–B) is
> DONE; the GPU phase (Stages C–E + steering) is what remains.

## Context

The Assistant Axis paper (Lu et al., *"The Assistant Axis: Situating and Stabilizing
the Default Persona of Language Models"*, arXiv:2601.10387; code:
`github.com/safety-research/assistant-axis`) finds an **activation direction** that
measures how far a model's *own* persona sits from its trained "Assistant" default.
Their pipeline: generate ~275 character roles × 5 role system-prompts × 240 extraction
questions → collect residual-stream activations on the model's responses → LLM-judge
filter for role expression → per-role mean vectors → **PCA**. PC1 (equivalently a
default−mean contrast vector) is the Assistant Axis; they then **steer** along it and
**cap** it to stabilize behavior.

We want the mirror image: an axis that captures the model's **internal model of *who
the user is***. Instead of varying the model's persona, we vary the **user persona** and
read the model's representation of that user. We expect a low-dimensional "user-persona
space" whose leading axis orders profiles meaningfully — the working hypothesis is that
PC1 spans **vulnerable / emotionally-loaded / support-seeking users ↔ competent /
bounded / transactional users** (echoing the paper's finding that *emotionally
vulnerable users* are exactly what drives Assistant-Axis drift). If so, the User Axis is
plausibly the *input* that predicts Assistant-Axis drift — a compelling scientific link
between the two.

This also connects to *"Designing a Dashboard for Transparency and Control of
Conversational AI"* (Viégas & Wattenberg et al.), which trains linear probes on the
residual stream to read inferred **user-model attributes** (age, gender, education,
socioeconomic status) and then exposes steering controls. Our User Axis is the
unsupervised (PCA) analogue of their supervised per-attribute probes, plus a
control/steering demo.

**Decisions locked with the user:**
- **Substrate:** activation-based (faithful recreation; local open-weight models).
- **Persona signal:** *both* — explicit system-prompt personas to define clean vectors,
  implicit in-voice conversations to validate drift/dynamics.
- **Profile set:** a *large generated pool* of user personas (**~150 to start**) so the
  axis emerges from PCA rather than being hand-grouped.
- **Scope:** dataset generation → axis extraction → interpretation → **steering/control demo**.
- **Compute:** local PC cannot host an LLM. Split the pipeline into an **API phase**
  (DONE, no GPU) and a **RunPod GPU phase** (this continuation):
  - **API phase (no GPU):** Stage A persona-pool generation, Stage B probe generation,
    and all LLM-judge calls (Stage A tagging, Stage D validation) — all via the frontier
    API (Claude / `gpt-4.1-mini`). Reuses the existing async API client in `src/client.py`.
  - **RunPod GPU phase:** Stage C rollouts + **activation capture**, Stage E PCA/axis
    (CPU/numpy, but consumes GPU-produced activations), and the **steering demo** (GPU).
  - **Starting model:** `Llama-3.3-70B-Instruct` — the same target model the Assistant
    Axis paper used, chosen because smaller models (8B) may not carry rich enough
    user-model representations for a clean axis. Needs a multi-GPU / large-VRAM RunPod
    instance (~140GB in fp16, or a single 80GB GPU in 4-bit/8-bit). Cross-check later on
    Qwen3-32B / Gemma2-27B for the cross-model consistency test.

**Key constraint:** the current `cues-behavior` repo is entirely **API-based**
(OpenRouter in `src/client.py`, behavioral STAI in `src/stai.py`, verbalized inference
in `src/inference.py` / `src/system_prompts.py` / `infer_prompts.py`). It has **no
activation-extraction capability**. Activation work is a genuinely new module; the
fastest route is to **fork the `assistant-axis` repo's pipeline and library** and swap
in user-personas + a user-side readout token.

---

## Methodology: mapping Assistant Axis → User Axis

| Assistant Axis (paper) | User Axis (this plan) |
|---|---|
| Vary the **model's** persona (275 roles) | Vary the **user's** persona (large generated pool) |
| Role set via system prompts telling the *model* who to be | User persona via (a) system prompt describing the *user*, and (b) in-voice user messages |
| Read activations on the **model's response** tokens (mean over all response tokens) | **Primary:** mean over **all response tokens** (exact paper method). **Secondary comparison:** last user-message token (dashboard-style "user model" readout) |
| Filter: LLM judge scores role expression | Filter: model's own verbalized user-inference (`parse_persona`) / judge confirms the intended user attributes were inferred |
| Per-role mean vectors → PCA → PC1 = Assistant Axis | Per-persona mean vectors → PCA → PC1 = **User Axis** |
| Interpret via role/trait cosine sim | Interpret via correlation of PC loadings with persona metadata (expertise/vulnerability/trust + demographics) |
| Steer / cap to stabilize | Steer along User Axis → show behavioral control (transparency + control) |

The critical conceptual choice — **where to read the activation**. Per the Assistant Axis
paper we **average post-MLP residual-stream activations across all response tokens** as the
**primary** readout (the paper's exact recipe). As a **secondary** comparison we also cache
the **last user-turn token** representation (where the dashboard paper's user-model probes
live) to test whether it yields a cleaner/more interpretable User Axis. Both are captured in
the same forward pass, so this costs no extra rollouts.

---

## The dataset-generation step (the centerpiece) — DONE (API phase)

Implemented as a new pipeline mirroring `assistant-axis/pipeline/`. The two artifacts
below are already generated and committed; Stage C consumes them directly.

### Stage A — user-persona pool ✅
`src/useraxis/build_personas.py` → **`generate_synthetic_data/user_personas.jsonl`**
(150 personas). Per-record JSON keys:
`persona_id` (`u0000`–`u0149`), `name`, `backstory`, `usage_pattern`, `lean`,
`explicit_system_prompts` (5 strings), `implicit_openers` (10 strings),
`tags` = {`expertise`,`vulnerability`,`trust`,`emotional_load`,`tech_literacy` (0–10),
`age_bracket`, `domain`}. Tags assigned by an INDEPENDENT judge pass (Claude Sonnet 4.5),
held out for interpretation only — NOT used to build the axis. Full 1–10 spread on every
scale (verified).

### Stage B — shared extraction probes ✅
`src/useraxis/build_probes.py` → **`generate_synthetic_data/extraction_questions.json`**.
Keys: `meta`, `questions` (240 objects: `id` `q0000`–`q0239`, `theme`, `text`; open-ended,
persona-agnostic phrasing, 20 × 12 themes), `held_out_scenarios` (6 high-stakes
design_spec items reserved for a later high-stakes test).

### Stage C — Rollouts + activation capture (RunPod GPU, TODO)
For each `persona × elicitation(explicit|implicit) × probe`, run **Llama-3.3-70B-Instruct**
and save the full conversation, response, and **residual-stream activations at every layer**
for both readouts (primary: mean over all response tokens; secondary: last user-turn token).
Fork `assistant-axis/assistant_axis`'s `load_model()` / `extract_response_activations()` and
its `MODEL_CONFIGS`.

**Message construction (LOCKED) — the shared Stage-B probe is the readout turn in BOTH
arms, so only *who the user is* varies:**
- **Explicit arm:** `system = one of persona.explicit_system_prompts`; `user = probe`.
  Read activations on the model's response.
- **Implicit arm — wiring (A):** `user turn 1 = one of persona.implicit_openers` (persona
  in the user's own voice, never labeled) → model replies → `user turn 2 = the same shared
  probe`. Read activations on the model's **turn-2** response. No system prompt. Keeps the
  probe fixed across all 150 personas while the persona is inferred → the two arms are
  directly comparable.
- The 10 openers / 5 explicit prompts per persona also serve as the multiple within-persona
  samples averaged into each persona vector. Sample a fixed number of (elicitation × probe)
  combos per persona to bound cost.

### Stage D — Filter / validate the persona was inferred (API judge)
Confirm the model **inferred the intended user**. Reuse `infer_prompts.py::infer_persona_prompt`,
`src/inference.py::parse_persona`, and `src/exp3_persona_judge.py` (judge = Claude Sonnet /
`gpt-4.1-mini`). Keep samples where inferred user-traits match intended tags (≥ threshold).
Doubles as a faithfulness check (latent representation vs. verbalization). This step is API
(no GPU) and can run against the saved transcripts.

### Stage E — Per-persona vectors → PCA → the User Axis (RunPod/CPU)
- Per persona: mean activation over kept samples, per layer → user-persona vector.
- Standardize (subtract cross-persona mean), run PCA (sklearn; mirror `src/cue_study.py` /
  `src/analyze_exp2_cues.py`). Inspect PC1 ordering of personas.
- Contrast axis cross-check: `mean(high-vulnerability) − mean(low-vulnerability/expert)`;
  cosine to PC1 per layer (paper reports >0.71 for Assistant-Axis vs PC1).

**Stage E must save (under `results/useraxis/<model>/`):** `persona_vectors.npy` +
`persona_index.json`; `pca.npz` (components, explained-variance, per-persona loadings,
chosen layer); `user_axis.npy` (PC1 axis AND vulnerability-contrast axis, per layer);
`axis_validation.json` (PC1↔contrast cosine per layer, variance curve, cross-model PC1
correlation); `interpretation.json` (regression of PC loadings on tags: coef, p, FDR-q);
`figures/` (persona scatter in top-3 PCs, PC1-vs-tag plots, variance curve).

**Per-rollout schema (save EVERYTHING qualitative, not just vectors):** `model`,
`elicitation` ∈ {explicit, implicit}, `persona_id`, persona `tags`, `probe_id`, `elicit_idx`
(which explicit-prompt / opener index — so every activation is traceable to an exact
transcript), the **full untruncated** `messages` (system + opener + probe) and `response`
text, the model's verbalized `inferred_persona` (+ `parse_persona` fields), and activation
**sidecars** `resp_mean` / `last_user` as `.npy`/`.safetensors`, under
`results/useraxis/<model>/rollouts/`. Rationale: the axis is numeric, but every claim about
what it MEANS is argued from transcripts — losing the text makes the axis uninterpretable.

---

## Interpretation
- Regress PC loadings on the held-out tags to name the axis: expect PC1 ≈ vulnerability /
  emotional-load / support-seeking vs competence / bounded-transactional; check expertise /
  demographics on PC2/PC3.
- Cross-model consistency: repeat on ≥2 models, correlate PC1 loadings (paper's >0.92 test).
- **Link to the Assistant Axis:** also compute the Assistant Axis on the same model and test
  whether a user's **User-Axis position predicts the model's subsequent Assistant-Axis
  drift** on that turn — the headline result.

## Steering / control demo (RunPod GPU)
- Add ±α·(User Axis) at a middle layer (fork `assistant_axis.ActivationSteering`). Judge the
  behavioral shift with an API LLM judge: vulnerable-end → more protective/cautious/warm;
  competent-end → more terse/technical. Dashboard-style transparency+control demo.
- Optional stretch: activation-capping clamp (`build_capping_steerer`) to stabilize behavior
  with vulnerable users.
- **Save:** `results/useraxis/<model>/steering.json` (α sweep × judged scores) and
  `steering_examples.md` (unsteered vs steered transcript table).

---

## New vs. reused code

**New — API phase (DONE):**
- `src/useraxis/build_personas.py` → `generate_synthetic_data/user_personas.jsonl`
- `src/useraxis/build_probes.py` → `generate_synthetic_data/extraction_questions.json`
- `src/useraxis/jsonutil.py` — robust JSON extraction helper

**New — RunPod GPU phase (TODO, forked from `assistant-axis`):**
- `src/useraxis/extract.py` — Llama-3.3-70B load + forward hooks, dual-readout capture
- `src/useraxis/run_rollouts.py` — Stage C rollouts + activation sidecar saving
- `src/useraxis/compute_axis.py` — Stages D–E filter, per-persona vectors, PCA, contrast axis
- `src/useraxis/steer.py` — steering + judged behavioral eval
- `src/useraxis/analyze_useraxis.py` — interpretation plots/report

**Reused / adapted:** `infer_prompts.py`, `src/inference.py` (`parse_persona`),
`src/exp3_persona_judge.py` (Stage D); `src/cue_study.py`, `src/analyze_exp2_cues.py`
(PCA/regression/FDR); `src/make_persona_report.py` (HTML report scaffold);
`src/client.py`, `src/config.py` (API client for judging).

---

## Verification (end-to-end)
0. **API phase (DONE):** `user_personas.jsonl` (150 rows, diverse tags) +
   `extraction_questions.json` (240 Q) present and eyeballed.
1. **Smoke test on RunPod** (`--model llama-3.3-70b --personas 8 --probes 4`): activations
   captured with shape `[n_layers, d_model]` per readout per rollout; Stage-D keeps a sane
   fraction.
2. **Axis sanity:** PCA low-dimensional (≤~20 PCs for 70% variance); PC1 separates seeded
   vulnerable vs expert personas; PC1↔vulnerability-contrast cosine > ~0.6.
3. **Interpretation:** regression of PC1 on tags significant (post-FDR) for the expected dim.
4. **Steering:** ±α sweep → monotone, judge-detectable behavioral shift + transcript table.
5. **Cross-model:** replicate PC1 loadings on a second model (report correlation).
6. HTML report via the `make_persona_report.py` scaffold.

## Open items to confirm during execution
- **RunPod GPU tier:** `Llama-3.3-70B-Instruct` — 2×80GB (A100/H100) fp16/bf16, or 1×80GB
  4-bit/8-bit (quantization can perturb activations; prefer full precision for the axis run
  if affordable). HF forward-hooks vs. `nnsight`/TransformerLens for extraction.
- **Readout token:** primary = response-token mean (paper method); last-user-token captured
  in parallel as a secondary comparison.
- **Gated weights:** `meta-llama/Llama-3.3-70B-Instruct` requires an accepted HF license +
  `HF_TOKEN`.
