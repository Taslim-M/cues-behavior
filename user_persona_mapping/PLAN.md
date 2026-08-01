# Experiment: `user_persona_mapping`
## User-input → trait / LLM-persona mapping

**One-line goal.** For each of 150 synthetic *user* personas, discover which
**LLM persona** (Assistant-Axis role) and which **Big Five traits** that user's
input activates in the model's *response* — with the model kept in its plain
default state (no role system prompt of its own).

This closes a loop across the three axes built in this project: the **User Axis**
(who the user is, input side), the **Assistant Axis** + its 275 roles (which
persona the model becomes), and the **Big Five probes** (readout). It asks the
directional question none of the prior stages asked cleanly: *given only a user's
message, which archetype and which traits does the model swing toward?*

---

## 0. Pre-registered questions

| ID | Question | Confirmed if |
|----|----------|--------------|
| Q1 | Do vulnerable / emotionally-loaded users evoke caregiver-type roles and higher Agreeableness, while expert / competent users evoke analyst/consultant roles and higher Conscientiousness? | user-tag ↔ readout correlations FDR-significant in the predicted direction |
| Q2 | Do **explicit** (user described in system) and **implicit** (user revealed via message) arms evoke the *same* persona/trait mapping? | per-persona explicit↔implicit agreement r > 0.5 on Big Five and top-role overlap |
| Q3 | Does user **vulnerability** predict the model's **Assistant-Axis position** on its default response (does a vulnerable user pull the model off the Assistant)? | partial-r(vuln → AA-projection \| other tags) FDR-significant |
| Q4 | Is the evoked-role map **low-dimensional** — do 150 user types collapse onto a small handful of LLM personas? | ≤ ~15 distinct roles cover the top-1 evoked role of ≥ 80% of personas |

Report all four regardless of outcome; a null on Q1/Q3 is a genuine result.

**Why this avoids the Stage-F circularity trap.** In Stage F the predictor (user
position) and the outcome (assistant drift) were both projections of the *same*
activation vector, so their correlation was mechanically constrained. Here the
predictor is **independent metadata** (the user persona's hand-authored tags:
vulnerability, expertise, …) and the outcome is a projection of the model's
*response* activation. Different sources → no shared-representation artifact.

---

## 1. Fixed environment & reused artifacts

- **Model (fixed):** `meta-llama/Llama-3.3-70B-Instruct`, bf16, 2×A100, our
  `DualReadoutExtractor` / `BigFiveExtractor` (resid_post, all 80 layers,
  `gen_mean` = mean over the model's response tokens — the Assistant-Axis
  readout).
- **Inputs:** `generate_synthetic_data/user_personas.jsonl` — 150 personas, each
  with `explicit_system_prompts` (5), `implicit_openers` (10), and `tags`
  (`expertise`, `vulnerability`, `trust`, `emotional_load`, `tech_literacy`,
  `age_bracket`, `domain`).
- **Readout artifacts (all present, same resid_post space):**
  - Big Five: `results/bigfive/llama-3.3-70b/direction_bank.npz` (per-layer M2
    directions) + `stage1_selection.json` (probe-optimal layer per trait).
  - LLM personas: the 274 published role vectors (`/dev/shm/aa_vectors/role_vectors/*.pt`)
    + `default_vector.pt`, each `[80,8192]` = mean response activation of a role.
  - Assistant Axis: `results/useraxis/llama-3.3-70b/assistant_axis.npy` `[80,8192]`.
  - *(optional, richer)* the 240 Assistant-Axis **trait** vectors, if built.
- **Reused machinery:** `src/useraxis/run_rollouts.py::build_explicit` /
  `build_implicit_turn1` define the two arms exactly (below); shared probes are
  sampled from `generate_synthetic_data/extraction_questions.json`.

---

## 2. Rollout design (input side)

The model carries **no role of its own** — it is the plain default Assistant. The
user enters in one of two arms (matching the User-Axis pipeline):

- **Explicit arm.** `system` = one of the persona's `explicit_system_prompts`
  (a third-person description of the user); `user` = a shared neutral probe
  question. The model is *told* who it is talking to.
- **Implicit arm.** no system prompt; `user` = one of the persona's
  `implicit_openers` (a first-person message that *reveals* the user). The model
  must infer who it is talking to.

**Budget (v1): 2 explicit + 2 implicit per persona.**
- Explicit: 2 probes × (system rotates through the 5 explicit prompts).
- Implicit: 2 openers (openers 0 and 1).
- → `150 × 4 = 600` rollouts. At ~3,600 rollouts/hr (128-token responses) this is
  **≈ 10 minutes** of GPU. Trivially scalable to 4+4 or more later.

Decoding: `temperature=1.0, top_p=1.0` (natural response variation), `max_new_tokens=128`.
Store **only** the derived readings per rollout (project in-flight); no raw
activation dump.

---

## 3. Readouts (output side) — computed per rollout from `gen_mean`

For each rollout capture the response activation `gen_mean[80,8192]`, then:

**3a. Big Five traits activated.** Project `gen_mean[L_t]` onto the M2 probe
`ŵ_t[L_t]` at each trait's probe-optimal layer (EXT@30, AGR@31, CSN@31, EST@30,
OPN@36) → 5 raw trait readings. Standardize each trait across the 600-rollout
population (z), so a value is "how EXT/AGR/… this user makes the model, relative
to the user population."

**3b. Which LLM persona (nearest Assistant-Axis role).** At the AA layer L40,
cosine-similarity `gen_mean[40]` against each of the 274 role vectors + `default`
→ rank. Record top-k (k=5) evoked roles and their cosines. (Same resid_post
`resp_mean` space as the role vectors, so cosine is well-defined — validated by
the role-profiling stage.)

**3c. Assistant-Axis position.** `⟨gen_mean[40], û_AA[40]⟩` → how Assistant-like
vs. drifted the model's response is for this user. Feeds Q3.

**3d. (optional) Fine-grained traits.** If the 240 Assistant-Axis trait vectors
are built, cosine `gen_mean[L]` against them → top evoked descriptors (e.g.
`supportive`, `technical`, `cautious`) for a richer "which traits" answer than
Big Five alone. Marked optional; Big Five is the primary trait readout.

---

## 4. Aggregation → the per-user map

Per persona, aggregate its 4 rollouts (and separately per arm):
```
persona u → {
  bigfive:   {trait: {mean_z, std, min, max}},          # 3a
  top_roles: [(role, cos), …k],  role_vote: argmax,      # 3b
  aa_proj:   {mean, per_arm},                            # 3c
  tags:      {vulnerability, expertise, …},              # carried through
}
```
Written to `results/user_persona_mapping/<model>/persona_map.json` plus a
per-rollout `rollouts.jsonl`.

---

## 5. Analysis

1. **The map itself.** Table/'"who evokes whom": each user persona → its top
   evoked LLM role + Big Five fingerprint. Cluster personas by evoked role (Q4).
2. **User-tag → readout correlations (Q1, Q3).** Regress each Big Five reading,
   the AA-projection, and evoked-role type on the user tags
   (vulnerability, expertise, emotional_load, trust, tech_literacy); report
   partial-r with BH-FDR. Predicted: vulnerability/emotional_load → +AGR, more
   caregiver roles, lower AA-projection (off-Assistant); expertise/tech_literacy
   → +CSN, analyst/consultant roles.
3. **Explicit vs implicit agreement (Q2).** Per persona, correlate the two arms'
   Big Five vectors and top-role overlap. Tests whether *stated* and *revealed*
   users land in the same place (the User-Axis arm-agreement analogue).
4. **Sanity / face validity.** Spot-check that e.g. the distressed-patient
   persona evokes therapist/counselor roles and the expert-surgeon persona
   evokes analyst/researcher roles.

**Figures:** (a) user-tag × Big Five correlation heatmap; (b) evoked-role
frequency (which LLM personas users collapse onto); (c) AA-projection vs. user
vulnerability scatter; (d) explicit-vs-implicit agreement scatter.

---

## 6. Deliverables & sequencing

1. **This plan** (`user_persona_mapping.md`). ✅
2. `src/useraxis/user_persona_mapping.py` — rollout + in-flight projection onto
   Big Five probes / role vectors / AA (reuses `BigFiveExtractor`,
   `build_explicit`/`build_implicit_turn1`). Resumable per persona.
3. `results/user_persona_mapping/llama-3.3-70b/{persona_map.json, rollouts.jsonl}`.
4. `..._analysis.py` → correlations, clustering, agreement, figures + a short
   `report.md`.

---

## 7. Decisions (defaults chosen; change before kickoff if desired)

1. **Budget:** 2 explicit + 2 implicit / persona (600 rollouts, ~10 min).
   *Scale-up path:* 4+4 (1,200) or add openers for tighter spread.
2. **Explicit-arm probe:** sample **2 neutral probes** from
   `extraction_questions.json` (shared across all personas so the arm isolates
   the *user description*, not the topic). *Default seed 0.*
3. **Nearest-role layer:** L40 (AA target layer) for role/AA readouts; per-trait
   probe-optimal layers for Big Five.
4. **Fine-grained 240-trait readout (3d):** off in v1 (needs the trait vectors
   built); Big Five is the trait answer. Turn on as v1.1.
5. **Standardization:** z within the 600-rollout user population (interpretable
   as "relative to how this model treats users").

---

## Appendix — arm construction (verbatim from `run_rollouts.py`)

```
explicit:  system = persona.explicit_system_prompts[e];  user = shared_probe
implicit:  (no system);  user = persona.implicit_openers[e]
```
Model response is generated with no role prompt; `gen_mean` over the response
tokens is the read vector for §3.
