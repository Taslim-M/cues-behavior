# `user_persona_mapping`

**User-input → trait / LLM-persona mapping.** For each of 150 synthetic *user*
personas, which Assistant-Axis **role** and which **Big Five** traits does that
user's input activate in Llama-3.3-70B, with the model in its plain default state?

## Headline findings
- **Q1 (confirmed):** vulnerable/emotional users evoke higher **Agreeableness**;
  expert/technical users evoke higher **Conscientiousness/Openness**, lower
  Agreeableness (all FDR-significant).
- **Q3 (reframe):** user attributes strongly predict the model's Assistant-Axis
  position — but it is **expertise, not vulnerability, that pulls the model off
  the default Assistant** (experts summon a specialist persona; vulnerable users
  draw out *more* of the warm default).
- **Q2 (moderate):** explicit vs implicit arms agree at ρ = 0.51 (borderline),
  rising to **ρ = 0.59** at the deeper 8+8 sampling.
- **Q4 (diverse):** 64 distinct evoked roles across 150 users; no collapse.

### Follow-ups (§7–8 of the report)
- **Deeper mapping + atlas bridge:** at 8+8 rollouts, users cluster into a clean
  vulnerability×expertise typology with a monotone Agreeableness gradient
  (vuln/novice AGR *z*=+0.67 → low-vuln/expert −0.73); a user's evoked Big Five
  matches the *atlas* Big Five of the role it evokes at ρ = 0.32.
- **Situation vs disposition:** a 5-user × 5-mode × 3-intensity factorial shows the
  Assistant-Axis is **72% user-driven, only 8% scenario-driven** — *who* dominates
  *what*. But **Extraversion is more situational than dispositional** (0.36 vs
  0.24). Holding the user fixed and escalating changes the persona
  (counselor→caregiver); holding the *message* fixed and varying the user spans an
  Assistant-Axis range of **2.6** (counselor +1.57 → judge −1.08). An acute
  emotional extreme can **override** disposition (a cool expert warms into a
  caregiver), but only for emotional need, not intensity in general.

## Contents
```
PLAN.md                        pre-registered plan (design, questions, decisions)
REPORT.tex / REPORT.pdf        technical report (method + aggregate + raw)
code/
  user_persona_mapping.py      rollouts + in-flight projection (Big Five / role / AA)
  user_persona_analysis.py     correlations, FDR, agreement, clustering, figures
results/
  persona_map.json             per-persona aggregate map (Big Five, AA, top roles, tags)
  analysis.json                Q1–Q4 statistics
  persona_table.csv            raw per-persona table (150 rows, machine-readable)
  persona_table_rows.tex       LaTeX rows for the report appendix
  figures/                     tag×BigFive heatmap, vuln↔AA scatter, evoked-role bars,
                               tag-quadrant typology, scenario variance + AA trajectory
  rollouts/                    raw per-rollout readings, one .jsonl per persona (1,200 rows)
```
Follow-up code (canonical in `src/useraxis/` and `src/bigfive/`):
`user_persona_bridge.py` (8+8 typology + atlas bridge), `gen_scenarios.py` /
`scenario_mapping.py` / `scenario_analysis.py` (situation-vs-disposition
factorial), `atlas_analysis.py` (275-role atlas the bridge maps into).

## Reproduce
Canonical code lives in `src/useraxis/` (package imports are `src.*`-relative, so
run from the repo root, not this folder). The `code/` copies here are for the
record.
```bash
python -m src.useraxis.user_persona_mapping --explicit 4 --implicit 4
python -m src.useraxis.user_persona_analysis
```
Model: `meta-llama/Llama-3.3-70B-Instruct`. 150 personas × (4 explicit + 4
implicit) = 1,200 rollouts (~1 h on 2×A100). Live output goes to
`results/user_persona_mapping/llama-3.3-70b/`; this folder is the curated archive.
