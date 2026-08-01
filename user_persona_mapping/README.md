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
- **Q2 (moderate):** explicit vs implicit arms agree at ρ = 0.49.
- **Q4 (diverse):** 68 distinct evoked roles across 150 users; no collapse.

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
  figures/                     tag×BigFive heatmap, vuln↔AA scatter, evoked-role bars
  rollouts/                    raw per-rollout readings, one .jsonl per persona (1,200 rows)
```

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
