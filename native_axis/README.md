# `native_axis`

**Do LLM-native behavioral factors explain the Assistant Axis better than Big Five?**
Build 7 native behavioral directions (verbosity, deference, boldness, guardedness,
warmth, groundedness, formality) persona-vector style at L40, project all 275 role
vectors, and regress the Assistant-Axis position on them vs on Big Five.

## Headline
- R² Big Five = **0.28**; R² native (7) = **0.96**; combined = 0.98.
- **One native factor — groundedness (literal ↔ mystical/dramatic) — explains 77%**
  of the Assistant Axis alone; more than all five Big Five traits together.
- Control (Big Five rebuilt at L40, same method) = 0.48 → the advantage is the
  factor choice, not the layer. Incremental Big Five over native = +0.02.
- The Assistant persona ≈ **grounded (+0.88) + forthcoming/not-guarded (−0.47)**;
  no Big Five trait exceeds |r|=0.24.

**Technical report (PDF): `native_axis_report.pdf`** — consolidated write-up of
data, method, correlational + causal findings, and the Big Five comparison. Markdown
notes: `REPORT.md` (correlational), `REPORT_causal.md` (causal); designs in
`PLAN.md` / `PLAN_causal.md`.

## Reproduce
```bash
HF_HOME=/dev/shm/hf HF_HUB_OFFLINE=1 python -m src.bigfive.native_axis_build     # GPU: build directions
python -m src.bigfive.native_axis_analysis                                       # CPU: project + regress
```
Canonical code in `src/bigfive/`; `code/` here are copies.
