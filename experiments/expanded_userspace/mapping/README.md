# Expanded user-space → internal traits & personas (independent)

Maps 289 decorrelated-factorial users to the model's evoked Big Five,
Assistant-Axis, and nearest LLM persona — by projecting already-captured
`resp_mean` activations (no new generation). Independent of the 150-user study.

## Headline
- The Assistant-Axis is driven by **age (η²=0.23)** and **competence (0.14)** —
  **not** vulnerability (0.01) or emotional_load (0.00). The 150-user "vulnerability
  → warmth" story was a proxy for younger/less-expert users.
- Clean one-factor levers: **Openness ← competence**, **Stability ← emotional_load**,
  **Agreeableness ← comm_style**, **Extra/Consc ← trust**.
- Factors act near-independently (joint R² ≈ Σ single η²).
- Stance is set by *who the user is*, not the topic (domain AA η²=0.04).

See `REPORT.md` (full), `PLAN.md` (design).

## Reproduce
```bash
python -m src.useraxis.expanded_mapping           # project resp_mean -> persona_map.json
python -m src.useraxis.expanded_factor_analysis   # eta^2 / OLS / role assoc
```
Reads the expanded_userspace rollouts + Big Five bank + AA vector + role vectors
(/dev/shm/aa_vectors). Canonical code in `src/useraxis/`; `code/` are copies.
