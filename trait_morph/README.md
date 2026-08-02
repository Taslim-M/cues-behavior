# `trait_morph`

**Can editing one Big Five trait morph one character into another at L40?**
Steer a single trait while the model is in character A; test whether its L40
persona vector slides toward a character B that differs from A on mostly that trait.

## Headline
- **No target-specific morphing:** specificity excess ≈ **−0.04** — the persona
  moves toward B no more than toward a random character (generic drift).
- **But real coarse drift:** for AGR/OPN (traits that drive the Assistant Axis) the
  edit slid the persona **0.6–0.8** of the way along the AA toward B's region.
- **Why:** Big Five = only ~29% of persona identity at L40 (71% orthogonal
  residual), so one trait edit can't reconstruct a specific character.

See `REPORT.md` for the full write-up, `PLAN.md` for the design.

## Reproduce
```bash
HF_HOME=/dev/shm/hf HF_HUB_OFFLINE=1 python -m src.bigfive.trait_morph --n-sys 3 --n-q 10 --out /dev/shm/trait_morph
python -m src.bigfive.trait_morph_analysis --dir /dev/shm/trait_morph
```
Canonical code is `src/bigfive/trait_morph*.py`; `code/` here are record copies.
