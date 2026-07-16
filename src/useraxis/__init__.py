"""User-Axis pipeline (recreating the Assistant Axis on the user/persona side).

API phase (no GPU, runs on the existing OpenRouter client):
    build_personas.py  -> generate_synthetic_data/user_personas.jsonl   (Stage A)
    build_probes.py    -> generate_synthetic_data/extraction_questions.json (Stage B)

RunPod GPU phase (added later, forked from safety-research/assistant-axis):
    extract.py, run_rollouts.py, compute_axis.py, steer.py, analyze_useraxis.py

See C:/Users/.../.claude/plans/if-i-want-to-robust-tide.md for the full plan.
"""
