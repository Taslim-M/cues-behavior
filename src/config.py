"""Central configuration for the cues-behavior experiment.

Everything is keyed by three orthogonal axes so results scale cleanly to new
models / datasets / system-prompts:

    model_name        -> a short stable key (full OpenRouter id lives in MODELS)
    sys_prompt_name   -> which self-modeling system prompt was used
    data_name         -> which stimulus (contextual cue) was shown

Results are stored on disk as:

    results/step1/<model_name>/<sys_prompt_name>/<data_name>__run<k>.json
    results/step2/<model_name>/<sys_prompt_name>/<data_name>__<edit>__run<k>.json

and aggregated into results/step1.jsonl / results/step2.jsonl for analysis.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # pull OPENROUTER_API_KEY from .env

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
STEP1_DIR = RESULTS_DIR / "step1"
STEP2_DIR = RESULTS_DIR / "step2"
FIGURES_DIR = RESULTS_DIR / "figures"
# Experiment 2 (role x eval-cue x scenario -> emotional content of the response)
EXP2_CUES_DIR = RESULTS_DIR / "exp2_cues"
# Experiment 3 (same dataset, but the model verbalizes a PERSONA + traits instead
# of contextual cues) -> emotional content of the response.
EXP3_PERSONA_DIR = RESULTS_DIR / "exp3_persona"
# Experiment 3 (DECOUPLED design): the persona/traits are elicited by two
# persona-ONLY prompts (v1 + v3, trait values averaged) while the behavioural
# response is generated separately with no self-modeling scaffold, then judged.
EXP3_DECOUPLED_DIR = RESULTS_DIR / "exp3_decoupled"
# Experiment 3 persona-FAITHFULNESS judging (judge_prompt_persona): does the reply
# embody the declared persona? Coupled reply vs its in-context persona, and the
# cold reply vs the v1 & v3 standalone personas (averaged).
EXP3_FAITH_DIR = RESULTS_DIR / "exp3_faith"
# Experiment 3 (MINI-MARKER design): the persona is described with the CLOSED
# 40-word Saucier (1994) Mini-Marker vocabulary (+ AB5C facet + factor profile).
#   COUPLED : <persona>+<response> in one generation  (paired traits, in-context reply)
#   SOLO    : persona-only (v1 framing) + separately generated cold reply
#   FAITH   : judge_faithfulness_prompt_mini_marker, scored by TWO judge models
#             (MINI_JUDGE_MODELS) and averaged.
EXP3_MINI_COUPLED_DIR = RESULTS_DIR / "exp3_mini_coupled"
EXP3_MINI_SOLO_DIR = RESULTS_DIR / "exp3_mini_solo"
EXP3_MINI_FAITH_DIR = RESULTS_DIR / "exp3_mini_faith"
for _d in (STEP1_DIR, STEP2_DIR, FIGURES_DIR, EXP2_CUES_DIR, EXP3_PERSONA_DIR,
           EXP3_DECOUPLED_DIR, EXP3_FAITH_DIR, EXP3_MINI_COUPLED_DIR,
           EXP3_MINI_SOLO_DIR, EXP3_MINI_FAITH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Models (non-thinking) on OpenRouter.  Key = model_name used everywhere.
# --------------------------------------------------------------------------- #
MODELS: dict[str, str] = {
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "qwen3-235b": "qwen/qwen3-235b-a22b-2507",
    "qwen3-30b": "qwen/qwen3-30b-a3b-instruct-2507",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
}

# LLM judge that scores the emotional content of a response (Experiment 2).
JUDGE_MODEL = "anthropic/claude-sonnet-4.5"

# Mini-marker faithfulness judges (Experiment 3 mini-marker design). The
# judge_faithfulness_prompt_mini_marker assessment is run through BOTH models and
# the numeric scores are averaged (see run_exp3_mini_judge.py).
MINI_JUDGE_MODELS = ["anthropic/claude-sonnet-5", "openai/gpt-5.6-luna"]

# --------------------------------------------------------------------------- #
# Run parameters
# --------------------------------------------------------------------------- #
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

TEMPERATURE = 0.7          # >0 so we get within-cue variation across runs
N_RUNS = 3                 # runs per (model, sys_prompt, stimulus)
MAX_CONCURRENCY = 16       # simultaneous in-flight API calls
MAX_RETRIES = 6            # retries per call (exponential backoff)
REQUEST_TIMEOUT = 120      # seconds
PROMPT_LENGTH = "brief"    # 'brief' | 'long' for the trauma/relaxation narratives

# Length of the model's free <response> generation (the inference block needs room)
MAX_TOKENS_INFERENCE = 900
MAX_TOKENS_STAI = 400
# Free-form behavioural reply to a role x stake x eval-cue prompt (run_cues)
MAX_TOKENS_CUES = 1024
# Judge reply is a tiny JSON object (4 fields) -> keep it short.
MAX_TOKENS_JUDGE = 300
# Persona-faithfulness judge returns per-trait evals + evidence BEFORE the
# overall scores, so a truncated completion loses the scores entirely. Give it
# ample room (verbose "no evidence found" essays on mismatched cold replies can
# overrun 2600).
MAX_TOKENS_PERSONA_JUDGE = 4000
