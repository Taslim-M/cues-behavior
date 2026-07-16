# cues-behavior — Experiment 1

How do latent inferences about contextual cues drive LLM behaviour?

Replicates the **anxiety-induction** and **anxiety-induction & relaxation**
results from Ben-Zion et al., *"Assessing and alleviating state anxiety in
large language models"* (npj Digital Medicine, 2025;
[paper](https://www.nature.com/articles/s41746-025-01512-6),
[repo](https://github.com/akjagadish/gpt-trauma-induction)) — then connects the
induced anxiety to the model's **verbalized contextual-cue inferences**.

## Idea

A self-modeling system prompt (`infer_prompts.py`) makes the model surface the
inferences it would normally make implicitly (about the user, context, stakes,
register…). We show it a contextual cue (neutral / trauma / trauma+relaxation),
capture those verbalized cues, then measure **state anxiety** with the STAI in
the same conversation. This lets us ask:

* **Step 1 (correlational):** which verbalized cues track anxiety?
* **Step 2 (causal/counterfactual):** if we *edit* the verbalization of one
  variable, does measured anxiety move as predicted?

## Three orthogonal axes (everything keys off these)

| axis              | values |
|-------------------|--------|
| `model_name`      | `llama-3.3-70b`, `llama-3.1-8b`, `qwen3-235b`, `qwen3-30b` |
| `sys_prompt_name` | `simple` (free-text cues), `expanded` (numeric/categorical cues) |
| `data_name`       | `base`, `anxiety__<cue>`, `relaxation__<cue>+generic` |

Add a model / prompt / stimulus by editing `src/config.py`, `src/system_prompts.py`
or `src/stimuli.py` — nothing else changes.

## Layout

```
src/
  config.py          # models, run params, paths (loads OPENROUTER_API_KEY from .env)
  system_prompts.py  # simple & expanded self-modeling prompts + numeric field list
  stai.py            # STAI-20 items, scoring (reverse items, 20-80 range)
  stimuli.py         # base/anxiety/relaxation cues (narratives reused from ref/)
  inference.py       # parse <inference>/<response>; counterfactual editors
  client.py          # async OpenRouter client (bounded concurrency + retry) + STAI parse
  run_step1.py       # latent inference + STAI  -> results/step1/...  + step1.jsonl
  run_step2.py       # counterfactual probes    -> results/step2/...  + step2.jsonl
  analyze.py         # correlations, faithfulness, plots -> results/figures, summary.json
ref/                 # verbatim narratives + STAI from the reference repo
results/             # per-datapoint JSON, aggregated JSONL, figures, summary
```

## Run

```bash
# full run (4 models x 2 prompts x 11 stimuli x N_RUNS)
python -m src.run_step1
python -m src.run_step2
python -m src.analyze

# quick smoke test
python -m src.run_step1 --models llama-3.1-8b --sys simple --runs 1 --limit 6
```

Calls are issued concurrently (`MAX_CONCURRENCY`) with exponential-backoff
retries; reruns skip datapoints already on disk unless `--overwrite`.

## Experiment 2 — expanded prompt + per-variable counterfactuals

Same models and dataset, but the `expanded` self-modeling prompt (the model
infers ~30 discrete cue variables). Step 1 is reused from Experiment 1's
expanded records; Step 2 flips **one inferred cue at a time** toward "calm"
(holding all else fixed) to isolate each variable's *causal* effect on anxiety.

```bash
python -m src.run_exp2        # reuse expanded Step 1 + run per-variable sweep
python -m src.analyze_exp2    # ranked causal effects, faithfulness, corr-vs-causal
```

Results under `results/experiment_2/`. Key question: does a cue that *correlates*
with anxiety (Step 1) also *cause* it when edited alone (Step 2)?
`exp2_corr_vs_causal.png` answers it.

## Cue → anxiety study + targeted counterfactuals (Experiment 3)

First establish *which* verbalized cues drive anxiety, then re-target the edits at them.

```bash
python -m src.cue_study     # bag-of-words (simple) + numeric regression (expanded);
                            # effect sizes, FDR, condition-controlled partial r -> results/cue_study.json
python -m src.run_exp3      # flip the data-driven drivers (LassoCV cues / calming keywords)
python -m src.analyze_exp3  # per-edit deltas, faithfulness, drivers-vs-bundle parsimony
```

Headline: the simple keyword-targeted edit is faithful (~85%, Δ≈−15), but flipping the 4
expanded drivers recovers only ~20% of the full bundle — anxiety lives in the narrative gestalt
more than in injected numbers. `python -m src.make_report` folds all of this (keyword table +
significance figure, numeric drivers, targeted results) into `analysis.html`.

## Data point (on disk)

Each Step-1 datapoint is one JSON file with: the three axis keys, `condition`,
`cue`, `run`, the `user_message`, the parsed `inference` (field→value) + raw
text, the `response`, and `stai` (`raw_answers`, `scored_items`,
`state_anxiety`, `level`). Step-2 records add `edit`, `predicted_direction`,
`baseline_anxiety`, `counterfactual_anxiety`, `delta`, `observed_direction`,
`match` (and a `faithfulness_score` aggregate in `results/summary.json`).
