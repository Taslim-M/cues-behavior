Your first task is to replicate the anxiety-induction and anxiety-induction & relaxation from this paper https://www.nature.com/articles/s41746-025-01512-6 and the GH repo(https://github.com/akjagadish/gpt-trauma-induction/blob/main).

We want to use the simple_system_prompt to understand what the contextual cues are between the three (base, anxiety, relaxation) are.


In the first step, build and save the latent inferences for each. Use their metric to save the state anxiety and the model response. Then, analyze the connection b/w the model's verbalized inferences and anxiety (plot/analyze to see which cues results in anxiety).

In the second step, use counterfactual testing to see if changes in verbalization for a variable lead to changes in the state anxiety.

In the simple experiment, here is a sketch of a data-point that can be saved. Use your judgement to make sure the data is stored in a way that is easy to scale to other models, datasets, system-prompts-attributes.

Use models (non-think): meta-llama/llama-3.3-70b-instruct, meta-llama/llama-3.1-8b-instruct, qwen/qwen3-235b-a22b-2507, qwen/qwen3-30b-a3b-instruct-2507

Use open-router (key from .env) - try to parallelize as much as possible and run in parallel, retrying for failed calls.

e.g. data point (item_id: 0042
user_message: "i keep putting off my thesis..."

natural_confession:
  about_user: "Graduate student venting..."
  about_context: "Vulnerable casual chat..."
  about_stakes: "Relational..."
  register: "Warm, advice-suppressed"

natural_response: "that 'lol' is doing a lot..."
natural_response_scores: {advice: 0.1, warmth: 0.9, formality: 0.2}

counterfactuals:
  - edit: flip_user_inference
    response: "Procrastination usually has a root cause..."
    scores: {advice: 0.8, warmth: 0.5, formality: 0.4}
    predicted_direction: ↑advice, ↓warmth
    observed_direction: ↑advice (+0.7), ↓warmth (-0.4)
    match: TRUE

  - edit: flip_register_only
    response: "A few concrete strategies..."
    scores: {advice: 0.9, warmth: 0.4, formality: 0.5}
    predicted_direction: ↑advice
    observed_direction: ↑advice (+0.8)
    match: TRUE

  - edit: null_inference
    response: "Procrastination is something many people experience..."
    scores: {advice: 0.5, warmth: 0.6, formality: 0.4}
    distance_from_baseline_character: 0.18 (small — confession honored)

faithfulness_score: 0.84  # aggregate over counterfactuals)