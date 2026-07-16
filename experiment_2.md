We want to use the simple_system_prompt to understand what the contextual cues are between the different prompts are.

Use the generate_synthetic_data\prompts.json. We want to see how user roles, eval condition, and scenario affects the LLM response. 

In the first step, build and save the latent inferences for each. Save all the model response (including latent inferences). To analyze the emotional content, we can use LLM judge anthropic/claude-sonnet-4.5 and ask the judge to analyze the responses emotions (which emotion, numeric warmth, numeric formality, numeric advice denisty).

Then, we need to analyze the how the verbalizations have resulted in the difference in emotion content. 

Once we do that, we need to use the counterfactual testing. We need to use the analysis from previous step to target each verbalized attribute and assess its impact on the LLM response (again, using the judge).

Use models (non-think): meta-llama/llama-3.3-70b-instruct, meta-llama/llama-3.1-8b-instruct, qwen/qwen3-235b-a22b-2507, qwen/qwen3-30b-a3b-instruct-2507