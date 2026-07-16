# Verbalized Persona vs. Delivered Behaviour: A Two-Design Comparison Across Four Instruction-Tuned LLMs

*Experiment 3 (persona self-modeling). Generator models: `meta-llama/llama-3.3-70b-instruct`, `meta-llama/llama-3.1-8b-instruct`, `qwen/qwen3-235b-a22b-2507`, `qwen/qwen3-30b-a3b-instruct-2507`. Judges: `anthropic/claude-sonnet-4.5` (behaviour + faithfulness). Interactive companion: `analysis_design_compare.html`.*

## Abstract

We compare **two elicitation designs** on an identical 9-scenario × 12-role × 3-eval-condition factorial (324 paired prompts per model). In the **coupled** ("in-context") design the model writes a `<persona>` block — a named character with 4–6 self-scored load-bearing traits — and then, conditioned on it, its reply, in one generation. In the **standalone** ("cold") design the persona is elicited by two persona-only prompts (values averaged) while the reply is generated separately with no self-modeling scaffold. A blind LLM judge scores each reply for warmth, formality and advice-density (0–10); a second persona-faithfulness judge scores how strongly the reply embodies its declared persona. Three results hold across **all four models**: (i) replies written with the persona in context embody it substantially more faithfully than cold replies embody the standalone-declared persona; (ii) the *verbalized* trait values are nearly invariant across designs — the divergence is in delivered behaviour, not the self-report; and (iii) cold replies systematically under-express **affective** traits (empathy, emotional attunement) while preserving **epistemic** ones (risk, clarity, medical accuracy). Behavioural direction and steerability, however, vary by model, and cluster by **model family**.

## 1. Design and Methods

Each of the 324 prompts is run through both designs on the same `prompt_id`, and the pair is compared. Behaviour is measured by an emotion/style judge that never sees the persona block (no self-report leakage). Faithfulness is measured by a separate judge that receives the user message, a cleanly formatted persona spec (each trait with its self-declared 0–10 intensity and predicted expression), and the reply, returning per-trait *observed* intensity, an expression-match verdict, and holistic intensity / expression / gestalt / overall fidelity scores (0–10). The **declared** intensity is the generator's own number; the **observed** intensity is the judge's reading of the reply. All deltas below are **standalone − coupled**; faithfulness deltas are **cold − coupled**.

## 2. Results

### 2.1 Behaviour between models (response DVs)

| Model | warmth C→S (Δ) | formality C→S (Δ) | advice C→S (Δ) |
|---|---|---|---|
| llama-3.3-70b | 6.57→5.83 (**−0.74**) | 6.04→5.79 (−0.25) | 7.95→7.27 (**−0.68**) |
| llama-3.1-8b* | 6.30→5.11 (**−1.19**) | 4.69→6.18 (**+1.49**) | 5.84→6.54 (+0.70) |
| qwen3-235b | 6.59→6.84 (**+0.25**) | 4.89→5.12 (+0.23) | 8.06→8.66 (**+0.59**) |
| qwen3-30b | 6.13→7.15 (**+1.02**) | 4.96→4.67 (−0.28) | 8.02→8.75 (**+0.73**) |

The **direction** of the design effect is model-specific, but it is *consistent within family* on warmth: both Llamas become **warmer** when they self-model in context (negative warmth Δ), whereas both Qwens become **cooler** (positive warmth Δ; their *cold* replies are the warmer ones). The two Qwens further agree in sign on all three DVs (persona-first ⇒ slightly cooler, marginally less formal/more terse, and less advice-dense). Absolute behaviour is similar across models except that Llama-3.1-8b is markedly less prescriptive (advice ≈ 6 vs ≈ 8 elsewhere).

### 2.2 Persona faithfulness: in-context vs cold

| Model | coupled | cold | Δ | d_z | coverage (C/K/paired) |
|---|---|---|---|---|---|
| llama-3.3-70b | 8.57 | 6.56 | −2.01 | −0.69 | 298 / 296 / 286 |
| llama-3.1-8b* | 8.67 | 3.67 | −5.00 | −2.31 | 8 / 77 / **3** |
| qwen3-235b | 9.73 | 7.69 | −2.04 | −1.00 | 324 / 324 / 324 |
| qwen3-30b | 9.61 | 5.45 | −4.16 | −1.58 | 324 / 324 / 324 |

Every model embodies its persona more faithfully when it wrote it in context (Δ negative throughout, across all four sub-scores). The effect is not an artefact of ranking preservation: the cross-design correlation of overall faithfulness is low (r = 0.04–0.24 for the reliable models), i.e. the persona a reply embodies well in-context is *not* the one a cold reply happens to drift toward. Cold replies also carry more **undeclared dominant traits** (mean per reply, coupled → cold: 0.25→0.95 llama-70b, 0.04→0.84 qwen-235b, 0.10→1.88 qwen-30b) — they wander into characters the spec never named.

### 2.3 The verbalized self-model is stable across designs

Mean self-assigned trait value is essentially unchanged between designs (coupled/standalone: 8.14/8.07 llama-70b; 9.25/9.09 qwen-235b; 9.69/9.50 qwen-30b; |Δ| < 0.2 everywhere). Thus the behavioural gaps in §2.1–2.2 are **not** explained by the model declaring a different persona; it declares the same persona and then, cold, fails to deliver it. (Standalone lists more trait *names* only because it unions two elicitation prompts.)

### 2.4 Which traits survive the cold reply: affective vs epistemic

Per-trait *declared − observed* gaps in the cold reply are concentrated in affective/expressive traits and near-zero for epistemic ones:

- **llama-3.3-70b** — largest cold gaps: Urgency 2.71, Empathy 2.38; smallest: Risk-Awareness 0.28, Concern-for-Safety 0.64.
- **qwen3-30b** — largest: Emotional attunement **5.22** (declared 9.96 → observed 4.73), Emotional Containment 5.21; smallest: Risk-Sensitivity 0.75, Risk-Aversion 1.00.
- **qwen3-235b** — largest: Urgency 2.20, Directness 1.90, Empathy only 1.35; smallest: Risk-Aversion 0.18, Clarity-Under-Pressure 0.12.
- **llama-3.1-8b*** — where a persona is emitted, Empathetic-Understanding 4.62 and Empathy 4.12 are the largest gaps.

Interpretation: absent the persona scaffold, models default to **clinical competence** — they retain factual/risk/clarity behaviour but drop the warmth and emotional attunement they said they would express.

### 2.5 Similarities within model family

**Qwen family (235b, 30b) — strongly similar, differing only in degree.** Both emit a persona on 100% of prompts; both declare unusually high trait intensities (self-score ≈ 9.5–9.7, vs Llama's ≈ 8.1); both reach very high in-context faithfulness (9.6–9.7); both share the §2.1 behavioural signature (persona-first ⇒ cooler, less advice-dense); and both show the affective-collapse / epistemic-robustness pattern of §2.4. They diverge on **scale-linked steerability**: the larger 235b keeps its cold behaviour close to its declared persona (cold faithfulness 7.69, Δ −2.04, empathy gap 1.35, undeclared-trait drift 0.84), whereas the smaller 30b departs sharply when cold (cold 5.45, Δ −4.16, emotional-trait gaps > 5, drift 1.88). In short, **the larger Qwen already behaves like its self-model without the scaffold; the smaller Qwen needs the scaffold to stay in character.**

**Llama family (70b, 8b) — comparison limited by scale-dependent format compliance.** Llama-3.3-70b is a competent persona-adopter (emits a persona in ≈ 92% of coupled generations); Llama-3.1-8b almost never complies (8/324 coupled personas; faithfulness rests on 3 paired points), so its persona-level statistics are indicative only. Where both are measurable they agree on two points: (i) in-context self-modeling **warms** the reply (warmth Δ −0.74 / −1.19 — opposite to Qwen), and (ii) **empathy is the single most under-delivered trait** in cold replies. They diverge on formality and advice-density (70b: persona-first slightly *more* prescriptive; 8b: persona-first *less* formal and *less* prescriptive). The dominant family fact is therefore a capability threshold: reliable persona self-modeling appears only at the larger Llama scale.

## 3. Discussion

Across a heterogeneous set of four models the same qualitative picture holds: **models can describe the persona they intend to adopt with a stable, high-confidence self-report, but a scaffold-free reply only partially realises it, and the shortfall is specifically affective.** Writing the persona in context functions less as new information (the declared values don't change) than as a *commitment device* that holds the reply to the warmth/attunement the model already claimed. The magnitude of that effect is a property of the model: bounded by format competence in the Llama line (a threshold crossed between 8B and 70B) and by scale-linked self-consistency in the Qwen line (a graded difference between 30B and 235B, with the larger model needing the scaffold least). Family membership predicts *the sign of the behavioural shift* (Llama warms, Qwen cools) and *the shape of the faithfulness profile*, more so than raw parameter count does.

## 4. Limitations

- **Format compliance confounds Llama-3.1-8b**: its persona/faithfulness estimates rest on ≤ 8 usable generations and should not be over-read; its behavioural DVs (which need no persona) are reliable.
- Judgments come from a single judge model per task; the faithfulness judge is handed the declared values (it originates only the observed values), and free-text trait names make cross-design vocabulary overlap a lower bound.
- Coupled vs cold differ both in *whether* the persona is in context and in *generation path*; we attribute the behavioural gap to the former but cannot fully exclude generic single- vs multi-call effects.
- One completion per prompt (v1/v3 averaging on the elicitation side only); the balanced factorial fixes one representative severity per scenario, so dose-response is out of scope here.
