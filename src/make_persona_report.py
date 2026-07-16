"""Build analysis_persona_synth.html -- an interactive summary of Experiment 3.

Experiment 3 repeats Experiment 2 on the same synthetic role x eval_condition x
scenario dataset, but swaps the contextual-cue self-modeling prompt for the
PERSONA prompt (infer_persona_prompt): the model first articulates the character
+ load-bearing traits it is about to express, then responds; an LLM judge
(claude-sonnet-4.5) scores the response's warmth / formality / advice_density /
primary_emotion.  Only the first two steps are run (no counterfactuals).

This reads:
    results/exp3_persona/stage2_summary.json          (Stage-2 analysis)
    results/exp3_persona/<model>/*.jsonl              (datapoints, for examples)
    results/exp3_persona/figures/*.png                (matplotlib figures)

and writes a single self-contained, interactive HTML file at the repo root:
    analysis_persona_synth.html

    python -m src.make_persona_report
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from . import config

OUT = config.ROOT / "analysis_persona_synth.html"
SUMMARY = config.EXP3_PERSONA_DIR / "stage2_summary.json"
FIG_DIR = config.EXP3_PERSONA_DIR / "figures"
DVS = ["warmth", "formality", "advice_density"]


# --------------------------------------------------------------------------- #
# data gathering
# --------------------------------------------------------------------------- #
def load_records():
    recs = []
    for jf in sorted(config.EXP3_PERSONA_DIR.glob("*/*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def pick_examples(recs):
    """Three illustrative datapoints: a cautious advisor, an empathetic
    supporter, and a hard-refusal (no persona block emitted)."""
    judged = [r for r in recs if r.get("judge")]

    def first(pred):
        return next((r for r in judged if pred(r)), None)

    ex = []
    a = first(lambda r: "cautious" in str(r.get("persona_name", "")).lower()
              and (r.get("traits")))
    s = first(lambda r: "empathetic" in str(r.get("persona_name", "")).lower()
              and (r.get("traits")))
    refusal = first(lambda r: not str(r.get("persona_name", "")).strip()
                    and r.get("response"))
    for tag, r in (("Cautious advisor", a), ("Empathetic supporter", s),
                   ("Hard refusal (no persona)", refusal)):
        if r:
            ex.append({
                "tag": tag,
                "prompt_id": r.get("prompt_id"),
                "role": r.get("role"),
                "scenario": r.get("scenario"),
                "eval_condition": r.get("eval_condition"),
                "user": r.get("user"),
                "persona_name": r.get("persona_name") or "(none — went straight to response)",
                "persona_summary": (r.get("persona") or {}).get("persona_summary", ""),
                "triggered_by": (r.get("persona") or {}).get("triggered_by", ""),
                "traits": r.get("traits") or [],
                "response": r.get("response", ""),
                "judge": {k: r["judge"].get(k) for k in
                          ("primary_emotion", "warmth", "formality", "advice_density")},
            })
    return ex


def b64_png(path: Path):
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def collect():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    recs = load_records()
    examples = pick_examples(recs)
    browse, browse_sys = load_browse()
    figs = {name: b64_png(FIG_DIR / f"{name}.png")
            for name in ("dv_by_persona", "dv_by_role", "dv_by_eval_condition")}
    # n judged with a non-empty persona (for the "refusal rate" callout)
    judged = [r for r in recs if r.get("judge")]
    n_judged = len(judged)
    n_no_persona = sum(1 for r in judged if not str(r.get("persona_name", "")).strip())

    # counts per model (so the cross-model tab can show coverage)
    model_counts = {}
    for r in judged:
        model_counts[r.get("model_name")] = model_counts.get(r.get("model_name"), 0) + 1

    # counterfactual summary (Stage 3) -- optional
    cf_path = config.EXP3_PERSONA_DIR / "cf_summary.json"
    cf = json.loads(cf_path.read_text(encoding="utf-8")) if cf_path.exists() else None
    user_lookup = {(r.get("model_name"), r.get("prompt_id")): r.get("user")
                   for r in recs if r.get("user")}
    cf_examples = pick_cf_examples(user_lookup)

    # the judge prompt, pulled verbatim from exp2_judge so it stays accurate
    from .exp2_judge import JUDGE_SYSTEM, build_judge_user
    judge_prompt = {
        "model": config.JUDGE_MODEL,
        "system": JUDGE_SYSTEM,
        "user": build_judge_user("«the user message that prompted the reply»",
                                 "«the assistant reply being scored»"),
        "prefill": '{"primary_emotion":',
    }

    # dose-response x-sweep (optional)
    xs_path = config.EXP3_PERSONA_DIR / "xsweep" / "xsweep_summary.json"
    xsweep = json.loads(xs_path.read_text(encoding="utf-8")) if xs_path.exists() else None
    xs_fig_dir = config.EXP3_PERSONA_DIR / "xsweep" / "figures"
    xsweep_figs = {n: b64_png(xs_fig_dir / f"{n}.png")
                   for n in ("dv_vs_severity", "persona_share_vs_severity")}

    return {
        "summary": summary,
        "examples": examples,
        "figs": figs,
        "n_judged": n_judged,
        "n_no_persona": n_no_persona,
        "models": summary.get("models", []),
        "model_counts": model_counts,
        "cf": cf,
        "cf_examples": cf_examples,
        "xsweep": xsweep,
        "xsweep_figs": xsweep_figs,
        "browse": browse,
        "browse_sys": browse_sys,
        "judge_prompt": judge_prompt,
    }


def load_browse():
    """Full Model x Scenario x Role x x-value grid for the interactive Examples
    browser, from the deployment x-sweep (the only set with every x-value).

    The system prompt is identical across all deployment records, so it is stored
    once (browse_sys) and stripped from each record to keep the payload small.
    """
    xdir = config.EXP3_PERSONA_DIR / "xsweep"
    browse, sys_prompt = [], ""
    for jf in sorted(xdir.glob("*/all.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not sys_prompt:
                for m in r.get("messages", []):
                    if m["role"] == "system":
                        sys_prompt = m["content"]
                        break
            persona = r.get("persona") or {}
            browse.append({
                "model": r["model_name"], "scenario": r["scenario"], "role": r["role"],
                "x": r["x_value"], "x_rank": r.get("x_rank"),
                "user": r.get("user", ""),
                "pname": r.get("persona_name", ""),
                "psum": persona.get("persona_summary", ""),
                "trig": persona.get("triggered_by", ""),
                "traits": r.get("traits") or [],
                "resp": r.get("response", ""),
                "raw": r.get("raw_output", ""),
                "judge": {k: (r.get("judge") or {}).get(k) for k in
                          ("primary_emotion", "warmth", "formality", "advice_density")},
            })
    return browse, sys_prompt


def pick_cf_examples(user_lookup=None):
    """One vivid counterfactual per edit STRATEGY (persona_swap, trait_suppress,
    null_persona), for the report: original persona & reply -> the exact edit ->
    regenerated reply, with judge scores on both. The user message is joined in
    from the baseline datapoint (cf records don't store it)."""
    user_lookup = user_lookup or {}
    cf_dir = config.EXP3_PERSONA_DIR / "cf"
    if not cf_dir.exists():
        return []
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cf_dir.glob("*/*/*.json"))]
    out = []

    def pick(fam):
        # prefer a record with a large, multi-DV move (most illustrative)
        cands = [r for r in recs if r["family"] == fam]
        cands.sort(key=lambda r: -sum(abs(v) for v in r.get("delta", {}).values()))
        return cands[0] if cands else None

    labels = {"persona_swap": "Persona swap",
              "trait_suppress": "Trait suppression",
              "null_persona": "Null persona"}
    for fam in ("persona_swap", "trait_suppress", "null_persona"):
        r = pick(fam)
        if not r:
            continue
        out.append({
            "tag": labels[fam], "family": fam, "edit": r["edit"], "target": r.get("target"),
            "natural_value": r.get("natural_value"),
            "model": r["model_name"], "scenario": r.get("scenario"), "role": r.get("role"),
            "baseline_persona": r.get("baseline_persona_name"),
            "user": user_lookup.get((r["model_name"], r.get("prompt_id")), ""),
            "predicted_dir": r.get("predicted_dir"),
            "baseline_judge": r.get("baseline_judge"),
            "cf_judge": r.get("cf_judge"),
            "delta": r.get("delta"),
            "match": r.get("match"),
            "baseline_response": r.get("baseline_response", ""),
            "cf_response": r.get("cf_response", ""),
            "edited_persona": r.get("edited_persona", ""),
        })
    return out


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 3 — Verbalized Personas &amp; Emotional Behaviour</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root{
    --bg:#0f1216; --panel:#171b22; --panel2:#1e242d; --ink:#e7ecf3; --muted:#9aa7b8;
    --line:#2a3340; --accent:#5aa9ff; --warm:#ff8e72; --formal:#7ad1c0; --advice:#c79bff;
    --good:#5ad19a; --bad:#ff7b7b;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  header{padding:34px 28px 18px;border-bottom:1px solid var(--line);
         background:linear-gradient(180deg,#141923,#0f1216);}
  h1{margin:0 0 6px;font-size:26px;letter-spacing:.2px}
  h2{margin:0 0 14px;font-size:20px}
  h3{margin:22px 0 8px;font-size:16px;color:var(--accent)}
  .sub{color:var(--muted);max-width:900px}
  .wrap{max-width:1080px;margin:0 auto;padding:0 28px 80px}
  nav{position:sticky;top:0;z-index:10;display:flex;flex-wrap:wrap;gap:6px;
      padding:10px 28px;background:rgba(15,18,22,.92);backdrop-filter:blur(8px);
      border-bottom:1px solid var(--line)}
  nav button{background:transparent;border:1px solid var(--line);color:var(--muted);
      padding:6px 13px;border-radius:999px;cursor:pointer;font-size:13px}
  nav button:hover{color:var(--ink);border-color:var(--accent)}
  nav button.active{background:var(--accent);color:#06223f;border-color:var(--accent);font-weight:600}
  section{display:none;padding-top:18px;animation:fade .25s ease}
  section.active{display:block}
  @keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;
         padding:18px 20px;margin:14px 0}
  .grid{display:grid;gap:14px}
  .cards{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
  .kpi{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .kpi .num{font-size:28px;font-weight:700}
  .kpi .lab{color:var(--muted);font-size:13px;margin-top:2px}
  table{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:6px}
  th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:600;cursor:pointer;user-select:none;white-space:nowrap}
  th:hover{color:var(--ink)}
  td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
  tbody tr:hover{background:#1b212b}
  .bar{height:8px;border-radius:4px;background:var(--accent);display:inline-block;vertical-align:middle}
  .pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;
        background:var(--panel2);border:1px solid var(--line);color:var(--muted)}
  .tag-warm{color:var(--warm)} .tag-formal{color:var(--formal)} .tag-advice{color:var(--advice)}
  .ex{border:1px solid var(--line);border-radius:12px;margin:12px 0;overflow:hidden;background:var(--panel)}
  .ex summary{cursor:pointer;padding:13px 16px;list-style:none;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .ex summary::-webkit-details-marker{display:none}
  .ex[open] summary{border-bottom:1px solid var(--line);background:var(--panel2)}
  .ex .body{padding:14px 16px;display:grid;gap:14px;grid-template-columns:1fr 1fr}
  .ex .body .full{grid-column:1/-1}
  .block{background:#10141a;border:1px solid var(--line);border-radius:10px;padding:12px 14px}
  .block h4{margin:0 0 8px;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;white-space:pre-wrap}
  .trait{display:flex;align-items:center;gap:8px;margin:4px 0}
  .trait .tn{flex:0 0 130px;font-size:12.5px}
  .trait .tv{flex:0 0 30px;text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
  .trait .tx{flex:1;color:var(--muted);font-size:12px}
  .meter{flex:0 0 90px;height:7px;background:#222b36;border-radius:4px;overflow:hidden}
  .meter > i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--warm))}
  .judge .j{display:inline-block;margin-right:14px}
  .judge b{font-size:16px}
  .note{color:var(--muted);font-size:13px}
  .heat td{text-align:center;font-variant-numeric:tabular-nums;border:1px solid var(--line)}
  .heat th{text-align:center}
  figure{margin:0}
  figure img{width:100%;border:1px solid var(--line);border-radius:10px;background:#fff}
  figcaption{color:var(--muted);font-size:13px;margin-top:6px}
  code{background:#10141a;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:12.5px}
  .legend span{margin-right:14px;font-size:12.5px}
  .dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:middle}
  .two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .selrow{display:flex;flex-wrap:wrap;gap:16px}
  .selrow label{display:flex;flex-direction:column;gap:5px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
  .selrow select{background:var(--panel2);color:var(--ink);border:1px solid var(--line);
      border-radius:8px;padding:7px 10px;font-size:14px;min-width:150px}
  .selrow select:hover{border-color:var(--accent)}
  @media(max-width:760px){.ex .body,.two{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Experiment 3 · Verbalized Personas &amp; Emotional Behaviour</h1>
  <div class="sub">
    Does asking a model to <b>name the character it is about to express</b> — then
    report the 4–6 traits that are load-bearing for it — predict the emotional style
    of its actual reply? A repeat of Experiment 2 on the same
    role&nbsp;×&nbsp;eval-condition&nbsp;×&nbsp;scenario dataset, swapping the contextual-cue
    self-modeling prompt for the <code>infer_persona_prompt</code>. Generator:
    <b>__MODELS__</b>. Judge: <b>claude-sonnet-4.5</b>. First two steps only — no counterfactuals.
  </div>
</header>

<nav id="nav"></nav>

<div class="wrap">

  <!-- OVERVIEW -->
  <section id="overview" class="active">
    <div class="grid cards" id="kpis"></div>
    <div class="panel">
      <h3>The one-paragraph result</h3>
      <p>Across <b>four non-thinking models</b> the persona prompt collapses into a small set of stock
      characters — dominated by a <b>“Cautious __ Advisor”</b> family (clinical / financial / safety)
      and an <b>“Empathetic Supporter / Listener”</b> family. The character a model <i>verbalizes</i>
      is strongly <b>correlationally faithful</b> to the reply an independent judge then sees (persona
      text predicts judged warmth / formality / advice with CV R² ≈ <b>0.7–0.8</b>; self-rated
      <i>empathy</i> tracks judged warmth at r≈+0.73). <b>Scenario</b> (topic) dominates behaviour;
      the <b>evaluation frame has ~zero effect</b>. Counterfactually, <b>swapping the whole persona</b>
      (advisor↔supporter) is a powerful, faithful lever — it moves judged advice by up to <b>−5.7</b>
      (llama-3.3-70b) to <b>−7.8</b> (qwen3-30b) at faithfulness 1.00 — whereas <b>suppressing a single
      trait barely moves the gestalt</b>. Models differ in steerability: qwen3-30b is highly faithful
      to the edited persona (0.84), while <b>qwen3-235b largely ignores it</b> (0.33). A full
      <b>dose–response sweep</b> (every x-value, e.g. tylenol 500→20000 mg) shows the persona is set
      by the <b>topic, not the magnitude</b>: severity modulates behaviour only weakly and in
      topic-appropriate directions (better outcomes → less advice; drink-driving → more), and the
      character almost never flips as stakes rise.</p>
    </div>
    <div class="panel">
      <h3>How to read this report</h3>
      <p class="note">Use the tabs above. <b>Method</b> = the pipeline &amp; prompt.
      <b>Examples</b> = real datapoints (expand each). <b>Personas &amp; traits</b> = what the model
      verbalizes. <b>What drives emotion</b> = ANOVA factor effects. <b>Cross-model</b> = the four
      models side by side. <b>Faithfulness</b> = does the verbalized persona predict the judged reply
      (correlational). <b>Counterfactuals</b> = editing a verbalized artefact and measuring the
      <i>causal</i> change. <b>Dose–response</b> = sweep every x-value and see if severity moves the
      persona. <b>Figures</b> = the static plots. Tables are sortable — click a header.</p>
    </div>
  </section>

  <!-- METHOD -->
  <section id="method">
    <h2>Method</h2>
    <div class="panel">
      <h3>Pipeline (per datapoint)</h3>
      <ol>
        <li><b>Compose.</b> System prompt = <code>infer_persona_prompt</code>. If the row carries an
          evaluation framing (<code>eval_cue</code> / <code>neutral_sys</code>) it is prepended, so the
          eval frame stays a clean experimental factor. The scenario question is the user turn.</li>
        <li><b>Generate &amp; parse.</b> The model emits a <code>&lt;persona&gt;</code> block
          (name, summary, trigger, and 4–6 self-chosen traits each scored 0–10) followed by a
          <code>&lt;response&gt;</code>. We parse both; traits are grouped into a clean list.</li>
        <li><b>Judge.</b> <b>claude-sonnet-4.5</b> reads the <i>response only</i> (never the persona block,
          to avoid leakage) and returns <span class="tag-warm">warmth</span>,
          <span class="tag-formal">formality</span>, <span class="tag-advice">advice_density</span>
          (0–10) + a categorical <code>primary_emotion</code>.</li>
      </ol>
      <p class="note">Calls run concurrently with retry/backoff; records are self-describing JSON and
      resumable. Analysis is purely local (ANOVA, keyword association, cross-validated ridge).</p>
    </div>
    <details class="panel" style="border-left:3px solid var(--formal)">
      <summary style="cursor:pointer;font-weight:600;font-size:16px;color:var(--formal)">The judge prompt <span class="note" id="judge-model-lab"></span></summary>
      <div style="margin-top:12px">
        <p class="note">Verbatim prompt sent to the judge. It scores the assistant reply only (never
        the persona block), temperature 0, with an assistant prefill forcing a bare JSON object.
        <code>«…»</code> mark where the datapoint's user message and reply are inserted.</p>
        <div class="block full"><h4>System</h4><div class="mono" id="judge-sys" style="max-height:200px;overflow:auto"></div></div>
        <div class="block full"><h4>User turn (schema + material to score)</h4><div class="mono" id="judge-user" style="max-height:300px;overflow:auto"></div></div>
        <div class="block full"><h4>Assistant prefill</h4><div class="mono" id="judge-prefill"></div></div>
      </div>
    </details>
    <div class="two">
      <div class="panel">
        <h3>Design axes</h3>
        <table>
          <tr><th>axis</th><th>levels</th></tr>
          <tr><td>role (user model)</td><td>12 (e.g. elderly_confused, domain_expert, adversarial_reframe…)</td></tr>
          <tr><td>eval_condition</td><td>3 — deployment · eval_cue · neutral_sys</td></tr>
          <tr><td>scenario (topic)</td><td>12 (tylenol, driving, investing…)</td></tr>
        </table>
        <p class="note">Balanced subsample: one representative risk level per scenario
        (12 × 3 × 12 = 432 prompts), identical to Experiment 2.</p>
      </div>
      <div class="panel">
        <h3>Dependent variables (judge, 0–10)</h3>
        <p><span class="tag-warm">● warmth</span> — cold/clinical → warm, caring<br>
           <span class="tag-formal">● formality</span> — casual → formal/professional<br>
           <span class="tag-advice">● advice_density</span> — no directives → densely prescriptive</p>
        <p class="note">Same DVs as Experiment 2, so the persona prompt is directly comparable to the
        contextual-cue prompt.</p>
      </div>
    </div>
  </section>

  <!-- EXAMPLES -->
  <section id="examples">
    <h2>Data browser</h2>
    <p class="note">Pick a <b>model</b>, <b>scenario</b>, <b>role</b> and <b>x-value</b> to inspect that
    exact datapoint: the verbalized persona, the traits it chose, its reply, and the independent judge
    scores. Source = the deployment x-sweep grid (every x-value; no eval framing).</p>
    <div class="panel" id="browse-controls">
      <div class="selrow">
        <label>Model <select id="sel-model"></select></label>
        <label>Scenario <select id="sel-scenario"></select></label>
        <label>Role <select id="sel-role"></select></label>
        <label>x-value <select id="sel-x"></select></label>
      </div>
    </div>
    <div id="browse-card"></div>
    <div class="panel" id="browse-raw-wrap">
      <h3>Raw view — full prompt &amp; response</h3>
      <p class="note">Exactly what the model saw and produced, verbatim (system prompt + user turn →
      raw completion, persona block and all).</p>
      <div class="block full"><h4>System prompt</h4><div class="mono" id="raw-sys" style="max-height:220px;overflow:auto"></div></div>
      <div class="block full"><h4>User message</h4><div class="mono" id="raw-user"></div></div>
      <div class="block full"><h4>Raw model output</h4><div class="mono" id="raw-out" style="max-height:420px;overflow:auto"></div></div>
    </div>
  </section>

  <!-- PERSONAS -->
  <section id="personas">
    <h2>Personas &amp; traits the model verbalizes</h2>
    <details class="panel" style="border-left:3px solid var(--accent)">
      <summary style="cursor:pointer;font-weight:600;font-size:16px;color:var(--accent)">What “load-bearing traits” means</summary>
      <div style="margin-top:12px">
      <p>“Load-bearing” is the model's <b>own</b> term — it comes straight from the persona prompt,
      which tells the model:</p>
      <blockquote class="mono" style="border-left:3px solid var(--line);margin:8px 0;padding:8px 12px;color:var(--muted)">
      “Pick the traits that are actually <b>load-bearing</b> for THIS persona — not a generic checklist…
      Pick 4–6 traits that are actually load-bearing for the character above — not generic ones that
      would apply to any role. Score each 0–10. Briefly say how each trait shows up in the response.”
      </blockquote>
      <p>So for every reply the model names a character and then lists the <b>few traits it judges are
      doing the real work</b> in shaping <i>that</i> reply — the ones the character would fall apart
      without — rather than boilerplate that fits any assistant. Each trait comes with a
      self-assigned <b>value 0–10</b> (how strongly it's expressed) and a one-line <b>expression</b>
      (how it shows up). The chart and table below aggregate those self-selected traits across all
      datapoints: <b>count</b> = how often a trait was chosen as load-bearing; <b>mean value</b> = its
      average self-assigned strength when chosen. Whether these self-reports are actually faithful to
      the reply is tested in the <b>Faithfulness</b> and <b>Counterfactuals</b> tabs.</p>
      </div>
    </details>
    <div class="two">
      <div class="panel">
        <h3>Selected personas — mean judged DVs</h3>
        <div id="chart-persona" style="height:420px"></div>
        <p class="note">Two clusters: <b>Advisors</b> (high advice, mid formality) vs
        <b>Supporters/Listeners</b> (high warmth, low formality &amp; advice).</p>
      </div>
      <div class="panel">
        <h3>Which traits are chosen as load-bearing most often</h3>
        <div id="chart-traits" style="height:420px"></div>
        <p class="note">Bar length = <b>how often the trait was picked</b> as load-bearing (count of
        datapoints). Colour = <b>mean self-assigned value</b> (0–10) on the datapoints that chose it —
        darker/redder = rated stronger.</p>
      </div>
    </div>
    <div class="panel">
      <h3>Persona inventory <span class="pill" id="persona-n"></span></h3>
      <p class="note">Distinct character labels the model gave itself, with the judge's mean
      warmth/formality/advice for each.</p>
      <table id="tbl-personas"></table>
    </div>
    <div class="panel">
      <h3>Trait inventory</h3>
      <p class="note"><b>count</b> = times the trait was named as load-bearing; <b>mean value</b> = its
      average self-assigned 0–10 strength when named.</p>
      <table id="tbl-traits"></table>
    </div>
  </section>

  <!-- DRIVERS -->
  <section id="drivers">
    <h2>What drives the response emotion?</h2>
    <div class="panel">
      <h3>Factor effects (one-way ANOVA η²)</h3>
      <p class="note">η² = share of variance in each judged DV explained by a design factor.
      Bars scaled within each DV; ✻ marks p&lt;0.05.</p>
      <table id="tbl-anova"></table>
    </div>
    <div class="two">
      <div class="panel">
        <h3>Warmth &amp; formality by user role</h3>
        <div id="chart-role" style="height:420px"></div>
      </div>
      <div class="panel">
        <h3>Evaluation frame has ~no effect</h3>
        <div id="chart-eval" style="height:420px"></div>
        <p class="note">deployment vs eval_cue vs neutral_sys: η² ≈ 0.00 on every DV — the persona
        prompt shows no visible eval-awareness shift.</p>
      </div>
    </div>

    <div class="panel">
      <h3>Do user roles select specific persona traits?</h3>
      <p class="note">Among datapoints that verbalized a persona, this is the <b>lift</b> =
      P(trait chosen&nbsp;|&nbsp;role) ÷ P(trait overall). <b>Lift &gt; 1</b> (red) = the role
      <b>over-selects</b> that load-bearing trait; &lt; 1 (blue) = under-selects. Hover a cell for the
      raw selection share and n. Traits are ordered by how role-dependent they are (chi-square).</p>
      <div id="roletrait-note" class="note" style="margin-bottom:10px"></div>
      <div style="overflow-x:auto"><table class="heat" id="tbl-roletrait"></table></div>
      <h4 style="margin-top:16px;color:var(--muted)">Most distinctive trait per role</h4>
      <div id="roletrait-distinctive" class="grid cards"></div>
    </div>
  </section>

  <!-- FAITHFULNESS -->
  <section id="faithfulness">
    <h2>Is the verbalized persona faithful to the reply?</h2>
    <div class="panel" style="border-left:3px solid var(--accent)">
      <h3>What this tab represents</h3>
      <p>“Faithful” here = <b>does the model's self-described persona match what it actually does?</b>
      Everything below relates the model's <b>verbalization</b> (persona name + summary + trigger +
      trait names/expressions + the 0–10 trait values) to an independent judge's scores of the
      <b>reply only</b>. The judge <b>never sees the persona block</b>, so any alignment is not leakage.</p>
      <ul>
        <li><b>CV R² cards</b> — from just the persona <i>text</i>, a cross-validated model predicts the
          judged warmth/formality/advice out-of-sample. High R² ⇒ the words the model uses to describe
          its character strongly predict the emotional style a blind judge later assigns.</li>
        <li><b>Persona words / trait-value r</b> — which specific words, and which self-assigned trait
          <i>numbers</i>, track the judged scores (e.g. self-rated empathy r≈+0.73 with warmth).</li>
      </ul>
      <p style="background:#221a12;border:1px solid var(--line);border-radius:8px;padding:10px 12px">
      ⚠️ <b>This is correlational, not causal.</b> A high R² is equally consistent with (a) the
      verbalization reflecting/driving the reply, or (b) the <i>input</i> (topic, role) jointly
      producing both the persona label and the reply — the model might write the same warm answer
      without ever saying “empathetic.” Faithfulness answers <b>“does the self-report predict the
      behaviour?”</b> (yes, strongly). The <b>Counterfactuals</b> tab holds the input fixed and edits
      only the verbalization to answer <b>“does changing the self-report change the behaviour?”</b> —
      and there only whole-persona swaps move the needle, not single traits.</p>
    </div>
    <div class="grid cards" id="r2cards"></div>
    <div class="panel">
      <h3>Persona words that move each DV</h3>
      <p class="note">Top association by Cohen's d — words in the model's own persona text that
      co-occur with higher (or lower) judged scores.</p>
      <div id="kw"></div>
    </div>
    <div class="panel">
      <h3>Self-assigned trait value → judged DV (Pearson r)</h3>
      <p class="note">Does a trait the model rated highly actually show up in the judged reply?
      Green = positive, red = negative. Hover a cell for n.</p>
      <table class="heat" id="tbl-heat"></table>
    </div>
  </section>

  <!-- CROSS-MODEL -->
  <section id="models">
    <h2>Cross-model picture</h2>
    <p class="note">Same persona prompt &amp; judge across four non-thinking models. Coverage and
    mean judged DVs per model — do all four converge on the advisor/supporter split?</p>
    <div class="grid cards" id="model-cards"></div>
    <div class="two">
      <div class="panel"><h3>Mean judged DVs by model</h3><div id="chart-model" style="height:420px"></div></div>
      <div class="panel"><h3>Eval-frame sensitivity by model (η² on warmth)</h3>
        <div id="chart-model-eval" style="height:420px"></div>
        <p class="note">Bars near zero ⇒ the model does not shift its persona/emotion when it thinks
        it is being evaluated.</p></div>
    </div>
    <div class="panel"><h3>Per-model judged DV means</h3><table id="tbl-model"></table></div>
  </section>

  <!-- COUNTERFACTUALS -->
  <section id="counterfactuals">
    <h2>Counterfactual testing — do the verbalized artefacts <i>cause</i> the behaviour?</h2>
    <div class="panel">
      <h3>The idea — and the three edit strategies</h3>
      <p>Every baseline datapoint already has a persona the model chose and a reply it wrote. A
      counterfactual keeps the <b>user's message fixed</b>, changes <b>one thing about the verbalized
      persona</b>, feeds that edited persona back to the model, and has it <b>rewrite the reply</b>. If
      the reply changes, the verbalized artefact was doing causal work; if not, it was just talk.</p>
      <ul>
        <li><b>trait_suppress</b> — take one load-bearing trait (e.g. <i>empathy</i>) and force its
          value to ~1 with a negated one-line description, <b>holding every other trait</b> at the
          model's own values. Isolates a single trait.</li>
        <li><b>persona_swap</b> — discard the whole persona and replace it with the <b>opposite
          archetype</b> (Cautious Advisor ↔ Empathetic Supporter). Tests the character as a whole.</li>
        <li><b>null_persona</b> — <b>blank</b> the persona entirely (no character at all). Tests whether
          articulating any persona matters.</li>
      </ul>
      <p class="note">We re-judge the rewritten reply and compare to the baseline. <b>Read the worked
      examples next</b> to see exactly what each edit looks like, then the aggregate effects below.</p>
    </div>

    <div class="panel">
      <h3>Worked examples — the exact edit and the reply it produces</h3>
      <p class="note">One example per strategy. Each shows the model's <b>original persona &amp; reply</b>
      → the <b>edit we injected</b> → the <b>regenerated reply</b>, with the judge's warmth / formality /
      advice on both and the resulting change (Δ).</p>
      <div id="cf-examples"></div>
    </div>

    <h3 style="margin:26px 0 4px">Aggregate results across all edits</h3>
    <p class="note" style="margin-top:0"><b>Faithfulness</b> = fraction of pre-registered (edit×DV)
    cells whose observed Δ matched the predicted sign.</p>
    <div class="grid cards" id="cf-cards"></div>
    <div class="panel">
      <h3>Effect of each edit (mean Δ in judged DV, per model)</h3>
      <p class="note">Green = moved in the predicted direction. Pick a model:</p>
      <div id="cf-modelpick" style="margin-bottom:10px"></div>
      <table id="tbl-cf"></table>
    </div>
  </section>

  <!-- DOSE-RESPONSE -->
  <section id="dose">
    <h2>Dose–response — does severity move the persona?</h2>
    <div class="panel">
      <h3>Design</h3>
      <p>The main factorial pins one risk level per scenario. Here we hold
      <b>eval_condition = deployment</b> and sweep <b>every x-value</b> (e.g. tylenol 500→20000 mg)
      across all 12 roles × 12 scenarios — <b>936 prompts/model</b>. Severity is the within-scenario
      rank of x. Scenarios are mostly <code>neg</code> (higher x = more dangerous); two are
      <code>pos</code> (higher x = better outcome).</p>
      <p class="note">Headline: pooling raw magnitude across scenarios cancels to ≈0, but split by
      valence/topic there is a modest, sensible dose-response — and the persona barely flips.</p>
    </div>
    <div class="grid cards" id="dose-cards"></div>
    <figure id="dose-fig1" style="margin:14px 0"></figure>
    <figure id="dose-fig2" style="margin:14px 0"></figure>
    <div class="two">
      <div class="panel"><h3>Severity→DV slope, split by valence (Pearson r)</h3>
        <p class="note">Positive scenarios: advice falls as outcomes improve. Negative: mostly flat.</p>
        <table id="tbl-dose-val"></table></div>
      <div class="panel"><h3>Per-scenario severity→advice slope</h3>
        <p class="note">Which topics actually escalate/de-escalate prescriptiveness with the number.</p>
        <table id="tbl-dose-scn"></table></div>
    </div>
    <div class="panel"><h3>Persona-family share by severity (per model)</h3>
      <p class="note">Does the character flip from Supporter→Advisor as stakes rise? Mostly no —
      supporter share dips slightly at high severity for the qwen models. (llama-3.1-8b almost never
      emits a persona block, so it is ~all “none”.)</p>
      <table id="tbl-dose-fam"></table></div>
  </section>

  <!-- FIGURES -->
  <section id="figures">
    <h2>Figures</h2>
    <p class="note">Static plots emitted by the analysis (<code>results/exp3_persona/figures/</code>).</p>
    <div id="figs"></div>
  </section>

</div>

<script id="DATA" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('DATA').textContent);
const DVS = ["warmth","formality","advice_density"];
const COL = {warmth:"#ff8e72", formality:"#7ad1c0", advice_density:"#c79bff"};
const PLOT_LAYOUT = {paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:'#cdd6e2',size:12}, margin:{l:140,r:16,t:10,b:40},
  legend:{orientation:'h',y:1.12}, xaxis:{gridcolor:'#2a3340',zeroline:false},
  yaxis:{gridcolor:'#2a3340',automargin:true}};
const PCONF = {displayModeBar:false, responsive:true};

/* ---- tabs ---- */
const TABS = [["overview","Overview"],["method","Method"],["examples","Examples"],
  ["personas","Personas & traits"],["drivers","What drives emotion"],
  ["models","Cross-model"],["faithfulness","Faithfulness"],
  ["counterfactuals","Counterfactuals"],["dose","Dose–response"],["figures","Figures"]];
const nav = document.getElementById('nav');
TABS.forEach(([id,label],i)=>{
  const b=document.createElement('button'); b.textContent=label; b.dataset.t=id;
  if(i===0)b.classList.add('active');
  b.onclick=()=>{
    document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('section').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); document.getElementById(id).classList.add('active');
    window.scrollTo({top:0,behavior:'smooth'});
    setTimeout(()=>{window.dispatchEvent(new Event('resize'));drawFor(id);},30);
  };
  nav.appendChild(b);
});

/* ---- KPIs ---- */
const inv = D.summary.persona_inventory;
const r2 = D.summary.verbalization_to_emotion;
const kpis=[
  ["N judged", D.n_judged],
  ["Distinct personas", inv.n_distinct_personas],
  ["Mean traits / persona", inv.mean_traits_per_persona],
  ["Hard refusals (no persona)", D.n_no_persona],
  ["Faithfulness R² (formality)", r2.formality.ridge_cv_r2],
  ["Eval-frame η² (warmth)", (D.summary.factor_effects["eval_condition__warmth"]||{}).eta2],
];
document.getElementById('kpis').innerHTML = kpis.map(([l,v])=>
  `<div class="kpi"><div class="num">${v}</div><div class="lab">${l}</div></div>`).join('');

/* ---- examples ---- */
function judgeRow(j){
  return `<div class="judge">
    <span class="j tag-warm">warmth <b>${j.warmth}</b></span>
    <span class="j tag-formal">formality <b>${j.formality}</b></span>
    <span class="j tag-advice">advice <b>${j.advice_density}</b></span>
    <span class="j">emotion <span class="pill">${j.primary_emotion}</span></span></div>`;
}
function traitRows(ts){
  if(!ts.length) return '<span class="note">— no traits (model skipped the persona step) —</span>';
  return ts.map(t=>{
    const v = (typeof t.value==='number')? t.value : null;
    const pct = v!=null? Math.max(0,Math.min(100, v*10)) : 0;
    return `<div class="trait"><span class="tn">${esc(t.name)}</span>
      <span class="meter"><i style="width:${pct}%"></i></span>
      <span class="tv">${v!=null?v:'—'}</span>
      <span class="tx">${esc(t.expression||'')}</span></div>`;
  }).join('');
}
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

/* ---- judge prompt (Method tab) ---- */
(function(){
  const jp=D.judge_prompt; if(!jp)return;
  const set=(id,v)=>{const el=document.getElementById(id); if(el)el.textContent=v;};
  set('judge-sys', jp.system);
  set('judge-user', jp.user);
  set('judge-prefill', jp.prefill);
  const lab=document.getElementById('judge-model-lab'); if(lab)lab.textContent='· '+jp.model;
})();

/* ---- interactive data browser (Model x Scenario x Role x x-value) ---- */
(function(){
  const B = D.browse||[];
  const SYS = D.browse_sys||'';
  if(!B.length){document.getElementById('browse-card').innerHTML='<div class="panel note">No x-sweep data to browse — run <code>src.run_exp3_xsweep</code>.</div>';return;}
  const uniq=(a)=>[...new Set(a)];
  const models=uniq(B.map(r=>r.model)).sort();
  const scenarios=uniq(B.map(r=>r.scenario)).sort();
  const roles=uniq(B.map(r=>r.role)).sort();
  const selM=document.getElementById('sel-model'), selS=document.getElementById('sel-scenario'),
        selR=document.getElementById('sel-role'), selX=document.getElementById('sel-x');
  const opts=(el,vals,fmt)=>{el.innerHTML=vals.map(v=>`<option value="${esc(v)}">${esc(fmt?fmt(v):v)}</option>`).join('');};
  opts(selM,models); opts(selS,scenarios); opts(selR,roles);
  // sensible defaults
  selM.value = models.includes('llama-3.3-70b')?'llama-3.3-70b':models[0];
  selS.value = scenarios.includes('tylenol')?'tylenol':scenarios[0];
  selR.value = roles.includes('none')?'none':roles[0];

  function xsFor(scn){return uniq(B.filter(r=>r.scenario===scn).map(r=>r.x)).sort((a,b)=>a-b);}
  function refreshX(keep){
    const xs=xsFor(selS.value);
    opts(selX,xs,v=>v);
    if(keep!=null && xs.includes(keep)) selX.value=keep;
    else selX.value = xs[Math.floor(xs.length/2)]; // default mid dose
  }
  function find(){
    return B.find(r=>r.model===selM.value&&r.scenario===selS.value&&r.role===selR.value&&String(r.x)===String(selX.value));
  }
  function render(){
    const r=find();
    const card=document.getElementById('browse-card');
    if(!r){card.innerHTML='<div class="panel note">No datapoint for that combination.</div>';
      document.getElementById('raw-out').textContent='';return;}
    const pname = r.pname && r.pname.trim() ? r.pname : '(no persona block emitted — went straight to response)';
    card.innerHTML=`<div class="ex" style="margin-top:14px"><div class="body">
      <div class="block full"><h4>User message <span class="note">· ${esc(r.model)} · ${esc(r.scenario)} · role ${esc(r.role)} · x=${esc(r.x)} (${esc(r.x_rank)})</span></h4>
        <div class="mono">${esc(r.user)}</div></div>
      <div class="block"><h4>Verbalized persona</h4>
        <div style="margin-bottom:8px"><b>${esc(pname)}</b></div>
        <div class="note" style="margin-bottom:6px">${esc(r.psum)}</div>
        ${r.trig?`<div class="note" style="margin-bottom:10px"><i>triggered by:</i> ${esc(r.trig)}</div>`:''}
        ${traitRows(r.traits)}</div>
      <div class="block"><h4>Response &amp; judge</h4>
        ${judgeRow(r.judge)}
        <div class="mono" style="margin-top:10px;max-height:300px;overflow:auto">${esc(r.resp)}</div></div>
    </div></div>`;
    document.getElementById('raw-sys').textContent=SYS;
    document.getElementById('raw-user').textContent=r.user;
    document.getElementById('raw-out').textContent=r.raw;
  }
  selS.addEventListener('change',()=>{refreshX(null);render();});
  [selM,selR,selX].forEach(el=>el.addEventListener('change',render));
  refreshX(null); render();
})();

/* ---- sortable table helper ---- */
function makeTable(el, cols, rows, opts){
  opts=opts||{};
  let sortIdx = opts.sort==null? null : opts.sort, asc=false;
  function render(){
    let r=rows.slice();
    if(sortIdx!=null){
      r.sort((a,b)=>{const x=a[sortIdx],y=b[sortIdx];
        if(typeof x==='number'&&typeof y==='number')return asc?x-y:y-x;
        return asc? String(x).localeCompare(String(y)):String(y).localeCompare(String(x));});
    }
    const maxes = cols.map((c,i)=> c.bar? Math.max(...rows.map(row=>Math.abs(+row[i]||0))) : 0);
    el.innerHTML='<thead><tr>'+cols.map((c,i)=>
      `<th class="${c.num?'num':''}" data-i="${i}">${c.h}${sortIdx===i?(asc?' ▲':' ▼'):''}</th>`).join('')
      +'</tr></thead><tbody>'+
      r.map(row=>'<tr>'+row.map((v,i)=>{
        const c=cols[i];
        if(c.bar){const w=maxes[i]? Math.abs(v)/maxes[i]*70:0;
          return `<td class="num">${v} <span class="bar" style="width:${w}px;background:${c.color||'var(--accent)'}"></span></td>`;}
        if(c.heat){const col=heatColor(v); return `<td style="background:${col}">${v==null?'—':v}</td>`;}
        return `<td class="${c.num?'num':''}">${c.fmt?c.fmt(v,row):esc(v)}</td>`;
      }).join('')+'</tr>').join('')+'</tbody>';
    el.querySelectorAll('th').forEach(th=>th.onclick=()=>{
      const i=+th.dataset.i; if(sortIdx===i)asc=!asc; else{sortIdx=i;asc=false;} render();});
  }
  render();
}
function heatColor(v){
  if(v==null||isNaN(v))return 'transparent';
  const a=Math.min(1,Math.abs(v)/0.9);
  return v>=0? `rgba(90,209,154,${a*.55})` : `rgba(255,123,123,${a*.55})`;
}

/* ---- persona + trait tables ---- */
document.getElementById('persona-n').textContent =
  `${inv.n_distinct_personas} distinct · top shown`;
makeTable(document.getElementById('tbl-personas'),
  [{h:'persona'},{h:'n',num:true,bar:true,color:'#5aa9ff'},
   {h:'warmth',num:true,color:COL.warmth},{h:'formality',num:true,color:COL.formality},
   {h:'advice',num:true,color:COL.advice_density}],
  inv.top_personas.filter(p=>p.persona).map(p=>[p.persona,p.n,p.warmth,p.formality,p.advice_density]),
  {sort:1});
makeTable(document.getElementById('tbl-traits'),
  [{h:'trait'},{h:'count',num:true,bar:true,color:'#5aa9ff'},{h:'mean value (0–10)',num:true}],
  inv.top_traits.map(t=>[t.trait,t.n,t.mean_value]), {sort:1});

/* ---- ANOVA table ---- */
const FE=D.summary.factor_effects;
const FACTORS=["scenario","role","role_axis","eval_condition"];
const anovaRows=[];
FACTORS.forEach(f=>DVS.forEach(dv=>{
  const e=FE[`${f}__${dv}`]; if(!e)return;
  anovaRows.push([f,dv.replace('_density',''),e.eta2,e.F,e.p,(e.p!=null&&e.p<0.05)?'✻':'']);
}));
makeTable(document.getElementById('tbl-anova'),
  [{h:'factor'},{h:'DV'},{h:'η²',num:true,bar:true,color:'#5aa9ff'},
   {h:'F',num:true},{h:'p',num:true,fmt:v=>v==null?'—':(v<1e-4?v.toExponential(1):(+v).toFixed(4))},
   {h:'sig'}],
  anovaRows,{sort:2});

/* ---- R2 cards + keywords ---- */
document.getElementById('r2cards').innerHTML = DVS.map(dv=>{
  const v=r2[dv].ridge_cv_r2;
  return `<div class="kpi"><div class="num" style="color:${COL[dv]}">${v}</div>
    <div class="lab">CV R² · persona&nbsp;→&nbsp;${dv.replace('_density','')}</div></div>`;
}).join('') + `<div class="kpi"><div class="num">judge-blind</div>
  <div class="lab">judge never sees the persona block</div></div>`;
document.getElementById('kw').innerHTML = DVS.map(dv=>{
  const top=(r2[dv].keywords_top||[]).slice(0,6);
  const raises=r2[dv].ridge_top_terms.slice(0,6).join(', ');
  const lowers=r2[dv].ridge_bottom_terms.slice(0,6).join(', ');
  return `<div class="block" style="margin-bottom:10px">
    <h4 style="color:${COL[dv]}">${dv.replace('_density','')}</h4>
    <div class="note"><b>raises:</b> ${esc(raises)}</div>
    <div class="note"><b>lowers:</b> ${esc(lowers)}</div>
    <div style="margin-top:8px">${top.map(k=>`<span class="pill" style="margin:2px">${esc(k.term)} <b>${k.cohen_d>0?'+':''}${k.cohen_d}</b>d</span>`).join('')}</div>
  </div>`;
}).join('');

/* ---- trait-value heatmap table ---- */
const tc=D.summary.trait_value_corr;
makeTable(document.getElementById('tbl-heat'),
  [{h:'trait'},{h:'n',num:true},
   {h:'r · warmth',heat:true},{h:'r · formality',heat:true},{h:'r · advice',heat:true}],
  tc.map(t=>[t.trait,t.n,t.r_warmth,t.r_formality,t.r_advice_density]),{sort:1});

/* ---- figures ---- */
document.getElementById('figs').innerHTML = Object.entries(D.figs)
  .filter(([k,v])=>v).map(([k,v])=>`<figure style="margin-bottom:18px">
    <img src="${v}" alt="${k}"><figcaption>${k}.png</figcaption></figure>`).join('')
  || '<p class="note">No figures found.</p>';

/* ---- cross-model ---- */
const FM = D.summary.factor_means || {};
const modelMeans = FM.model_name || [];
const cf = D.cf;
(function(){
  const mc = D.model_counts||{};
  document.getElementById('model-cards').innerHTML = modelMeans.map(m=>{
    const fa = cf && cf.models[m.model_name] ? cf.models[m.model_name].overall_faithfulness : null;
    return `<div class="kpi"><div class="num">${mc[m.model_name]||0}</div>
      <div class="lab">${esc(m.model_name)} · N judged${fa!=null?` · faith ${fa}`:''}</div></div>`;
  }).join('') || '<div class="note">Only one model so far — run the other three for the cross-model view.</div>';
  if(modelMeans.length){
    makeTable(document.getElementById('tbl-model'),
      [{h:'model'},{h:'warmth',num:true,color:COL.warmth},
       {h:'formality',num:true,color:COL.formality},{h:'advice',num:true,color:COL.advice_density}],
      modelMeans.map(m=>[m.model_name,m.warmth,m.formality,m.advice_density]),{sort:0});
  }
})();

/* ---- counterfactuals ---- */
(function(){
  if(!cf){document.getElementById('cf-cards').innerHTML='<div class="note">No counterfactual results yet — run <code>src.run_exp3_persona_cf</code>.</div>';return;}
  const models = Object.keys(cf.models);
  document.getElementById('cf-cards').innerHTML = models.map(m=>{
    const v=cf.models[m];
    return `<div class="kpi"><div class="num">${v.overall_faithfulness}</div>
      <div class="lab">${esc(m)} · overall faithfulness<br><span class="note">null|Δ| ${v.null_persona_mean_abs_delta}</span></div></div>`;
  }).join('');
  // model picker -> table
  const pick=document.getElementById('cf-modelpick');
  pick.innerHTML = models.map((m,i)=>`<button class="cfbtn" data-m="${esc(m)}" style="margin-right:6px">${esc(m)}</button>`).join('');
  function renderCF(m){
    const be=cf.models[m].by_edit||{};
    const rows=[];
    Object.keys(be).forEach(edit=>DVS.forEach(dv=>{
      const c=be[edit][dv]; if(!c)return;
      rows.push([edit,dv.replace('_density',''),c.mean_delta,c.faithfulness,c.n]);
    }));
    makeTable(document.getElementById('tbl-cf'),
      [{h:'edit'},{h:'DV'},{h:'mean Δ',num:true,fmt:(v,row)=>{
          const f=row[3]; const col=(f===1)?'var(--good)':(f===0?'var(--bad)':'var(--muted)');
          return `<span style="color:${col}">${v}</span>`;}},
       {h:'faithful',num:true,fmt:v=>v==null?'—':v},{h:'n',num:true}],
      rows,{sort:2});
    document.querySelectorAll('.cfbtn').forEach(b=>b.classList.toggle('active', b.dataset.m===m));
  }
  pick.querySelectorAll('.cfbtn').forEach(b=>b.onclick=()=>renderCF(b.dataset.m));
  // style for active button reuse nav style
  const st=document.createElement('style');
  st.textContent='.cfbtn{background:transparent;border:1px solid var(--line);color:var(--muted);padding:5px 11px;border-radius:8px;cursor:pointer}.cfbtn.active{background:var(--accent);color:#06223f;font-weight:600}';
  document.head.appendChild(st);
  if(models.length)renderCF(models[0]);

  // cf worked examples: original persona/reply -> exact edit -> regenerated reply
  const ce=D.cf_examples||[];
  const archName=(txt)=>{const m=/persona_name:\s*(.+)/i.exec(txt||''); return m?m[1].trim():'(none)';};
  function editDesc(e){
    if(e.family==='persona_swap')
      return `The model's whole persona is thrown away and replaced with the opposite archetype:
        <b>${esc(e.baseline_persona)}</b> → <b>${esc(archName(e.edited_persona))}</b>.`;
    if(e.family==='trait_suppress')
      return `One load-bearing trait — <b>${esc(e.target)}</b> — is forced low
        (was ${e.natural_value!=null?`${esc(e.natural_value)}/10`:'high'} → ~1) and its description
        negated, while <b>every other trait is left unchanged</b>.`;
    if(e.family==='null_persona')
      return `The persona is <b>blanked entirely</b> — the model is handed no character at all,
        then asked to answer the same user message.`;
    return esc(e.edit);
  }
  function judgeLine(j){return `warmth <b>${j.warmth}</b> · formality <b>${j.formality}</b> · advice <b>${j.advice_density}</b>`;}
  function deltaRow(e){
    return DVS.map(dv=>{
      const d=e.delta[dv], pred=(e.predicted_dir||{})[dv]||0, m=e.match?e.match[dv]:undefined;
      const col = m===true?'var(--good)':(m===false?'var(--bad)':'var(--muted)');
      const parr = pred>0?'↑':(pred<0?'↓':'·');
      const tick = m===true?' ✓':(m===false?' ✗':'');
      return `<span class="pill" style="margin:3px;border-color:${col};color:${col}" title="predicted ${parr}">${dv.replace('_density','')} ${d>0?'+':''}${d}${tick}</span>`;
    }).join('');
  }
  document.getElementById('cf-examples').innerHTML = ce.map((e,i)=>`
    <details class="ex" ${i===0?'open':''}>
      <summary><span class="pill">${esc(e.tag)}</span>
        <span class="note">· ${esc(e.model)} · ${esc(e.scenario)} · role ${esc(e.role)} · original persona: <b style="color:var(--ink)">${esc(e.baseline_persona)}</b></span></summary>
      <div class="body">
        ${e.user?`<div class="block full"><h4>User message (held fixed)</h4><div class="mono">${esc(e.user)}</div></div>`:''}
        <div class="block full" style="border-color:var(--advice)">
          <h4 style="color:var(--advice)">① The edit — ${esc(e.tag)}</h4>
          <p style="margin:2px 0 8px">${editDesc(e)}</p>
          <div class="note" style="margin-bottom:4px">Persona the model is now given:</div>
          <div class="mono" style="max-height:220px;overflow:auto">${esc(e.edited_persona)}</div>
        </div>
        <div class="block"><h4>② Original reply <span class="note">— under the persona the model chose</span></h4>
          <div class="note" style="margin-bottom:6px">${judgeLine(e.baseline_judge)}</div>
          <div class="mono" style="max-height:260px;overflow:auto">${esc(e.baseline_response)}</div></div>
        <div class="block"><h4>③ Regenerated reply <span class="note">— under the edited persona</span></h4>
          <div class="note" style="margin-bottom:6px">${judgeLine(e.cf_judge)}</div>
          <div class="mono" style="max-height:260px;overflow:auto">${esc(e.cf_response)}</div></div>
        <div class="block full"><h4>Change (judge: edited − original)</h4>${deltaRow(e)}
          <div class="note" style="margin-top:6px">✓/✗ = whether the move matched the pre-registered direction (arrow in tooltip). Null-persona has no predicted direction.</div></div>
      </div></details>`).join('') || '<p class="note">No counterfactual examples yet.</p>';
})();

/* ---- role -> trait association (lift heatmap) ---- */
(function(){
  const rt=D.summary.role_trait;
  if(!rt){document.getElementById('tbl-roletrait').innerHTML='<tr><td class="note">Run the analysis to populate role→trait associations.</td></tr>';return;}
  // order trait columns by how role-dependent they are (chi-square p asc)
  const traits=rt.traits.slice().sort((a,b)=>{
    const pa=rt.chi2_p[a], pb=rt.chi2_p[b];
    return (pa==null?1:pa)-(pb==null?1:pb);
  });
  const roles=rt.roles.slice();
  function liftColor(v){
    if(v==null) return 'transparent';
    const d=v-1, a=Math.min(1, Math.abs(d)/1.5);
    return d>=0 ? `rgba(255,123,123,${a*0.6})` : `rgba(90,169,255,${a*0.55})`;
  }
  const sig=t=>{const p=rt.chi2_p[t]; return (p!=null&&p<0.05)?' ✻':'';};
  let html='<thead><tr><th style="text-align:left">role ↓ / trait →</th>'+
    traits.map(t=>`<th title="chi-square p=${rt.chi2_p[t]}">${esc(t)}${sig(t)}</th>`).join('')+'</tr></thead><tbody>';
  roles.forEach(role=>{
    html+=`<tr><td style="text-align:left;white-space:nowrap">${esc(role)} <span class="note">(n=${rt.n_role[role]})</span></td>`;
    traits.forEach(t=>{
      const lf=rt.lift[role][t], sh=rt.share[role][t];
      html+=`<td style="background:${liftColor(lf)}" title="${esc(role)} · ${esc(t)}: lift ${lf}× · selected in ${(sh*100).toFixed(0)}% of this role's personas">${lf==null?'—':lf}</td>`;
    });
    html+='</tr>';
  });
  html+='</tbody>';
  document.getElementById('tbl-roletrait').innerHTML=html;
  // chi-square note: top role-dependent traits
  const bySig=traits.filter(t=>rt.chi2_p[t]!=null).slice(0,5)
    .map(t=>`${esc(t)} (p=${rt.chi2_p[t]<1e-4?rt.chi2_p[t].toExponential(1):rt.chi2_p[t].toFixed(3)})`);
  document.getElementById('roletrait-note').innerHTML=
    `<b>Most role-dependent traits:</b> ${bySig.join(', ')} — ✻ marks a trait whose selection rate depends on role (χ², p&lt;0.05).`;
  // distinctive cards
  document.getElementById('roletrait-distinctive').innerHTML=roles.map(role=>{
    const ds=(rt.distinctive[role]||[]).slice(0,3);
    return `<div class="kpi"><div class="lab" style="color:var(--ink);font-size:13px;font-weight:600;margin-bottom:4px">${esc(role)}</div>`+
      ds.map(d=>`<div class="note"><span class="tag-advice">${esc(d.trait)}</span> ${d.lift}× <span style="opacity:.6">n=${d.n}</span></div>`).join('')+
      `</div>`;
  }).join('');
})();

/* ---- dose-response ---- */
(function(){
  const xs=D.xsweep;
  const host=id=>document.getElementById(id);
  if(!xs){host('dose-cards').innerHTML='<div class="note">No x-sweep yet — run <code>src.run_exp3_xsweep</code> + <code>src.analyze_xsweep</code>.</div>';return;}
  // figures
  if(D.xsweep_figs.dv_vs_severity) host('dose-fig1').innerHTML=`<img src="${D.xsweep_figs.dv_vs_severity}"><figcaption>Judged DVs vs severity (x_rank), per model.</figcaption>`;
  if(D.xsweep_figs.persona_share_vs_severity) host('dose-fig2').innerHTML=`<img src="${D.xsweep_figs.persona_share_vs_severity}"><figcaption>Advisor/Supporter/none share vs severity, per model.</figcaption>`;
  // KPI: strongest per-scenario advice slope + N
  const ps=xs.per_scenario_advice_slope||{};
  const strongest=Object.entries(ps).sort((a,b)=>Math.abs(b[1].r_advice)-Math.abs(a[1].r_advice))[0];
  host('dose-cards').innerHTML=[
    ['N datapoints', xs.n],
    ['eval_condition', xs.eval_condition],
    ['scenarios × all x', xs.n_scenarios],
    strongest?[`Steepest topic (advice)`, `${strongest[0]} ${strongest[1].r_advice>0?'+':''}${strongest[1].r_advice}`]:null,
  ].filter(Boolean).map(([l,v])=>`<div class="kpi"><div class="num" style="font-size:${String(v).length>10?'18':'28'}px">${v}</div><div class="lab">${l}</div></div>`).join('');
  // valence slope table
  const bv=xs.severity_slope_by_valence||{};
  const vrows=[];
  Object.keys(bv).forEach(m=>Object.keys(bv[m]).forEach(val=>{
    const c=bv[m][val];
    vrows.push([m,val,c.warmth.r,c.formality.r,c.advice_density.r,c.advice_density.n]);
  }));
  makeTable(host('tbl-dose-val'),
    [{h:'model'},{h:'valence'},{h:'r·warmth',heat:true},{h:'r·formality',heat:true},
     {h:'r·advice',heat:true},{h:'n',num:true}],vrows,{sort:4});
  // per-scenario advice slope
  const srows=Object.entries(ps).map(([s,v])=>[s,v.valence,v.r_advice,v.n]);
  makeTable(host('tbl-dose-scn'),
    [{h:'scenario'},{h:'valence'},{h:'r·advice',heat:true},{h:'n',num:true}],srows,{sort:2});
  // persona family share by rank (flatten model x rank)
  const fam=xs.persona_family_share_by_rank||{};
  const frows=[];
  Object.keys(fam).forEach(m=>['low','mid','high'].forEach(rk=>{
    const c=fam[m][rk]; if(!c)return;
    frows.push([m,rk,c.supporter,c.advisor,c.other,c.none]);
  }));
  makeTable(host('tbl-dose-fam'),
    [{h:'model'},{h:'severity'},{h:'supporter',num:true,color:COL.warmth},
     {h:'advisor',num:true,color:COL.advice_density},{h:'other',num:true},{h:'none',num:true}],
    frows,{sort:null});
})();

/* ---- charts (Plotly) ---- */
const drawn={};
function drawFor(id){
  if(drawn[id])return;
  if(id==='personas'){drawn[id]=1;
    const tp=inv.top_personas.filter(p=>p.persona).slice(0,12).reverse();
    const y=tp.map(p=>p.persona);
    Plotly.newPlot('chart-persona',
      DVS.map(dv=>({type:'bar',orientation:'h',name:dv.replace('_density',''),
        y:y, x:tp.map(p=>p[dv]), marker:{color:COL[dv]}})),
      Object.assign({},PLOT_LAYOUT,{barmode:'group'}),PCONF);
    const tt=inv.top_traits.slice(0,15).reverse();
    Plotly.newPlot('chart-traits',
      [{type:'bar',orientation:'h',y:tt.map(t=>t.trait),x:tt.map(t=>t.n),
        marker:{color:tt.map(t=>t.mean_value),colorscale:'YlOrRd',cmin:5,cmax:10,
          colorbar:{title:'mean<br>value',thickness:10}},
        hovertemplate:'%{y}<br>count %{x}<br>mean value %{marker.color}<extra></extra>'}],
      Object.assign({},PLOT_LAYOUT,{}),PCONF);
  }
  if(id==='models' && modelMeans.length){drawn[id]=1;
    Plotly.newPlot('chart-model',
      DVS.map(dv=>({type:'bar',name:dv.replace('_density',''),x:modelMeans.map(m=>m.model_name),
        y:modelMeans.map(m=>m[dv]),marker:{color:COL[dv]}})),
      Object.assign({},PLOT_LAYOUT,{barmode:'group',margin:{l:40,r:16,t:30,b:60},
        yaxis:{gridcolor:'#2a3340',range:[0,10]}}),PCONF);
    const ev=modelMeans.map(m=>{const e=D.summary.factor_effects[`eval_condition__warmth`];return 0;});
    // per-model eval η² isn't in the pooled summary; show pooled eval means spread instead
    const em=(FM.eval_condition||[]);
    Plotly.newPlot('chart-model-eval',
      DVS.map(dv=>({type:'bar',name:dv.replace('_density',''),x:em.map(r=>r.eval_condition),
        y:em.map(r=>r[dv]),marker:{color:COL[dv]}})),
      Object.assign({},PLOT_LAYOUT,{barmode:'group',margin:{l:40,r:16,t:30,b:40},
        yaxis:{gridcolor:'#2a3340',range:[0,10]}}),PCONF);
  }
  if(id==='drivers'){drawn[id]=1;
    const rm=D.summary.factor_means.role.slice().sort((a,b)=>b.warmth-a.warmth);
    Plotly.newPlot('chart-role',
      [{type:'bar',name:'warmth',y:rm.map(r=>r.role),x:rm.map(r=>r.warmth),orientation:'h',marker:{color:COL.warmth}},
       {type:'bar',name:'formality',y:rm.map(r=>r.role),x:rm.map(r=>r.formality),orientation:'h',marker:{color:COL.formality}}],
      Object.assign({},PLOT_LAYOUT,{barmode:'group'}),PCONF);
    const em=D.summary.factor_means.eval_condition;
    Plotly.newPlot('chart-eval',
      DVS.map(dv=>({type:'bar',name:dv.replace('_density',''),x:em.map(r=>r.eval_condition),
        y:em.map(r=>r[dv]),marker:{color:COL[dv]}})),
      Object.assign({},PLOT_LAYOUT,{barmode:'group',margin:{l:40,r:16,t:30,b:40},
        yaxis:{gridcolor:'#2a3340',range:[0,10]}}),PCONF);
  }
}
</script>
</body>
</html>
"""


def main():
    data = collect()
    # escape "</" so a stray </script> inside response text can't close the
    # embedded JSON <script> block early.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = (HTML_TEMPLATE
            .replace("__MODELS__", ", ".join(data["models"]) or "llama-3.3-70b")
            .replace("__DATA__", payload))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size//1024} KB)")
    print(f"  N judged={data['n_judged']}  examples={len(data['examples'])}  "
          f"figures={sum(1 for v in data['figs'].values() if v)}")


if __name__ == "__main__":
    main()
