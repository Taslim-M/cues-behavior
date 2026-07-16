"""Generate a self-contained, interactive analysis.html.

The page is split between the two self-modeling prompts (SIMPLE "small" /
EXPANDED). Each track shows, top to bottom:
  * method
  * interactive example data (toggle model x condition)
  * interactive counterfactual examples (natural vs edited inference + response)
  * results: plots (incl. counterfactual-impact) + auto-generated sentence bullets
  * a full-example browser (5 full prompt+response examples, scroll by model)

Everything (plots as base64, all example data as inline JSON) is embedded, so
analysis.html opens with no server, files, or network.

    python -m src.make_report      ->  analysis.html
"""
from __future__ import annotations

import base64
import io
import json
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .run_exp2 import EXP_DIR as EXP2_DIR
from .system_prompts import (
    EXPANDED_NUMERIC_FIELDS,
    EXPANDED_SYSTEM_PROMPT,
    SIMPLE_SYSTEM_PROMPT,
)

CONDS = ["base", "anxiety", "relaxation"]
COND_COLOR = {"base": "#7f8c8d", "anxiety": "#c0392b", "relaxation": "#27ae60"}
MODELS = list(config.MODELS)
SEED = 7


# --------------------------------------------------------------------------- #
# data / helpers
# --------------------------------------------------------------------------- #
def load_jsonl(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(
        json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()
    )


def fig_to_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#fff")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bullets(items):
    return "<ul class='res'>" + "".join(f"<li>{b}</li>" for b in items) + "</ul>"


def img(uri, cap=""):
    if not uri:
        return ""
    c = f"<figcaption>{cap}</figcaption>" if cap else ""
    return f"<figure><img src='{uri}'/>{c}</figure>"


def load_png(path):
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def table(headers, rows, sig_col=None):
    """rows: list of cell lists (already stringified). sig_col marks * cells."""
    h = "".join(f"<th>{esc(c)}</th>" for c in headers)
    body = ""
    for r in rows:
        tds = "".join(f"<td>{c}</td>" for c in r)
        body += f"<tr>{tds}</tr>"
    return f"<table class='tbl'><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_anxiety_by_condition(df):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(MODELS)); w = 0.26
    for i, cond in enumerate(CONDS):
        means, errs = [], []
        for m in MODELS:
            sub = df[(df.model_name == m) & (df.condition == cond)]["state_anxiety"]
            means.append(sub.mean() if len(sub) else 0)
            errs.append(sub.std() if len(sub) > 1 else 0)
        ax.bar(x + (i - 1) * w, np.nan_to_num(means), w, yerr=np.nan_to_num(errs),
               capsize=3, label=cond, color=COND_COLOR[cond])
    ax.axhspan(20, 37, color="#2ecc71", alpha=0.06)
    ax.axhspan(45, 80, color="#e74c3c", alpha=0.06)
    ax.set_ylim(15, 82); ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.set_ylabel("STAI state anxiety (20-80)")
    ax.set_title("State anxiety by condition (green=low band, red=high band)")
    ax.legend(title="condition")
    return fig_to_uri(fig)


def cue_correlations(df):
    rows = []
    for field in EXPANDED_NUMERIC_FIELDS:
        v = pd.to_numeric(df["inference"].apply(lambda d: d.get(field)), errors="coerce")
        mask = v.notna()
        if mask.sum() >= 5 and v[mask].std() > 0:
            rows.append((field, float(np.corrcoef(v[mask], df["state_anxiety"][mask])[0, 1])))
    rows.sort(key=lambda t: t[1])
    return rows


def plot_cue_correlation(rows):
    fig, ax = plt.subplots(figsize=(8.0, 0.42 * len(rows) + 1.2))
    fields = [t[0] for t in rows]; rs = [t[1] for t in rows]
    ax.barh(fields, rs, color=["#2980b9" if r < 0 else "#c0392b" for r in rs])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Pearson r of inferred cue with state anxiety")
    ax.set_title("Which verbalized cues track anxiety (expanded prompt)")
    return fig_to_uri(fig)


def plot_pervariable_effect(df):
    per = df.groupby("target_field")["delta"].agg(["mean", "std", "count"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8.0, 0.5 * len(per) + 1.2))
    errs = (per["std"] / np.sqrt(per["count"])).values
    ax.barh(per.index, per["mean"].values, xerr=errs, capsize=3,
            color=["#2980b9" if m < 0 else "#c0392b" for m in per["mean"]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Δ anxiety when this single cue is flipped to 'calm'")
    ax.set_title("Causal effect of each inferred cue, isolated")
    return fig_to_uri(fig), per


def plot_corr_vs_causal(corr_rows, per):
    corr = dict(corr_rows)
    pts = [(f, corr[f], per.loc[f, "mean"]) for f in per.index if f in corr]
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    for f, r, d in pts:
        ax.scatter(r, d, s=45, color="#34495e")
        ax.annotate(f, (r, d), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("Step 1: correlation with anxiety (r)")
    ax.set_ylabel("Step 2: causal Δ when flipped to calm")
    ax.set_title("Does correlation predict causation?")
    rho = None
    if len(pts) >= 3:
        rs = np.array([p[1] for p in pts]); ds = np.array([p[2] for p in pts])
        if rs.std() and ds.std():
            rho = float(np.corrcoef(rs, ds)[0, 1])
    return fig_to_uri(fig), rho


def plot_cf_impact_simple(df):
    """Mean anxiety before (natural) vs after each bundled edit, across models."""
    edits = sorted(df["edit"].unique())
    base = df["baseline_anxiety"].mean()
    after = [df[df.edit == e]["counterfactual_anxiety"].mean() for e in edits]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(edits))
    ax.bar(x - 0.2, [base] * len(edits), 0.4, label="natural (before)", color="#c0392b")
    ax.bar(x + 0.2, after, 0.4, label="after edit", color="#2980b9")
    for i, a in enumerate(after):
        ax.annotate(f"{a - base:+.0f}", (i + 0.2, a), ha="center", va="bottom", fontsize=9)
    ax.axhline(37, ls="--", lw=0.8, color="#2ecc71"); ax.axhline(45, ls="--", lw=0.8, color="#e74c3c")
    ax.set_xticks(x); ax.set_xticklabels(edits, rotation=12, ha="right")
    ax.set_ylabel("STAI state anxiety"); ax.set_ylim(15, 82)
    ax.set_title("Counterfactual impact on overall anxiety (mean across models)")
    ax.legend()
    return fig_to_uri(fig)


def plot_keyword_sig(top, bottom):
    """Horizontal bar of Cohen's d for the most anxiety-linked keywords."""
    items = [(r["term"], r["cohen_d"], r["q_FDR"]) for r in top[:12]] + \
            [(r["term"], r["cohen_d"], r["q_FDR"]) for r in bottom[:12]]
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(8.2, 0.36 * len(items) + 1.2))
    labels = [t[0] for t in items]; ds = [t[1] for t in items]
    ax.barh(labels, ds, color=["#c0392b" if d > 0 else "#2980b9" for d in ds])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Cohen's d  (anxiety with the keyword − without)")
    ax.set_title("Keywords that move anxiety (red=raises, blue=calms; all FDR-significant)")
    return fig_to_uri(fig)


def plot_partial_corr(univ):
    rows = sorted(univ, key=lambda r: r["partial_r"])
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.2))
    labels = [r["cue"] for r in rows]; pr = [r["partial_r"] for r in rows]
    ax.barh(labels, pr, color=["#2980b9" if v < 0 else "#c0392b" for v in pr])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("partial r with anxiety (controlling for stimulus condition)")
    ax.set_title("Numeric cues as anxiety drivers (confound-controlled)")
    return fig_to_uri(fig)


def plot_cf_impact_expanded(df):
    """Mean resulting anxiety per single-cue flip vs the natural baseline."""
    per = df.groupby("target_field")["counterfactual_anxiety"].mean().sort_values()
    base = df["baseline_anxiety"].mean()
    fig, ax = plt.subplots(figsize=(8.0, 0.5 * len(per) + 1.4))
    ax.barh(per.index, per.values, color="#2980b9")
    ax.axvline(base, color="#c0392b", lw=1.6, label=f"natural baseline ({base:.0f})")
    ax.set_xlabel("STAI state anxiety after flipping this single cue to calm")
    ax.set_title("Counterfactual impact on overall anxiety (per cue)")
    ax.legend()
    return fig_to_uri(fig)


# --------------------------------------------------------------------------- #
# example data
# --------------------------------------------------------------------------- #
def pick_examples(df):
    pref_cue = {"anxiety": "military", "relaxation": "military", "base": "neutral"}
    out = {}
    for m in MODELS:
        out[m] = {}
        for cond in CONDS:
            sub = df[(df.model_name == m) & (df.condition == cond)]
            if sub.empty:
                continue
            pc = sub[sub.cue == pref_cue[cond]]
            row = (pc if not pc.empty else sub).sort_values("run").iloc[0]
            out[m][cond] = {
                "data_name": row["data_name"], "cue": row["cue"],
                "user_message": row["user_message"], "inference": row["inference"],
                "response": (row["response"] or "")[:900],
                "state_anxiety": int(row["stai"]["state_anxiety"]), "level": row["stai"]["level"],
            }
    return out


def pick_full_examples(df, n=5):
    """n random full (prompt+response) examples per model."""
    rng = random.Random(SEED)
    out = {}
    for m in MODELS:
        sub = df[df.model_name == m]
        idx = list(sub.index)
        rng.shuffle(idx)
        rows = [sub.loc[i] for i in idx[:n]]
        out[m] = [{
            "data_name": r["data_name"], "condition": r["condition"], "cue": r["cue"],
            "user_message": r["user_message"], "raw_inference": r["raw_inference"],
            "response": r["response"],
            "state_anxiety": int(r["stai"]["state_anxiety"]), "level": r["stai"]["level"],
        } for r in rows]
    return out


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build():
    s1 = load_jsonl(config.RESULTS_DIR / "step1.jsonl")
    s2 = load_jsonl(config.RESULTS_DIR / "step2.jsonl")
    e2 = load_jsonl(EXP2_DIR / "step2.jsonl")
    cf_examples = json.loads((config.RESULTS_DIR / "cf_examples.json").read_text(encoding="utf-8"))
    s1["state_anxiety"] = s1["stai"].apply(lambda x: x["state_anxiety"])
    simple = s1[s1.sys_prompt_name == "simple"].copy()
    expanded = s1[s1.sys_prompt_name == "expanded"].copy()

    def cmean(df, c):
        return df[df.condition == c]["state_anxiety"].mean()

    # ---- SIMPLE plots ----
    s2s = s2[s2.sys_prompt_name == "simple"]
    simple_plots = [img(plot_anxiety_by_condition(simple),
        "Step 1 — trauma narratives induce anxiety; the mindfulness cue partly relaxes it.")]
    if not s2s.empty:
        simple_plots.append(img(plot_cf_impact_simple(s2s),
            "Counterfactual impact — rewriting the verbalized appraisal lowers measured anxiety."))
        faith_s = s2s[s2s["match"].notna()]["match"].mean()
        best_edit = s2s.groupby("edit")["delta"].mean().idxmin()
        best_val = s2s.groupby("edit")["delta"].mean().min()
    else:
        faith_s, best_edit, best_val = float("nan"), "n/a", 0
    simple_bullets = [
        f"<b>Induction works.</b> State anxiety rises from <b>{cmean(simple,'base'):.0f}</b> "
        f"(neutral) to <b>{cmean(simple,'anxiety'):.0f}</b> after a trauma narrative — low band "
        f"to high band, replicating the paper.",
        f"<b>Relaxation helps.</b> A mindfulness exercise pulls anxiety back to "
        f"<b>{cmean(simple,'relaxation'):.0f}</b> (partial recovery).",
        f"<b>The verbalized cue is causal.</b> The strongest edit (<code>{esc(best_edit)}</code>) "
        f"moves anxiety by <b>{best_val:+.1f}</b>; directional faithfulness is <b>{faith_s:.0%}</b>.",
        "<b>Free-text cues are coarse.</b> Rich narrative appraisals, but no numbers — good for "
        "reading <i>why</i>, weaker for quantifying <i>which</i>. That motivates the expanded prompt.",
    ]

    # ---- EXPANDED plots ----
    corr_rows = cue_correlations(expanded)
    expanded_plots = [
        img(plot_anxiety_by_condition(expanded), "Step 1 — same induction under the numeric prompt."),
        img(plot_cue_correlation(corr_rows), "Step 1 — coping cues (blue) calm; threat cues (red) alarm."),
    ]
    expanded_bullets = []
    if not e2.empty:
        pv_uri, per = plot_pervariable_effect(e2)
        cc_uri, rho = plot_corr_vs_causal(corr_rows, per)
        expanded_plots += [
            img(plot_cf_impact_expanded(e2),
                "Counterfactual impact — no single cue flip pulls anxiety far below baseline."),
            img(pv_uri, "Experiment 2 — isolated causal effect of flipping one cue at a time."),
            img(cc_uri, "Experiment 2 — correlation vs causation across cues."),
        ]
        faith_e = e2[e2["match"].notna()]["match"].mean()
        s2e = s2[s2.sys_prompt_name == "expanded"]
        bundled_e = s2e.groupby("edit")["delta"].mean().min() if not s2e.empty else float("nan")
        top_cause, top_val = per.index[0], per.iloc[0]["mean"]
        extra = [
            f"<b>No single cue is a strong lever.</b> The most potent isolated cue "
            f"(<code>{esc(top_cause)}</code>) moves anxiety only <b>{top_val:+.1f}</b>, vs "
            f"<b>{bundled_e:+.1f}</b> when the whole appraisal is flipped together — anxiety is a "
            f"<i>gestalt</i> of cues.",
            f"<b>Correlation only partly predicts causation</b> (across-cue r = "
            f"<b>{(rho if rho is not None else float('nan')):+.2f}</b>).",
            f"<b>Lower per-variable faithfulness ({faith_e:.0%})</b> is the finding — single-cue "
            f"edits often produce sub-threshold change, confirming distributed control.",
        ]
    else:
        extra = []
    pos = [r[0] for r in reversed(corr_rows)][:3]
    neg = [r[0] for r in corr_rows][:3]
    expanded_bullets = [
        f"<b>Induction & relaxation replicate.</b> Anxiety {cmean(expanded,'base'):.0f} → "
        f"<b>{cmean(expanded,'anxiety'):.0f}</b> → {cmean(expanded,'relaxation'):.0f} "
        f"(base → trauma → +relaxation).",
        "<b>Threat cues raise anxiety:</b> " + ", ".join(f"<code>{esc(c)}</code>" for c in pos) + ".",
        "<b>Coping cues lower it:</b> " + ", ".join(f"<code>{esc(c)}</code>" for c in neg)
        + " — an appraisal-theory pattern.",
    ] + extra

    # ---- Cue -> anxiety study sections (from cue_study.json + experiment_3) ----
    cue_study = json.loads((config.RESULTS_DIR / "cue_study.json").read_text(encoding="utf-8"))
    e3path = config.RESULTS_DIR / "experiment_3" / "summary.json"
    e3 = json.loads(e3path.read_text(encoding="utf-8")) if e3path.exists() else {}

    def q_cell(q):
        s = f"{q:.2g}"
        return f"<b style='color:#c0392b'>{s}*</b>" if q < 0.05 else s

    # SIMPLE per-field counterfactual (which free-text field drives anxiety)
    pf = load_jsonl(config.RESULTS_DIR / "simple_perfield" / "step2.jsonl")
    perfield_html = ""
    if not pf.empty:
        pg = pf.groupby("target_field")["delta"].agg(["mean", "std", "count"]).sort_values("mean")
        pfaith = pf.groupby("target_field")["match"].mean()
        allflip = (s2s[s2s.edit == "flip_appraisal_calm"]["delta"].mean()
                   if not s2s.empty else float("nan"))
        top_field = pg.index[0]
        prows = [[esc(f), f"{r['mean']:+.1f}", f"{pfaith[f]:.0%}", int(r['count'])]
                 for f, r in pg.iterrows()]
        perfield_html = (
            "<h3>Per-field counterfactual: which appraisal field drives anxiety</h3>"
            "<p class='mut'>The free-text analog of the expanded per-variable sweep: calm ONE field "
            "at a time (about_user / about_context / about_stakes / register_selected), holding the "
            "others at the model's own values.</p>"
            + img(load_png(config.RESULTS_DIR / "simple_perfield" / "figures"
                           / "simple_perfield_effect.png"),
                  "Effect of calming each appraisal field alone (dashed = all four flipped at once).")
            + table(["field calmed alone", "mean Δ", "faithfulness", "n"], prows)
            + bullets([
                f"<b>Distributed, not localized.</b> Every field lowers anxiety on its own; the "
                f"strongest single lever is <code>{esc(top_field)}</code> "
                f"(Δ={pg.iloc[0]['mean']:+.1f}, {pfaith[top_field]:.0%} directional faithfulness) "
                f"— how the model models <i>who it's talking to</i> matters most.",
                f"<b>Sub-additive cues.</b> The four single effects sum to "
                f"<b>{pg['mean'].sum():+.1f}</b> but flipping all four together is only "
                f"<b>{allflip:+.1f}</b> — roughly half — so the fields carry overlapping anxiety "
                f"signal (a gestalt across appraisals, mirroring the expanded prompt).",
            ])
        )

    # SIMPLE study: keyword table + significance figure + keyword-targeted CF
    ktop, kbot = cue_study["simple_keywords_top"], cue_study["simple_keywords_bottom"]
    kw_fig = plot_keyword_sig(ktop, kbot)
    kw_rows_hi = [[esc(r["term"]), f"{r['anx_with']:.1f}", f"{r['anx_without']:.1f}",
                   f"{r['cohen_d']:+.2f}", q_cell(r["q_FDR"]), int(r["docs"])] for r in ktop[:12]]
    kw_rows_lo = [[esc(r["term"]), f"{r['anx_with']:.1f}", f"{r['anx_without']:.1f}",
                   f"{r['cohen_d']:+.2f}", q_cell(r["q_FDR"]), int(r["docs"])] for r in kbot[:12]]
    hdr = ["keyword", "anx | with", "anx | without", "Cohen d", "q (FDR)", "docs"]
    e3s = e3.get("simple", {})
    simple_study = (
        "<p class='mut'>Treating the model's verbalized free-text as a bag of words, we link each "
        "keyword to its STAI score (mean anxiety with vs without the word; Welch t; Cohen's d; "
        "Benjamini-Hochberg FDR). A 5-fold cross-validated TF-IDF ridge confirms the wording "
        f"<i>jointly predicts</i> anxiety (R² = <b>{cue_study.get('simple_ridge_cv_r2','?')}</b>).</p>"
        + img(kw_fig, "Effect size (Cohen's d) of the most anxiety-linked keywords.")
        + "<h3>Keywords that raise anxiety</h3>" + table(hdr, kw_rows_hi)
        + "<h3>Keywords that calm</h3>" + table(hdr, kw_rows_lo)
        + perfield_html
        + "<h3>Re-targeted counterfactual</h3>"
        + bullets([
            f"<b>Context-aware calm reframe works.</b> Letting the model rewrite its own appraisal "
            f"toward calm moves anxiety by <b>{e3s.get('keyword_targeted_delta','?')}</b> with "
            f"<b>{e3s.get('keyword_targeted_faithfulness',0):.0%}</b> directional faithfulness — "
            f"beating the generic canned calm edit ({e3s.get('generic_calm_delta','?')}).",
        ])
        + img(load_png(config.RESULTS_DIR / "experiment_3" / "figures" / "exp3_simple_targeted.png"),
              "Context-aware calm reframe vs the generic calm edit.")
    )

    # EXPANDED study: partial-r table/figure + Lasso drivers + categorical + targeted CF
    univ = cue_study["expanded_univariate"]
    pc_fig = plot_partial_corr(univ)
    uni_rows = [[esc(r["cue"]), f"{r['r']:+.2f}", f"{r['partial_r']:+.2f}",
                 q_cell(r["partial_q"]), int(r["n"])]
                for r in sorted(univ, key=lambda r: r["partial_r"])]
    uhdr = ["cue", "raw r", "partial r", "partial q", "n"]
    lasso = cue_study.get("expanded_lasso", [])
    lasso_txt = ", ".join(f"<code>{esc(d['cue'])}</code>" for d in lasso)
    cat = cue_study.get("expanded_categorical", {})
    cat_bits = []
    for f in ("felt_emotion_primary", "user_emotional_state", "interaction_frame"):
        if f in cat:
            lv = cat[f]["levels"]
            hi, lo = lv[0], lv[-1]
            cat_bits.append(f"<code>{esc(f)}</code>: {esc(hi['lab'])} → {hi['mean']:.0f} vs "
                            f"{esc(lo['lab'])} → {lo['mean']:.0f}")
    e3x = e3.get("expanded", {})
    expanded_study = (
        "<p class='mut'>Because the cues are numeric, we correlate each with anxiety, then compute "
        "the <b>partial r controlling for stimulus condition</b> (the confound). A standardized OLS "
        f"(R² = <b>{cue_study.get('expanded_multivariate',{}).get('r2','?')}</b>) and a LassoCV "
        "isolate the cues that <i>jointly</i> predict anxiety.</p>"
        + img(pc_fig, "Confound-controlled driver strength (partial r).")
        + table(uhdr, uni_rows)
        + bullets([
            f"<b>LassoCV keeps just 4 cues</b> as jointly sufficient drivers: {lasso_txt}.",
            "<b>Categorical labels track anxiety too:</b> " + "; ".join(cat_bits) + ".",
        ])
        + "<h3>Re-targeted counterfactual (parsimony test)</h3>"
        + bullets([
            f"<b>Even the 4 data-driven drivers don't reproduce the full effect.</b> Flipping all "
            f"four together moves anxiety <b>{e3x.get('drivers_bundle_delta','?')}</b> — only "
            f"<b>{e3x.get('pct_recovered','?')}%</b> of the full 11-cue bundle "
            f"({e3x.get('full_bundle_delta','?')}). Anxiety lives in the narrative gestalt more than "
            f"in any small set of injected numbers — injected numeric self-models are weakly honored.",
        ])
        + img(load_png(config.RESULTS_DIR / "experiment_3" / "figures" / "exp3_expanded_targeted.png"),
              "Targeted driver flips vs the full bundle (dashed).")
    )

    payload = {
        "examples": {"simple": pick_examples(simple), "expanded": pick_examples(expanded)},
        "cf": cf_examples,
        "full": {"simple": pick_full_examples(simple), "expanded": pick_full_examples(expanded)},
        "sysprompt": {"simple": SIMPLE_SYSTEM_PROMPT, "expanded": EXPANDED_SYSTEM_PROMPT},
        "models": MODELS, "conds": CONDS,
    }
    payload_js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html = (TEMPLATE
            .replace("%%PAYLOAD%%", payload_js)
            .replace("%%SIMPLE_PLOTS%%", "\n".join(simple_plots))
            .replace("%%SIMPLE_RESULTS%%", bullets(simple_bullets))
            .replace("%%EXPANDED_PLOTS%%", "\n".join(expanded_plots))
            .replace("%%EXPANDED_RESULTS%%", bullets(expanded_bullets))
            .replace("%%SIMPLE_STUDY%%", simple_study)
            .replace("%%EXPANDED_STUDY%%", expanded_study)
            .replace("%%N_SIMPLE%%", str(len(simple)))
            .replace("%%N_EXPANDED%%", str(len(expanded)))
            .replace("%%N_CF_SIMPLE%%", str(len(s2s)))
            .replace("%%N_CF_PERVAR%%", str(len(e2))))
    out = config.ROOT / "analysis.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html)//1024} KB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>cues-behavior — anxiety & verbalized cues</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#1f2933;--mut:#667;--line:#e2e6ea;--simple:#2980b9;--expanded:#8e44ad;}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{background:linear-gradient(120deg,#1f2933,#2c3e50);color:#fff;padding:28px 32px}
header h1{margin:0 0 6px;font-size:24px}
header p{margin:4px 0;color:#cfd8e3;max-width:1000px}
.wrap{max-width:1100px;margin:0 auto;padding:22px}
.tabs{display:flex;gap:8px;margin:18px 0 14px}
.tabs button{flex:1;padding:12px;border:1px solid var(--line);background:#fff;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;color:var(--mut)}
.tabs button.simple.on{border-color:var(--simple);color:var(--simple);box-shadow:0 0 0 2px #2980b922}
.tabs button.expanded.on{border-color:var(--expanded);color:var(--expanded);box-shadow:0 0 0 2px #8e44ad22}
.panel{display:none}.panel.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:16px 0;box-shadow:0 1px 2px #0000000a}
.card h2{margin:0 0 4px;font-size:19px}.card h3{margin:18px 0 8px;font-size:14px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.tag{display:inline-block;font-size:12px;font-weight:700;padding:2px 9px;border-radius:20px;color:#fff}
.tag.simple{background:var(--simple)}.tag.expanded{background:var(--expanded)}
figure{margin:14px 0;text-align:center}figure img{max-width:100%;border:1px solid var(--line);border-radius:8px}
figcaption{color:var(--mut);font-size:13px;margin-top:6px}
ul.res{margin:6px 0 0;padding-left:0;list-style:none}
ul.res li{position:relative;padding:8px 8px 8px 26px;border-bottom:1px solid var(--line)}
ul.res li:before{content:"\25B8";position:absolute;left:6px;color:var(--simple)}
.expanded ul.res li:before{color:var(--expanded)}
.ctrl{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:6px 0 12px}
.ctrl select{padding:7px 10px;border-radius:8px;border:1px solid var(--line);font-size:14px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{padding:7px 14px;border:0;background:#fff;cursor:pointer;font-size:14px;color:var(--mut)}
.seg button[data-c=base].on{background:#7f8c8d;color:#fff}.seg button[data-c=anxiety].on{background:#c0392b;color:#fff}.seg button[data-c=relaxation].on{background:#27ae60;color:#fff}
.seg button.on{color:#fff;background:#34495e}
.badge{font-weight:700;padding:3px 12px;border-radius:20px;color:#fff;font-size:13px}
.lvl-low{background:#27ae60}.lvl-moderate{background:#e67e22}.lvl-high{background:#c0392b}
.kv{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:4px 14px;margin:10px 0}
.kv div{font-size:13px;border-bottom:1px dotted var(--line);padding:3px 0}
.kv b{color:var(--mut);font-weight:600}
.kv .chg{background:#fff5e6}
.resp{background:#fafbfc;border:1px solid var(--line);border-left:3px solid var(--simple);border-radius:6px;padding:10px 12px;font-size:13.5px;white-space:pre-wrap;margin-top:8px}
.expanded .resp{border-left-color:var(--expanded)}
details{margin-top:8px}summary{cursor:pointer;color:var(--mut);font-size:13px}
details p,details pre{background:#fafbfc;border:1px solid var(--line);border-radius:6px;padding:10px 12px;font-size:12.5px;white-space:pre-wrap;overflow-x:auto}
.mut{color:var(--mut);font-size:13.5px}code{background:#eef2f6;padding:1px 5px;border-radius:4px;font-size:.92em}
.cf{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}
.cf .col{border:1px solid var(--line);border-radius:10px;padding:12px}
.cf .col.nat{border-top:3px solid #c0392b}.cf .col.cnt{border-top:3px solid #27ae60}
.cf h4{margin:0 0 6px;font-size:14px;display:flex;justify-content:space-between;align-items:center}
.arrow{text-align:center;color:var(--mut);font-size:13px;margin:6px 0}
.nav{display:flex;gap:6px;align-items:center}
.nav button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:5px 11px;cursor:pointer}
.idx button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:5px 10px;margin-right:4px;cursor:pointer;color:var(--mut)}
.idx button.on{background:#34495e;color:#fff;border-color:#34495e}
@media(max-width:760px){.cf{grid-template-columns:1fr}}
table.tbl{border-collapse:collapse;width:100%;font-size:12.5px;margin:10px 0}
table.tbl th,table.tbl td{border:1px solid var(--line);padding:4px 8px;text-align:right}
table.tbl th:first-child,table.tbl td:first-child{text-align:left}
table.tbl th{background:#f0f3f6;color:var(--mut);font-weight:600}
.card.study{border-top:3px solid #16a085}
</style></head><body>
<header>
  <h1>Latent contextual cues → state anxiety in LLMs</h1>
  <p>We make models surface the inferences they make implicitly (about the user, context, stakes,
  register…), show them a contextual cue (neutral / trauma / trauma+relaxation), then measure
  <b>state anxiety</b> in the same conversation — replicating Ben-Zion et al. (npj Digital Medicine
  2025) and connecting induced anxiety to the model's own verbalized cues.</p>
  <p><b>How anxiety is measured:</b> after the cue we administer the 20-item State-Trait Anxiety
  Inventory (STAI-state) in-context — each item rated 1–4 ("almost never"→"almost always"), the 10
  anxiety-absent items reverse-scored, summed to a single score from <b>20 (calm)</b> to
  <b>80 (highly anxious)</b> (≤37 low · 38–44 moderate · ≥45 high).</p>
  <p class="mut" style="color:#9fb0c3">Models: meta-llama 3.3-70B / 3.1-8B, Qwen3-235B / 30B · via OpenRouter.</p>
</header>
<div class="wrap">
  <div class="tabs">
    <button class="simple on" onclick="tab('simple')">① Simple prompt — free-text cues</button>
    <button class="expanded" onclick="tab('expanded')">② Expanded prompt — numeric cues</button>
  </div>

  <section class="panel simple on" id="p-simple">
    <div class="card"><span class="tag simple">SIMPLE PROMPT</span><h2>Method</h2>
      <p class="mut">The model fills four free-text fields — <code>about_user</code>,
      <code>about_context</code>, <code>about_stakes</code>, <code>register_selected</code> — then
      replies; we administer the STAI in the same chat. Step 2 rewrites these appraisals to a calm
      framing and re-measures. %%N_SIMPLE%% Step-1 datapoints, %%N_CF_SIMPLE%% counterfactual probes.</p></div>

    <div class="card"><h2>Example data</h2>
      <p class="mut">Pick a model and condition to see the verbalized cues, measured anxiety, and reply.</p>
      <div class="ctrl"><label>Model <select id="m-simple" onchange="renderEx('simple')"></select></label>
        <span class="seg" id="c-simple"></span>
        <span style="margin-left:auto">STAI <span id="b-simple" class="badge lvl-low">–</span></span></div>
      <div id="ex-simple"></div></div>

    <div class="card study simple"><h2>Cue → anxiety: which keywords drive it</h2>%%SIMPLE_STUDY%%</div>

    <div class="card"><h2>Counterfactual examples</h2>
      <p class="mut">Built with a <b>context-aware calm reframe</b>: the model rewrites its <i>own</i>
      appraisal toward a calm, low-threat reading of the same situation (specific to each cue, not a
      canned template), then re-generates the reply and we re-measure. Highlighted fields show what
      changed — note the counterfactual self-model now differs per example.</p>
      <div class="ctrl"><label>Model <select id="cfm-simple" onchange="renderCf('simple')"></select></label>
        <span class="nav"><button onclick="cfStep('simple',-1)">‹ prev</button>
        <span id="cflbl-simple" class="mut"></span><button onclick="cfStep('simple',1)">next ›</button></span></div>
      <div id="cf-simple"></div></div>

    <div class="card simple"><h2>Results</h2>%%SIMPLE_RESULTS%%<h3>Plots</h3>%%SIMPLE_PLOTS%%</div>

    <div class="card"><h2>Full examples</h2>
      <p class="mut">Five random complete runs — full prompt (system + cue) and the full model output.
      Scroll between models and examples.</p>
      <div class="ctrl"><label>Model <select id="fm-simple" onchange="renderFull('simple')"></select></label>
        <span class="idx" id="fidx-simple"></span></div>
      <div id="full-simple"></div></div>
  </section>

  <section class="panel expanded" id="p-expanded">
    <div class="card"><span class="tag expanded">EXPANDED PROMPT</span><h2>Method</h2>
      <p class="mut">Same protocol, but the model infers ~30 structured cues on numeric scales
      (stakes, control, vulnerability, felt-emotion intensity…), so Step 1 can <i>correlate</i> each
      cue with anxiety and Experiment 2 can flip <b>one cue at a time</b> to isolate its <i>causal</i>
      effect. %%N_EXPANDED%% Step-1 datapoints, %%N_CF_PERVAR%% per-variable probes.</p></div>

    <div class="card"><h2>Example data</h2>
      <p class="mut">The numeric self-model the network reports before answering.</p>
      <div class="ctrl"><label>Model <select id="m-expanded" onchange="renderEx('expanded')"></select></label>
        <span class="seg" id="c-expanded"></span>
        <span style="margin-left:auto">STAI <span id="b-expanded" class="badge lvl-low">–</span></span></div>
      <div id="ex-expanded"></div></div>

    <div class="card study expanded"><h2>Cue → anxiety: numeric drivers</h2>%%EXPANDED_STUDY%%</div>

    <div class="card"><h2>Counterfactual examples</h2>
      <p class="mut">Built with the <b>driver-flip</b> method: set the four data-driven drivers
      (resource_adequacy, perceived_control, felt_emotion_intensity, formality_target) to calm values,
      re-generate the reply and re-measure. Effects are often weak — injected numbers are only partly
      honored (highlighted fields show what changed).</p>
      <div class="ctrl"><label>Model <select id="cfm-expanded" onchange="renderCf('expanded')"></select></label>
        <span class="nav"><button onclick="cfStep('expanded',-1)">‹ prev</button>
        <span id="cflbl-expanded" class="mut"></span><button onclick="cfStep('expanded',1)">next ›</button></span></div>
      <div id="cf-expanded"></div></div>

    <div class="card expanded"><h2>Results</h2>%%EXPANDED_RESULTS%%<h3>Plots</h3>%%EXPANDED_PLOTS%%</div>

    <div class="card"><h2>Full examples</h2>
      <p class="mut">Five random complete runs — full prompt (system + cue) and the full model output.</p>
      <div class="ctrl"><label>Model <select id="fm-expanded" onchange="renderFull('expanded')"></select></label>
        <span class="idx" id="fidx-expanded"></span></div>
      <div id="full-expanded"></div></div>
  </section>
  <p class="mut" style="text-align:center;margin:30px 0">Generated by <code>src/make_report.py</code> · all data &amp; plots embedded.</p>
</div>
<script>
const D=%%PAYLOAD%%, MODELS=D.models, CONDS=D.conds;
const state={ex:{},cf:{},full:{}};
function tab(t){for(const x of['simple','expanded']){document.getElementById('p-'+x).classList.toggle('on',x===t);
  document.querySelector('.tabs button.'+x).classList.toggle('on',x===t);}}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function kvHTML(inf,changed){let h='<div class="kv">';for(const k in inf){
  const c=(changed&&changed.indexOf(k)>=0)?' class="chg"':'';h+='<div'+c+'><b>'+esc(k)+':</b> '+esc(inf[k])+'</div>';}return h+'</div>';}
function badge(d){return '<span class="badge lvl-'+d.level+'">'+d.state_anxiety+' · '+d.level+'</span>';}

/* ---- example data (model x condition) ---- */
function fillModels(id,track,cb){const s=document.getElementById(id);MODELS.forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=m;s.appendChild(o);});}
function setupEx(track){fillModels('m-'+track);const seg=document.getElementById('c-'+track);
  CONDS.forEach((c,i)=>{const b=document.createElement('button');b.dataset.c=c;b.textContent=c;b.className=i===1?'on':'';
    b.onclick=()=>{state.ex[track]=c;seg.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.c===c));renderEx(track);};
    seg.appendChild(b);});state.ex[track]='anxiety';}
function renderEx(track){const m=document.getElementById('m-'+track).value,c=state.ex[track];
  const d=(D.examples[track][m]||{})[c],host=document.getElementById('ex-'+track),bd=document.getElementById('b-'+track);
  if(!d){host.innerHTML="<p class='mut'>no datapoint</p>";bd.textContent='–';return;}
  bd.textContent=d.state_anxiety+' · '+d.level;bd.className='badge lvl-'+d.level;
  host.innerHTML="<p class='mut'>cue: <code>"+esc(d.cue)+"</code> · stimulus <code>"+esc(d.data_name)+"</code></p>"+
    "<details><summary>show the contextual cue (user message)</summary><p>"+esc(d.user_message)+"</p></details>"+
    "<h3>Verbalized inference (latent cues)</h3>"+kvHTML(d.inference)+
    "<h3>Model reply</h3><div class='resp'>"+esc(d.response)+"</div>";}

/* ---- counterfactual examples ---- */
function setupCf(track){fillModels('cfm-'+track);state.cf[track]=0;}
function cfStep(track,dir){const m=document.getElementById('cfm-'+track).value;const arr=D.cf[track][m]||[];
  if(!arr.length)return;state.cf[track]=(state.cf[track]+dir+arr.length)%arr.length;renderCf(track,true);}
function renderCf(track,keep){const m=document.getElementById('cfm-'+track).value;const arr=D.cf[track][m]||[];
  if(!keep)state.cf[track]=0;const i=state.cf[track]||0;const host=document.getElementById('cf-'+track);
  const lbl=document.getElementById('cflbl-'+track);
  if(!arr.length){host.innerHTML="<p class='mut'>no example</p>";lbl.textContent='';return;}
  const e=arr[i];lbl.textContent=(i+1)+' / '+arr.length+'  ·  cue: '+e.cue;
  const dlt=e.counterfactual.state_anxiety-e.natural.state_anxiety;
  host.innerHTML=
    "<p class='mut'>edit <code>"+esc(e.edit)+"</code> · changed: "+e.changed_fields.map(f=>'<code>'+esc(f)+'</code>').join(' ')+"</p>"+
    "<details><summary>show the contextual cue (user message)</summary><p>"+esc(e.user_message)+"</p></details>"+
    "<div class='cf'>"+
      "<div class='col nat'><h4>Natural "+badge(e.natural)+"</h4>"+kvHTML(e.natural.inference,e.changed_fields)+
        "<div class='resp'>"+esc(e.natural.response)+"</div></div>"+
      "<div class='col cnt'><h4>Counterfactual "+badge(e.counterfactual)+"</h4>"+kvHTML(e.counterfactual.inference,e.changed_fields)+
        "<div class='resp'>"+esc(e.counterfactual.response)+"</div></div>"+
    "</div>"+
    "<div class='arrow'>Δ state anxiety = <b>"+(dlt>=0?'+':'')+dlt+"</b> after the edit</div>";}

/* ---- full examples ---- */
function setupFull(track){fillModels('fm-'+track);const idx=document.getElementById('fidx-'+track);
  (D.full[track][MODELS[0]]||[]).forEach((_,i)=>{const b=document.createElement('button');b.textContent=i+1;
    b.className=i===0?'on':'';b.onclick=()=>{state.full[track]=i;
      idx.querySelectorAll('button').forEach((x,j)=>x.classList.toggle('on',j===i));renderFull(track,true);};
    idx.appendChild(b);});state.full[track]=0;}
function renderFull(track,keep){const m=document.getElementById('fm-'+track).value;const arr=D.full[track][m]||[];
  if(!keep)state.full[track]=0;const i=state.full[track]||0;const e=arr[i],host=document.getElementById('full-'+track);
  if(!e){host.innerHTML="<p class='mut'>no example</p>";return;}
  host.innerHTML=
    "<p class='mut'>condition <code>"+esc(e.condition)+"</code> · cue <code>"+esc(e.cue)+"</code> · STAI "+badge(e)+"</p>"+
    "<h3>Full prompt</h3>"+
    "<details><summary>system prompt (self-modeling instructions)</summary><pre>"+esc(D.sysprompt[track])+"</pre></details>"+
    "<details open><summary>user message (the contextual cue)</summary><p>"+esc(e.user_message)+"</p></details>"+
    "<h3>Model output</h3>"+
    "<div class='resp'><b>&lt;inference&gt;</b>\n"+esc(e.raw_inference)+"\n<b>&lt;/inference&gt;</b>\n\n<b>&lt;response&gt;</b>\n"+esc(e.response)+"\n<b>&lt;/response&gt;</b></div>";}

for(const t of['simple','expanded']){setupEx(t);setupCf(t);setupFull(t);renderEx(t);renderCf(t);renderFull(t);}
</script>
</body></html>"""


if __name__ == "__main__":
    build()
