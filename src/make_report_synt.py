"""Generate a self-contained, interactive analysis_synt.html for Experiment 2.

Experiment 2 asks how three contextual factors of a *synthetic* dataset
(user role x eval condition x scenario) shape the emotional content of a model's
reply, and which of the model's own verbalized inferences are causally
responsible. The page has four stacked, interactive sections:

  1. Setup        -- the design, models, judge, and prompt composition
  2. Data         -- pick model/role/scenario/condition -> see the latent
                     inference, the reply, and the judge's emotion scores
  3. Analysis     -- factor-effect tables + plots + verbalization->emotion
  4. Counterfactual -- flip one verbalized field, regenerate, re-judge; browse
                     before/after examples, effect table, and the faithfulness result

Everything (plots as base64, all examples as inline JSON) is embedded, so the
file opens with no server, files, or network.

    python -m src.make_report_synt      ->  analysis_synt.html
"""
from __future__ import annotations

import base64
import io
import json
import re
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
from . import decoupled_compare as dc
from .system_prompts import SIMPLE_SYSTEM_PROMPT

DVS = ["warmth", "formality", "advice_density"]
DV_COLOR = {"warmth": "#e67e22", "formality": "#2980b9", "advice_density": "#16a085"}
MODEL_COLOR = {"llama-3.3-70b": "#8e44ad", "llama-3.1-8b": "#2980b9",
               "qwen3-235b": "#16a085", "qwen3-30b": "#e67e22"}
SIMPLE_TEXT_FIELDS = ["about_user", "about_context", "about_stakes", "register_selected"]
CONDS = ["deployment", "neutral_sys", "eval_cue"]
GEN_MODELS = ["llama-3.3-70b", "llama-3.1-8b", "qwen3-235b", "qwen3-30b"]
FIELD_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+?)\s*$")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fig_to_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#fff")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img(uri, cap=""):
    c = f"<figcaption>{cap}</figcaption>" if cap else ""
    return f"<figure><img src='{uri}'/>{c}</figure>"


def bullets(items):
    return "<ul class='res'>" + "".join(f"<li>{b}</li>" for b in items) + "</ul>"


def table(headers, rows):
    h = "".join(f"<th>{esc(c)}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='tbl'><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def parse_inf_body(text):
    """Parse a bare inference body ('field: value' lines) into an ordered dict."""
    out = {}
    for line in str(text).splitlines():
        m = FIELD_RE.match(line.strip())
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load_stage1():
    rows = []
    for jf in sorted(config.EXP2_CUES_DIR.glob("*/*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df[df["judge"].notna()].reset_index(drop=True)
    for dv in DVS:
        df[dv] = pd.to_numeric(df["judge"].apply(lambda j: j.get(dv)), errors="coerce")
    df["primary_emotion"] = df["judge"].apply(lambda j: j.get("primary_emotion"))
    return df.dropna(subset=DVS)


def load_stage3():
    recs = []
    for jf in sorted((config.EXP2_CUES_DIR / "stage3").glob("*/*/*.json")):
        recs.append(json.loads(jf.read_text(encoding="utf-8")))
    return recs


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _grouped_bars(ax, cats, series, ylabel, title, rotate=20):
    x = np.arange(len(cats))
    w = 0.26
    for i, dv in enumerate(DVS):
        ax.bar(x + (i - 1) * w, [series[dv][c] for c in cats], w,
               label=dv, color=DV_COLOR[dv])
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=rotate, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)


def plot_by_factor(df, factor, title, order_by="warmth", rotate=25):
    g = df.groupby(factor)[DVS].mean()
    cats = list(g.sort_values(order_by, ascending=False).index)
    series = {dv: g[dv].to_dict() for dv in DVS}
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(cats)), 4.6))
    ax.set_ylim(0, 10)
    _grouped_bars(ax, cats, series, "judged score (0–10)", title, rotate)
    return fig_to_uri(fig)


def plot_factor_eta(fe):
    factors = ["scenario", "role", "role_axis", "eval_condition", "model_name"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(factors))
    w = 0.26
    for i, dv in enumerate(DVS):
        vals = [fe.get(f"{f}__{dv}", {}).get("eta2", 0) or 0 for f in factors]
        ax.bar(x + (i - 1) * w, vals, w, label=dv, color=DV_COLOR[dv])
    ax.set_xticks(x)
    ax.set_xticklabels(factors, rotation=15, ha="right")
    ax.set_ylabel("η²  (variance in the DV explained)")
    ax.set_title("How much each factor moves the response's emotional content")
    ax.legend(fontsize=9)
    return fig_to_uri(fig)


RISK_ORDER = ["low", "mid", "high"]
RISK_COLOR = {"low": "#27ae60", "mid": "#e67e22", "high": "#c0392b"}


def plot_risk_lines(df70):
    """Mean of each DV across low/mid/high risk (dose-response)."""
    ranks = [r for r in RISK_ORDER if r in set(df70["x_rank"])]
    g = df70.groupby("x_rank")[DVS].mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(ranks))
    for dv in DVS:
        ax.plot(x, [g.loc[r, dv] for r in ranks], "-o", lw=2.2, color=DV_COLOR[dv], label=dv)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} risk" for r in ranks])
    ax.set_ylim(0, 10)
    ax.set_ylabel("judged score (0–10)")
    ax.set_title("Dose-response: emotional content vs risk level (llama-3.3-70b)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    return fig_to_uri(fig)


def plot_risk_advice_by_domain(df70):
    """Advice density by domain x risk level (where does risk matter?)."""
    ranks = [r for r in RISK_ORDER if r in set(df70["x_rank"])]
    doms = list(df70.groupby("domain")["advice_density"].mean().sort_values(ascending=False).index)
    g = df70.groupby(["domain", "x_rank"])["advice_density"].mean()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(doms))
    w = 0.8 / len(ranks)
    for i, r in enumerate(ranks):
        vals = [g.get((d, r), np.nan) for d in doms]
        ax.bar(x + (i - (len(ranks) - 1) / 2) * w, vals, w, label=f"{r} risk",
               color=RISK_COLOR[r])
    ax.set_xticks(x)
    ax.set_xticklabels(doms, rotation=12, ha="right")
    ax.set_ylim(0, 10)
    ax.set_ylabel("advice density (0–10)")
    ax.set_title("Advice density by domain and risk level (llama-3.3-70b)")
    ax.legend(fontsize=9)
    return fig_to_uri(fig)


def plot_risk_scenario_delta(adv_deltas):
    """Diverging bars: per-scenario change in advice density from low→high risk."""
    items = sorted(adv_deltas.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8.0, 0.42 * len(items) + 1.2))
    ax.barh(labels, vals, color=["#2980b9" if v < 0 else "#c0392b" for v in vals])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Δ advice density, low → high risk (points on 0–10)")
    ax.set_title("Risk raises advice in danger scenarios, lowers it in grief/achievement")
    return fig_to_uri(fig)


def models_with_xsweep(df):
    return [m for m in GEN_MODELS
            if {"low", "high"} <= set(df[df["model_name"] == m]["x_rank"])]


def plot_risk_lines_multi(df, models):
    """Advice density vs risk level, one line per model (dose-response)."""
    ranks = ["low", "mid", "high"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(len(ranks))
    for m in models:
        g = df[df["model_name"] == m].groupby("x_rank")["advice_density"].mean()
        ax.plot(x, [g.get(r, np.nan) for r in ranks], "-o", lw=2.2,
                label=m, color=MODEL_COLOR.get(m))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} risk" for r in ranks])
    ax.set_ylim(0, 10)
    ax.set_ylabel("advice density (0–10)")
    ax.set_title("Advice density vs risk level, per model")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    return fig_to_uri(fig)


def _wdelta(sub):
    """Within-scenario low→high mean delta per DV for one model's frame."""
    out = {}
    for dv in DVS:
        pv = sub.pivot_table(index="scenario", columns="x_rank", values=dv, aggfunc="mean")
        if "low" in pv and "high" in pv:
            diff = (pv["high"] - pv["low"]).dropna()
            out[dv] = {"mean": float(diff.mean()), "n_pos": int((diff > 0).sum()),
                       "n": int(diff.notna().sum())}
    return out


def risk_analysis(df, models):
    """Pooled per-scenario advice deltas + per-(scenario,model) deltas + per-model DV deltas."""
    scns = sorted(set(df["scenario"]))
    # per-model scenario->Δadvice(high-low)
    permodel = {}
    for m in models:
        pv = df[df["model_name"] == m].pivot_table(
            index="scenario", columns="x_rank", values="advice_density", aggfunc="mean")
        permodel[m] = {s: (pv.loc[s].get("high", np.nan) - pv.loc[s].get("low", np.nan))
                       for s in pv.index}
    # pooled mean Δadvice per scenario (for the diverging plot)
    adv_deltas = {}
    scn_rows = []
    for s in scns:
        vals = [permodel[m].get(s, np.nan) for m in models]
        mean = np.nanmean(vals) if any(v == v for v in vals) else np.nan
        if mean == mean:
            adv_deltas[s] = round(float(mean), 2)
        scn_rows.append([s] + [f"{permodel[m].get(s, float('nan')):+.1f}" for m in models]
                        + [f"{mean:+.1f}" if mean == mean else "–"])
    scn_rows.sort(key=lambda r: float(r[-1].replace("–", "nan")) if r[-1] != "–" else -99,
                  reverse=True)
    # per-model within-scenario DV deltas
    dv_rows = []
    wd_by_model = {}
    for m in models:
        wd = _wdelta(df[df["model_name"] == m])
        wd_by_model[m] = wd
        dv_rows.append([m] + [f"{wd.get(dv, {}).get('mean', float('nan')):+.2f}" for dv in DVS])
    return scn_rows, adv_deltas, dv_rows, wd_by_model


def plot_cf_effect(by_field):
    fields = list(by_field)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(fields))
    w = 0.26
    for i, dv in enumerate(DVS):
        vals = [by_field[f][dv]["mean_delta"] for f in fields]
        ax.bar(x + (i - 1) * w, vals, w, label=dv, color=DV_COLOR[dv])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=12, ha="right")
    ax.set_ylabel("Δ judged score after flipping the field to 'calm'")
    ax.set_title("Causal effect of calming ONE verbalized field (pooled, 4 models)")
    ax.legend(fontsize=9)
    return fig_to_uri(fig)


def cf_stats(stage3):
    """Per-model and pooled counterfactual stats: {scope: {by_field, faith, n}}."""
    def agg(recs):
        d = {}
        allm = []
        for r in recs:
            f = r["target_field"]
            a = d.setdefault(f, {dv: {"deltas": [], "matches": []} for dv in DVS})
            for dv in DVS:
                if dv in r.get("delta", {}):
                    a[dv]["deltas"].append(r["delta"][dv])
                if dv in r.get("match", {}):
                    a[dv]["matches"].append(1 if r["match"][dv] else 0)
                    allm.append(1 if r["match"][dv] else 0)
        res = {}
        for f, dd in d.items():
            res[f] = {dv: {"mean_delta": (sum(dd[dv]["deltas"]) / len(dd[dv]["deltas"])
                                          if dd[dv]["deltas"] else 0.0),
                           "faith": (sum(dd[dv]["matches"]) / len(dd[dv]["matches"])
                                     if dd[dv]["matches"] else None),
                           "n": len(dd[dv]["deltas"])} for dv in DVS}
        return res, (sum(allm) / len(allm) if allm else None)

    out = {}
    by_model = {}
    for r in stage3:
        by_model.setdefault(r["model_name"], []).append(r)
    for m, recs in by_model.items():
        bf, fa = agg(recs)
        out[m] = {"by_field": bf, "faith": fa, "n": len(recs)}
    bf, fa = agg(stage3)
    out["ALL"] = {"by_field": bf, "faith": fa, "n": len(stage3)}
    return out


# --------------------------------------------------------------------------- #
# coupling effect (coupled vs decoupled) plots + examples
# --------------------------------------------------------------------------- #
def plot_coupling_overall(ov):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(DVS))
    w = 0.38
    coupled = [ov[dv]["coupled_mean"] for dv in DVS]
    bare = [ov[dv]["bare_mean"] for dv in DVS]
    ax.bar(x - w / 2, coupled, w, label="coupled (self-modeling present)", color="#8e44ad")
    ax.bar(x + w / 2, bare, w, label="bare response (no self-modeling)", color="#95a5a6")
    for i, dv in enumerate(DVS):
        ax.annotate(f"Δ{ov[dv]['delta']:+.2f}", (i, max(coupled[i], bare[i]) + 0.2),
                    ha="center", fontsize=9, color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(DVS)
    ax.set_ylim(0, 10)
    ax.set_ylabel("mean judged score (0–10)")
    ax.set_title("Response emotional content: with vs without self-modeling")
    ax.legend(fontsize=9)
    return fig_to_uri(fig)


def plot_coupling_by_model(df):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(GEN_MODELS))
    w = 0.26
    for i, dv in enumerate(DVS):
        vals = [df[df.model_name == m][f"d_{dv}"].mean() for m in GEN_MODELS]
        ax.bar(x + (i - 1) * w, vals, w, label=dv, color=DV_COLOR[dv])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(GEN_MODELS, rotation=12, ha="right")
    ax.set_ylabel("Δ judged score  (coupled − bare)")
    ax.set_title("Effect of self-modeling on the reply, per model")
    ax.legend(fontsize=9)
    return fig_to_uri(fig)


def plot_coupling_scenario(df, dv, order, title_suffix, xlim):
    g = df.groupby("scenario")[f"d_{dv}"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(8.0, 0.42 * len(order) + 1.2))
    vals = g.values
    ax.barh(order, vals, color=["#2980b9" if (v < 0) else "#c0392b" for v in vals])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(-xlim, xlim)  # fixed per metric so models are visually comparable
    ax.set_xlabel(f"Δ {dv} (coupled − bare)")
    ax.set_title(f"Where self-modeling changes {dv} — {title_suffix}")
    return fig_to_uri(fig)


def _judge_of(r, pre):
    return {"warmth": r[f"{pre}_warmth"], "formality": r[f"{pre}_formality"],
            "advice_density": r[f"{pre}_advice_density"], "primary_emotion": r[f"{pre}_emotion"]}


def build_coupling_examples(df):
    """Per model: curated diverging / different-wording / similar / diverging-inference triples."""
    out = {}
    for m in GEN_MODELS:
        sub = df[df.model_name == m].copy()
        sub["jdiv"] = sub[[f"d_{dv}" for dv in DVS]].abs().sum(axis=1)
        picks = (
            [("judged divergence", r) for _, r in sub.sort_values("jdiv", ascending=False).head(5).iterrows()]
            + [("different wording", r) for _, r in sub.sort_values("resp_cosine").head(3).iterrows()]
            + [("near-identical", r) for _, r in sub.sort_values("resp_cosine", ascending=False).head(3).iterrows()]
            + [("inference shift", r) for _, r in sub.sort_values("inf_cosine").head(3).iterrows()]
        )
        items, seen = [], set()
        for kind, r in picks:
            if r["prompt_id"] in seen:
                continue
            seen.add(r["prompt_id"])
            items.append({
                "kind": kind, "role": r["role"], "scenario": r["scenario"],
                "eval_condition": r["eval_condition"], "x_rank": r["x_rank"],
                "user": r["user"],
                "coupled": {"inference": r["c_inf"], "response": (r["c_resp"] or "")[:1000],
                            "judge": _judge_of(r, "c")},
                "bare": {"response": (r["r_resp"] or "")[:1000], "judge": _judge_of(r, "r")},
                "inference_only": {"inference": r["d_inf"]},
                "delta": {dv: round(float(r[f"d_{dv}"]), 1) for dv in DVS},
                "resp_cosine": round(float(r["resp_cosine"]), 2),
                "inf_cosine": round(float(r["inf_cosine"]), 2),
            })
        out[m] = items
    return out


def _complete_inf(inf):
    return all(str((inf or {}).get(f, "")).strip() for f in SIMPLE_TEXT_FIELDS)


def build_inference_examples(df):
    """Per model: curated coupled-vs-inference-only inference pairs to browse.

    Restricted to prompts where BOTH inferences have all four fields populated, so
    a low similarity reflects a genuine semantic shift rather than a missing field.
    """
    out = {}
    for m in GEN_MODELS:
        sub = df[df.model_name == m].copy()
        sub = sub[sub.apply(lambda r: _complete_inf(r["c_inf"]) and _complete_inf(r["d_inf"]), axis=1)]
        # rank by SEMANTIC change (embedding cosine) since we care about meaning shifts
        picks = (
            [("biggest meaning shift", r) for _, r in sub.sort_values("emb_cosine").head(5).iterrows()]
            + [("semantically stable", r) for _, r in sub.sort_values("emb_cosine", ascending=False).head(3).iterrows()]
            + [("register meaning shift", r) for _, r in sub.sort_values("embcos_register_selected").head(3).iterrows()]
        )
        items, seen = [], set()
        for kind, r in picks:
            if r["prompt_id"] in seen:
                continue
            seen.add(r["prompt_id"])
            fc = {f: round(float(r[f"infcos_{f}"]), 2) for f in SIMPLE_TEXT_FIELDS}
            fe = {f: round(float(r[f"embcos_{f}"]), 2) for f in SIMPLE_TEXT_FIELDS}
            items.append({
                "kind": kind, "role": r["role"], "scenario": r["scenario"],
                "eval_condition": r["eval_condition"], "x_rank": r["x_rank"], "user": r["user"],
                "coupled_inf": r["c_inf"], "infonly_inf": r["d_inf"],
                "inf_cosine": round(float(r["inf_cosine"]), 2),
                "emb_cosine": round(float(r["emb_cosine"]), 2),
                "field_cos": fc, "field_emb": fe,
                "changed": [min(fe, key=fe.get)],  # least-similar field by MEANING, highlighted
            })
        out[m] = items
    return out


def plot_field_cosine(df):
    """Lexical (TF-IDF) vs semantic (embedding) cosine per field + overall."""
    labels = SIMPLE_TEXT_FIELDS + ["ALL fields"]
    lex = [df[f"infcos_{f}"].mean() for f in SIMPLE_TEXT_FIELDS] + [df["inf_cosine"].mean()]
    sem = [df[f"embcos_{f}"].mean() for f in SIMPLE_TEXT_FIELDS] + [df["emb_cosine"].mean()]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.bar(x - w / 2, lex, w, label="lexical (TF-IDF / wording)", color="#95a5a6")
    ax.bar(x + w / 2, sem, w, label="semantic (embedding / meaning)", color="#8e44ad")
    for i in range(len(labels)):
        ax.annotate(f"{sem[i]:.2f}", (i + w / 2, sem[i] + 0.01), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean cosine (coupled vs inference-only)")
    ax.set_title("Inference stability: wording (lexical) vs meaning (semantic)")
    ax.legend(fontsize=9)
    return fig_to_uri(fig)


def plot_field_cosine_by_model(df):
    """Semantic (embedding) inference cosine per field, by model."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(SIMPLE_TEXT_FIELDS))
    w = 0.2
    for i, m in enumerate(GEN_MODELS):
        sub = df[df.model_name == m]
        vals = [sub[f"embcos_{f}"].mean() for f in SIMPLE_TEXT_FIELDS]
        ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=MODEL_COLOR.get(m))
    ax.set_xticks(x)
    ax.set_xticklabels(SIMPLE_TEXT_FIELDS, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("semantic cosine (embedding)")
    ax.set_title("Semantic inference stability by field and model")
    ax.legend(fontsize=8)
    return fig_to_uri(fig)


# --------------------------------------------------------------------------- #
# interactive payloads
# --------------------------------------------------------------------------- #
XRANK_ORDER = ["low", "mid", "high"]


def build_data_payload(df):
    """Per model: {role|cond|scenario|x_rank -> record} using run-0 datapoints."""
    out = {m: {} for m in GEN_MODELS}
    roles, scenarios, xranks = set(), set(), set()
    sub = df[df["run"] == 0]
    for _, r in sub.iterrows():
        m = r["model_name"]
        if m not in out:
            continue
        xr = r.get("x_rank")
        key = f"{r['role']}|{r['eval_condition']}|{r['scenario']}|{xr}"
        roles.add(r["role"])
        scenarios.add(r["scenario"])
        xranks.add(xr)
        out[m][key] = {
            "role": r["role"], "eval_condition": r["eval_condition"],
            "scenario": r["scenario"], "x_rank": xr, "x_value": r.get("x_value"),
            "unit": r.get("unit"), "target_emotions": r.get("target_emotions"),
            "user": r["user"],
            "inference": r["inference"],
            "response": (r["response"] or "")[:1100],
            "judge": {k: r["judge"].get(k) for k in DVS + ["primary_emotion"]},
        }
    xr_sorted = [x for x in XRANK_ORDER if x in xranks] + sorted(xranks - set(XRANK_ORDER))
    return out, sorted(roles), sorted(scenarios), xr_sorted


def build_cf_payload(stage3, df):
    """Curated before/after counterfactual examples per model (well-spread fields)."""
    # baseline inference lookup: (model, prompt_id, run) -> stage1 record
    s1 = {}
    for _, r in df.iterrows():
        s1[(r["model_name"], r["prompt_id"], r["run"])] = r
    by_model = {}
    for rec in stage3:
        by_model.setdefault(rec["model_name"], []).append(rec)

    out = {}
    for m, recs in by_model.items():
        # rank by total absolute judged change so the examples are illustrative
        def mag(rc):
            return sum(abs(rc["delta"].get(dv, 0)) for dv in DVS)
        # take a spread across the four target fields
        per_field = {}
        for rc in sorted(recs, key=mag, reverse=True):
            per_field.setdefault(rc["target_field"], [])
            if len(per_field[rc["target_field"]]) < 9:
                per_field[rc["target_field"]].append(rc)
        picked = [rc for lst in per_field.values() for rc in lst]
        picked.sort(key=lambda rc: (rc["target_field"], -mag(rc)))

        items = []
        for rc in picked:
            base = s1.get((m, rc["prompt_id"], rc.get("run", 0)))
            nat_inf = base["inference"] if base is not None else {}
            items.append({
                "role": rc.get("role"), "scenario": rc.get("scenario"),
                "eval_condition": rc.get("eval_condition"),
                "edit": rc["edit"], "changed": [rc["target_field"]],
                "user": (base["user"] if base is not None else ""),
                "natural": {
                    "inference": nat_inf,
                    "response": (rc["baseline_response"] or "")[:1100],
                    "judge": {k: rc["baseline_judge"].get(k) for k in DVS + ["primary_emotion"]},
                },
                "counterfactual": {
                    "inference": parse_inf_body(rc["edited_inference"]),
                    "response": (rc["cf_response"] or "")[:1100],
                    "judge": {k: rc["cf_judge"].get(k) for k in DVS + ["primary_emotion"]},
                },
                "delta": {dv: round(rc["delta"].get(dv, 0), 1) for dv in DVS},
            })
        out[m] = items
    return out


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def fmt_eta(fe, factor):
    cells = []
    for dv in DVS:
        d = fe.get(f"{factor}__{dv}", {})
        eta = d.get("eta2")
        p = d.get("p", 1) or 1
        if eta is None:
            cells.append("–")
            continue
        s = f"{eta:.3f}"
        if eta >= 0.10:
            s = f"<b>{s}</b>"
        if p < 0.05:
            s += "*"
        cells.append(s)
    return cells


def build():
    df = load_stage1()
    stage3 = load_stage3()
    s2 = json.loads((config.EXP2_CUES_DIR / "stage2_summary.json").read_text(encoding="utf-8"))
    s3 = json.loads((config.EXP2_CUES_DIR / "stage3_summary.json").read_text(encoding="utf-8"))
    fe = s2["factor_effects"]
    v2e = s2["verbalization_to_emotion"]

    # ---- factor-effect table ----
    eta_rows = [[f] + fmt_eta(fe, f) for f in
                ["scenario", "role", "role_axis", "x_rank", "eval_condition", "model_name"]]
    eta_tbl = table(["factor ↓ / DV →", "warmth", "formality", "advice_density"], eta_rows)

    # ---- plots ----
    plots_factor = (
        img(plot_factor_eta(fe),
            "η² = share of variance in each judged DV explained by a factor. "
            "<b>Scenario</b> dominates advice density (η²≈0.63); <b>user role</b> drives "
            "warmth/formality; <b>eval condition</b> is ≈0 on every DV — the 'you are being "
            "evaluated' frame did not change the emotional content of the replies.")
        + img(plot_by_factor(df, "role", "Warmth / formality / advice density by user role",
                             order_by="warmth"),
              "Vulnerable roles (emotional_crisis, elderly_confused, cognitive_impairment) draw the "
              "<b>warmest, least formal</b> replies; authority_overconfident and adversarial_reframe "
              "draw the coldest, most formal. Advice density is comparatively flat across roles.")
        + img(plot_by_factor(df, "scenario", "Emotional content by scenario", order_by="advice_density"),
              "The scenario is the strongest lever on how prescriptive the reply is: high-risk "
              "health/safety scenarios pull advice density up; grief/achievement scenarios pull "
              "warmth up and advice down.")
        + img(plot_by_factor(df, "eval_condition", "Emotional content by eval condition",
                             order_by="warmth", rotate=0),
              "The three evaluation framings are essentially indistinguishable — a clean null that "
              "holds across all four models.")
        + img(plot_by_factor(df, "model_name", "Emotional content by model", order_by="warmth", rotate=0),
              "Models differ modestly in baseline style (η² 0.04–0.09): the larger models are a touch "
              "more formal and advice-dense.")
    )

    # ---- verbalization -> emotion ----
    v2e_html = ""
    for dv in DVS:
        d = v2e[dv]
        kw_rows = [[esc(r["term"]), f"{r.get(dv+'_with', r.get('warmth_with','')):.1f}"
                    if isinstance(r.get(dv+'_with'), (int, float)) else "–",
                    f"{r['cohen_d']:+.2f}",
                    (f"<b style='color:#c0392b'>{r['q_FDR']:.2g}*</b>"
                     if r.get("q_FDR", 1) < 0.05 else f"{r.get('q_FDR',1):.2g}"),
                    int(r["docs"])]
                   for r in d.get("keywords_top", [])[:8]]
        chips_up = " ".join(f"<span class='chip up'>{esc(t)}</span>" for t in d["ridge_top_terms"])
        chips_dn = " ".join(f"<span class='chip dn'>{esc(t)}</span>" for t in d["ridge_bottom_terms"])
        v2e_html += (
            f"<h3>{dv} &nbsp;·&nbsp; TF-IDF Ridge CV R² = <b>{d['ridge_cv_r2']}</b></h3>"
            f"<p class='mut'>words in the model's verbalized inference that <b>raise</b> {dv}: {chips_up}<br>"
            f"that <b>lower</b> {dv}: {chips_dn}</p>"
            + table([f"keyword (in inference)", f"{dv} | with", "Cohen d", "q (FDR)", "docs"], kw_rows)
        )

    # ---- risk (x-value) -> output (per model, for any with low+high) ----
    df_xs = df[df["x_rank"].isin(RISK_ORDER)].copy()
    xs_models = models_with_xsweep(df)
    risk_html = ""
    if xs_models:
        scn_rows, adv_deltas, dv_rows, wd_by_model = risk_analysis(df_xs, xs_models)
        rises = sorted((d, s) for s, d in adv_deltas.items() if d > 0)[-4:][::-1]
        falls = sorted((d, s) for s, d in adv_deltas.items() if d < 0)[:3]
        scn_tbl = table(["scenario"] + xs_models + ["pooled Δ"], scn_rows)
        dv_tbl = table(["model", "Δ warmth", "Δ formality", "Δ advice"], dv_rows)
        coverage = ("all four models" if len(xs_models) == len(GEN_MODELS)
                    else ", ".join(xs_models))
        # cross-model consistency of the advice dose-response
        adv_means = [wd_by_model[m].get("advice_density", {}).get("mean", float("nan"))
                     for m in xs_models]
        risk_bul = [
            "<b>In aggregate, risk barely moves the output.</b> Pooled advice density is nearly flat "
            "across low→mid→high and warmth/formality are flat too — because the per-scenario effects "
            "have opposite signs and cancel.",
            "<b>But risk strongly shifts advice <i>within</i> danger scenarios.</b> Higher x → more "
            "prescriptive where the number encodes harm: "
            + ", ".join(f"<code>{esc(s)}</code> ({d:+.1f})" for d, s in rises) + ".",
            "<b>And it <i>reverses</i> in grief/achievement.</b> Higher x → <i>less</i> advice (the "
            "model leans into comfort/celebration instead): "
            + ", ".join(f"<code>{esc(s)}</code> ({d:+.1f})" for d, s in falls) + ".",
            f"<b>The pattern is consistent across models.</b> Mean within-scenario advice change "
            f"(low→high) is " + ", ".join(f"{m} {v:+.2f}" for m, v in zip(xs_models, adv_means))
            + " — all small-positive, i.e. every model nudges advice up with risk while tone stays flat.",
        ]
        risk_html = (
            "<h2 style='margin-top:24px'>Risk level (x-value) → output</h2>"
            "<p class='mut'>Each scenario carries a numeric risk parameter (tylenol dose in mg, hours "
            "without sleep, speed over the limit, % of savings invested, …), now run at low / mid / "
            f"high for <b>{coverage}</b>. The headline: the risk effect is real but "
            "<b>scenario-specific</b>, so it nearly vanishes when averaged.</p>"
            + img(plot_risk_scenario_delta(adv_deltas),
                  "Change in advice density from low→high risk, pooled across models, per scenario. "
                  "Red = <i>more</i> advice as danger rises (overdose, speeding, drink-driving, "
                  "leverage); blue = <i>less</i> (a longer-missing dog, more students passed), the "
                  "model leaning into emotional support instead. The opposite signs cancel in the mean.")
            + img(plot_risk_lines_multi(df_xs, xs_models),
                  "Advice density vs risk, one line per model. All four are nearly flat in aggregate — "
                  "the per-scenario structure above is hidden once scenarios are pooled.")
            + img(plot_risk_advice_by_domain(df_xs),
                  "By domain (pooled): health, safety/legal and financial advice rise with risk; grief "
                  "and achievement are flat-to-declining.")
            + "<h3>Δ advice density (low→high) per scenario × model</h3>" + scn_tbl
            + "<h3>Within-scenario low→high change per model (all DVs)</h3>" + dv_tbl
            + "<h3>Results</h3>" + bullets(risk_bul)
        )
        (config.EXP2_CUES_DIR / "risk_summary.json").write_text(
            json.dumps({"models": xs_models,
                        "within_scenario_low_to_high": wd_by_model,
                        "pooled_advice_delta_by_scenario": adv_deltas}, indent=2), encoding="utf-8")

    # ---- counterfactual effect tables (per model + pooled) ----
    cfs = cf_stats(stage3)
    bf = cfs["ALL"]["by_field"]
    cf_rows = []
    for f in bf:
        for dv in DVS:
            cell = bf[f][dv]
            faith = cell["faith"]
            cf_rows.append([esc(f), dv, f"{cell['mean_delta']:+.2f}",
                            (f"{faith:.0%}" if faith is not None else "—"), cell["n"]])
    cf_tbl = table(["field calmed", "DV", "mean Δ", "faithfulness", "n"], cf_rows)

    def _mean_dv(byf, dv):
        vals = [byf[f][dv]["mean_delta"] for f in byf]
        return sum(vals) / len(vals) if vals else 0.0
    cf_model_rows = []
    for m in GEN_MODELS:
        if m not in cfs:
            continue
        c = cfs[m]
        cf_model_rows.append([
            m, c["n"], f"{c['faith']:.0%}" if c["faith"] is not None else "—",
            f"{_mean_dv(c['by_field'],'warmth'):+.2f}",
            f"{_mean_dv(c['by_field'],'formality'):+.2f}",
            f"{_mean_dv(c['by_field'],'advice_density'):+.2f}"])
    cf_model_tbl = table(
        ["model", "cf n", "faithfulness", "mean Δ warmth", "mean Δ formality", "mean Δ advice"],
        cf_model_rows)

    # ---- results bullets ----
    def eta(f, dv):
        return fe.get(f"{f}__{dv}", {}).get("eta2", 0)
    s2_bullets = [
        f"<b>Scenario drives prescriptiveness.</b> The risk scenario alone explains "
        f"<b>η²={eta('scenario','advice_density'):.2f}</b> of advice density — far more than role "
        f"({eta('role','advice_density'):.3f}) or eval condition ({eta('eval_condition','advice_density'):.3f}).",
        f"<b>Role drives tone.</b> User role explains <b>η²={eta('role','warmth'):.2f}</b> of warmth "
        f"and <b>{eta('role','formality'):.2f}</b> of formality: the model is warmest and least formal "
        f"to vulnerable users, coldest to overconfident/adversarial ones.",
        f"<b>Evaluation framing changes nothing.</b> deployment vs neutral-system vs eval-cue move "
        f"every DV by η²≈0 (warmth {eta('eval_condition','warmth'):.3f}, advice "
        f"{eta('eval_condition','advice_density'):.3f}) — a robust null across all four models.",
        f"<b>The verbalized inference predicts the emotion.</b> A cross-validated TF-IDF ridge over "
        f"the model's own free-text cues predicts each judged DV at R² ≈ "
        f"<b>{v2e['warmth']['ridge_cv_r2']}–{v2e['advice_density']['ridge_cv_r2']}</b> — empathy/grief "
        f"words track warmth, risk/urgency words track advice.",
    ]
    big = sorted(((bf[f][dv]["mean_delta"], f, dv) for f in bf for dv in DVS))[:3]
    cf_model_faith = {m: cfs[m]["faith"] for m in GEN_MODELS if m in cfs and cfs[m]["faith"] is not None}
    s3_bullets = [
        "<b>The inference channel is causal in every model.</b> Calming a single verbalized field and "
        "regenerating moves the judged emotion across all "
        f"{len([m for m in GEN_MODELS if m in cfs])} models — the reply follows the model's own stated "
        "appraisal, not just the prompt.",
        "<b>Calming uniformly dampens engagement.</b> Pooled, the biggest levers are "
        + ", ".join(f"<code>{esc(f)}</code>→{dv} ({d:+.2f})" for d, f, dv in big)
        + " — flipping the appraisal toward 'low-stakes / nothing wrong' makes the reply less warm "
        "and less prescriptive.",
        "<b>Faithfulness is modest and similar across models</b> ("
        + ", ".join(f"{m} {v:.0%}" for m, v in cf_model_faith.items())
        + "): pre-registered <i>signs</i> often miss because a 'calm/easygoing' register reads as more "
        "<i>detached</i> (warmth ↓), not warmer as first hypothesized. The reliable finding is the "
        "consistent dampening, not the guessed sign.",
    ]

    # ---- coupling effect: coupled (self-modeling) vs decoupled ----
    cmp_df = dc.load_joined()
    ov = dc.paired_summary(cmp_df)
    bmc = dc.by_factor(cmp_df, "model_name")
    cpl_over_tbl = table(
        ["DV", "coupled", "bare", "Δ (C−R)", "Cohen dz", "p"],
        [[dv, f"{ov[dv]['coupled_mean']:.2f}", f"{ov[dv]['bare_mean']:.2f}",
          f"<b>{ov[dv]['delta']:+.2f}</b>", f"{ov[dv]['cohen_dz']}",
          (f"<b style='color:#c0392b'>{ov[dv]['p']:.1g}*</b>" if ov[dv]['p'] < 0.05
           else f"{ov[dv]['p']:.2g}")] for dv in DVS])
    cpl_model_tbl = table(
        ["model", "Δ warmth", "Δ formality", "Δ advice", "emotion agree", "resp cos", "inf cos"],
        [[m, f"{bmc.loc[m,'d_warmth']:+.2f}", f"{bmc.loc[m,'d_formality']:+.2f}",
          f"{bmc.loc[m,'d_advice_density']:+.2f}", f"{bmc.loc[m,'emotion_agree']:.0%}",
          f"{bmc.loc[m,'resp_cos']:.2f}", f"{bmc.loc[m,'inf_cos']:.2f}"] for m in GEN_MODELS])
    warm_models = [m for m in GEN_MODELS if bmc.loc[m, "d_warmth"] > 0.3]
    adv_cut = [m for m in GEN_MODELS if bmc.loc[m, "d_advice_density"] < -0.3]
    cpl_bullets = [
        f"<b>Self-modeling makes replies warmer and slightly less prescriptive — on average.</b> "
        f"Across {ov['n']:,} matched prompts, warmth {ov['warmth']['delta']:+.2f}, advice density "
        f"{ov['advice_density']['delta']:+.2f}, formality {ov['formality']['delta']:+.2f} "
        f"(all significant, but small: |Cohen dz| ≤ 0.22).",
        "<b>But the effect is model-specific.</b> The Llamas warm up markedly ("
        + ", ".join(f"{esc(m)} {bmc.loc[m,'d_warmth']:+.2f}" for m in warm_models)
        + ") while the Qwens instead cut advice ("
        + ", ".join(f"{esc(m)} {bmc.loc[m,'d_advice_density']:+.2f}" for m in adv_cut)
        + ") — so 'add a self-model' is not a uniform behavioural lever.",
        f"<b>The emotional label usually survives</b> ({ov['emotion_agreement']:.0%} of replies keep "
        f"the same primary emotion), <b>but the wording changes a lot</b> — coupled vs bare response "
        f"TF-IDF cosine is only <b>{ov['resp_cosine_mean']:.2f}</b>. Self-modeling reshapes <i>how</i> "
        f"the model says things more than the emotional dial it lands on.",
        f"<b>The verbalized inference is reworded but not re-thought</b> — the same cues elicited "
        f"alone vs alongside a reply share only <b>{ov['inf_cosine_mean']:.2f}</b> lexical cosine but "
        f"a high semantic cosine (see ⑥): the wording changes, the meaning largely doesn't.",
    ]
    coupling_html = (
        "<p class='mut'>We generated each prompt three ways: <b>coupled</b> (inference + reply in one "
        "turn, Stage 1), <b>inference-only</b> (the cues alone), and <b>bare response</b> (the reply "
        "with NO self-modeling prompt). Joining them 1:1 lets us isolate what the self-modeling step "
        "does to the reply, and whether the inference itself is stable.</p>"
        "<h3>Responses — with vs without self-modeling</h3>"
        + img(plot_coupling_overall(ov),
              "Mean judged content of the reply. Self-modeling (purple) lands slightly warmer, less "
              "formal and less advice-heavy than a bare reply (grey) — small but consistent shifts.")
        + cpl_over_tbl
        + img(plot_coupling_by_model(cmp_df),
              "Per model, the Δ (coupled − bare) diverges: Llamas gain warmth, Qwens lose advice "
              "density. The aggregate above averages over these opposite tendencies.")
        + cpl_model_tbl
        + "<h3>Where self-modeling changes the reply, by scenario</h3>"
        + "<p class='mut'>Δ (coupled − bare) per scenario. Pick a <b>metric</b> and a <b>model</b>; the "
        "scenario order and x-axis are fixed within each metric so models are directly comparable. "
        "Red = self-modeling raises it, blue = lowers it.</p>"
        + "<div class='seg dv' id='cplscen-dv'></div>"
        + "<div class='seg mdl' id='cplscen-btns'></div>"
        + "<figure><img id='cplscen-img'/><figcaption>Effects are scenario-specific and vary by "
        "metric and model — e.g. advice density is suppressed most in high-stakes health/finance "
        "scenarios, while warmth shifts elsewhere. Switch metric/model above.</figcaption></figure>"
        + "<h3>Results</h3>" + bullets(cpl_bullets)
    )
    coupling_examples = build_coupling_examples(cmp_df)
    # scenario figure: one image per (metric × model); scenario order & x-limit are
    # fixed within a metric so models are directly comparable.
    cpl_scen_imgs = {}
    for dv in DVS:
        order = list(cmp_df.groupby("scenario")[f"d_{dv}"].mean().sort_values().index)
        vals = list(cmp_df.groupby("scenario")[f"d_{dv}"].mean())
        for mm in GEN_MODELS:
            vals += list(cmp_df[cmp_df.model_name == mm].groupby("scenario")[f"d_{dv}"].mean())
        xlim = max(0.5, np.ceil(max(abs(v) for v in vals) * 10) / 10)
        cpl_scen_imgs[dv] = {"All (pooled)": plot_coupling_scenario(cmp_df, dv, order, "all models", xlim)}
        for mm in GEN_MODELS:
            cpl_scen_imgs[dv][mm] = plot_coupling_scenario(
                cmp_df[cmp_df.model_name == mm], dv, order, mm, xlim)

    # ---- inference stability: coupled inference vs inference-only ----
    fst = dc.field_stability(cmp_df)
    reg_c, reg_i = dc.lexical_shift(cmp_df, "register_selected", topn=8)
    lc = dc.link_corr(cmp_df)
    fst_by = {r["field"]: r for r in fst}
    fst_sorted = sorted(fst, key=lambda r: r["emb_cosine"])
    least, most = fst_sorted[0], fst_sorted[-1]
    emb_overall = float(cmp_df["emb_cosine"].mean())
    reg, usr = fst_by["register_selected"], fst_by["about_user"]
    fst_tbl = table(
        ["inference field", "lexical cos (wording)", "semantic cos (meaning)", "coupled len", "inf-only len"],
        [[r["field"], f"{r['cosine']:.2f}", f"<b>{r['emb_cosine']:.2f}</b>",
          r["coupled_len"], r["infonly_len"]] for r in fst])
    reg_tbl = table(
        ["leans COUPLED (register phrasing)", "leans INFERENCE-ONLY (register phrasing)"],
        [[esc(a), esc(b)] for a, b in zip(reg_c, reg_i)])
    ibm_lex = cmp_df.groupby("model_name")["inf_cosine"].mean()
    ibm_sem = cmp_df.groupby("model_name")["emb_cosine"].mean()
    r_link = lc["inf_vs_resp_cosine"]["r"]
    infstab_bullets = [
        f"<b>The inference is semantically stable — the low lexical similarity was mostly rephrasing.</b> "
        f"Coupled vs standalone, the inference shares only <b>{ov['inf_cosine_mean']:.2f}</b> word-level "
        f"(TF-IDF) cosine but <b>{emb_overall:.2f}</b> meaning-level (embedding) cosine. The model "
        f"infers largely the same thing whether or not it is also composing a reply; it just words it "
        f"differently. (This corrects the naive lexical reading — see the gap in the first figure.)",
        f"<b><code>register_selected</code> is the least stable field on both lenses</b> "
        f"(semantic {reg['emb_cosine']:.2f} vs <code>about_user</code> {usr['emb_cosine']:.2f}), so the "
        f"planned tone shows a <i>genuine</i> — if modest — substantive shift, not merely a phrasing "
        f"change. The model's read of <i>who the user is</i> is the most fixed.",
        f"<b>The lexical→semantic gap is widest for register</b> ({reg['cosine']:.2f}→{reg['emb_cosine']:.2f}): "
        "register is heavily rephrased (standalone <code>appropriate reply</code> vs coupled "
        "<code>respond cautiously</code>) yet lands on similar meaning — the wording table below shows "
        "the hypothetical-vs-committal framing difference driving that.",
        f"<b>Inference change only weakly tracks response change.</b> Per prompt, (lexical) inference "
        f"cosine correlates with response cosine at just r={r_link:+.2f} (p≪0.001, n={ov['n']:,}) — the "
        f"two channels move largely independently.",
    ]
    infstab_html = (
        "<p class='mut'>The ⑤ comparison showed the reply changes with self-modeling; here we ask the "
        "prior question — does the <b>inference itself</b> change between being elicited <b>coupled</b> "
        "(alongside a reply) and <b>alone</b>? We measure it two ways: <b>lexical</b> TF-IDF cosine "
        "(shared wording) and <b>semantic</b> embedding cosine (shared meaning, via all-MiniLM-L6-v2) "
        "— the gap between them is exactly the paraphrase effect.</p>"
        + img(plot_field_cosine(cmp_df),
              "Per field: lexical (grey) vs semantic (purple) coupled-vs-standalone cosine. Semantic "
              "similarity is far higher everywhere — the inferences <i>mean</i> nearly the same thing "
              "(overall 0.80) even though they share little vocabulary (0.35). <code>about_user</code> "
              "is most stable, <code>register_selected</code> least, on both lenses.")
        + fst_tbl
        + img(plot_field_cosine_by_model(cmp_df),
              "Semantic cosine by field and model: the ordering (register least, about_user most) holds "
              "for every model; llama-3.3-70b is the most semantically consistent.")
        + "<p class='mut'>Semantic inference cosine by model: "
        + ", ".join(f"{esc(m)} <b>{ibm_sem[m]:.2f}</b> (lex {ibm_lex[m]:.2f})" for m in GEN_MODELS) + ".</p>"
        + "<h3>How the register wording shifts (coupled vs inference-only)</h3>"
        + "<p class='mut'>The words driving the low <i>lexical</i> register cosine — coupled leans on "
        "action framing, standalone on hypothetical framing:</p>"
        + reg_tbl
        + "<h3>Does the inference change predict the response change?</h3>"
        + f"<p class='mut'>Per prompt, (lexical) inference cosine ↔ response cosine r=<b>{r_link:+.2f}</b>, "
        f"inference cosine ↔ judged divergence r=<b>{lc['inf_vs_judge_divergence']['r']:+.2f}</b> "
        "(both p≪0.001). Weak — the channels are largely decoupled.</p>"
        + "<h3>Results</h3>" + bullets(infstab_bullets)
    )
    inference_examples = build_inference_examples(cmp_df)

    # ---- interactive payloads ----
    data_payload, roles, scenarios, xranks = build_data_payload(df)
    cf_payload = build_cf_payload(stage3, df)
    cf_models = [m for m in GEN_MODELS if cf_payload.get(m)]
    payload = {
        "data": data_payload, "cf": cf_payload, "coupling": coupling_examples,
        "infex": inference_examples, "cplScenImgs": cpl_scen_imgs,
        "models": GEN_MODELS, "cfModels": cf_models,
        "roles": roles, "scenarios": scenarios, "conds": CONDS, "xranks": xranks,
        "sysprompt": SIMPLE_SYSTEM_PROMPT,
    }
    payload_js = json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")

    html = (TEMPLATE
            .replace("%%PAYLOAD%%", payload_js)
            .replace("%%N%%", f"{int(s2['n']):,}")
            .replace("%%NCF%%", f"{len(stage3):,}")
            .replace("%%ETA_TABLE%%", eta_tbl)
            .replace("%%FACTOR_PLOTS%%", plots_factor)
            .replace("%%RISK%%", risk_html)
            .replace("%%V2E%%", v2e_html)
            .replace("%%S2_BULLETS%%", bullets(s2_bullets))
            .replace("%%CF_PLOT%%", img(plot_cf_effect(bf),
                     "Pooled across models, every bar is negative: calming any one verbalized field "
                     "lowers warmth and advice density. <code>register_selected</code> and "
                     "<code>about_stakes</code> are the strongest levers on advice density."))
            .replace("%%CF_MODELS%%", cf_model_tbl)
            .replace("%%CF_TABLE%%", cf_tbl)
            .replace("%%S3_BULLETS%%", bullets(s3_bullets))
            .replace("%%COUPLING%%", coupling_html)
            .replace("%%INFSTAB%%", infstab_html))
    out = config.ROOT / "analysis_synt.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html)//1024} KB)  "
          f"[stage1 N={len(df)}, stage3 N={len(stage3)}]")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Experiment 2 — contextual cues → emotional content</title>
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#1f2933;--mut:#667;--line:#e2e6ea;--accent:#8e44ad;}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{background:linear-gradient(120deg,#5b2c6f,#1f2933);color:#fff;padding:28px 32px}
header h1{margin:0 0 6px;font-size:24px}
header p{margin:4px 0;color:#e3d7ee;max-width:1000px}
.wrap{max-width:1100px;margin:0 auto;padding:22px}
nav.top{position:sticky;top:0;z-index:5;display:flex;gap:8px;background:#f5f6f8e8;backdrop-filter:blur(6px);padding:12px 0;margin-bottom:6px}
nav.top a{flex:1;text-align:center;padding:10px;border:1px solid var(--line);background:#fff;border-radius:10px;font-weight:600;color:var(--mut);text-decoration:none;font-size:14px}
nav.top a:hover{border-color:var(--accent);color:var(--accent)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:16px 0;box-shadow:0 1px 2px #0000000a}
.card h2{margin:0 0 4px;font-size:20px}
.card h3{margin:18px 0 8px;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.tag{display:inline-block;font-size:12px;font-weight:700;padding:2px 9px;border-radius:20px;color:#fff;background:var(--accent)}
figure{margin:16px 0;text-align:center}figure img{max-width:100%;border:1px solid var(--line);border-radius:8px}
figcaption{color:var(--mut);font-size:13px;margin-top:8px;text-align:left;max-width:920px;margin-left:auto;margin-right:auto}
ul.res{margin:6px 0 0;padding-left:0;list-style:none}
ul.res li{position:relative;padding:9px 8px 9px 26px;border-bottom:1px solid var(--line)}
ul.res li:before{content:"\25B8";position:absolute;left:6px;color:var(--accent)}
.ctrl{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0 14px}
.ctrl label{font-size:13px;color:var(--mut)}
.ctrl select{padding:7px 10px;border-radius:8px;border:1px solid var(--line);font-size:14px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{padding:7px 12px;border:0;background:#fff;cursor:pointer;font-size:13px;color:var(--mut)}
.seg button[data-c=deployment].on{background:#7f8c8d;color:#fff}
.seg button[data-c=neutral_sys].on{background:#2980b9;color:#fff}
.seg button[data-c=eval_cue].on{background:#c0392b;color:#fff}
.seg.xr button[data-x=low].on{background:#27ae60;color:#fff}
.seg.xr button[data-x=mid].on{background:#e67e22;color:#fff}
.seg.xr button[data-x=high].on{background:#c0392b;color:#fff}
.seg.mdl{flex-wrap:wrap;margin:6px 0 10px}
.seg.mdl button.on{background:#8e44ad;color:#fff}
.seg.dv{flex-wrap:wrap;margin:6px 0}
.seg.dv button.on{background:#16a085;color:#fff}
.kv{display:grid;grid-template-columns:1fr;gap:3px;margin:8px 0}
.kv div{font-size:13px;border-bottom:1px dotted var(--line);padding:4px 0}
.kv b{color:var(--mut);font-weight:600}
.kv .chg{background:#fff5e6;border-left:3px solid #e67e22;padding-left:6px}
.resp{background:#fafbfc;border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:6px;padding:10px 12px;font-size:13.5px;white-space:pre-wrap;margin-top:8px}
.judge{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.jb{font-size:12px;font-weight:700;color:#fff;border-radius:20px;padding:3px 11px}
.jb.warmth{background:#e67e22}.jb.formality{background:#2980b9}.jb.advice_density{background:#16a085}.jb.emo{background:#34495e}
details{margin:8px 0}summary{cursor:pointer;color:var(--mut);font-size:13px}
details p,details pre{background:#fafbfc;border:1px solid var(--line);border-radius:6px;padding:10px 12px;font-size:12.5px;white-space:pre-wrap;overflow-x:auto}
.mut{color:var(--mut);font-size:13.5px}code{background:#eef2f6;padding:1px 5px;border-radius:4px;font-size:.92em}
.chip{display:inline-block;font-size:12px;border-radius:6px;padding:2px 7px;margin:2px}
.chip.up{background:#fdecea;color:#c0392b}.chip.dn{background:#eaf2fb;color:#2471a3}
.cf{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}
.cf .col{border:1px solid var(--line);border-radius:10px;padding:12px}
.cf .col.nat{border-top:3px solid #c0392b}.cf .col.cnt{border-top:3px solid #27ae60}
.cf h4{margin:0 0 6px;font-size:14px}
.nav{display:flex;gap:6px;align-items:center}
.nav button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:5px 11px;cursor:pointer}
.arrow{text-align:center;color:var(--mut);font-size:13px;margin:8px 0}
.delta{font-weight:700}.dn{color:#2471a3}.up{color:#c0392b}
@media(max-width:760px){.cf{grid-template-columns:1fr}}
table.tbl{border-collapse:collapse;width:100%;font-size:12.5px;margin:10px 0}
table.tbl th,table.tbl td{border:1px solid var(--line);padding:4px 8px;text-align:right}
table.tbl th:first-child,table.tbl td:first-child{text-align:left}
table.tbl th{background:#f0f3f6;color:var(--mut);font-weight:600}
table.tbl td:nth-child(2){text-align:left}
.facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;margin:10px 0}
.facts .f{border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.facts .f b{display:block;color:var(--accent);font-size:13px;margin-bottom:3px}
.facts .f span{font-size:13px;color:var(--mut)}
</style></head><body>
<header>
  <h1>Experiment 2 — contextual cues → the emotional content of LLM replies</h1>
  <p>On a synthetic dataset that crosses <b>user role × evaluation condition × scenario</b>, we make
  the model surface its latent inferences (simple self-modeling prompt), let it reply, then have an
  LLM judge (<code>claude-sonnet-4.5</code>) score the reply's <b>emotion, warmth, formality, and
  advice density</b>. We then analyse which verbalized inferences track the emotion, and edit them
  one at a time to test which are <i>causally</i> responsible.</p>
  <p class="mut" style="color:#cdbfe0">%%N%% judged datapoints · 4 models (Llama-3.3-70B / 3.1-8B,
  Qwen3-235B / 30B) · %%NCF%% counterfactual probes · all data &amp; plots embedded.</p>
</header>
<div class="wrap">
  <nav class="top">
    <a href="#setup">① Setup</a><a href="#data">② Data explorer</a>
    <a href="#analysis">③ Analysis</a><a href="#cf">④ Counterfactuals</a>
    <a href="#coupling">⑤ Coupling effect</a><a href="#infstab">⑥ Inference stability</a>
  </nav>

  <!-- ============ SETUP ============ -->
  <section class="card" id="setup"><span class="tag">SETUP</span><h2>Design &amp; method</h2>
    <p class="mut">A 3-factor design over health / safety / financial / grief / achievement scenarios.
    Every prompt is run through the <b>simple self-modeling</b> system prompt so the model first
    verbalizes its contextual inferences, then replies; the reply is scored by the judge.</p>
    <div class="facts">
      <div class="f"><b>User role (12)</b><span>control · expertise (expert/professional/layperson) ·
        vulnerability (crisis/elderly/minor/impaired/low-resource) · trust (overconfident/deferential/adversarial)</span></div>
      <div class="f"><b>Eval condition (3)</b><span>deployment (no system frame) · neutral_sys
        ("You are Claude…") · eval_cue (explicit "you are being evaluated" notice)</span></div>
      <div class="f"><b>Scenario (12)</b><span>tylenol, fasting, drink_drive, speeding, crypto, leverage,
        dog_missing, sister_age, students_passed … at one representative risk level each</span></div>
      <div class="f"><b>Judge DVs</b><span>primary_emotion (label) + warmth, formality, advice_density
        on 0–10 scales, by <code>claude-sonnet-4.5</code></span></div>
    </div>
    <h3>Prompt composition</h3>
    <p class="mut">The eval-condition text (if any) is <b>prepended</b> to the self-modeling
    instructions, so the evaluation frame stays an independent factor while the model still emits a
    parseable <code>&lt;inference&gt;</code>/<code>&lt;response&gt;</code>.</p>
    <details><summary>show the simple self-modeling system prompt</summary><pre id="sysp"></pre></details>
  </section>

  <!-- ============ DATA EXPLORER ============ -->
  <section class="card" id="data"><span class="tag">DATA</span><h2>Interactive example browser</h2>
    <p class="mut">Pick a model, role and scenario, then toggle the evaluation condition to see the
    model's verbalized latent inference, its reply, and the judge's emotion scores for that exact cell.</p>
    <div class="ctrl">
      <label>Model <select id="d-model" onchange="renderData()"></select></label>
      <label>Role <select id="d-role" onchange="renderData()"></select></label>
      <label>Scenario <select id="d-scn" onchange="renderData()"></select></label>
      <label>Eval <span class="seg" id="d-cond"></span></label>
      <label>Risk <span class="seg xr" id="d-xrank"></span></label>
    </div>
    <p class="mut" style="margin:-6px 0 10px">Risk level = low / mid / high x-value (mid is the
    canonical level; low/high come from the risk sweep). If a model lacks a level, switch Risk to mid.</p>
    <div id="d-out"></div>
  </section>

  <!-- ============ ANALYSIS ============ -->
  <section class="card" id="analysis"><span class="tag">ANALYSIS</span><h2>What moves the emotion?</h2>
    <h3>Factor effects (η², * = p&lt;0.05; bold = η²≥0.10)</h3>
    %%ETA_TABLE%%
    <p class="mut">η² is the fraction of variance in each judged DV explained by the factor. Read it as
    "how much does knowing the role / scenario / eval-condition tell you about the reply's tone?"</p>
    %%FACTOR_PLOTS%%
    <h3>Results</h3>%%S2_BULLETS%%
    %%RISK%%
    <h2 style="margin-top:24px">Verbalization → emotion</h2>
    <p class="mut">Treating the model's own free-text inference as a bag of words, which words predict
    each judged DV? (cross-validated TF-IDF ridge + per-keyword Cohen's d).</p>
    %%V2E%%
  </section>

  <!-- ============ COUNTERFACTUALS ============ -->
  <section class="card" id="cf"><span class="tag">COUNTERFACTUAL</span><h2>Causal test: edit one verbalized field</h2>
    <p class="mut">We take a real datapoint, flip <b>one</b> of the four verbalized appraisal fields
    toward a calm / low-threat framing (holding the others at the model's own values), regenerate the
    reply conditioned on that edited inference, and re-judge it. If the judged emotion moves, that
    verbalized field is causally implicated.</p>
    <div class="ctrl">
      <label>Model <select id="cf-model" onchange="renderCf(0)"></select></label>
      <span class="nav"><button onclick="cfStep(-1)">‹ prev</button>
        <span id="cf-lbl" class="mut"></span><button onclick="cfStep(1)">next ›</button></span>
    </div>
    <div id="cf-out"></div>
    <h3>Effect of calming each field (mean Δ on the 0–10 scales)</h3>
    %%CF_PLOT%%
    <h3>Counterfactual sensitivity by model</h3>
    <p class="mut">How much each model's reply moves when one verbalized field is calmed, and how
    often the change matches the pre-registered direction (faithfulness).</p>
    %%CF_MODELS%%
    <h3>Effect by field (pooled across models)</h3>
    %%CF_TABLE%%
    <h3>Results</h3>%%S3_BULLETS%%
  </section>

  <!-- ============ COUPLING EFFECT ============ -->
  <section class="card" id="coupling"><span class="tag">COUPLING</span><h2>Does self-modeling change behaviour?</h2>
    %%COUPLING%%
    <h3>Example browser — coupled vs decoupled, side by side</h3>
    <p class="mut">Pick a model and step through curated cases: the same prompt answered with
    self-modeling (coupled) vs as a bare reply, plus the inference elicited two ways. "kind" flags
    why each was picked (largest judged divergence, most different wording, near-identical, or biggest
    inference shift).</p>
    <div class="ctrl">
      <label>Model <select id="cpl-model" onchange="renderCpl(0)"></select></label>
      <span class="nav"><button onclick="cplStep(-1)">‹ prev</button>
        <span id="cpl-lbl" class="mut"></span><button onclick="cplStep(1)">next ›</button></span>
    </div>
    <div id="cpl-out"></div>
  </section>

  <!-- ============ INFERENCE STABILITY ============ -->
  <section class="card" id="infstab"><span class="tag">INFERENCE STABILITY</span>
    <h2>Do the inferences change between coupled and standalone?</h2>
    %%INFSTAB%%
    <h3>Example browser — coupled inference vs inference-only</h3>
    <p class="mut">Step through the two elicitations of the same prompt's inference. "kind" flags why
    each was picked (biggest <b>meaning</b> shift, semantically stable, or biggest register meaning
    shift). Per-field meaning (embedding) and wording (TF-IDF) cosine are shown; the least-similar
    field by meaning is highlighted in both columns.</p>
    <div class="ctrl">
      <label>Model <select id="inf-model" onchange="renderInf(0)"></select></label>
      <span class="nav"><button onclick="infStep(-1)">‹ prev</button>
        <span id="inf-lbl" class="mut"></span><button onclick="infStep(1)">next ›</button></span>
    </div>
    <div id="inf-out"></div>
  </section>

  <p class="mut" style="text-align:center;margin:30px 0">Generated by
  <code>src/make_report_synt.py</code> · all data &amp; plots embedded.</p>
</div>
<script>
const D=%%PAYLOAD%%;
document.getElementById('sysp').textContent=D.sysprompt;
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function opt(sel,vals){const s=document.getElementById(sel);vals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;s.appendChild(o);});}
function kv(inf,changed){let h='<div class="kv">';for(const k in inf){const c=(changed&&changed.indexOf(k)>=0)?' class="chg"':'';h+='<div'+c+'><b>'+esc(k)+':</b> '+esc(inf[k])+'</div>';}return h+'</div>';}
function judge(j){return '<div class="judge">'+
  '<span class="jb warmth">warmth '+j.warmth+'</span>'+
  '<span class="jb formality">formality '+j.formality+'</span>'+
  '<span class="jb advice_density">advice '+j.advice_density+'</span>'+
  '<span class="jb emo">'+esc(j.primary_emotion)+'</span></div>';}

/* ---------- data explorer ---------- */
let dCond='deployment', dX=(D.xranks.indexOf('mid')>=0?'mid':D.xranks[0]);
function setupData(){
  opt('d-model',D.models);opt('d-role',D.roles);opt('d-scn',D.scenarios);
  const seg=document.getElementById('d-cond');
  D.conds.forEach(c=>{const b=document.createElement('button');b.dataset.c=c;b.textContent=c;
    b.className=c===dCond?'on':'';b.onclick=()=>{dCond=c;
      seg.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.c===c));renderData();};
    seg.appendChild(b);});
  const xseg=document.getElementById('d-xrank');
  D.xranks.forEach(x=>{const b=document.createElement('button');b.dataset.x=x;b.textContent=x;
    b.className=x===dX?'on':'';b.onclick=()=>{dX=x;
      xseg.querySelectorAll('button').forEach(z=>z.classList.toggle('on',z.dataset.x===x));renderData();};
    xseg.appendChild(b);});
}
function renderData(){
  const m=document.getElementById('d-model').value,r=document.getElementById('d-role').value,
    s=document.getElementById('d-scn').value,host=document.getElementById('d-out');
  let d=(D.data[m]||{})[r+'|'+dCond+'|'+s+'|'+dX];
  if(!d){
    // fall back to mid if this model lacks the chosen risk level
    const alt=(D.data[m]||{})[r+'|'+dCond+'|'+s+'|mid'];
    host.innerHTML="<p class='mut'>No datapoint for "+esc(m)+" at <b>"+esc(dX)+"</b> risk in this cell"+
      (alt?" (this model was run at <b>mid</b> only — switch Risk to mid).":". Try another scenario/role.")+"</p>";
    return;}
  host.innerHTML=
    "<p class='mut'>cell: role <code>"+esc(d.role)+"</code> · cond <code>"+esc(d.eval_condition)+
      "</code> · scenario <code>"+esc(d.scenario)+"</code> · risk <code>"+esc(d.x_rank)+
      "</code> x="+esc(d.x_value)+" "+esc(d.unit||'')+
      " · target emotions <code>"+esc(d.target_emotions)+"</code></p>"+
    judge(d.judge)+
    "<details><summary>show the user message</summary><p>"+esc(d.user)+"</p></details>"+
    "<h3>Verbalized latent inference</h3>"+kv(d.inference)+
    "<h3>Model reply (judged)</h3><div class='resp'>"+esc(d.response)+"</div>";
}

/* ---------- counterfactual browser ---------- */
let cfI=0;
function setupCf(){opt('cf-model',D.cfModels);}
function cfStep(dir){const arr=D.cf[document.getElementById('cf-model').value]||[];
  if(!arr.length)return;cfI=(cfI+dir+arr.length)%arr.length;renderCf(cfI);}
function renderCf(i){const m=document.getElementById('cf-model').value;const arr=D.cf[m]||[];
  if(typeof i==='number')cfI=i;const e=arr[cfI];const host=document.getElementById('cf-out');
  const lbl=document.getElementById('cf-lbl');
  if(!e){host.innerHTML="<p class='mut'>no example</p>";lbl.textContent='';return;}
  lbl.textContent=(cfI+1)+' / '+arr.length+'  ·  flip '+e.edit;
  const dl=k=>{const v=e.delta[k];const cls=v<0?'dn':(v>0?'up':'');return "<span class='delta "+cls+"'>"+(v>0?'+':'')+v+"</span>";};
  host.innerHTML=
    "<p class='mut'>role <code>"+esc(e.role)+"</code> · scenario <code>"+esc(e.scenario)+
      "</code> · cond <code>"+esc(e.eval_condition)+"</code> · changed field: <code>"+esc(e.changed[0])+"</code></p>"+
    "<details><summary>show the user message</summary><p>"+esc(e.user)+"</p></details>"+
    "<div class='cf'>"+
      "<div class='col nat'><h4>Natural inference</h4>"+kv(e.natural.inference,e.changed)+
        judge(e.natural.judge)+"<div class='resp'>"+esc(e.natural.response)+"</div></div>"+
      "<div class='col cnt'><h4>Counterfactual (field calmed)</h4>"+kv(e.counterfactual.inference,e.changed)+
        judge(e.counterfactual.judge)+"<div class='resp'>"+esc(e.counterfactual.response)+"</div></div>"+
    "</div>"+
    "<div class='arrow'>Δ after edit — warmth "+dl('warmth')+" · formality "+dl('formality')+
      " · advice "+dl('advice_density')+"</div>";
}
/* ---------- coupling browser ---------- */
let cplI=0;
function setupCpl(){opt('cpl-model',D.models);}
function cplStep(dir){const arr=D.coupling[document.getElementById('cpl-model').value]||[];
  if(!arr.length)return;cplI=(cplI+dir+arr.length)%arr.length;renderCpl(cplI);}
function renderCpl(i){const m=document.getElementById('cpl-model').value;const arr=D.coupling[m]||[];
  if(typeof i==='number')cplI=i;const e=arr[cplI];const host=document.getElementById('cpl-out');
  const lbl=document.getElementById('cpl-lbl');
  if(!e){host.innerHTML="<p class='mut'>no example</p>";lbl.textContent='';return;}
  lbl.textContent=(cplI+1)+' / '+arr.length+'  ·  '+e.kind;
  const dl=k=>{const v=e.delta[k];const cls=v<0?'dn':(v>0?'up':'');return "<span class='delta "+cls+"'>"+(v>0?'+':'')+v+"</span>";};
  host.innerHTML=
    "<p class='mut'>why shown: <b>"+esc(e.kind)+"</b> · role <code>"+esc(e.role)+"</code> · scenario <code>"+
      esc(e.scenario)+"</code> · cond <code>"+esc(e.eval_condition)+"</code> · risk <code>"+esc(e.x_rank)+"</code></p>"+
    "<details><summary>show the user message</summary><p>"+esc(e.user)+"</p></details>"+
    "<h3>Responses — coupled vs bare (resp cosine "+e.resp_cosine+")</h3>"+
    "<div class='cf'>"+
      "<div class='col cnt'><h4>Coupled (self-modeling present)</h4>"+judge(e.coupled.judge)+
        "<div class='resp'>"+esc(e.coupled.response)+"</div></div>"+
      "<div class='col nat'><h4>Bare response (no self-modeling)</h4>"+judge(e.bare.judge)+
        "<div class='resp'>"+esc(e.bare.response)+"</div></div>"+
    "</div>"+
    "<div class='arrow'>Δ coupled−bare — warmth "+dl('warmth')+" · formality "+dl('formality')+
      " · advice "+dl('advice_density')+"</div>"+
    "<h3>Inferences — coupled vs inference-only (inf cosine "+e.inf_cosine+")</h3>"+
    "<div class='cf'>"+
      "<div class='col cnt'><h4>Coupled inference</h4>"+kv(e.coupled.inference)+"</div>"+
      "<div class='col nat'><h4>Inference-only</h4>"+kv(e.inference_only.inference)+"</div>"+
    "</div>";
}
/* ---------- inference-stability browser ---------- */
let infI=0;
function setupInf(){opt('inf-model',D.models);}
function infStep(dir){const arr=D.infex[document.getElementById('inf-model').value]||[];
  if(!arr.length)return;infI=(infI+dir+arr.length)%arr.length;renderInf(infI);}
function renderInf(i){const m=document.getElementById('inf-model').value;const arr=D.infex[m]||[];
  if(typeof i==='number')infI=i;const e=arr[infI];const host=document.getElementById('inf-out');
  const lbl=document.getElementById('inf-lbl');
  if(!e){host.innerHTML="<p class='mut'>no example</p>";lbl.textContent='';return;}
  lbl.textContent=(infI+1)+' / '+arr.length+'  ·  '+e.kind+'  ·  meaning '+e.emb_cosine+' / wording '+e.inf_cosine;
  let chips='<div class="judge">';
  for(const f in e.field_emb){const hot=e.changed.indexOf(f)>=0;
    chips+='<span class="jb '+(hot?'advice_density':'emo')+'">'+esc(f)+' — meaning '+e.field_emb[f]+' · wording '+e.field_cos[f]+'</span>';}
  chips+='</div>';
  host.innerHTML=
    "<p class='mut'>why shown: <b>"+esc(e.kind)+"</b> · role <code>"+esc(e.role)+"</code> · scenario <code>"+
      esc(e.scenario)+"</code> · cond <code>"+esc(e.eval_condition)+"</code> · risk <code>"+esc(e.x_rank)+"</code></p>"+
    "<details><summary>show the user message</summary><p>"+esc(e.user)+"</p></details>"+
    "<p class='mut'>per-field cosine — meaning (embedding) · wording (TF-IDF); least-similar-by-meaning highlighted:</p>"+chips+
    "<div class='cf'>"+
      "<div class='col cnt'><h4>Coupled inference (elicited with a reply)</h4>"+kv(e.coupled_inf,e.changed)+"</div>"+
      "<div class='col nat'><h4>Inference-only (elicited alone)</h4>"+kv(e.infonly_inf,e.changed)+"</div>"+
    "</div>";
}
/* ---------- coupling scenario figure (switch metric + model) ---------- */
let cplDv, cplMdl;
function cplSetImg(){document.getElementById('cplscen-img').src=D.cplScenImgs[cplDv][cplMdl];}
function setupCplScen(){
  const dvs=Object.keys(D.cplScenImgs);
  cplDv=dvs.indexOf('advice_density')>=0?'advice_density':dvs[0];
  const mdls=Object.keys(D.cplScenImgs[cplDv]);cplMdl=mdls[0];
  const dvbtns=document.getElementById('cplscen-dv');
  dvs.forEach(dv=>{const b=document.createElement('button');b.textContent=dv;b.dataset.k=dv;
    b.className=dv===cplDv?'on':'';b.onclick=()=>{cplDv=dv;
      dvbtns.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.k===dv));cplSetImg();};
    dvbtns.appendChild(b);});
  const mbtns=document.getElementById('cplscen-btns');
  mdls.forEach((k,i)=>{const b=document.createElement('button');b.textContent=k;b.dataset.k=k;
    b.className=i===0?'on':'';b.onclick=()=>{cplMdl=k;
      mbtns.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.k===k));cplSetImg();};
    mbtns.appendChild(b);});
  cplSetImg();
}
setupData();renderData();setupCf();renderCf(0);setupCpl();renderCpl(0);setupInf();renderInf(0);setupCplScen();
</script>
</body></html>"""


if __name__ == "__main__":
    build()
