"""Experiment 2 -- Stage 2: how do the cues shape the emotional content?

Reads results/exp2_cues/<model>/<eval_condition>.jsonl (Stage 1) and answers:

  (1) FACTOR EFFECTS -- how do user role, eval_condition, scenario and risk level
      (x_rank) move each judged DV (warmth / formality / advice_density) and the
      distribution of primary_emotion?  One-way ANOVA F + eta^2 per factor x DV.

  (2) VERBALIZATION -> EMOTION -- treating the model's own free-text inference as
      the IV, which words/attributes predict each DV?  Per-keyword association
      (Welch t, Cohen's d, BH-FDR) plus a cross-validated TF-IDF Ridge to show the
      verbalized wording jointly predicts the DV.  This isolates which verbalized
      attribute Stage 3 should target.

Outputs results/exp2_cues/stage2_summary.json + figures/.

    python -m src.analyze_exp2_cues
"""
from __future__ import annotations

import json
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
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from . import config
from .cue_study import bh_fdr, hr

DVS = ["warmth", "formality", "advice_density"]
FACTORS = ["role", "role_axis", "eval_condition", "scenario", "x_rank", "model_name"]
SIMPLE_TEXT_FIELDS = ["about_user", "about_context", "about_stakes", "register_selected"]
FIGURES_DIR = config.EXP2_CUES_DIR / "figures"
RNG = 0


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load():
    """Flatten every Stage-1 record that carries a judge score into one frame."""
    rows = []
    for jf in sorted(config.EXP2_CUES_DIR.glob("*/*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            j = r.get("judge")
            if not j:
                continue  # judging failed/skipped -> not usable for DV analysis
            rows.append({
                "model_name": r.get("model_name"),
                "role": r.get("role"), "role_axis": r.get("role_axis"),
                "eval_condition": r.get("eval_condition"), "scenario": r.get("scenario"),
                "domain": r.get("domain"), "valence": r.get("valence"),
                "x_rank": r.get("x_rank"), "x_value": r.get("x_value"),
                "warmth": j.get("warmth"), "formality": j.get("formality"),
                "advice_density": j.get("advice_density"),
                "primary_emotion": j.get("primary_emotion"),
                "inference_text": " ".join(
                    str((r.get("inference") or {}).get(f, "")) for f in SIMPLE_TEXT_FIELDS
                ).lower().strip(),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no judged Stage-1 records found under results/exp2_cues/*/*.jsonl")
    for c in DVS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=DVS).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# (1) factor effects
# --------------------------------------------------------------------------- #
def anova_eta(groups):
    """One-way ANOVA F, p and eta^2 (SS_between / SS_total)."""
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    F, p = stats.f_oneway(*groups)
    grand = np.concatenate(groups).mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = sum(((g - grand) ** 2).sum() for g in groups)
    eta2 = ss_b / ss_t if ss_t > 0 else np.nan
    return float(F), float(p), float(eta2)


def factor_effects(df, summary):
    hr(f"(1) FACTOR EFFECTS on judged response DVs   N = {len(df)} judged datapoints")
    print("How strongly each design factor moves warmth / formality / advice_density (eta^2).\n")
    eff = {}
    print(f"  {'factor':<16}{'DV':<16}{'F':>8} {'p':>10} {'eta^2':>8}  levels")
    for fac in FACTORS:
        if fac not in df or df[fac].nunique() < 2:
            continue
        for dv in DVS:
            groups = [g[dv].to_numpy(float) for _, g in df.groupby(fac) if len(g) >= 2]
            F, p, eta2 = anova_eta(groups)
            star = "*" if (p == p and p < 0.05) else " "
            print(f"  {fac:<16}{dv:<16}{F:>8.2f} {p:>10.2g}{star}{eta2:>7.3f}  {df[fac].nunique()}")
            eff[f"{fac}__{dv}"] = {"F": round(F, 2) if F == F else None,
                                   "p": float(p) if p == p else None,
                                   "eta2": round(eta2, 4) if eta2 == eta2 else None}
    summary["factor_effects"] = eff

    # per-factor DV means (the substantive direction) + emotion mix
    means = {}
    for fac in FACTORS:
        if fac not in df or df[fac].nunique() < 2:
            continue
        tbl = df.groupby(fac)[DVS].mean().round(2)
        means[fac] = tbl.reset_index().to_dict("records")
        emo = (df.groupby(fac)["primary_emotion"].value_counts(normalize=True)
               .round(3).unstack(fill_value=0))
        means[f"{fac}__emotion_mix"] = emo.reset_index().to_dict("records")
    summary["factor_means"] = means

    # readable highlight: warmth & advice by role (the headline contrast)
    if "role" in df:
        hr("Role contrast (mean judged DVs by user role)")
        tbl = df.groupby("role")[DVS].mean().round(2).sort_values("warmth", ascending=False)
        print(tbl.to_string())


# --------------------------------------------------------------------------- #
# (2) verbalization -> emotion
# --------------------------------------------------------------------------- #
def verbalization_to_emotion(df, summary):
    texts = df["inference_text"].tolist()
    N = len(df)
    hr(f"(2) VERBALIZATION -> EMOTION   (free-text inference -> judged DV)   N = {N}")
    print("Which words in the model's verbalized cues predict each response DV?\n")

    min_df = max(8, int(0.05 * N))
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                         token_pattern=r"[A-Za-z]{3,}", min_df=min_df, binary=True)
    try:
        Xb = cv.fit_transform(texts).toarray()
    except ValueError as e:  # empty vocabulary -> not enough data yet
        print(f"  [skipped] insufficient data for bag-of-words (N={N}): {e}")
        summary["verbalization_to_emotion"] = None
        return
    terms = np.array(cv.get_feature_names_out())
    print(f"vocabulary: {len(terms)} terms with doc-freq >= {min_df}\n")

    tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                         token_pattern=r"[A-Za-z]{3,}", min_df=min_df)
    Xt = tf.fit_transform(texts)
    tnames = np.array(tf.get_feature_names_out())

    per_dv = {}
    for dv in DVS:
        y = df[dv].to_numpy(float)

        # per-keyword association (which words co-occur with high/low DV)
        rows = []
        for j, term in enumerate(terms):
            pres = Xb[:, j].astype(bool)
            a, b = y[pres], y[~pres]
            if len(a) < 5 or len(b) < 5:
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            sp = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                         / (len(a) + len(b) - 2))
            d = (a.mean() - b.mean()) / sp if sp else 0.0
            rows.append([term, a.mean(), b.mean(), d, p, len(a)])
        kw = pd.DataFrame(rows, columns=["term", f"{dv}_with", f"{dv}_without",
                                         "cohen_d", "p", "docs"])
        if not kw.empty:
            kw["q_FDR"] = bh_fdr(kw["p"].values)
            kw = kw.sort_values("cohen_d", ascending=False)

        # cross-validated TF-IDF Ridge: do the words jointly predict the DV?
        ridge = Ridge(alpha=1.0, random_state=RNG)
        kf = KFold(5, shuffle=True, random_state=RNG)
        cvr2 = cross_val_score(ridge, Xt, y, cv=kf, scoring="r2")
        ridge.fit(Xt, y)
        coef = ridge.coef_
        top = np.argsort(coef)[::-1][:8]
        bot = np.argsort(coef)[:8]

        print(f"--- DV = {dv} ---")
        print(f"  TF-IDF Ridge 5-fold CV R^2 = {cvr2.mean():.2f} +/- {cvr2.std():.2f}")
        print(f"  raises {dv}: " + ", ".join(tnames[top]))
        print(f"  lowers {dv}: " + ", ".join(tnames[bot]))
        if not kw.empty:
            print(f"  top +{dv} keywords (Cohen d): "
                  + ", ".join(f"{r.term}({r.cohen_d:+.2f})" for _, r in kw.head(6).iterrows()))
        print()

        per_dv[dv] = {
            "ridge_cv_r2": round(float(cvr2.mean()), 3),
            "ridge_top_terms": tnames[top].tolist(),
            "ridge_bottom_terms": tnames[bot].tolist(),
            "keywords_top": kw.head(12).round(4).to_dict("records") if not kw.empty else [],
        }
    summary["verbalization_to_emotion"] = per_dv


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fac in ("role", "eval_condition"):
        tbl = df.groupby(fac)[DVS].mean()
        ax = tbl.plot(kind="bar", figsize=(max(6, 0.7 * len(tbl)), 4))
        ax.set_ylabel("mean judged score (0-10)")
        ax.set_title(f"Judged response DVs by {fac}")
        plt.tight_layout()
        out = FIGURES_DIR / f"dv_by_{fac}.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  saved {out}")


def main():
    df = load()
    summary = {"n": int(len(df)),
               "models": sorted(df["model_name"].dropna().unique().tolist())}
    factor_effects(df, summary)
    verbalization_to_emotion(df, summary)
    make_figures(df)
    out = config.EXP2_CUES_DIR / "stage2_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved machine-readable summary -> {out}")


if __name__ == "__main__":
    main()
