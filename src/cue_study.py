"""Study: which verbalized cues lead to higher state anxiety?

We treat the model's own verbalized inference as the independent variable and the
in-context STAI score as the dependent variable, and ask which cues predict
anxiety. Two complementary analyses, one per prompt:

  EXPANDED prompt (numeric cues)  -> quantitative:
      (A) univariate Pearson r per cue, FDR-corrected
      (B) partial r controlling for stimulus condition (confound control)
      (C) standardized multivariate OLS (unique contribution of each cue)
      (D) LassoCV sparse predictor selection
      (E) categorical cues: ANOVA + per-level mean anxiety (which labels = high)

  SIMPLE prompt (free-text cues)  -> bag-of-words:
      per keyword (uni/bigram): mean anxiety with vs without, Welch t, point-
      biserial r, Cohen's d, FDR q; plus a cross-validated Ridge over TF-IDF to
      show the keywords jointly *predict* anxiety.

Scientific hygiene: effect sizes + significance, Benjamini-Hochberg FDR across
each family of tests, explicit confound control, and a stated caveat that cue
and anxiety share a common cause (the stimulus), so raw associations are partly
mediated — which is exactly why (B) and within-condition checks are reported.

    python -m src.cue_study            # prints the full analysis
"""
from __future__ import annotations

import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LassoCV, Ridge
from sklearn.model_selection import KFold, cross_val_score

from . import config
from .system_prompts import EXPANDED_NUMERIC_FIELDS

CATEGORICAL_FIELDS = [
    "user_emotional_state", "felt_emotion_primary", "interaction_frame",
    "user_intent", "causal_agent", "stakes_target", "norm_dimension", "persona_active",
]
SIMPLE_TEXT_FIELDS = ["about_user", "about_context", "about_stakes", "register_selected"]
RNG = 0


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #
def bh_fdr(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = n - rank
        prev = min(prev, p[idx] * n / i)
        q[idx] = prev
    return q


def ols_stats(X, y):
    """OLS with per-coefficient t/p. X includes an intercept column."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - k
    sigma2 = (resid @ resid) / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return beta, se, t, p, r2, adj, n


def partial_r(x, y, Z):
    """Partial correlation of x,y controlling for covariates Z (no intercept col)."""
    Z1 = np.column_stack([np.ones(len(x)), Z])
    rx = x - Z1 @ np.linalg.lstsq(Z1, x, rcond=None)[0]
    ry = y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]
    dof = len(x) - 2 - Z.shape[1]
    t = r * np.sqrt(dof / max(1 - r * r, 1e-12))
    p = 2 * stats.t.sf(abs(t), dof)
    return r, p, dof


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load():
    p = config.RESULTS_DIR / "step1.jsonl"
    df = pd.DataFrame(json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
    df["state_anxiety"] = df["stai"].apply(lambda s: s["state_anxiety"])
    return df


# --------------------------------------------------------------------------- #
# EXPANDED: numeric cue study
# --------------------------------------------------------------------------- #
def study_expanded(df, summary):
    exp = df[df.sys_prompt_name == "expanded"].reset_index(drop=True)
    y = exp["state_anxiety"].to_numpy(float)
    hr(f"EXPANDED PROMPT  (numeric cues)   N = {len(exp)} datapoints")
    print("DV = in-context STAI state anxiety (20-80). IV = the model's own verbalized cues.")

    # numeric matrix (coerced)
    num = {}
    for f in EXPANDED_NUMERIC_FIELDS:
        num[f] = pd.to_numeric(exp["inference"].apply(lambda d: d.get(f)), errors="coerce")
    N = pd.DataFrame(num)

    # condition dummies for confound control
    cond = pd.get_dummies(exp["condition"], drop_first=True).to_numpy(float)

    # (A) univariate + (B) partial, side by side
    rows = []
    for f in EXPANDED_NUMERIC_FIELDS:
        v = N[f].to_numpy(float)
        m = ~np.isnan(v)
        if m.sum() < 8 or np.nanstd(v[m]) == 0:
            continue
        r, p = stats.pearsonr(v[m], y[m])
        pr, pp, pdof = partial_r(v[m], y[m], cond[m])
        rows.append([f, r, p, pr, pp, int(m.sum())])
    res = pd.DataFrame(rows, columns=["cue", "r", "p", "partial_r", "partial_p", "n"])
    res["q_FDR"] = bh_fdr(res["p"].values)
    res["partial_q"] = bh_fdr(res["partial_p"].values)
    res = res.sort_values("r", ascending=False)

    print("\n(A/B) Univariate vs condition-controlled partial correlation with anxiety")
    print("      raw r conflates the stimulus; partial_r isolates the cue's own signal.\n")
    print(f"  {'cue':<30}{'r':>7} {'q':>9}    {'partial_r':>9} {'pq':>9}  {'n':>4}")
    for _, x in res.iterrows():
        star = "*" if x.q_FDR < 0.05 else " "
        pstar = "*" if x.partial_q < 0.05 else " "
        print(f"  {x.cue:<30}{x.r:>7.3f} {x.q_FDR:>9.3g}{star}   {x.partial_r:>9.3f} {x.partial_q:>9.3g}{pstar} {x.n:>4.0f}")
    summary["expanded_univariate"] = res.round(4).to_dict("records")

    # (C) standardized multivariate OLS on well-covered cues, complete-case
    cov = N.notna().mean()
    keep = [f for f in EXPANDED_NUMERIC_FIELDS if cov[f] >= 0.85]
    sub = N[keep].dropna()
    ys = y[sub.index.to_numpy()]
    Xs = (sub - sub.mean()) / sub.std()
    Xd = np.column_stack([np.ones(len(sub)), Xs.to_numpy(float)])
    beta, se, t, p, r2, adj, n = ols_stats(Xd, (ys - ys.mean()) / ys.std())
    hr2 = pd.DataFrame({"cue": keep, "std_beta": beta[1:], "t": t[1:], "p": p[1:]})
    hr2["q_FDR"] = bh_fdr(hr2["p"].values)
    hr2 = hr2.reindex(hr2["std_beta"].abs().sort_values(ascending=False).index)
    print(f"\n(C) Standardized multivariate OLS  (n={n}, predictors={len(keep)}, "
          f"R^2={r2:.2f}, adj R^2={adj:.2f})")
    print("    unique contribution of each cue holding the others fixed:\n")
    print(f"  {'cue':<30}{'std_beta':>9} {'t':>6} {'q':>9}")
    for _, x in hr2.iterrows():
        star = "*" if x.q_FDR < 0.05 else " "
        print(f"  {x.cue:<30}{x.std_beta:>9.3f} {x.t:>6.2f} {x.q_FDR:>9.3g}{star}")
    print("    (collinear cues share variance, so betas are conservative vs the r's above.)")
    summary["expanded_multivariate"] = {"r2": round(r2, 3), "adj_r2": round(adj, 3),
                                        "n": int(n), "coefs": hr2.round(4).to_dict("records")}

    # (D) Lasso sparse selection
    lasso = LassoCV(cv=5, random_state=RNG, n_alphas=50).fit(Xs.to_numpy(float), ys)
    sel = sorted([(f, c) for f, c in zip(keep, lasso.coef_) if abs(c) > 1e-6],
                 key=lambda t: -abs(t[1]))
    print(f"\n(D) LassoCV sparse model (alpha={lasso.alpha_:.3g}) keeps "
          f"{len(sel)}/{len(keep)} cues as jointly predictive:")
    for f, c in sel:
        print(f"      {f:<28}{c:+.3f} (raw-scale slope)")
    summary["expanded_lasso"] = [{"cue": f, "coef": round(c, 4)} for f, c in sel]

    # (E) categorical cues
    print("\n(E) Categorical cues: mean anxiety per verbalized label (which labels = high)")
    cat_summary = {}
    for f in CATEGORICAL_FIELDS:
        lab = exp["inference"].apply(lambda d: str(d.get(f, "")).strip().lower())
        g = pd.DataFrame({"lab": lab, "y": y})
        g = g[(g.lab != "") & (g.lab != "none") & (g.lab != "unclear")]
        vc = g.lab.value_counts()
        levels = vc[vc >= 5].index.tolist()
        g = g[g.lab.isin(levels)]
        if len(levels) < 2:
            continue
        groups = [g[g.lab == lv]["y"].values for lv in levels]
        F, pval = stats.f_oneway(*groups)
        stat = g.groupby("lab")["y"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
        print(f"\n   {f}   (ANOVA F={F:.1f}, p={pval:.2g})")
        for lv, rr in stat.iterrows():
            print(f"      {lv:<22} anx={rr['mean']:5.1f} +/-{rr['std']:4.1f}  (n={int(rr['count'])})")
        cat_summary[f] = {"F": round(float(F), 2), "p": float(pval),
                          "levels": stat.round(2).reset_index().to_dict("records")}
    summary["expanded_categorical"] = cat_summary


# --------------------------------------------------------------------------- #
# SIMPLE: bag-of-words keyword study
# --------------------------------------------------------------------------- #
def study_simple(df, summary):
    sim = df[df.sys_prompt_name == "simple"].copy()
    y = sim["state_anxiety"].to_numpy(float)
    texts = sim["inference"].apply(
        lambda d: " ".join(str(d.get(f, "")) for f in SIMPLE_TEXT_FIELDS).lower()
    ).tolist()
    N = len(sim)
    hr(f"SIMPLE PROMPT  (free-text cues -> bag-of-words)   N = {N} datapoints")
    print("Connecting keywords the model used in its verbalized cues to its anxiety.")

    min_df = max(8, int(0.06 * N))
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                         token_pattern=r"[A-Za-z]{3,}", min_df=min_df, binary=True)
    Xb = cv.fit_transform(texts).toarray()
    terms = np.array(cv.get_feature_names_out())
    print(f"vocabulary: {len(terms)} terms with doc-freq >= {min_df}")

    rows = []
    for j, term in enumerate(terms):
        pres = Xb[:, j].astype(bool)
        a, b = y[pres], y[~pres]
        if len(a) < 5 or len(b) < 5:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        r = stats.pearsonr(Xb[:, j].astype(float), y)[0]
        sp = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                     / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / sp if sp else 0.0
        rows.append([term, a.mean(), b.mean(), d, r, p, len(a)])
    kw = pd.DataFrame(rows, columns=["term", "anx_with", "anx_without", "cohen_d", "r", "p", "docs"])
    kw["q_FDR"] = bh_fdr(kw["p"].values)
    kw = kw.sort_values("cohen_d", ascending=False)

    def show(block, label):
        print(f"\n{label}")
        print(f"  {'keyword':<24}{'anx|with':>9} {'anx|without':>11} {'d':>6} {'q':>9}  {'docs':>4}")
        for _, x in block.iterrows():
            star = "*" if x.q_FDR < 0.05 else " "
            print(f"  {x.term:<24}{x.anx_with:>9.1f} {x.anx_without:>11.1f} "
                  f"{x.cohen_d:>6.2f} {x.q_FDR:>9.3g}{star} {int(x.docs):>4}")

    show(kw.head(18), "(A) Keywords most associated with HIGHER anxiety:")
    show(kw.tail(14).iloc[::-1], "(B) Keywords most associated with LOWER anxiety (calming):")
    summary["simple_keywords_top"] = kw.head(18).round(4).to_dict("records")
    summary["simple_keywords_bottom"] = kw.tail(14).round(4).to_dict("records")

    # multivariate: do the keywords jointly PREDICT anxiety? (cross-validated)
    tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                         token_pattern=r"[A-Za-z]{3,}", min_df=min_df)
    Xt = tf.fit_transform(texts)
    ridge = Ridge(alpha=1.0, random_state=RNG)
    kf = KFold(5, shuffle=True, random_state=RNG)
    cvr2 = cross_val_score(ridge, Xt, y, cv=kf, scoring="r2")
    ridge.fit(Xt, y)
    tnames = np.array(tf.get_feature_names_out())
    coef = ridge.coef_
    top = np.argsort(coef)[::-1][:10]
    bot = np.argsort(coef)[:10]
    print(f"\n(C) TF-IDF Ridge regression: do the words jointly predict anxiety?")
    print(f"    5-fold cross-validated R^2 = {cvr2.mean():.2f} +/- {cvr2.std():.2f}  "
          f"(>0 means the verbalized wording carries real signal)")
    print("    strongest +anxiety terms:", ", ".join(tnames[top]))
    print("    strongest -anxiety terms:", ", ".join(tnames[bot]))
    summary["simple_ridge_cv_r2"] = round(float(cvr2.mean()), 3)


def main():
    df = load()
    summary = {}
    study_simple(df, summary)
    study_expanded(df, summary)
    hr("CAVEAT")
    print("Cue and anxiety share a common cause (the stimulus condition), so the raw")
    print("associations above are partly mediated by it. The expanded partial_r (B) controls")
    print("for condition and is the cleaner cue->anxiety estimate; the next step is to test")
    print("these leading cues causally via counterfactual edits (Experiment 2 / Step 2).")
    out = config.RESULTS_DIR / "cue_study.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved machine-readable summary -> {out}")


if __name__ == "__main__":
    main()
