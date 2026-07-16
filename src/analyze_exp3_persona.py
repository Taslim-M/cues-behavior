"""Experiment 3 -- Stage 2: how do the verbalized PERSONAS shape the response?

Reads results/exp3_persona/<model>/<eval_condition>.jsonl (Stage 1) and answers,
in parallel to Experiment 2's Stage-2 analysis but for the persona prompt:

  (0) PERSONA INVENTORY -- what characters does the model actually select, which
      traits recur, and what numeric values does it assign them?  (This is the
      "understand the verbalized persona traits" deliverable.)

  (1) FACTOR EFFECTS -- how do user role, eval_condition, scenario and risk level
      (x_rank) move each judged DV (warmth / formality / advice_density) and the
      distribution of primary_emotion?  One-way ANOVA F + eta^2 per factor x DV.

  (2) PERSONA -> EMOTION -- treating the model's own verbalized persona (name +
      summary + triggered_by + trait names/expressions) as the IV, which words
      predict each DV?  Per-keyword association (Welch t, Cohen's d, BH-FDR) plus
      a cross-validated TF-IDF Ridge.  Plus mean DVs per selected persona, and the
      correlation between a trait's self-assigned value and the judged DVs.

Outputs results/exp3_persona/stage2_summary.json + figures/.

    python -m src.analyze_exp3_persona
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

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
FIGURES_DIR = config.EXP3_PERSONA_DIR / "figures"
RNG = 0


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def _persona_text(rec) -> str:
    """Concatenate the model's verbalized persona into one analysable string:
    persona_name + summary + triggered_by + every trait name + expression."""
    p = rec.get("persona") or {}
    parts = [str(p.get("persona_name", "")),
             str(p.get("persona_summary", "")),
             str(p.get("triggered_by", ""))]
    for t in (rec.get("traits") or []):
        parts.append(str(t.get("name", "")))
        parts.append(str(t.get("expression", "")))
    return " ".join(parts).lower().strip()


def _norm_persona_name(name: str) -> str:
    """Coarsen a persona label for the inventory (lowercase, collapse whitespace)."""
    return re.sub(r"\s+", " ", str(name).strip().lower()).strip("[]. ")


def load():
    """Flatten every Stage-1 record that carries a judge score into one frame,
    plus keep the raw traits list alongside for the persona inventory."""
    rows, trait_rows = [], []
    for jf in sorted(config.EXP3_PERSONA_DIR.glob("*/*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            j = r.get("judge")
            if not j:
                continue  # judging failed/skipped -> not usable for DV analysis
            pname = _norm_persona_name(r.get("persona_name", ""))
            rows.append({
                "model_name": r.get("model_name"),
                "role": r.get("role"), "role_axis": r.get("role_axis"),
                "eval_condition": r.get("eval_condition"), "scenario": r.get("scenario"),
                "domain": r.get("domain"), "valence": r.get("valence"),
                "x_rank": r.get("x_rank"), "x_value": r.get("x_value"),
                "warmth": j.get("warmth"), "formality": j.get("formality"),
                "advice_density": j.get("advice_density"),
                "primary_emotion": j.get("primary_emotion"),
                "persona_name": pname,
                "n_traits": len(r.get("traits") or []),
                "persona_text": _persona_text(r),
            })
            for t in (r.get("traits") or []):
                v = t.get("value")
                v = float(v) if isinstance(v, (int, float)) else np.nan
                trait_rows.append({
                    "model_name": r.get("model_name"),
                    "role": r.get("role"), "role_axis": r.get("role_axis"),
                    "persona_name": pname,
                    "trait_name": re.sub(r"\s+", " ", str(t.get("name", "")).strip().lower()),
                    "trait_value": v,
                    "warmth": j.get("warmth"), "formality": j.get("formality"),
                    "advice_density": j.get("advice_density"),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no judged Stage-1 records found under results/exp3_persona/*/*.jsonl")
    for c in DVS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    tdf = pd.DataFrame(trait_rows)
    return df.dropna(subset=DVS).reset_index(drop=True), tdf


# --------------------------------------------------------------------------- #
# (0) persona inventory
# --------------------------------------------------------------------------- #
def persona_inventory(df, tdf, summary):
    hr(f"(0) PERSONA INVENTORY   N = {len(df)} judged datapoints")
    print("Which characters does the model verbalize, and with which traits?\n")

    print(f"distinct persona labels: {df['persona_name'].nunique()}   "
          f"mean traits/persona: {df['n_traits'].mean():.2f}\n")

    top_personas = df["persona_name"].value_counts().head(20)
    print("Top selected personas (count, mean judged DVs):")
    print(f"  {'persona':<34}{'n':>5}{'warmth':>8}{'formal':>8}{'advice':>8}")
    persona_table = []
    for name, n in top_personas.items():
        sub = df[df["persona_name"] == name]
        w, f_, a = sub["warmth"].mean(), sub["formality"].mean(), sub["advice_density"].mean()
        print(f"  {name[:33]:<34}{n:>5}{w:>8.2f}{f_:>8.2f}{a:>8.2f}")
        persona_table.append({"persona": name, "n": int(n),
                              "warmth": round(float(w), 2),
                              "formality": round(float(f_), 2),
                              "advice_density": round(float(a), 2)})

    # most frequent trait names + their mean self-assigned value
    trait_summary = []
    if not tdf.empty:
        hr("Most frequent verbalized traits (count, mean self-assigned 0-10 value)")
        grp = (tdf.groupby("trait_name")
               .agg(n=("trait_name", "size"), mean_value=("trait_value", "mean"))
               .sort_values("n", ascending=False).head(25))
        print(f"  {'trait':<28}{'n':>5}{'mean_value':>12}")
        for name, row in grp.iterrows():
            if not name:
                continue
            print(f"  {name[:27]:<28}{int(row['n']):>5}{row['mean_value']:>12.2f}")
            trait_summary.append({"trait": name, "n": int(row["n"]),
                                  "mean_value": round(float(row["mean_value"]), 2)})

    summary["persona_inventory"] = {
        "n_distinct_personas": int(df["persona_name"].nunique()),
        "mean_traits_per_persona": round(float(df["n_traits"].mean()), 2),
        "top_personas": persona_table,
        "top_traits": trait_summary,
    }


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

    if "role" in df:
        hr("Role contrast (mean judged DVs by user role)")
        tbl = df.groupby("role")[DVS].mean().round(2).sort_values("warmth", ascending=False)
        print(tbl.to_string())


# --------------------------------------------------------------------------- #
# (1b) role -> verbalized trait association
# --------------------------------------------------------------------------- #
def role_trait_association(df, tdf, summary, top_k=15, min_cell=4):
    """Do particular user roles select particular load-bearing traits?

    Among datapoints that actually verbalized a persona (n_traits>0), for each
    (role, trait) we compute the selection rate P(trait | role) and the LIFT
    P(trait|role) / P(trait) -- lift>1 means the role over-selects that trait. A
    chi-square test per trait flags whether its selection rate depends on role.
    """
    if tdf.empty or "role" not in tdf:
        summary["role_trait"] = None
        return
    hr("(1b) ROLE -> VERBALIZED TRAIT   (which traits each user role selects)")

    # datapoints that emitted >=1 trait, per role (the honest denominator)
    emitted = df[df["n_traits"] > 0]
    n_role = emitted.groupby("role").size().to_dict()
    n_total = int(len(emitted))
    roles = sorted([r for r in n_role if n_role[r] >= 10])

    top_traits = [t for t in tdf["trait_name"].value_counts().head(top_k).index if t]
    # presence count per (role, trait): a trait appears >=once in a datapoint
    pres = (tdf[tdf["trait_name"].isin(top_traits)]
            .groupby(["role", "trait_name"]).size().unstack(fill_value=0))

    lift, share, distinctive, chi2 = {}, {}, {}, {}
    base = {t: int(tdf["trait_name"].eq(t).sum()) / n_total for t in top_traits}
    for role in roles:
        share[role], lift[role] = {}, {}
        for t in top_traits:
            c = int(pres.loc[role, t]) if (role in pres.index and t in pres.columns) else 0
            p_rt = c / n_role[role]
            share[role][t] = round(p_rt, 3)
            lift[role][t] = round(p_rt / base[t], 2) if base[t] > 0 else None
        # this role's distinctive traits: highest lift with enough support
        cand = [(t, lift[role][t], int(pres.loc[role, t]) if role in pres.index else 0)
                for t in top_traits if lift[role][t] is not None]
        cand = [(t, lf, c) for t, lf, c in cand if c >= min_cell]
        cand.sort(key=lambda x: -x[1])
        distinctive[role] = [{"trait": t, "lift": lf, "n": c} for t, lf, c in cand[:4]]

    # chi-square: is trait selection independent of role?
    for t in top_traits:
        yes = np.array([int(pres.loc[r, t]) if (r in pres.index and t in pres.columns) else 0
                        for r in roles], float)
        no = np.array([n_role[r] for r in roles], float) - yes
        table = np.vstack([yes, no])
        try:
            _, p, _, _ = stats.chi2_contingency(table)
        except ValueError:
            p = np.nan
        chi2[t] = float(p) if p == p else None

    # print a compact readout
    print(f"Among {n_total} persona-emitting datapoints; lift>1 = role favours the trait.\n")
    print("Most role-dependent traits (chi-square p):")
    for t, p in sorted(chi2.items(), key=lambda kv: (kv[1] is None, kv[1])):
        star = "*" if (p is not None and p < 0.05) else " "
        print(f"  {t[:26]:<27} p={p:.2g}{star}" if p is not None else f"  {t[:26]:<27} p=nan")
    print("\nDistinctive trait per role (top lift, n>=4):")
    for role in roles:
        ds = ", ".join(f"{d['trait']}({d['lift']}x,n={d['n']})" for d in distinctive[role][:3])
        print(f"  {role:<24} {ds}")

    summary["role_trait"] = {
        "traits": top_traits, "roles": roles,
        "base_rate": {t: round(base[t], 3) for t in top_traits},
        "n_role": {r: int(n_role[r]) for r in roles},
        "share": share, "lift": lift,
        "distinctive": distinctive, "chi2_p": chi2,
    }


# --------------------------------------------------------------------------- #
# (2) persona verbalization -> emotion
# --------------------------------------------------------------------------- #
def verbalization_to_emotion(df, tdf, summary):
    texts = df["persona_text"].tolist()
    N = len(df)
    hr(f"(2) PERSONA -> EMOTION   (verbalized persona text -> judged DV)   N = {N}")
    print("Which words in the model's verbalized persona predict each response DV?\n")

    min_df = max(8, int(0.05 * N))
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                         token_pattern=r"[A-Za-z]{3,}", min_df=min_df, binary=True)
    try:
        Xb = cv.fit_transform(texts).toarray()
    except ValueError as e:  # empty vocabulary -> not enough data yet
        print(f"  [skipped] insufficient data for bag-of-words (N={N}): {e}")
        summary["verbalization_to_emotion"] = None
    else:
        terms = np.array(cv.get_feature_names_out())
        print(f"vocabulary: {len(terms)} terms with doc-freq >= {min_df}\n")

        tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                             token_pattern=r"[A-Za-z]{3,}", min_df=min_df)
        Xt = tf.fit_transform(texts)
        tnames = np.array(tf.get_feature_names_out())

        per_dv = {}
        for dv in DVS:
            y = df[dv].to_numpy(float)
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
                print(f"  top +{dv} persona words (Cohen d): "
                      + ", ".join(f"{r.term}({r.cohen_d:+.2f})" for _, r in kw.head(6).iterrows()))
            print()

            per_dv[dv] = {
                "ridge_cv_r2": round(float(cvr2.mean()), 3),
                "ridge_top_terms": tnames[top].tolist(),
                "ridge_bottom_terms": tnames[bot].tolist(),
                "keywords_top": kw.head(12).round(4).to_dict("records") if not kw.empty else [],
            }
        summary["verbalization_to_emotion"] = per_dv

    # (2b) does a trait's SELF-ASSIGNED value correlate with the judged DV?
    # Pool across personas for the recurrent, semantically clear traits.
    if not tdf.empty:
        hr("Trait self-value -> judged DV   (Pearson r over datapoints carrying the trait)")
        trait_corr = []
        common = (tdf.dropna(subset=["trait_value"])
                  .groupby("trait_name").size().sort_values(ascending=False))
        common = [t for t in common.index if t][:15]
        print(f"  {'trait':<26}{'n':>5}" + "".join(f"{dv[:8]:>10}" for dv in DVS))
        for tname in common:
            sub = tdf[(tdf["trait_name"] == tname)].dropna(subset=["trait_value"])
            if len(sub) < 10:
                continue
            cell = {"trait": tname, "n": int(len(sub))}
            line = f"  {tname[:25]:<26}{len(sub):>5}"
            for dv in DVS:
                if sub[dv].nunique() < 2 or sub["trait_value"].nunique() < 2:
                    r = np.nan
                else:
                    r = stats.pearsonr(sub["trait_value"], sub[dv].astype(float))[0]
                cell[f"r_{dv}"] = round(float(r), 3) if r == r else None
                line += f"{r:>10.2f}" if r == r else f"{'nan':>10}"
            print(line)
            trait_corr.append(cell)
        summary["trait_value_corr"] = trait_corr


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def make_figures(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fac in ("role", "eval_condition"):
        if fac not in df or df[fac].nunique() < 2:
            continue
        tbl = df.groupby(fac)[DVS].mean()
        ax = tbl.plot(kind="bar", figsize=(max(6, 0.7 * len(tbl)), 4))
        ax.set_ylabel("mean judged score (0-10)")
        ax.set_title(f"Judged response DVs by {fac} (persona prompt)")
        plt.tight_layout()
        out = FIGURES_DIR / f"dv_by_{fac}.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  saved {out}")

    # headline: mean DVs for the most-selected personas
    top = df["persona_name"].value_counts().head(12).index.tolist()
    sub = df[df["persona_name"].isin(top)]
    if not sub.empty:
        tbl = sub.groupby("persona_name")[DVS].mean().reindex(top)
        ax = tbl.plot(kind="barh", figsize=(8, max(4, 0.5 * len(tbl))))
        ax.set_xlabel("mean judged score (0-10)")
        ax.set_title("Judged response DVs by selected persona")
        plt.tight_layout()
        out = FIGURES_DIR / "dv_by_persona.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"  saved {out}")


def main():
    df, tdf = load()
    summary = {"n": int(len(df)),
               "models": sorted(df["model_name"].dropna().unique().tolist())}
    persona_inventory(df, tdf, summary)
    factor_effects(df, summary)
    role_trait_association(df, tdf, summary)
    verbalization_to_emotion(df, tdf, summary)
    make_figures(df)
    out = config.EXP3_PERSONA_DIR / "stage2_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved machine-readable summary -> {out}")


if __name__ == "__main__":
    main()
