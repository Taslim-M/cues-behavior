"""Coupling-effect comparison: does self-modeling change inference / behaviour?

We generated the same x-sweep prompts three ways:

  C = coupled       : <inference> + <response> in one turn (Stage 1, run 0)
  I = inference-only : the four contextual cues, elicited alone (decoupled)
  R = bare response  : the reply to the plain prompt, NO self-modeling (decoupled)

Joining C, I and R 1:1 on (model, prompt_id) lets us ask two questions:

  * Inferences   : does asking for a reply too (C) change the verbalized cues vs
                   eliciting the inference alone (I)?           -> compare C.inf vs I.inf
  * Responses    : does having first produced an inference (C) change the reply's
                   emotional content vs a bare reply (R)?       -> compare C.judge vs R.judge

For each joined triple we compute judged-metric deltas (warmth / formality /
advice_density, coupled − bare), primary-emotion agreement, and TF-IDF cosine
similarity of the response texts and of the inference texts. Aggregated by
scenario / role / model / eval_condition / x_rank.

    python -m src.decoupled_compare      # prints the comparison, saves summary json
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
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .run_exp2_cues import GEN_MODELS

DVS = ["warmth", "formality", "advice_density"]
INF_FIELDS = ["about_user", "about_context", "about_stakes", "register_selected"]
DECOUPLED_DIR = config.EXP2_CUES_DIR / "decoupled"
EMBCOS_PATH = config.EXP2_CUES_DIR / "decoupled_embcos.json"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"

_EMB = None


def _emb_model():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer(EMB_MODEL_NAME)
    return _EMB


def _embed_row_cosine(texts_a, texts_b):
    """Row-wise *semantic* cosine via sentence embeddings (0 if either side empty)."""
    model = _emb_model()
    Ea = model.encode(list(texts_a), batch_size=256, normalize_embeddings=True,
                      show_progress_bar=False)
    Eb = model.encode(list(texts_b), batch_size=256, normalize_embeddings=True,
                      show_progress_bar=False)
    cos = (np.asarray(Ea) * np.asarray(Eb)).sum(axis=1)  # normalized -> dot == cosine
    empty = np.array([not str(a).strip() or not str(b).strip()
                      for a, b in zip(texts_a, texts_b)])
    return np.where(empty, 0.0, cos)


# --------------------------------------------------------------------------- #
# load + join
# --------------------------------------------------------------------------- #
def _inf_text(inf):
    return " ".join(str((inf or {}).get(f, "")) for f in INF_FIELDS).strip()


def _load_coupled():
    """Stage-1 run-0 records keyed (model, prompt_id)."""
    out = {}
    for m in GEN_MODELS:
        for jf in sorted((config.EXP2_CUES_DIR / m).glob("*.jsonl")):
            for line in jf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("run") != 0 or not r.get("judge"):
                    continue
                out[(m, r["prompt_id"])] = r
    return out


def _load_decoupled():
    out = {}
    for m in GEN_MODELS:
        for jf in sorted((DECOUPLED_DIR / m).glob("*.jsonl")):
            for line in jf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r["bare_response"].get("judge"):
                    continue
                out[(m, r["prompt_id"])] = r
    return out


def _row_cosine(texts_a, texts_b):
    """Row-wise TF-IDF cosine similarity between two aligned text lists."""
    vec = TfidfVectorizer(stop_words="english", token_pattern=r"[A-Za-z]{3,}", min_df=3)
    vec.fit(texts_a + texts_b)
    A, B = vec.transform(texts_a), vec.transform(texts_b)
    num = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    da = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    db = np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel())
    den = da * db
    return np.where(den > 0, num / np.where(den == 0, 1, den), 0.0)


def compute_and_save_embcos(df=None):
    """Compute semantic (embedding) cosine per field + overall; cache to disk.

    Keyed 'model|prompt_id'. Run once (or after new data); the report and
    load_joined() then read the cache instead of re-encoding ~50k texts.
    """
    if df is None:
        df = load_joined(with_emb=False)
    overall = _embed_row_cosine(
        [_inf_text(x) for x in df["c_inf"]], [_inf_text(x) for x in df["d_inf"]])
    per_field = {}
    for f in INF_FIELDS:
        per_field[f] = _embed_row_cosine(
            [str((x or {}).get(f, "")) for x in df["c_inf"]],
            [str((x or {}).get(f, "")) for x in df["d_inf"]])
    cache = {}
    keys = (df["model_name"] + "|" + df["prompt_id"]).tolist()
    for i, k in enumerate(keys):
        cache[k] = {"emb_cos": round(float(overall[i]), 4),
                    **{f"emb_cos_{f}": round(float(per_field[f][i]), 4) for f in INF_FIELDS}}
    EMBCOS_PATH.write_text(json.dumps(cache), encoding="utf-8")
    print(f"saved embedding cosines for {len(cache)} pairs ({EMB_MODEL_NAME}) -> {EMBCOS_PATH}")
    return cache


def load_joined(with_emb=True):
    """One row per (model, prompt_id) present in BOTH coupled and decoupled."""
    C, D = _load_coupled(), _load_decoupled()
    keys = sorted(set(C) & set(D))
    rows = []
    for (m, pid) in keys:
        c, d = C[(m, pid)], D[(m, pid)]
        cj, dj = c["judge"], d["bare_response"]["judge"]
        rows.append({
            "model_name": m, "prompt_id": pid,
            "role": c.get("role"), "role_axis": c.get("role_axis"),
            "scenario": c.get("scenario"), "domain": c.get("domain"),
            "eval_condition": c.get("eval_condition"), "x_rank": c.get("x_rank"),
            "user": c.get("user"),
            # coupled
            "c_inf": c.get("inference", {}), "c_resp": c.get("response", ""),
            "c_warmth": cj["warmth"], "c_formality": cj["formality"],
            "c_advice_density": cj["advice_density"], "c_emotion": cj["primary_emotion"],
            # decoupled
            "d_inf": d["inference_only"].get("inference", {}),
            "r_resp": d["bare_response"].get("response", ""),
            "r_warmth": dj["warmth"], "r_formality": dj["formality"],
            "r_advice_density": dj["advice_density"], "r_emotion": dj["primary_emotion"],
        })
    df = pd.DataFrame(rows)
    # judged deltas (coupled − bare): >0 => self-modeling raises this DV
    for dv in DVS:
        df[f"d_{dv}"] = df[f"c_{dv}"] - df[f"r_{dv}"]
    df["emotion_match"] = df["c_emotion"] == df["r_emotion"]
    # content similarity
    df["resp_cosine"] = _row_cosine(df["c_resp"].tolist(), df["r_resp"].tolist())
    df["inf_cosine"] = _row_cosine(
        [_inf_text(x) for x in df["c_inf"]], [_inf_text(x) for x in df["d_inf"]])
    df["c_resp_len"] = df["c_resp"].str.len()
    df["r_resp_len"] = df["r_resp"].str.len()
    # per-field inference similarity + length (coupled 'c' vs inference-only 'd')
    for f in INF_FIELDS:
        a = [str((x or {}).get(f, "")) for x in df["c_inf"]]
        b = [str((x or {}).get(f, "")) for x in df["d_inf"]]
        df[f"infcos_{f}"] = _row_cosine(a, b)
        df[f"clen_{f}"] = [len(t) for t in a]
        df[f"dlen_{f}"] = [len(t) for t in b]
    # merge cached semantic (embedding) cosines if available
    if with_emb and EMBCOS_PATH.exists():
        cache = json.loads(EMBCOS_PATH.read_text(encoding="utf-8"))
        keys = df["model_name"] + "|" + df["prompt_id"]
        df["emb_cosine"] = keys.map(lambda k: cache.get(k, {}).get("emb_cos", np.nan))
        for f in INF_FIELDS:
            df[f"embcos_{f}"] = keys.map(lambda k: cache.get(k, {}).get(f"emb_cos_{f}", np.nan))
    return df


# --------------------------------------------------------------------------- #
# inference-stability analysis (coupled inference vs inference-only)
# --------------------------------------------------------------------------- #
def field_stability(df):
    """Per inference field: lexical + semantic cosine, and mean lengths."""
    rows = []
    for f in INF_FIELDS:
        row = {
            "field": f,
            "cosine": round(float(df[f"infcos_{f}"].mean()), 3),
            "coupled_len": int(df[f"clen_{f}"].mean()),
            "infonly_len": int(df[f"dlen_{f}"].mean()),
        }
        if f"embcos_{f}" in df:
            row["emb_cosine"] = round(float(df[f"embcos_{f}"].mean()), 3)
        rows.append(row)
    return rows


def lexical_shift(df, field, topn=10, min_df=20):
    """Terms whose use in `field` leans coupled vs inference-only (log-odds)."""
    from sklearn.feature_extraction.text import CountVectorizer
    a = [str((x or {}).get(field, "")).lower() for x in df["c_inf"]]
    b = [str((x or {}).get(field, "")).lower() for x in df["d_inf"]]
    cv = CountVectorizer(stop_words="english", token_pattern=r"[A-Za-z]{3,}",
                         ngram_range=(1, 2), min_df=min_df)
    X = cv.fit_transform(a + b)
    terms = np.array(cv.get_feature_names_out())
    na = len(a)
    ca = np.asarray(X[:na].sum(axis=0)).ravel().astype(float)
    cb = np.asarray(X[na:].sum(axis=0)).ravel().astype(float)
    ta, tb = ca.sum(), cb.sum()
    lo = (np.log((ca + 0.5) / (ta - ca + 0.5)) - np.log((cb + 0.5) / (tb - cb + 0.5)))
    order = np.argsort(lo)
    coupled_more = [terms[i] for i in order[::-1][:topn]]
    infonly_more = [terms[i] for i in order[:topn]]
    return coupled_more, infonly_more


def link_corr(df):
    """Does a bigger inference change go with a bigger response change?"""
    jdiv = df[[f"d_{dv}" for dv in DVS]].abs().sum(axis=1).to_numpy(float)
    r1, p1 = stats.pearsonr(df["inf_cosine"], df["resp_cosine"])
    r2, p2 = stats.pearsonr(df["inf_cosine"], jdiv)
    return {"inf_vs_resp_cosine": {"r": round(float(r1), 3), "p": float(p1)},
            "inf_vs_judge_divergence": {"r": round(float(r2), 3), "p": float(p2)}}


# --------------------------------------------------------------------------- #
# aggregates
# --------------------------------------------------------------------------- #
def paired_summary(df):
    """Overall paired comparison (coupled vs bare) per judged DV."""
    out = {}
    for dv in DVS:
        a, b = df[f"c_{dv}"].to_numpy(float), df[f"r_{dv}"].to_numpy(float)
        t, p = stats.ttest_rel(a, b)
        sd = (a - b).std(ddof=1)
        out[dv] = {"coupled_mean": round(float(a.mean()), 3),
                   "bare_mean": round(float(b.mean()), 3),
                   "delta": round(float((a - b).mean()), 3),
                   "cohen_dz": round(float((a - b).mean() / sd), 3) if sd else None,
                   "t": round(float(t), 2), "p": float(p)}
    out["emotion_agreement"] = round(float(df["emotion_match"].mean()), 3)
    out["resp_cosine_mean"] = round(float(df["resp_cosine"].mean()), 3)
    out["inf_cosine_mean"] = round(float(df["inf_cosine"].mean()), 3)
    out["n"] = int(len(df))
    return out


def by_factor(df, factor):
    """Per-level: mean judged delta per DV, emotion agreement, response/inference cosine."""
    g = df.groupby(factor)
    tbl = g[[f"d_{dv}" for dv in DVS]].mean()
    tbl["emotion_agree"] = g["emotion_match"].mean()
    tbl["resp_cos"] = g["resp_cosine"].mean()
    tbl["inf_cos"] = g["inf_cosine"].mean()
    return tbl.round(3)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true",
                    help="(re)compute + cache semantic embedding cosines, then report")
    a = ap.parse_args()
    if a.embed:
        compute_and_save_embcos()

    df = load_joined()
    summary = {"overall": paired_summary(df)}
    if "emb_cosine" in df:
        summary["overall"]["inf_emb_cosine_mean"] = round(float(df["emb_cosine"].mean()), 3)
        print("Inference stability — lexical (TF-IDF) vs semantic (embedding) cosine:")
        for r in field_stability(df):
            print(f"  {r['field']:<20} lexical {r['cosine']:.2f}   semantic {r.get('emb_cosine','?')}")
        print(f"  {'OVERALL':<20} lexical {df['inf_cosine'].mean():.2f}   "
              f"semantic {df['emb_cosine'].mean():.2f}\n")
    for fac in ["model_name", "scenario", "role", "eval_condition", "x_rank", "domain"]:
        summary[f"by_{fac}"] = by_factor(df, fac).reset_index().to_dict("records")

    o = summary["overall"]
    print(f"Coupling effect  (coupled = self-modeling present, bare = absent)   N = {o['n']} pairs\n")
    print("Judged response deltas (coupled − bare):")
    for dv in DVS:
        s = o[dv]
        star = "*" if s["p"] < 0.05 else " "
        print(f"  {dv:<16} coupled {s['coupled_mean']:5.2f}  bare {s['bare_mean']:5.2f}  "
              f"Δ {s['delta']:+.2f}{star}  dz={s['cohen_dz']}")
    print(f"\n  primary-emotion agreement : {o['emotion_agreement']:.0%}")
    print(f"  response text cosine (C vs R): {o['resp_cosine_mean']:.2f}")
    print(f"  inference text cosine (C vs I): {o['inf_cosine_mean']:.2f}")

    print("\nBy model (judged Δ coupled−bare):")
    print(by_factor(df, "model_name")[[f"d_{dv}" for dv in DVS] + ["emotion_agree", "resp_cos", "inf_cos"]].to_string())

    out = config.EXP2_CUES_DIR / "decoupled_compare_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
