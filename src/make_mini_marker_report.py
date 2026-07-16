"""Build an interactive HTML for EXPERIMENT 4 -- the MINI-MARKER design comparison.

Closed-vocabulary (40-word Saucier Mini-Marker) analogue of make_design_compare_report.
(Data still lives under results/exp3_mini_* on disk; "Experiment 4" is the label.)

Samples where the model did not produce a usable persona-and-reply (empty reply or
no committed Mini-Marker persona) are counted as REJECTIONS, reported in the
Overview, and dropped from every comparison (pairwise: a prompt is removed if
EITHER design rejected).

COUPLED  (results/exp3_mini_coupled/<model>/*.jsonl)
    the model writes its <persona> (mini-marker traits + AB5C facets) and then,
    conditioned on it, its <response>, in ONE generation -> "paired traits" +
    in-context reply.
SOLO     (results/exp3_mini_solo/<model>/*.jsonl)
    the mini-marker persona is elicited ALONE (v1 framing), and the reply is
    generated separately with NO self-modeling scaffold -> "solo traits" + cold
    reply. (One framing only -- no v1/v3 averaging.)
FAITH    (results/exp3_mini_faith/<model>/all.jsonl)
    judge_faithfulness_prompt_mini_marker scored by TWO judges
    (claude-sonnet-5 + gpt-5.6-luna), numeric scores AVERAGED.

Paired on prompt_id (run 0). Questions:
  1. TRAITS   -- does the closed-vocab self-model differ coupled vs solo?
  2. RESPONSE -- does behaviour differ in-context vs cold?
  3. FAITH    -- how faithfully does each reply express its committed Mini-Marker
                 persona (fidelity / purity / overall), and does coupling help?

    python -m src.make_mini_marker_report   ->  analysis_mini_marker.html
"""
from __future__ import annotations

import glob
import json
import math
import re
from collections import Counter, defaultdict

from . import config
from .exp3_mini_judge import JUDGE_MINI_SYSTEM, build_judge_user, format_mini_profile

OUT = config.ROOT / "analysis_experiment_4.html"

DVS = ("warmth", "formality", "advice_density")
FAITH_KEYS = ("fidelity_score", "purity_score", "overall_score")
VERDICTS = ("FAITHFUL", "UNDER", "OVER", "ABSENT", "INVERTED")

# --------------------------------------------------------------------------- #
# Mini-Marker closed vocabulary -> (factor, polarity)
# --------------------------------------------------------------------------- #
_MARKERS = {
    "I": {"+": ["Bold", "Energetic", "Extraverted", "Talkative"],
          "-": ["Bashful", "Quiet", "Shy", "Withdrawn"]},
    "II": {"+": ["Cooperative", "Kind", "Sympathetic", "Warm"],
           "-": ["Cold", "Harsh", "Rude", "Unsympathetic"]},
    "III": {"+": ["Efficient", "Organized", "Practical", "Systematic"],
            "-": ["Careless", "Disorganized", "Inefficient", "Sloppy"]},
    "IV": {"+": ["Relaxed", "Unenvious"],
           "-": ["Envious", "Fretful", "Jealous", "Moody", "Temperamental", "Touchy"]},
    "V": {"+": ["Complex", "Creative", "Deep", "Imaginative", "Intellectual", "Philosophical"],
          "-": ["Uncreative", "Unintellectual"]},
}
FACTORS = ("I", "II", "III", "IV", "V")
FACTOR_LABEL = {"I": "I Extraversion", "II": "II Agreeableness",
                "III": "III Conscientiousness", "IV": "IV Emot. Stability",
                "V": "V Intellect/Openness"}
MARKER_MAP = {}          # normalized marker -> (factor, polarity)
for _f, _pol in _MARKERS.items():
    for _p, _words in _pol.items():
        for _w in _words:
            MARKER_MAP[_w.lower()] = (_f, _p)


def classify(name):
    """Return (factor, polarity) for a trait name, matching a Mini-Marker word
    even if the model wrote extra text. None if outside the closed vocabulary."""
    n = re.sub(r"[^a-z]+", " ", str(name).lower()).strip()
    if n in MARKER_MAP:
        return MARKER_MAP[n]
    for tok in n.split():
        if tok in MARKER_MAP:
            return MARKER_MAP[tok]
    return None


# --------------------------------------------------------------------------- #
# stats helpers
# --------------------------------------------------------------------------- #
def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pstdev(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def pearson(xs, ys):
    pairs = [(a, b) for a, b in zip(xs, ys) if a is not None and b is not None]
    n = len(pairs)
    if n < 2:
        return None
    mx = sum(a for a, _ in pairs) / n
    my = sum(b for _, b in pairs) / n
    num = sum((a - mx) * (b - my) for a, b in pairs)
    dx = math.sqrt(sum((a - mx) ** 2 for a, _ in pairs))
    dy = math.sqrt(sum((b - my) ** 2 for _, b in pairs))
    return num / (dx * dy) if dx > 0 and dy > 0 else None


def norm_name(name):
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def paired_wilcoxon(cs, ss):
    """Wilcoxon signed-rank test on paired (coupled, cold) scores.

    H0: the median of the within-prompt difference (cold - coupled) is 0. This is
    the right test for this metric: the faithfulness scores are paired by prompt,
    bounded 0-10, two-judge averages with strong ceiling effects, so a normal-
    theory paired t-test is not warranted -- a non-parametric paired test is.

    Zeros (ties) are dropped; |differences| are ranked with average ranks for ties
    and the variance is tie-corrected. p comes from the continuity-corrected normal
    approximation, which at n in the hundreds is indistinguishable from exact.
    Effect size = matched-pairs rank-biserial correlation r = (T+ - T-)/(T+ + T-);
    its SIGN gives the direction (negative => coupled scored higher than cold).
    Returns None if fewer than 6 non-tied pairs.
    """
    diffs = [b - a for a, b in zip(cs, ss) if a is not None and b is not None]
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n < 6:
        return None
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0        # 1-based average rank for the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    t_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    t_minus = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    t_tot = t_plus + t_minus
    exp = n * (n + 1) / 4.0
    tie_term = sum(c ** 3 - c for c in Counter(abs(d) for d in diffs).values())
    var = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var <= 0:
        return None
    z = (abs(t_plus - exp) - 0.5) / math.sqrt(var)     # continuity-corrected
    # two-sided p = 2*(1-Phi(|z|)) via erfc for tail accuracy (no underflow at large z)
    p = math.erfc(abs(z) / math.sqrt(2))
    r_rb = (t_plus - t_minus) / t_tot if t_tot else 0.0
    return {"test": "wilcoxon_signed_rank", "W": round(min(t_plus, t_minus), 1),
            "z": round(z, 3), "p": p, "n": n, "rank_biserial": round(r_rb, 3),
            "significant": bool(p < 0.05)}


# --------------------------------------------------------------------------- #
# loading + pairing
# --------------------------------------------------------------------------- #
def _load(dirpath):
    recs = []
    for jf in glob.glob(str(dirpath / "*.jsonl")):
        for line in open(jf, encoding="utf-8"):
            if line.strip():
                recs.append(json.loads(line))
    return recs


def _index_run0(recs):
    out = {}
    for r in recs:
        if r.get("run", 0) != 0:
            continue
        out.setdefault(r["prompt_id"], r)
    return out


def _load_faith(model):
    p = config.EXP3_MINI_FAITH_DIR / model / "all.jsonl"
    out = {}
    if p.exists():
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["prompt_id"]] = r
    return out


def models_with_both():
    have = []
    for m in config.MODELS:
        c = config.EXP3_MINI_COUPLED_DIR / m
        s = config.EXP3_MINI_SOLO_DIR / m
        if (c.exists() and s.exists() and glob.glob(str(c / "*.jsonl"))
                and glob.glob(str(s / "*.jsonl"))):
            have.append(m)
    return have


# --------------------------------------------------------------------------- #
# trait analysis (closed vocabulary)
# --------------------------------------------------------------------------- #
def traitlist(rec):
    return [{"name": t.get("name", ""), "facet": t.get("facet", ""),
             "value": _num(t.get("value"))}
            for t in (rec.get("traits") or []) if str(t.get("name", "")).strip()]


def trait_stats(list_of_traitlists):
    counts = [len(tl) for tl in list_of_traitlists]
    marker = Counter()
    marker_vals = defaultdict(list)
    factor = Counter()
    factor_vals = defaultdict(list)
    polarity = Counter()
    in_vocab = out_vocab = 0
    allvals = []
    for tl in list_of_traitlists:
        for t in tl:
            v = t["value"]
            if v is not None:
                allvals.append(v)
            cl = classify(t["name"])
            if cl is None:
                out_vocab += 1
                continue
            in_vocab += 1
            f, pol = cl
            key = re.sub(r"[^a-z]+", " ", t["name"].lower()).strip()
            key = key if key in MARKER_MAP else next(
                (tok for tok in key.split() if tok in MARKER_MAP), key)
            marker[key] += 1
            factor[f] += 1
            polarity[pol] += 1
            if v is not None:
                marker_vals[key].append(v)
                factor_vals[f].append(v)
    top = [{"name": k.title(), "key": k, "factor": MARKER_MAP[k][0],
            "pol": MARKER_MAP[k][1], "count": c,
            "mean_val": round(mean(marker_vals[k]) or 0, 2)}
           for k, c in marker.most_common(24)]
    return {
        "n_dp": len(list_of_traitlists),
        "mean_count": round(mean(counts) or 0, 2),
        "value_mean": round(mean(allvals) or 0, 2),
        "value_sd": round(pstdev(allvals), 2),
        "in_vocab_rate": round(in_vocab / (in_vocab + out_vocab), 3) if (in_vocab + out_vocab) else None,
        "n_out_vocab": out_vocab,
        "factor_dist": {f: factor.get(f, 0) for f in FACTORS},
        "factor_mean_val": {f: round(mean(factor_vals[f]) or 0, 2) for f in FACTORS},
        "polarity": {"+": polarity.get("+", 0), "-": polarity.get("-", 0)},
        "top": top,
        "_marker": marker, "_marker_vals": marker_vals,
    }


def factor_profile_means(recs):
    acc = {f: [] for f in FACTORS}
    for r in recs:
        fp = r.get("factor_profile") or {}
        for f in FACTORS:
            if f in fp:
                acc[f].append(_num(fp[f]))
    return {f: round(mean(acc[f]) or 0, 2) if acc[f] else None for f in FACTORS}


# --------------------------------------------------------------------------- #
# faith helpers
# --------------------------------------------------------------------------- #
def _avg(fr, side):
    return ((fr.get(side) or {}).get("avg")) or {}


def _judges(fr, side):
    return (fr.get(side) or {}).get("judges") or {}


def _verdict_counts(fr, side):
    """Sum per-trait verdicts across both judges for one side."""
    c = Counter()
    for jm in _judges(fr, side).values():
        j = jm.get("judge") or {}
        for t in j.get("trait_evaluations") or []:
            v = t.get("verdict")
            if v in VERDICTS:
                c[v] += 1
    return c


def _per_trait(fr, side, acc):
    """Fold each judge's per-trait (name, stated, observed, verdict) into acc."""
    for jm in _judges(fr, side).values():
        j = jm.get("judge") or {}
        for t in j.get("trait_evaluations") or []:
            cl = classify(t.get("name"))
            if cl is None:
                continue
            key = re.sub(r"[^a-z]+", " ", str(t.get("name")).lower()).strip()
            key = key if key in MARKER_MAP else next(
                (tok for tok in key.split() if tok in MARKER_MAP), key)
            a = acc[key]
            a["n"] += 1
            s, o = _num(t.get("stated_value")), _num(t.get("observed_value"))
            if s is not None:
                a["stated"].append(s)
            if o is not None:
                a["obs"].append(o)
            if t.get("verdict") in VERDICTS:
                a["verdict"][t["verdict"]] += 1


def _n_scored_traits(rec):
    return sum(1 for t in (rec.get("traits") or []) if _num(t.get("value")) is not None)


def is_rejection(rec):
    """A sample is a 'rejection' (model didn't respond usefully) if it produced no
    behavioural reply OR no committed Mini-Marker persona (0 numeric-valued traits).
    Such a datapoint cannot enter the trait / response / faithfulness comparison."""
    if not (rec.get("response") or "").strip():
        return True
    if _n_scored_traits(rec) == 0:
        return True
    return False


def _msgs_to_pair(messages):
    """[{role,content}] -> {'system':..., 'user':...} for display (last of each)."""
    out = {"system": "", "user": ""}
    for m in messages or []:
        if m.get("role") in out:
            out[m["role"]] = m.get("content", "")
    return out


def build_showcase(kept, coupled, solo, faith_idx):
    """One example prompt showing the EXACT prompts fed to the model: the coupled
    generation prompt, the solo generation prompts (persona elicitation + cold
    reply), and the faithfulness-judge prompt (system + assembled user turn)."""
    if not kept:
        return None
    # prefer a clean 'deployment' prompt (no eval frame) with a full persona+reply
    def score(pid):
        c = coupled[pid]
        return (c.get("eval_condition") == "deployment",
                _n_scored_traits(c) >= 4, len((c.get("response") or "")) > 250)
    pid = max(kept, key=score)
    c, s = coupled[pid], solo[pid]
    coup = _msgs_to_pair(c.get("messages"))
    solo_p = _msgs_to_pair(s.get("elicit_messages"))
    solo_r = _msgs_to_pair(s.get("response_messages"))
    # reconstruct the judge prompt actually used for this prompt's coupled side
    prof, _ = format_mini_profile(c.get("persona") or {}, c.get("traits") or [],
                                  c.get("persona_name", ""), c.get("factor_profile") or {})
    judge_user = build_judge_user(c.get("user", ""), prof, c.get("response", ""))
    return {
        "prompt_id": pid, "scenario": c.get("scenario"), "role": c.get("role"),
        "condition": c.get("eval_condition"), "user": c.get("user", ""),
        "coupled": {"system": coup["system"], "user": coup["user"]},
        "solo_persona": {"system": solo_p["system"], "user": solo_p["user"]},
        "solo_response": {"system": solo_r["system"], "user": solo_r["user"]},
        "judge": {"system": JUDGE_MINI_SYSTEM, "user": judge_user},
    }


def analyse_model(model):
    coupled = _index_run0(_load(config.EXP3_MINI_COUPLED_DIR / model))
    solo = _index_run0(_load(config.EXP3_MINI_SOLO_DIR / model))
    faith_idx = _load_faith(model)
    all_shared = sorted(set(coupled) & set(solo))
    rej_c = [pid for pid in all_shared if is_rejection(coupled[pid])]
    rej_s = [pid for pid in all_shared if is_rejection(solo[pid])]
    rej_any = set(rej_c) | set(rej_s)
    shared = [pid for pid in all_shared if pid not in rej_any]
    rejection = {
        "total": len(all_shared), "coupled": len(rej_c), "solo": len(rej_s),
        "dropped": len(rej_any), "retained": len(shared),
        "pct_dropped": round(100 * len(rej_any) / len(all_shared), 1) if all_shared else 0.0,
    }
    showcase = build_showcase(shared, coupled, solo, faith_idx)

    c_tl, s_tl, c_recs, s_recs = [], [], [], []
    dv_c = {k: [] for k in DVS}
    dv_s = {k: [] for k in DVS}
    scn_delta = defaultdict(lambda: {k: [] for k in DVS})
    cond_delta = defaultdict(lambda: {k: [] for k in DVS})
    role_delta = defaultdict(lambda: {k: [] for k in DVS})
    emo_c, emo_s = Counter(), Counter()
    len_c, len_s = [], []
    browse = []

    # faith accumulators
    fc = {k: [] for k in FAITH_KEYS}
    fk = {k: [] for k in FAITH_KEYS}
    fpair = {k: {"c": [], "s": []} for k in FAITH_KEYS}
    f_scn = defaultdict(lambda: {"c": [], "s": []})
    f_cond = defaultdict(lambda: {"c": [], "s": []})
    f_role = defaultdict(lambda: {"c": [], "s": []})
    inv_c, inv_s, leak_c, leak_s = [], [], [], []
    vc_c, vc_s = Counter(), Counter()
    judge_agree = {"claude": [], "gpt": []}   # overall_score per judge (pooled)
    fhist_c, fhist_s = [0] * 11, [0] * 11
    pt_c, pt_s = (defaultdict(lambda: {"n": 0, "stated": [], "obs": [], "verdict": Counter()}),
                  defaultdict(lambda: {"n": 0, "stated": [], "obs": [], "verdict": Counter()}))
    n_fc = n_fs = 0

    for pid in shared:
        rc, rs = coupled[pid], solo[pid]
        ct, st = traitlist(rc), traitlist(rs)
        c_tl.append(ct); s_tl.append(st); c_recs.append(rc); s_recs.append(rs)

        jc, js = rc.get("judge") or {}, rs.get("judge") or {}
        for k in DVS:
            a, b = _num(jc.get(k)), _num(js.get(k))
            if a is not None and b is not None:
                dv_c[k].append(a); dv_s[k].append(b)
                scn_delta[rc["scenario"]][k].append(b - a)
                cond_delta[rc["eval_condition"]][k].append(b - a)
                role_delta[rc["role"]][k].append(b - a)
        if jc.get("primary_emotion"):
            emo_c[jc["primary_emotion"]] += 1
        if js.get("primary_emotion"):
            emo_s[js["primary_emotion"]] += 1
        len_c.append(len(rc.get("response", "")))
        len_s.append(len(rs.get("response", "")))

        fr = faith_idx.get(pid)
        cav, sav = (_avg(fr, "coupled"), _avg(fr, "cold")) if fr else ({}, {})
        if fr and cav:
            n_fc += 1
            for k in FAITH_KEYS:
                if cav.get(k) is not None:
                    fc[k].append(cav[k])
            if cav.get("n_inversions") is not None:
                inv_c.append(cav["n_inversions"])
            if cav.get("n_leakage") is not None:
                leak_c.append(cav["n_leakage"])
            if cav.get("overall_score") is not None:
                fhist_c[int(round(cav["overall_score"]))] += 1
            vc_c += _verdict_counts(fr, "coupled")
            _per_trait(fr, "coupled", pt_c)
        if fr and sav:
            n_fs += 1
            for k in FAITH_KEYS:
                if sav.get(k) is not None:
                    fk[k].append(sav[k])
            if sav.get("n_inversions") is not None:
                inv_s.append(sav["n_inversions"])
            if sav.get("n_leakage") is not None:
                leak_s.append(sav["n_leakage"])
            if sav.get("overall_score") is not None:
                fhist_s[int(round(sav["overall_score"]))] += 1
            vc_s += _verdict_counts(fr, "cold")
            _per_trait(fr, "cold", pt_s)
        if fr and cav and sav:
            for k in FAITH_KEYS:
                a, b = cav.get(k), sav.get(k)
                if a is not None and b is not None:
                    fpair[k]["c"].append(a); fpair[k]["s"].append(b)
            oc, os_ = cav.get("overall_score"), sav.get("overall_score")
            if oc is not None and os_ is not None:
                f_scn[rc["scenario"]]["c"].append(oc); f_scn[rc["scenario"]]["s"].append(os_)
                f_cond[rc["eval_condition"]]["c"].append(oc); f_cond[rc["eval_condition"]]["s"].append(os_)
                f_role[rc["role"]]["c"].append(oc); f_role[rc["role"]]["s"].append(os_)
        # judge-vs-judge agreement (pool coupled + cold), overall_score
        if fr:
            for side in ("coupled", "cold"):
                jm = _judges(fr, side)
                cj = (jm.get("anthropic/claude-sonnet-5") or {}).get("judge") or {}
                gj = (jm.get("openai/gpt-5.6-luna") or {}).get("judge") or {}
                a, b = _num(cj.get("overall_score")), _num(gj.get("overall_score"))
                if a is not None and b is not None:
                    judge_agree["claude"].append(a); judge_agree["gpt"].append(b)

        browse.append({
            "prompt_id": pid, "scenario": rc["scenario"], "role": rc["role"],
            "condition": rc["eval_condition"], "user": rc.get("user", ""),
            "c_name": rc.get("persona_name", ""),
            "c_traits": ct, "c_factor": rc.get("factor_profile") or {},
            "c_resp": rc.get("response", ""),
            "c_dv": {k: jc.get(k) for k in DVS + ("primary_emotion",)},
            "s_name": rs.get("persona_name", ""),
            "s_traits": st, "s_factor": rs.get("factor_profile") or {},
            "s_resp": rs.get("response", ""),
            "s_dv": {k: js.get(k) for k in DVS + ("primary_emotion",)},
            "faith": _compact_faith(fr) if fr else None,
        })

    cstat, sstat = trait_stats(c_tl), trait_stats(s_tl)

    # vocab overlap on the closed set (markers used >=3x in a design)
    THRESH = 3
    cset = {k for k, v in cstat["_marker"].items() if v >= THRESH}
    sset = {k for k, v in sstat["_marker"].items() if v >= THRESH}
    union = cset | sset
    jaccard = round(len(cset & sset) / len(union), 3) if union else 0.0

    # shared markers value comparison (>=5x both)
    shared_vals = []
    for k in cset & sset:
        cn, sn = len(cstat["_marker_vals"][k]), len(sstat["_marker_vals"][k])
        if cn >= 5 and sn >= 5:
            cm, sm = mean(cstat["_marker_vals"][k]), mean(sstat["_marker_vals"][k])
            shared_vals.append({"name": k.title(), "coupled_mean": round(cm, 2),
                                "coupled_n": cn, "solo_mean": round(sm, 2),
                                "solo_n": sn, "delta": round(sm - cm, 2)})
    shared_vals.sort(key=lambda d: -abs(d["delta"]))

    resp_dv = {}
    for k in DVS:
        deltas = [b - a for a, b in zip(dv_c[k], dv_s[k])]
        dsd = pstdev(deltas)
        resp_dv[k] = {
            "coupled_mean": round(mean(dv_c[k]) or 0, 2),
            "cold_mean": round(mean(dv_s[k]) or 0, 2),
            "delta_mean": round(mean(deltas) or 0, 2),
            "d_z": round((mean(deltas) or 0) / dsd, 2) if dsd > 0 else None,
            "pct_up": round(100 * sum(1 for d in deltas if d > 0) / len(deltas), 1) if deltas else None,
            "r": round(pearson(dv_c[k], dv_s[k]) or 0, 2),
            "n": len(deltas),
        }
    by_scn = [{"scenario": s, **{k: round(mean(scn_delta[s][k]) or 0, 2) for k in DVS},
               "n": len(scn_delta[s][DVS[0]])} for s in sorted(scn_delta)]
    by_cond = [{"condition": c, **{k: round(mean(cond_delta[c][k]) or 0, 2) for k in DVS},
                "n": len(cond_delta[c][DVS[0]])} for c in sorted(cond_delta)]
    by_role = [{"role": r, **{k: round(mean(role_delta[r][k]) or 0, 2) for k in DVS},
                "n": len(role_delta[r][DVS[0]])}
               for r in sorted(role_delta, key=lambda x: -len(role_delta[x][DVS[0]]))]

    # faith aggregates
    def paired_stat(cs, ss):
        deltas = [b - a for a, b in zip(cs, ss)]
        dsd = pstdev(deltas)
        return {"coupled_mean": round(mean(cs), 2) if cs else None,
                "cold_mean": round(mean(ss), 2) if ss else None,
                "coupled_se": round(pstdev(cs) / math.sqrt(len(cs)), 3) if len(cs) > 1 else None,
                "cold_se": round(pstdev(ss) / math.sqrt(len(ss)), 3) if len(ss) > 1 else None,
                "delta_mean": round(mean(deltas), 2) if deltas else None,
                "d_z": round(mean(deltas) / dsd, 2) if dsd > 0 else None,
                "pct_cold_higher": round(100 * sum(1 for d in deltas if d > 0) / len(deltas), 1) if deltas else None,
                "r": round(pearson(cs, ss), 2) if len(cs) > 1 and pearson(cs, ss) is not None else None,
                "n": len(deltas),
                "wilcoxon": paired_wilcoxon(cs, ss)}
    faith_paired = {k: paired_stat(fpair[k]["c"], fpair[k]["s"]) for k in FAITH_KEYS}
    faith_means = {k: {"coupled": round(mean(fc[k]), 2) if fc[k] else None, "coupled_n": len(fc[k]),
                       "cold": round(mean(fk[k]), 2) if fk[k] else None, "cold_n": len(fk[k])}
                   for k in FAITH_KEYS}
    f_by_scn = [{"scenario": s, "coupled": round(mean(f_scn[s]["c"]), 2),
                 "cold": round(mean(f_scn[s]["s"]), 2),
                 "delta": round(mean(f_scn[s]["s"]) - mean(f_scn[s]["c"]), 2),
                 "n": len(f_scn[s]["c"])} for s in sorted(f_scn)]
    f_by_cond = [{"condition": c, "coupled": round(mean(f_cond[c]["c"]), 2),
                  "cold": round(mean(f_cond[c]["s"]), 2),
                  "delta": round(mean(f_cond[c]["s"]) - mean(f_cond[c]["c"]), 2),
                  "n": len(f_cond[c]["c"])} for c in sorted(f_cond)]
    f_by_role = [{"role": r, "coupled": round(mean(f_role[r]["c"]), 2),
                  "cold": round(mean(f_role[r]["s"]), 2),
                  "delta": round(mean(f_role[r]["s"]) - mean(f_role[r]["c"]), 2),
                  "n": len(f_role[r]["c"])}
                 for r in sorted(f_role, key=lambda x: -len(f_role[x]["c"]))]
    jr = pearson(judge_agree["claude"], judge_agree["gpt"])
    jmad = mean([abs(a - b) for a, b in zip(judge_agree["claude"], judge_agree["gpt"])])

    def _pt_rows(pt, top=16):
        rows = []
        for k, a in pt.items():
            if a["n"] < 4:
                continue
            tot = sum(a["verdict"].values()) or 1
            rows.append({"name": k.title(), "factor": MARKER_MAP.get(k, ("?",))[0],
                         "n": a["n"], "stated": round(mean(a["stated"]) or 0, 2),
                         "obs": round(mean(a["obs"]) or 0, 2),
                         "gap": round((mean(a["stated"]) or 0) - (mean(a["obs"]) or 0), 2),
                         "faithful_pct": round(100 * a["verdict"]["FAITHFUL"] / tot),
                         "inverted_pct": round(100 * a["verdict"]["INVERTED"] / tot)})
        rows.sort(key=lambda r: -r["n"])
        return rows[:top]

    def _pt_comb(top=14):
        """Per-marker faithfulness paired coupled-vs-cold (markers scored >=3x in
        both designs). 'faithful' = share of that marker's per-trait judgments
        rated FAITHFUL; 'obs' = mean observed intensity."""
        rows = []
        for k in set(pt_c) | set(pt_s):
            ac, as_ = pt_c.get(k), pt_s.get(k)
            nc = ac["n"] if ac else 0
            ns = as_["n"] if as_ else 0
            if nc < 3 or ns < 3:
                continue

            def fpct(a):
                tot = sum(a["verdict"].values()) or 1
                return round(100 * a["verdict"]["FAITHFUL"] / tot)
            rows.append({"name": k.title(), "factor": MARKER_MAP.get(k, ("?",))[0],
                         "n_coupled": nc, "n_cold": ns,
                         "faithful_coupled": fpct(ac), "faithful_cold": fpct(as_),
                         "obs_coupled": round(mean(ac["obs"]) or 0, 2),
                         "obs_cold": round(mean(as_["obs"]) or 0, 2)})
        rows.sort(key=lambda r: -(r["n_coupled"] + r["n_cold"]))
        return rows[:top]

    faith = {
        "available": bool(faith_idx),
        "n_coupled": n_fc, "n_cold": n_fs,
        "n_paired": faith_paired["overall_score"]["n"],
        "means": faith_means, "paired": faith_paired,
        "hist": {"coupled": fhist_c, "cold": fhist_s},
        "by_scenario": f_by_scn, "by_condition": f_by_cond, "by_role": f_by_role,
        "verdicts": {"coupled": {v: vc_c.get(v, 0) for v in VERDICTS},
                     "cold": {v: vc_s.get(v, 0) for v in VERDICTS}},
        "inversion": {"coupled": round(mean(inv_c), 3) if inv_c else None,
                      "cold": round(mean(inv_s), 3) if inv_s else None},
        "leakage": {"coupled": round(mean(leak_c), 3) if leak_c else None,
                    "cold": round(mean(leak_s), 3) if leak_s else None},
        "judge_agreement": {"r": round(jr, 3) if jr is not None else None,
                            "mean_abs_diff": round(jmad, 2) if jmad is not None else None,
                            "n": len(judge_agree["claude"]),
                            "models": config.MINI_JUDGE_MODELS},
        "scatter": [[a, b] for a, b in zip(fpair["overall_score"]["c"], fpair["overall_score"]["s"])],
        "per_trait_coupled": _pt_rows(pt_c), "per_trait_cold": _pt_rows(pt_s),
        "per_trait_combined": _pt_comb(),
    }

    def strip(st):
        return {k: v for k, v in st.items() if not k.startswith("_")}

    return {
        "meta": {"n_pairs": len(shared),
                 "rejection": rejection,
                 "scenarios": sorted({b["scenario"] for b in browse}),
                 "roles": sorted({b["role"] for b in browse}),
                 "conditions": sorted({b["condition"] for b in browse})},
        "showcase": showcase,
        "traits": {
            "coupled": strip(cstat), "solo": strip(sstat),
            "vocab_jaccard": jaccard, "shared_values": shared_vals,
            "factor_profile": {"coupled": factor_profile_means(c_recs),
                               "solo": factor_profile_means(s_recs)},
        },
        "responses": {"dv": resp_dv, "by_scenario": by_scn, "by_condition": by_cond,
                      "by_role": by_role,
                      "emotion": {"coupled": dict(emo_c), "solo": dict(emo_s)},
                      "length": {"coupled_chars": round(mean(len_c) or 0),
                                 "solo_chars": round(mean(len_s) or 0)}},
        "faith": faith,
        "browse": browse,
    }


def _compact_faith(fr):
    def side(s):
        av = _avg(fr, s)
        jm = _judges(fr, s)
        out = {"avg": {k: av.get(k) for k in FAITH_KEYS + ("n_inversions", "n_leakage")},
               "judges": {}}
        for model_id, blk in jm.items():
            j = blk.get("judge")
            short = "claude-sonnet-5" if "claude" in model_id else "gpt-5.6-luna"
            if j:
                out["judges"][short] = {k: j.get(k) for k in FAITH_KEYS}
                out["judges"][short]["rationale"] = (j.get("rationale") or "")[:400]
                out["judges"][short]["gestalt"] = (j.get("gestalt_match") or "")[:300]
                out["judges"][short]["traits"] = [
                    {"name": t.get("name"), "stated": t.get("stated_value"),
                     "obs": t.get("observed_value"), "verdict": t.get("verdict"),
                     "facet": t.get("facet_check")}
                    for t in (j.get("trait_evaluations") or [])][:6]
            else:
                out["judges"][short] = {"error": (blk.get("error") or "")[:120]}
        return out
    return {"coupled": side("coupled"), "cold": side("cold")}


# --------------------------------------------------------------------------- #
def build_payload():
    models = models_with_both()
    payload = {"models": {}, "factor_label": FACTOR_LABEL,
               "judges": config.MINI_JUDGE_MODELS}
    for m in models:
        payload["models"][m] = analyse_model(m)
    return payload


def build_html(payload):
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE.replace("/*DATA*/", data_json)


def main():
    payload = build_payload()
    if not payload["models"]:
        raise SystemExit("no models have BOTH exp3_mini_coupled and exp3_mini_solo results yet.")
    OUT.write_text(build_html(payload), encoding="utf-8")
    ms = list(payload["models"])
    print(f"wrote {OUT}  ({len(ms)} models: {', '.join(ms)})")
    for m in ms:
        d = payload["models"][m]
        f = d["faith"]
        rj = d["meta"]["rejection"]
        print(f"  {m}: {d['meta']['n_pairs']} retained pairs "
              f"(rejected: coupled {rj['coupled']}, solo {rj['solo']}, "
              f"dropped {rj['dropped']}/{rj['total']}); "
              f"faith overall coupled={f['means']['overall_score']['coupled']} "
              f"cold={f['means']['overall_score']['cold']}")


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 4 — Mini-Marker Design Comparison</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/svg2pdf.js@2.2.3/dist/svg2pdf.umd.min.js"></script>
<style>
:root{--c:#2563eb;--s:#db2777;--bg:#f8fafc;--ink:#0f172a;--mut:#64748b;--bd:#e2e8f0;--card:#fff;}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg);}
header{background:linear-gradient(120deg,#1e293b,#0f172a);color:#fff;padding:26px 30px;}
header h1{margin:0 0 4px;font-size:22px}
header p{margin:0;color:#cbd5e1;font-size:14px}
.legend{display:flex;gap:18px;margin-top:12px;font-size:13px;color:#e2e8f0}
.legend b{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:middle}
.wrap{max-width:1180px;margin:0 auto;padding:22px 26px 80px}
.controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:18px 0 6px}
select{padding:7px 10px;border:1px solid var(--bd);border-radius:8px;background:#fff;font-size:14px}
label{font-size:13px;color:var(--mut);font-weight:600;margin-right:4px}
nav{display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--bd);margin:14px 0 20px}
nav button{border:0;background:none;padding:10px 14px;font-size:14px;color:var(--mut);cursor:pointer;border-bottom:2px solid transparent}
nav button.active{color:var(--c);border-bottom-color:var(--c);font-weight:600}
.tab{display:none}.tab.active{display:block}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:18px 20px;margin:16px 0;}
.card h3{margin:0 0 4px;font-size:16px}
.card .sub{color:var(--mut);font-size:13px;margin:0 0 14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
@media(max-width:900px){.grid2,.grid3{grid-template-columns:1fr}}
.stat{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px 16px}
.stat .n{font-size:24px;font-weight:700}.stat .l{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.stat .d{font-size:13px;color:var(--mut);margin-top:3px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 9px;border-bottom:1px solid var(--bd);text-align:left}
th{color:var(--mut);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:#16a34a;font-weight:600}.neg{color:#dc2626;font-weight:600}
.pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:12px;font-weight:600}
.pc{background:#dbeafe;color:#1d4ed8}.ps{background:#fce7f3;color:#be185d}
.two{display:grid;grid-template-columns:1fr 1fr;gap:0;border:1px solid var(--bd);border-radius:12px;overflow:hidden}
.two > div{padding:14px 16px}
.two .h{font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:.03em;margin-bottom:8px}
.col-c{background:#f5f8ff;border-right:1px solid var(--bd)}
.col-s{background:#fff7fb}
.resp{white-space:pre-wrap;font-size:13px;background:#fff;border:1px solid var(--bd);border-radius:8px;padding:10px;max-height:340px;overflow:auto}
.tr{display:flex;justify-content:space-between;gap:8px;padding:2px 0;border-bottom:1px dashed var(--bd);font-size:13px}
.tr .v{font-variant-numeric:tabular-nums;color:var(--mut)}
.muted{color:var(--mut)}
.fct{display:inline-block;font-size:11px;font-weight:700;padding:0 5px;border-radius:4px;margin-right:4px}
.f1{background:#fee2e2;color:#b91c1c}.f2{background:#dcfce7;color:#15803d}.f3{background:#dbeafe;color:#1d4ed8}
.f4{background:#fef9c3;color:#a16207}.f5{background:#f3e8ff;color:#7e22ce}
.vd{display:inline-block;font-size:11px;font-weight:700;padding:0 6px;border-radius:4px}
.vF{background:#dcfce7;color:#15803d}.vU{background:#fef9c3;color:#a16207}.vO{background:#ffedd5;color:#c2410c}
.vA{background:#e2e8f0;color:#475569}.vI{background:#fee2e2;color:#b91c1c}
details{margin-top:8px}summary{cursor:pointer;color:var(--c);font-size:13px}
.note{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:12px 14px;font-size:13px;color:#78350f;margin:14px 0}
.cap{font-size:12.5px;color:#475569;background:#f8fafc;border-left:3px solid #cbd5e1;padding:7px 11px;margin-top:10px;border-radius:0 6px 6px 0;line-height:1.5}
.cap b{color:#334155}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}
.defs{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:6px 0 14px}
.defs>div{background:#f8fafc;border:1px solid var(--bd);border-radius:8px;padding:9px 11px;font-size:13px;line-height:1.5}
.defs .t{font-weight:700;color:var(--c);margin-right:4px}
@media(max-width:760px){.defs{grid-template-columns:1fr}}
.pbox{white-space:pre-wrap;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;background:#0f172a;color:#e2e8f0;border-radius:8px;padding:12px 14px;max-height:360px;overflow:auto;margin:6px 0 2px}
.btn{border:1px solid var(--bd);background:#fff;color:var(--c);padding:6px 12px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer}
.btn:hover{background:#f1f5f9}
.pdf-btn{position:absolute;top:4px;right:6px;z-index:6;border:1px solid var(--bd);background:rgba(255,255,255,.92);color:var(--c);font-size:11px;font-weight:700;padding:1px 7px;border-radius:6px;cursor:pointer;line-height:1.5}
.pdf-btn:hover{background:#eff6ff}
.figtools{display:flex;justify-content:flex-end;margin:0 0 6px}
.plabel{font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:var(--mut);margin-top:12px}
.pmeta{font-size:12px;color:var(--mut);margin:2px 0 6px}
details.show summary{font-size:14px;font-weight:600;color:var(--c);padding:6px 0}
</style></head>
<body>
<header>
  <h1>Experiment 4 — Mini-Marker Design Comparison</h1>
  <p>Closed-vocabulary (40-word Saucier Mini-Marker) persona <b>+ reply in one gen</b> (coupled) vs persona <b>elicited alone</b> + <b>cold</b> reply (solo). Faithfulness judged by two models, averaged.</p>
  <div class="legend">
    <span><b style="background:#2563eb"></b>Coupled — mini-marker persona + reply in one generation (traits in-context)</span>
    <span><b style="background:#db2777"></b>Solo — persona alone (v1); reply generated cold</span>
  </div>
</header>
<div class="wrap">
  <div class="controls">
    <div><label>Model</label><select id="model"></select></div>
    <div class="muted" id="pairinfo"></div>
  </div>
  <nav id="nav"></nav>
  <div id="tab-overview" class="tab"></div>
  <div id="tab-traits" class="tab"></div>
  <div id="tab-responses" class="tab"></div>
  <div id="tab-faith" class="tab"></div>
  <div id="tab-examples" class="tab"></div>
  <div id="tab-method" class="tab"></div>
</div>
<script>
const DATA = /*DATA*/;
const DVS = ["warmth","formality","advice_density"];
const FKEYS = ["fidelity_score","purity_score","overall_score"];
const FACTORS = ["I","II","III","IV","V"];
const VERDICTS = ["FAITHFUL","UNDER","OVER","ABSENT","INVERTED"];
const COL_C="#2563eb", COL_S="#db2777";
let M = null;

const $ = s => document.querySelector(s);
const esc = s => (s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const fmt = (x,d=2)=> x==null?"—":(typeof x==="number"?x.toFixed(d):x);
const sign = x => x==null?"—":(x>0?("+"+x.toFixed(2)):x.toFixed(2));
const cls = x => x>0?"pos":(x<0?"neg":"");
const fcls = f => ({"I":"f1","II":"f2","III":"f3","IV":"f4","V":"f5"}[f]||"");
const vcls = v => ({FAITHFUL:"vF",UNDER:"vU",OVER:"vO",ABSENT:"vA",INVERTED:"vI"}[v]||"");
const MODELCOL = ["#2563eb","#db2777","#0d9488","#d97706","#7c3aed"];
const pfmt = p => p==null?"—":(p<1e-4?"&lt;0.0001":(p<0.001?p.toExponential(1):p.toFixed(4)));
const SUMMARY_KEY = "__ALL__";

const TABS=[["overview","Overview"],["traits","Traits: coupled vs solo"],
  ["responses","Responses: in-context vs cold"],["faith","Faithfulness"],
  ["examples","Paired examples"],["method","Method"]];

function initModel(){
  const sel=$("#model"); sel.innerHTML="";
  const so=document.createElement("option");so.value=SUMMARY_KEY;so.textContent="★ All models — summary";sel.appendChild(so);
  Object.keys(DATA.models).forEach(m=>{const o=document.createElement("option");o.value=m;o.textContent=m;sel.appendChild(o);});
  sel.onchange=renderAll;
  sel.value=SUMMARY_KEY;   // land on the cross-model summary
}
function initNav(){
  const nav=$("#nav"); nav.innerHTML="";
  TABS.forEach(([id,label],i)=>{const b=document.createElement("button");b.textContent=label;
    b.onclick=()=>{document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
      document.querySelectorAll("nav button").forEach(x=>x.classList.remove("active"));
      $("#tab-"+id).classList.add("active");b.classList.add("active");};
    if(i===0)b.classList.add("active");nav.appendChild(b);});
}
function renderAll(){
  const val=$("#model").value;
  if(val===SUMMARY_KEY){
    const nm=Object.keys(DATA.models).length;
    $("#pairinfo").textContent=`Summary across ${nm} models — the most important take-away per tab`;
    summaryOverview();summaryTraits();summaryResponses();summaryFaith();
    $("#tab-examples").innerHTML=`<div class="note">Paired examples are per-model. Pick a single model from the dropdown to browse them.</div>`;
    renderMethod();
    setTimeout(addPdfButtons,60);
    return;
  }
  M=DATA.models[val];
  $("#pairinfo").textContent=`${M.meta.n_pairs} paired prompts · ${M.meta.scenarios.length} scenarios · ${M.meta.roles.length} roles · ${M.meta.conditions.length} conditions`;
  renderOverview();renderTraits();renderResponses();renderFaith();renderExamples();renderMethod();
  setTimeout(addPdfButtons,60);
}

/* ---------- cross-model summary (dropdown: ★ All models) ---------- */
const summaryModels = () => Object.keys(DATA.models);

function summaryOverview(){
  const models=summaryModels();
  const el=$("#tab-overview");
  el.innerHTML=`<div class="card"><h3>All four models at a glance</h3>
    <p class="sub">The headline result: a reply written together with its persona (<b style="color:${COL_C}">coupled</b>) is more faithful to that persona than a cold reply (<b style="color:${COL_S}">solo</b>). Δ = cold − coupled; negative ⇒ coupling helped.</p>
    <div id="so_faith"></div>
    <table id="so_tbl"></table>
    <div class="cap">Coupled beats cold on overall faithfulness for <b>every</b> model, and the paired Wilcoxon test is significant (p&lt;0.05) in all four.</div>
  </div>`;
  Plotly.newPlot("so_faith",[
    {x:models,y:models.map(m=>DATA.models[m].faith.paired.overall_score.coupled_mean),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:models,y:models.map(m=>DATA.models[m].faith.paired.overall_score.cold_mean),name:"cold",type:"bar",marker:{color:COL_S}},
  ],{title:{text:"Overall faithfulness (0–10), coupled vs cold",font:{size:14}},barmode:"group",height:360,
     margin:{t:36,r:10,b:80,l:50},legend:{orientation:"h",y:1.13},yaxis:{range:[0,10]},font:{size:12}},{displayModeBar:false,responsive:true});
  $("#so_tbl").innerHTML=`<thead><tr><th>model</th><th class="num">pairs</th><th class="num">dropped</th>
    <th class="num">in-vocab c/s</th><th class="num">overall coupled</th><th class="num">cold</th><th class="num">Δ</th><th class="num">Wilcoxon p</th><th>sig</th></tr></thead><tbody>`+
    models.map(m=>{const d=DATA.models[m];const o=d.faith.paired.overall_score;const w=o.wilcoxon;const rj=d.meta.rejection;const t=d.traits;
      return `<tr><td>${esc(m)}</td><td class="num">${d.meta.n_pairs}</td><td class="num">${rj.dropped}/${rj.total}</td>
        <td class="num">${fmt(100*(t.coupled.in_vocab_rate||0),0)}% / ${fmt(100*(t.solo.in_vocab_rate||0),0)}%</td>
        <td class="num">${fmt(o.coupled_mean)}</td><td class="num">${fmt(o.cold_mean)}</td><td class="num ${cls(o.delta_mean)}">${sign(o.delta_mean)}</td>
        <td class="num">${w?pfmt(w.p):"—"}</td><td>${w?(w.significant?'<span class="pos">yes</span>':'<span class="neg">no</span>'):"—"}</td></tr>`;}).join("")+`</tbody>`;
}

function summaryResponses(){
  const models=summaryModels();
  const el=$("#tab-responses");
  el.innerHTML=`<div class="card"><h3>How coupling changes response style — across all 4 models</h3>
    <p class="sub">For each judged DV, Δ = cold − coupled (positive ⇒ the cold/solo reply scored higher; negative ⇒ coupling raised it). One bar group per DV, one colour per model — so you can see whether the coupling effect points the same way across models.</p>
    <div id="sr_dv"></div>
    <table id="sr_tbl"></table>
    <div class="cap" id="sr_cap"></div>
  </div>`;
  const traces=models.map((m,i)=>({x:DVS,y:DVS.map(k=>DATA.models[m].responses.dv[k].delta_mean),
    name:m,type:"bar",marker:{color:MODELCOL[i%MODELCOL.length]}}));
  Plotly.newPlot("sr_dv",traces,{title:{text:"Δ (cold − coupled) per DV, by model",font:{size:14}},
    barmode:"group",height:380,margin:{t:36,r:10,b:56,l:54},legend:{orientation:"h",y:1.14},
    yaxis:{title:"Δ (cold − coupled)",zeroline:true,zerolinecolor:"#94a3b8"},font:{size:12}},{displayModeBar:false,responsive:true});
  $("#sr_tbl").innerHTML=`<thead><tr><th>model</th>${DVS.map(k=>`<th class="num">${k} Δ</th>`).join("")}</tr></thead><tbody>`+
    models.map(m=>`<tr><td>${esc(m)}</td>${DVS.map(k=>{const d=DATA.models[m].responses.dv[k].delta_mean;return `<td class="num ${cls(d)}">${sign(d)}</td>`;}).join("")}</tr>`).join("")+`</tbody>`;
  $("#sr_cap").innerHTML="<b>Direction agreement across models</b> — "+DVS.map(k=>{
    const ds=models.map(m=>DATA.models[m].responses.dv[k].delta_mean);
    const up=ds.filter(d=>d>0).length,dn=ds.filter(d=>d<0).length;
    const tag=up===models.length?`cold higher in all ${models.length}`:dn===models.length?`coupled higher in all ${models.length}`:`${up}↑ / ${dn}↓`;
    return `<b>${k}</b>: ${tag}`;}).join(" · ");
}

function summaryFaith(){
  const models=summaryModels().filter(m=>DATA.models[m].faith&&DATA.models[m].faith.available);
  const el=$("#tab-faith");
  if(!models.length){el.innerHTML=`<div class="note">No faithfulness judgments on disk yet.</div>`;return;}
  el.innerHTML=`<div class="card"><h3>Faithfulness — coupled vs cold across all 4 models</h3>
    <p class="sub">Averaged over two judges. Δ = cold − coupled (negative ⇒ coupling more faithful). Significance from the paired Wilcoxon signed-rank test on overall faithfulness.</p>
    <div id="sf_bar"></div>
    <table id="sf_tbl"></table>
    <div class="cap">Same pattern in every model: overall faithfulness is significantly higher coupled than cold (see p and the medium-to-large rank-biserial effect sizes).</div>
  </div>`;
  Plotly.newPlot("sf_bar",[
    {x:models,y:models.map(m=>DATA.models[m].faith.paired.overall_score.coupled_mean),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:models,y:models.map(m=>DATA.models[m].faith.paired.overall_score.cold_mean),name:"cold",type:"bar",marker:{color:COL_S}},
  ],{title:{text:"Overall faithfulness (0–10), coupled vs cold",font:{size:14}},barmode:"group",height:360,
     margin:{t:36,r:10,b:80,l:50},legend:{orientation:"h",y:1.13},yaxis:{range:[0,10]},font:{size:12}},{displayModeBar:false,responsive:true});
  const FK=[["fidelity_score","fidelity"],["purity_score","purity"],["overall_score","overall"]];
  $("#sf_tbl").innerHTML=`<thead><tr><th>model</th>${FK.map(x=>`<th class="num">${x[1]} Δ</th>`).join("")}<th class="num">overall p</th><th class="num">rank-biserial</th><th>sig</th></tr></thead><tbody>`+
    models.map(m=>{const P=DATA.models[m].faith.paired;const w=P.overall_score.wilcoxon;
      return `<tr><td>${esc(m)}</td>${FK.map(x=>`<td class="num ${cls(P[x[0]].delta_mean)}">${sign(P[x[0]].delta_mean)}</td>`).join("")}
        <td class="num">${w?pfmt(w.p):"—"}</td><td class="num ${w?cls(w.rank_biserial):''}">${w?fmt(w.rank_biserial,3):"—"}</td>
        <td>${w?(w.significant?'<span class="pos">yes</span>':'<span class="neg">no</span>'):"—"}</td></tr>`;}).join("")+`</tbody>`;
}

function summaryTraits(){
  const models=summaryModels();
  const el=$("#tab-traits");
  el.innerHTML=`<div class="card"><h3>Staying inside the closed vocabulary — across all 4 models</h3>
    <p class="sub">Share of declared traits that are valid Saucier Mini-Marker words (the model was told to use only those). Coupled vs solo, per model — a check that the closed-set instruction was followed either way.</p>
    <div id="st_inv"></div>
    <table id="st_tbl"></table>
    <div class="cap">All models stay in-vocabulary the large majority of the time, and the coupled and solo elicitations follow the closed set about equally.</div>
  </div>`;
  Plotly.newPlot("st_inv",[
    {x:models,y:models.map(m=>100*(DATA.models[m].traits.coupled.in_vocab_rate||0)),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:models,y:models.map(m=>100*(DATA.models[m].traits.solo.in_vocab_rate||0)),name:"solo",type:"bar",marker:{color:COL_S}},
  ],{title:{text:"In-vocabulary rate (%)",font:{size:14}},barmode:"group",height:340,
     margin:{t:36,r:10,b:80,l:50},legend:{orientation:"h",y:1.13},yaxis:{range:[0,100]},font:{size:12}},{displayModeBar:false,responsive:true});
  $("#st_tbl").innerHTML=`<thead><tr><th>model</th><th class="num">in-vocab coupled</th><th class="num">in-vocab solo</th><th class="num">traits/datapoint coupled</th><th class="num">traits/datapoint solo</th></tr></thead><tbody>`+
    models.map(m=>{const t=DATA.models[m].traits;return `<tr><td>${esc(m)}</td>
      <td class="num">${fmt(100*(t.coupled.in_vocab_rate||0),0)}%</td><td class="num">${fmt(100*(t.solo.in_vocab_rate||0),0)}%</td>
      <td class="num">${fmt(t.coupled.mean_count)}</td><td class="num">${fmt(t.solo.mean_count)}</td></tr>`;}).join("")+`</tbody>`;
}
function stat(n,l,d){return `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div>${d?`<div class="d">${d}</div>`:""}</div>`;}

/* ---------- overview ---------- */
function renderOverview(){
  const t=M.traits, r=M.responses, F=M.faith;
  let cards=`<div class="grid3">
    ${stat(M.meta.n_pairs,"paired prompts","same prompt_id, run 0, both designs")}
    ${stat(t.coupled.mean_count+" / "+t.solo.mean_count,"traits per datapoint","coupled / solo")}
    ${stat(fmt(100*(t.coupled.in_vocab_rate||0),0)+"% / "+fmt(100*(t.solo.in_vocab_rate||0),0)+"%","in-vocabulary","valid Mini-Marker names, coupled / solo")}
  </div>`;
  const dv=r.dv;
  let dvrows = DVS.map(k=>`<tr><td>${k}</td>
    <td class="num">${fmt(dv[k].coupled_mean)}</td><td class="num">${fmt(dv[k].cold_mean)}</td>
    <td class="num ${cls(dv[k].delta_mean)}">${sign(dv[k].delta_mean)}</td>
    <td class="num">${fmt(dv[k].d_z)}</td><td class="num">${fmt(dv[k].r)}</td></tr>`).join("");
  const headline = `<div class="card"><h3>Does writing a persona first change the behaviour?</h3>
    <p class="sub">Judged DVs on the SAME prompts. Δ = cold − coupled (positive ⇒ the cold reply scores higher). d<sub>z</sub> = paired effect size; r = cross-design correlation.</p>
    <table><thead><tr><th>DV</th><th class="num">coupled</th><th class="num">cold</th><th class="num">Δ</th><th class="num">d<sub>z</sub></th><th class="num">r</th></tr></thead>
    <tbody>${dvrows}</tbody></table></div>`;
  let faithCard="";
  if(F && F.available){
    const o=F.paired.overall_score;
    faithCard=`<div class="card"><h3>Is the reply faithful to its Mini-Marker persona — and does coupling matter?</h3>
      <p class="sub">Two judges (${esc(F.judge_agreement.models.join(" + "))}) score reply-vs-persona; numbers averaged. Δ = cold − coupled.</p>
      <div class="grid3">
        ${stat(fmt(o.coupled_mean)+" → "+fmt(o.cold_mean),"overall faithfulness","coupled → cold (paired, n="+o.n+")")}
        ${stat(sign(o.delta_mean),"Δ faithfulness","d_z "+fmt(o.d_z)+" · cold higher on "+fmt(o.pct_cold_higher,0)+"% of prompts")}
        ${stat(fmt(F.judge_agreement.r),"judge agreement r","claude-5 vs gpt-5.6, |Δ|="+fmt(F.judge_agreement.mean_abs_diff))}
      </div>
      <p class="sub" style="margin-top:8px">Full breakdown in the <b>Faithfulness</b> tab.</p></div>`;
  }
  // rejection stat
  const rj=M.meta.rejection;
  const rejCard=`<div class="card"><h3>Rejected samples — model did not respond usefully</h3>
    <p class="sub">A sample is <b>rejected</b> when the model produced no reply, or no committed Mini-Marker persona (0 scored traits). Rejected prompts are <b>excluded</b> from every comparison on this page — a prompt is dropped if <b>either</b> design rejected it, so trait / response / faithfulness numbers are all on the same retained set.</p>
    <div class="grid3">
      ${stat(rj.coupled,"coupled rejections","of "+rj.total+" prompts")}
      ${stat(rj.solo,"solo rejections","of "+rj.total+" prompts")}
      ${stat(rj.retained+" kept","after exclusion",rj.dropped+" dropped ("+fmt(rj.pct_dropped,1)+"% of "+rj.total+")")}
    </div></div>`;
  // exact-prompts showcase (collapsed)
  let showCard="";
  const S=M.showcase;
  if(S){
    const box=(label,meta,sys,usr)=>`<div class="plabel">${label}</div>${meta?`<div class="pmeta">${meta}</div>`:""}
      <div class="pbox"><b style="color:#93c5fd">[system]</b>\n${esc(sys||"(none)")}\n\n<b style="color:#93c5fd">[user]</b>\n${esc(usr)}</div>`;
    showCard=`<div class="card"><details class="show"><summary>▸ Show the exact prompts used (one example: ${esc(S.prompt_id)} · ${esc(S.scenario)} · ${esc(S.role)} · ${esc(S.condition)})</summary>
      <div class="cap" style="margin-top:10px"><b>User message</b> (identical across all three generation calls):</div>
      <div class="pbox">${esc(S.user)}</div>
      ${box("Generation — Version 1: COUPLED (persona + reply in one generation)","system = Mini-Marker persona prompt with a &lt;response&gt; block appended",S.coupled.system,S.coupled.user)}
      ${box("Generation — Version 2a: SOLO persona elicitation (stop after &lt;/persona&gt;)","the mini-marker persona is elicited alone",S.solo_persona.system,S.solo_persona.user)}
      ${box("Generation — Version 2b: SOLO cold reply (no self-modeling scaffold)","separate call; only the eval frame (if any) is kept as system",S.solo_response.system,S.solo_response.user)}
      ${box("Faithfulness judge (run through BOTH judge models)","system = judge_faithfulness_prompt_mini_marker; user = PROFILE + USER QUERY + RESPONSE (shown here for the COUPLED reply of this prompt)",S.judge.system,S.judge.user)}
    </details></div>`;
  }
  // plain-language primer (first thing in the Overview)
  const VOCAB={
    I:["Bold, Energetic, Extraverted, Talkative","Bashful, Quiet, Shy, Withdrawn"],
    II:["Cooperative, Kind, Sympathetic, Warm","Cold, Harsh, Rude, Unsympathetic"],
    III:["Efficient, Organized, Practical, Systematic","Careless, Disorganized, Inefficient, Sloppy"],
    IV:["Relaxed, Unenvious","Envious, Fretful, Jealous, Moody, Temperamental, Touchy"],
    V:["Complex, Creative, Deep, Imaginative, Intellectual, Philosophical","Uncreative, Unintellectual"]};
  const facRows=FACTORS.map(f=>`<tr>
      <td><span class="fct ${fcls(f)}">${f}</span>${esc(DATA.factor_label[f].replace(/^[IV]+ /,""))}</td>
      <td class="pos">+ ${esc(VOCAB[f][0])}</td>
      <td class="neg">− ${esc(VOCAB[f][1])}</td></tr>`).join("");
  const primer=`<div class="card">
    <h3>What this experiment measures — a plain-language primer</h3>
    <p class="sub">Start here if you don't come from personality psychology.</p>
    <p style="font-size:14px;line-height:1.65">Before answering a user, the model is asked to <b>declare the character it is about to play</b> — a "persona" — and then we check whether its <b>actual reply lives up to that declared character</b>. That match is what we call <b>faithfulness</b>. A model can <i>say</i> it will be a warm, gentle listener and then reply like a cold, clipped expert; faithfulness catches that gap. We elicit the persona two ways:</p>
    <div class="grid2">
      <div class="stat"><div class="l" style="color:${COL_C}">Coupled</div><div style="font-size:13.5px;margin-top:4px">The model names its persona <b>and writes the reply in one go</b> — the declared traits sit right next to the answer.</div></div>
      <div class="stat"><div class="l" style="color:${COL_S}">Solo</div><div style="font-size:13.5px;margin-top:4px">The model names its persona <b>on its own</b>; the reply is written <b>separately and cold</b>, without the declared traits in front of it.</div></div>
    </div>
    <p style="font-size:14px;line-height:1.65;margin-top:12px">Comparing the two answers one question: does <b>declaring a character first actually steer behaviour</b>, or is it just talk?</p>

    <h3 style="margin-top:18px">The trait vocabulary — the "Mini-Markers"</h3>
    <p style="font-size:14px;line-height:1.65">To keep descriptions comparable, the model may only describe its persona with a <b>fixed list of 40 adjectives</b>: the Saucier (1994) <b>Mini-Markers</b>, a standard shorthand for the <b>Big Five</b> personality factors. Each factor runs between a <b class="pos">positive (+)</b> and a <b class="neg">negative (−)</b> pole:</p>
    <table><thead><tr><th>Factor</th><th class="pos">+ pole markers</th><th class="neg">− pole markers</th></tr></thead><tbody>${facRows}</tbody></table>

    <h3 style="margin-top:18px">How to read a declared trait</h3>
    <div class="defs">
      <div><span class="t">name</span> — one adjective from the list above, e.g. <b>Warm</b> (its "home" factor is II Agreeableness).</div>
      <div><span class="t">value 0–10</span> — how strongly that pole is expressed. 0 = not at all, 5 = clearly present, 10 = dominates the reply.</div>
      <div><span class="t">facet (AB5C blend)</span> — e.g. <code>II+/I+</code>: mainly Agreeableness-positive, <i>tinged with</i> Extraversion-positive → "warm in an outgoing way". <code>/pure</code> = no secondary colour.</div>
      <div><span class="t">factor_profile</span> — the persona's overall 0–10 standing on <b>all five</b> factors at once, for placing it in personality space.</div>
    </div>

    <h3 style="margin-top:6px">How to read the faithfulness scores (0–10)</h3>
    <div class="defs">
      <div><span class="t">fidelity</span> — are the declared traits actually in the reply, at the right pole and strength?</div>
      <div><span class="t">purity</span> — did the reply stay in character — no slips to the <i>opposite</i> pole (inversions) and no strong extra traits it never declared (leakage)?</div>
      <div><span class="t">overall</span> — one representativeness score; a single opposite-pole slip caps it low.</div>
      <div><span class="t">two judges, averaged</span> — every reply is scored by ${esc(DATA.judges.join(" and "))}; we report the average and their agreement.</div>
    </div>
    <div class="cap"><b>Colour key:</b> throughout, <span style="color:${COL_C};font-weight:700">blue = coupled</span>, <span style="color:${COL_S};font-weight:700">pink = solo/cold</span>. Factor chips <span class="fct f1">I</span><span class="fct f2">II</span><span class="fct f3">III</span><span class="fct f4">IV</span><span class="fct f5">V</span> colour every trait by its Big-Five factor.</div>
  </div>`;
  $("#tab-overview").innerHTML = primer + cards + rejCard + headline + faithCard + showCard +
    `<div class="note">Models with <b>both</b> designs on disk appear in the dropdown. Trait names are constrained to the 40-word Saucier Mini-Marker set; "in-vocabulary" measures how often the model stayed inside it.</div>`;
}

/* ---------- traits ---------- */
function barTwo(divid,title,cats,cVals,sVals,ytitle){
  Plotly.newPlot(divid,[
    {x:cats,y:cVals,name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:cats,y:sVals,name:"solo",type:"bar",marker:{color:COL_S}},
  ],{title:{text:title,font:{size:14}},barmode:"group",margin:{t:36,r:10,b:80,l:50},
     height:320,legend:{orientation:"h",y:1.15},yaxis:{title:ytitle},font:{size:12}},
     {displayModeBar:false,responsive:true});
}
function renderTraits(){
  const t=M.traits, c=t.coupled, s=t.solo;
  const el=$("#tab-traits");
  el.innerHTML=`
    <div class="card"><h3>Big Five factor distribution of selected traits</h3>
      <p class="sub">Share of committed traits that fall under each Big-Five factor (by the trait's home factor). Does coupling shift which factors the model foregrounds?</p>
      <div id="tf_factor"></div>
      <div class="cap"><b>Reading:</b> Agreeableness (II) and Openness (V) usually dominate helper personas; watch for coupled vs solo divergence.</div>
    </div>
    <div class="card"><h3>Most-used Mini-Markers</h3>
      <p class="sub">Frequency of each closed-vocabulary adjective, coupled vs solo, with the model's own mean 0–10 value.</p>
      <div id="tf_markers"></div>
      <table id="tf_marker_tbl"></table>
    </div>
    <div class="card"><h3>Polarity balance &amp; factor-profile coordinates</h3>
      <div class="grid2">
        <div id="tf_polarity"></div>
        <div id="tf_profile"></div>
      </div>
      <div class="cap"><b>factor_profile</b> = the persona's self-rated 0–10 standing on each factor (declared in the spec), averaged over datapoints.</div>
    </div>
    <div class="card"><h3>Shared markers: value shift coupled → solo</h3>
      <p class="sub">Markers used ≥5× in BOTH designs. Δ = solo − coupled mean value.</p>
      <table id="tf_shared"></table>
    </div>`;
  // factor distribution (normalized share)
  const cs=FACTORS.map(f=>c.factor_dist[f]), ss=FACTORS.map(f=>s.factor_dist[f]);
  const cn=cs.reduce((a,b)=>a+b,0)||1, sn=ss.reduce((a,b)=>a+b,0)||1;
  barTwo("tf_factor","Factor share (%)",FACTORS.map(f=>DATA.factor_label[f]),
    cs.map(x=>100*x/cn),ss.map(x=>100*x/sn),"% of traits");
  // top markers (union of coupled top)
  const names=c.top.slice(0,14).map(x=>x.name);
  const cmap=Object.fromEntries(c.top.map(x=>[x.name,x.count]));
  const smap=Object.fromEntries(s.top.map(x=>[x.name,x.count]));
  barTwo("tf_markers","Marker frequency (top 14 by coupled)",names,
    names.map(n=>cmap[n]||0),names.map(n=>smap[n]||0),"count");
  $("#tf_marker_tbl").innerHTML=`<thead><tr><th>marker</th><th>factor</th>
    <th class="num">coupled n</th><th class="num">coupled x̄</th>
    <th class="num">solo n</th><th class="num">solo x̄</th></tr></thead><tbody>`+
    c.top.slice(0,16).map(x=>{const sm=s.top.find(y=>y.key===x.key);
      return `<tr><td><span class="fct ${fcls(x.factor)}">${x.factor}${x.pol}</span>${esc(x.name)}</td>
      <td>${DATA.factor_label[x.factor]}</td><td class="num">${x.count}</td><td class="num">${fmt(x.mean_val)}</td>
      <td class="num">${sm?sm.count:0}</td><td class="num">${sm?fmt(sm.mean_val):"—"}</td></tr>`;}).join("")+`</tbody>`;
  // polarity
  Plotly.newPlot("tf_polarity",[
    {x:["+ pole","− pole"],y:[c.polarity["+"],c.polarity["-"]],name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:["+ pole","− pole"],y:[s.polarity["+"],s.polarity["-"]],name:"solo",type:"bar",marker:{color:COL_S}},
  ],{title:{text:"Trait polarity",font:{size:14}},barmode:"group",height:300,
     margin:{t:36,r:10,b:40,l:50},legend:{orientation:"h",y:1.2},font:{size:12}},{displayModeBar:false,responsive:true});
  // factor profile
  const fp=t.factor_profile;
  Plotly.newPlot("tf_profile",[
    {x:FACTORS,y:FACTORS.map(f=>fp.coupled[f]),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:FACTORS,y:FACTORS.map(f=>fp.solo[f]),name:"solo",type:"bar",marker:{color:COL_S}},
  ],{title:{text:"Declared factor_profile (0–10)",font:{size:14}},barmode:"group",height:300,
     margin:{t:36,r:10,b:40,l:50},legend:{orientation:"h",y:1.2},yaxis:{range:[0,10]},font:{size:12}},{displayModeBar:false,responsive:true});
  // shared table
  $("#tf_shared").innerHTML=`<thead><tr><th>marker</th><th class="num">coupled x̄</th><th class="num">solo x̄</th><th class="num">Δ</th><th class="num">n</th></tr></thead><tbody>`+
    (t.shared_values.length? t.shared_values.map(d=>`<tr><td>${esc(d.name)}</td>
      <td class="num">${fmt(d.coupled_mean)}</td><td class="num">${fmt(d.solo_mean)}</td>
      <td class="num ${cls(d.delta)}">${sign(d.delta)}</td><td class="num">${d.coupled_n}/${d.solo_n}</td></tr>`).join("")
      : `<tr><td colspan="5" class="muted">no marker met the ≥5× in both threshold yet</td></tr>`)+`</tbody>`;
}

/* grouped Δ (cold − coupled) bars: one series per DV, one group per category */
const DVCOL={warmth:"#e11d48",formality:"#7c3aed",advice_density:"#0d9488"};
function deltaBars(divid,rows,key,title){
  const cats=rows.map(r=>r[key]);
  const traces=DVS.map(k=>({x:cats,y:rows.map(r=>r[k]),name:k,type:"bar",marker:{color:DVCOL[k]}}));
  Plotly.newPlot(divid,traces,{title:{text:title,font:{size:14}},barmode:"group",height:360,
    margin:{t:36,r:10,b:110,l:52},legend:{orientation:"h",y:1.13},
    yaxis:{title:"Δ (cold − coupled)",zeroline:true,zerolinecolor:"#94a3b8"},
    font:{size:12}},{displayModeBar:false,responsive:true});
}

/* ---------- responses ---------- */
function renderResponses(){
  const r=M.responses, el=$("#tab-responses");
  el.innerHTML=`
    <div class="card"><h3>Judged response DVs — in-context vs cold</h3>
      <p class="sub">Same claude-sonnet-4.5 DV judge as every Experiment-3 run. Δ = cold − coupled.</p>
      <div id="rf_dv"></div></div>
    <div class="card"><h3>Δ (cold − coupled) by scenario</h3>
      <p class="sub">Positive ⇒ the <b>cold</b> (solo) reply scored higher on that DV than the coupled reply, within the same scenario. Bars above 0 = coupling <i>lowered</i> the DV; below 0 = coupling <i>raised</i> it.</p>
      <div id="rf_scn_plot"></div>
      <table id="rf_scn"></table></div>
    <div class="card"><h3>Δ (cold − coupled) by user-type (role)</h3>
      <p class="sub">Same Δ, split by the user role the prompt casts (its "user-type"). Shows whether coupling shifts warmth / formality / advice more for some kinds of user than others. Roles ordered by sample size.</p>
      <div id="rf_role_plot"></div>
      <table id="rf_role"></table></div>
    <div class="card"><h3>Δ (cold − coupled) by eval condition</h3><table id="rf_cond"></table></div>
    <div class="card"><h3>Response length &amp; primary emotion</h3>
      <div class="grid2"><div id="rf_len"></div><div id="rf_emo"></div></div></div>`;
  Plotly.newPlot("rf_dv",[
    {x:DVS,y:DVS.map(k=>r.dv[k].coupled_mean),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:DVS,y:DVS.map(k=>r.dv[k].cold_mean),name:"cold",type:"bar",marker:{color:COL_S}},
  ],{title:{text:"Mean DV (0–10)",font:{size:14}},barmode:"group",height:320,
     margin:{t:36,r:10,b:50,l:50},legend:{orientation:"h",y:1.15},yaxis:{range:[0,10]},font:{size:12}},{displayModeBar:false,responsive:true});
  const dtbl=(rows,key)=>`<thead><tr><th>${key}</th>${DVS.map(k=>`<th class="num">${k}</th>`).join("")}<th class="num">n</th></tr></thead><tbody>`+
    rows.map(row=>`<tr><td>${esc(row[key])}</td>${DVS.map(k=>`<td class="num ${cls(row[k])}">${sign(row[k])}</td>`).join("")}<td class="num">${row.n}</td></tr>`).join("")+`</tbody>`;
  deltaBars("rf_scn_plot",r.by_scenario,"scenario","Δ (cold − coupled) per DV, by scenario");
  $("#rf_scn").innerHTML=dtbl(r.by_scenario,"scenario");
  deltaBars("rf_role_plot",r.by_role,"role","Δ (cold − coupled) per DV, by user-type (role)");
  $("#rf_role").innerHTML=dtbl(r.by_role,"role");
  $("#rf_cond").innerHTML=dtbl(r.by_condition,"condition");
  Plotly.newPlot("rf_len",[{x:["coupled","cold"],y:[r.length.coupled_chars,r.length.solo_chars],
    type:"bar",marker:{color:[COL_C,COL_S]}}],{title:{text:"Mean response length (chars)",font:{size:14}},
    height:300,margin:{t:36,r:10,b:40,l:60},font:{size:12}},{displayModeBar:false,responsive:true});
  const emos=[...new Set([...Object.keys(r.emotion.coupled),...Object.keys(r.emotion.solo)])]
    .sort((a,b)=>((r.emotion.coupled[b]||0)+(r.emotion.solo[b]||0))-((r.emotion.coupled[a]||0)+(r.emotion.solo[a]||0))).slice(0,8);
  Plotly.newPlot("rf_emo",[
    {x:emos,y:emos.map(e=>r.emotion.coupled[e]||0),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:emos,y:emos.map(e=>r.emotion.solo[e]||0),name:"cold",type:"bar",marker:{color:COL_S}},
  ],{title:{text:"Primary emotion",font:{size:14}},barmode:"group",height:300,
     margin:{t:36,r:10,b:70,l:40},legend:{orientation:"h",y:1.2},font:{size:12}},{displayModeBar:false,responsive:true});
}

/* ---------- faith ---------- */
function renderFaith(){
  const F=M.faith, el=$("#tab-faith");
  if(!F || !F.available){el.innerHTML=`<div class="note">No faithfulness judgments on disk yet for this model. Run <code>python -m src.run_exp3_mini_judge</code>.</div>`;return;}
  let frows=FKEYS.map(k=>{const p=F.paired[k];const m=F.means[k];
    return `<tr><td>${k.replace("_score","")}</td>
      <td class="num">${fmt(m.coupled)}</td><td class="num">${fmt(m.cold)}</td>
      <td class="num ${cls(p.delta_mean)}">${sign(p.delta_mean)}</td>
      <td class="num">${fmt(p.d_z)}</td><td class="num">${fmt(p.r)}</td></tr>`;}).join("");
  el.innerHTML=`
    <div class="card"><h3>Faithfulness scores — reply vs its committed persona</h3>
      <p class="sub">Averaged over two judges. <b>fidelity</b> = committed traits present at right pole/intensity; <b>purity</b> = no inversions/leaked traits; <b>overall</b> = representativeness (an inversion or dominant leak caps it). Δ = cold − coupled. Bars are paired means ± 1 SE.</p>
      <div id="ff_score_plot"></div>
      <table><thead><tr><th>score</th><th class="num">coupled</th><th class="num">cold</th><th class="num">Δ</th><th class="num">d<sub>z</sub></th><th class="num">r</th></tr></thead><tbody>${frows}</tbody></table>
      <div class="cap"><b>Main result:</b> whether the reply that was written right after committing to a persona (coupled) embodies it more faithfully than a cold reply matched to a persona elicited separately (solo).</div>
      <h3 style="margin-top:18px">Significance test</h3>
      <p class="sub"><b>Wilcoxon signed-rank test</b> (paired, two-sided). The scores are paired by prompt (same prompt_id, coupled vs cold), bounded 0–10, two-judge averages with strong ceiling effects — so a non-parametric paired test is appropriate rather than a normal-theory t-test. H<sub>0</sub>: the within-prompt difference (cold − coupled) has median 0. Zeros dropped; ties get average ranks with a tie-corrected variance; p from the continuity-corrected normal approximation (n≈300 ⇒ ≈ exact). Effect size = matched-pairs rank-biserial <code>r</code> (sign shows direction; negative ⇒ coupled more faithful).</p>
      <table id="ff_sig"></table>
      <div class="cap" id="ff_sig_note"></div>
    </div>
    <div class="card"><h3>Overall-faithfulness distribution</h3><div id="ff_hist"></div></div>
    <div class="card"><h3>Per-trait verdicts</h3>
      <p class="sub">Every committed trait gets a verdict from each judge. INVERTED (opposite pole) and ABSENT are the severe failures.</p>
      <div id="ff_verdict"></div></div>
    <div class="card"><h3>Inversion &amp; leakage rate (per reply)</h3>
      <div class="grid2"><div id="ff_inv"></div><div>
        <table><thead><tr><th></th><th class="num">coupled</th><th class="num">cold</th></tr></thead><tbody>
        <tr><td>mean inversions / reply</td><td class="num">${fmt(F.inversion.coupled)}</td><td class="num">${fmt(F.inversion.cold)}</td></tr>
        <tr><td>mean leaked traits / reply</td><td class="num">${fmt(F.leakage.coupled)}</td><td class="num">${fmt(F.leakage.cold)}</td></tr>
        </tbody></table>
        <div class="cap">Judge agreement (claude-5 vs gpt-5.6 overall): r=${fmt(F.judge_agreement.r)}, mean|Δ|=${fmt(F.judge_agreement.mean_abs_diff)} over ${F.judge_agreement.n} judged replies.</div>
      </div></div></div>
    <div class="card"><h3>Overall faithfulness by scenario</h3>
      <p class="sub">Mean averaged overall_score (0–10) per scenario, coupled vs cold. Δ = cold − coupled.</p>
      <div id="ff_scn_plot"></div>
      <table id="ff_scn"></table></div>
    <div class="card"><h3>Overall faithfulness by user-type (role)</h3>
      <p class="sub">Mean averaged overall_score (0–10) per user role, coupled vs cold. Δ = cold − coupled. Roles ordered by sample size.</p>
      <div id="ff_role_plot"></div>
      <table id="ff_role"></table></div>
    <div class="card"><h3>Overall faithfulness by eval condition</h3><table id="ff_cond"></table></div>
    <div class="card"><h3>Per-marker faithfulness — coupled vs cold</h3>
      <p class="sub">For each Mini-Marker, the share of its per-trait judgments rated <b>FAITHFUL</b> (correct pole, intensity within ±2), coupled vs cold. Markers scored ≥3× in both designs, ordered by sample size. Taller = the marker was delivered faithfully more often.</p>
      <div id="ff_pt_plot"></div></div>
    <div class="card"><h3>Per-marker fidelity detail (coupled)</h3>
      <p class="sub">For each Mini-Marker: mean declared vs observed intensity, gap, and % of verdicts FAITHFUL / INVERTED.</p>
      <table id="ff_pt_c"></table></div>
    <div class="card"><h3>Per-marker fidelity detail (cold)</h3><table id="ff_pt_s"></table></div>`;
  // faithfulness scores plot (paired means ± SE) + significance
  const fp=k=>F.paired[k];
  Plotly.newPlot("ff_score_plot",[
    {x:FKEYS.map(k=>k.replace("_score","")),y:FKEYS.map(k=>fp(k).coupled_mean),name:"coupled",
     type:"bar",marker:{color:COL_C},
     error_y:{type:"data",array:FKEYS.map(k=>fp(k).coupled_se),visible:true,color:"#1e3a8a",thickness:1.3}},
    {x:FKEYS.map(k=>k.replace("_score","")),y:FKEYS.map(k=>fp(k).cold_mean),name:"cold",
     type:"bar",marker:{color:COL_S},
     error_y:{type:"data",array:FKEYS.map(k=>fp(k).cold_se),visible:true,color:"#9d174d",thickness:1.3}},
  ],{title:{text:"Faithfulness scores — paired means ± 1 SE",font:{size:14}},barmode:"group",height:340,
     margin:{t:36,r:10,b:46,l:50},legend:{orientation:"h",y:1.15},yaxis:{range:[0,10],title:"score (0–10)"},
     font:{size:12}},{displayModeBar:false,responsive:true});
  const pfmt=p=>p==null?"—":(p<1e-4?"&lt;0.0001":(p<0.001?p.toExponential(1):p.toFixed(4)));
  $("#ff_sig").innerHTML=`<thead><tr><th>score</th><th class="num">n pairs</th><th class="num">W</th>
    <th class="num">z</th><th class="num">p (2-sided)</th><th class="num">rank-biserial</th><th>significant?</th></tr></thead><tbody>`+
    FKEYS.map(k=>{const w=fp(k).wilcoxon; if(!w) return `<tr><td>${k.replace("_score","")}</td><td colspan="6" class="muted">n/a</td></tr>`;
      return `<tr><td>${k.replace("_score","")}</td><td class="num">${w.n}</td><td class="num">${w.W}</td>
        <td class="num">${fmt(w.z)}</td><td class="num">${pfmt(w.p)}</td>
        <td class="num ${cls(w.rank_biserial)}">${fmt(w.rank_biserial,3)}</td>
        <td>${w.significant?'<span class="pos">yes (p&lt;0.05)</span>':'<span class="neg">no</span>'}</td></tr>`;}).join("")+`</tbody>`;
  const ov=fp("overall_score").wilcoxon;
  if(ov){
    const nsig=FKEYS.filter(k=>fp(k).wilcoxon&&fp(k).wilcoxon.significant).length;
    const dir=ov.rank_biserial<0?"coupled replies are <b>more</b> faithful than cold replies":"cold replies are more faithful than coupled";
    const bonf=0.05/FKEYS.length;
    const bonfSig=FKEYS.filter(k=>fp(k).wilcoxon&&fp(k).wilcoxon.p<bonf).length;
    $("#ff_sig_note").innerHTML=`<b>Verdict for ${esc($("#model").value)}:</b> overall faithfulness differs significantly between designs `+
      `(Wilcoxon p=${pfmt(ov.p)}, rank-biserial ${fmt(ov.rank_biserial,3)}, n=${ov.n}) — ${dir}. `+
      `${nsig}/${FKEYS.length} scores are significant at α=0.05; ${bonfSig}/${FKEYS.length} survive a Bonferroni correction for 3 tests (α=${bonf.toFixed(4)}).`;
  } else { $("#ff_sig_note").textContent="Too few paired scores for a significance test."; }
  Plotly.newPlot("ff_hist",[
    {x:[...Array(11).keys()],y:F.hist.coupled,name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:[...Array(11).keys()],y:F.hist.cold,name:"cold",type:"bar",marker:{color:COL_S}},
  ],{title:{text:"overall_score histogram",font:{size:14}},barmode:"group",height:300,
     margin:{t:36,r:10,b:40,l:50},legend:{orientation:"h",y:1.2},xaxis:{title:"score"},font:{size:12}},{displayModeBar:false,responsive:true});
  Plotly.newPlot("ff_verdict",[
    {x:VERDICTS,y:VERDICTS.map(v=>F.verdicts.coupled[v]),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:VERDICTS,y:VERDICTS.map(v=>F.verdicts.cold[v]),name:"cold",type:"bar",marker:{color:COL_S}},
  ],{title:{text:"per-trait verdict counts (both judges)",font:{size:14}},barmode:"group",height:300,
     margin:{t:36,r:10,b:50,l:50},legend:{orientation:"h",y:1.2},font:{size:12}},{displayModeBar:false,responsive:true});
  Plotly.newPlot("ff_inv",[
    {x:["inversions","leaked traits"],y:[F.inversion.coupled,F.leakage.coupled],name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:["inversions","leaked traits"],y:[F.inversion.cold,F.leakage.cold],name:"cold",type:"bar",marker:{color:COL_S}},
  ],{title:{text:"mean per reply",font:{size:14}},barmode:"group",height:300,
     margin:{t:36,r:10,b:40,l:50},legend:{orientation:"h",y:1.2},font:{size:12}},{displayModeBar:false,responsive:true});
  const ftbl=(rows,key)=>`<thead><tr><th>${key}</th><th class="num">coupled</th><th class="num">cold</th><th class="num">Δ</th><th class="num">n</th></tr></thead><tbody>`+
    rows.map(r=>`<tr><td>${esc(r[key])}</td><td class="num">${fmt(r.coupled)}</td><td class="num">${fmt(r.cold)}</td><td class="num ${cls(r.delta)}">${sign(r.delta)}</td><td class="num">${r.n}</td></tr>`).join("")+`</tbody>`;
  // overall faithfulness by scenario (coupled vs cold)
  const sc=F.by_scenario;
  Plotly.newPlot("ff_scn_plot",[
    {x:sc.map(r=>r.scenario),y:sc.map(r=>r.coupled),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:sc.map(r=>r.scenario),y:sc.map(r=>r.cold),name:"cold",type:"bar",marker:{color:COL_S}},
  ],{title:{text:"Overall faithfulness (0–10) by scenario",font:{size:14}},barmode:"group",height:360,
     margin:{t:36,r:10,b:110,l:50},legend:{orientation:"h",y:1.13},yaxis:{range:[0,10],title:"overall_score"},
     font:{size:12}},{displayModeBar:false,responsive:true});
  $("#ff_scn").innerHTML=ftbl(F.by_scenario,"scenario");
  // overall faithfulness by user-type (role), coupled vs cold
  const rl=F.by_role||[];
  Plotly.newPlot("ff_role_plot",[
    {x:rl.map(r=>r.role),y:rl.map(r=>r.coupled),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:rl.map(r=>r.role),y:rl.map(r=>r.cold),name:"cold",type:"bar",marker:{color:COL_S}},
  ],{title:{text:"Overall faithfulness (0–10) by user-type",font:{size:14}},barmode:"group",height:360,
     margin:{t:36,r:10,b:110,l:50},legend:{orientation:"h",y:1.13},yaxis:{range:[0,10],title:"overall_score"},
     font:{size:12}},{displayModeBar:false,responsive:true});
  $("#ff_role").innerHTML=ftbl(F.by_role,"role");
  $("#ff_cond").innerHTML=ftbl(F.by_condition,"condition");
  // per-marker faithfulness (share FAITHFUL), coupled vs cold
  const pc=F.per_trait_combined||[];
  if(pc.length){
    Plotly.newPlot("ff_pt_plot",[
      {x:pc.map(r=>r.name),y:pc.map(r=>r.faithful_coupled),name:"coupled",type:"bar",marker:{color:COL_C},
       customdata:pc.map(r=>r.n_coupled),hovertemplate:"%{x} coupled<br>%{y}% faithful (n=%{customdata})<extra></extra>"},
      {x:pc.map(r=>r.name),y:pc.map(r=>r.faithful_cold),name:"cold",type:"bar",marker:{color:COL_S},
       customdata:pc.map(r=>r.n_cold),hovertemplate:"%{x} cold<br>%{y}% faithful (n=%{customdata})<extra></extra>"},
    ],{title:{text:"% of judgments rated FAITHFUL, by marker",font:{size:14}},barmode:"group",height:380,
       margin:{t:36,r:10,b:90,l:50},legend:{orientation:"h",y:1.12},yaxis:{range:[0,100],title:"% FAITHFUL"},
       font:{size:12}},{displayModeBar:false,responsive:true});
  } else { $("#ff_pt_plot").innerHTML=`<p class="muted">not enough markers scored ≥3× in both designs yet</p>`; }
  const pttbl=rows=>`<thead><tr><th>marker</th><th class="num">n</th><th class="num">declared</th><th class="num">observed</th><th class="num">gap</th><th class="num">%faithful</th><th class="num">%inverted</th></tr></thead><tbody>`+
    (rows.length?rows.map(r=>`<tr><td><span class="fct ${fcls(r.factor)}">${r.factor}</span>${esc(r.name)}</td>
      <td class="num">${r.n}</td><td class="num">${fmt(r.stated)}</td><td class="num">${fmt(r.obs)}</td>
      <td class="num ${cls(r.gap)}">${sign(r.gap)}</td><td class="num">${r.faithful_pct}%</td>
      <td class="num ${r.inverted_pct>0?'neg':''}">${r.inverted_pct}%</td></tr>`).join("")
      :`<tr><td colspan="7" class="muted">not enough judged traits yet</td></tr>`)+`</tbody>`;
  $("#ff_pt_c").innerHTML=pttbl(F.per_trait_coupled);
  $("#ff_pt_s").innerHTML=pttbl(F.per_trait_cold);
}

/* ---------- examples ---------- */
function traitRows(traits){
  return traits.map(t=>{const f=classify_js(t.name);
    return `<div class="tr"><span>${f?`<span class="fct ${fcls(f)}">${f}</span>`:""}${esc(t.name)} <span class="muted">${esc(t.facet||"")}</span></span><span class="v">${fmt(t.value,0)}</span></div>`;}).join("");
}
const MARKER_JS={};
Object.entries({I:{"+":["Bold","Energetic","Extraverted","Talkative"],"-":["Bashful","Quiet","Shy","Withdrawn"]},
II:{"+":["Cooperative","Kind","Sympathetic","Warm"],"-":["Cold","Harsh","Rude","Unsympathetic"]},
III:{"+":["Efficient","Organized","Practical","Systematic"],"-":["Careless","Disorganized","Inefficient","Sloppy"]},
IV:{"+":["Relaxed","Unenvious"],"-":["Envious","Fretful","Jealous","Moody","Temperamental","Touchy"]},
V:{"+":["Complex","Creative","Deep","Imaginative","Intellectual","Philosophical"],"-":["Uncreative","Unintellectual"]}})
.forEach(([f,pols])=>Object.values(pols).flat().forEach(w=>MARKER_JS[w.toLowerCase()]=f));
function classify_js(name){const n=String(name||"").toLowerCase().replace(/[^a-z]+/g," ").trim();
  if(MARKER_JS[n])return MARKER_JS[n];for(const tok of n.split(" "))if(MARKER_JS[tok])return MARKER_JS[tok];return null;}
function judgeBlock(side){
  if(!side)return "<span class='muted'>—</span>";
  return Object.entries(side.judges).map(([jm,j])=>{
    if(j.error)return `<div class="muted">${esc(jm)}: error ${esc(j.error)}</div>`;
    const tv=(j.traits||[]).map(t=>`<span class="vd ${vcls(t.verdict)}">${esc(t.name)} ${t.stated}→${t.obs} ${t.verdict||""}</span>`).join(" ");
    return `<div style="margin-bottom:6px"><b>${esc(jm)}</b> — fid ${fmt(j.fidelity_score,0)} · pur ${fmt(j.purity_score,0)} · <b>ovr ${fmt(j.overall_score,0)}</b>
      <div style="margin:3px 0">${tv}</div>
      <div class="muted" style="font-size:12px">${esc(j.gestalt||"")}</div></div>`;}).join("");
}
function renderExamples(){
  const el=$("#tab-examples");
  const scns=["all",...M.meta.scenarios];
  el.innerHTML=`<div class="card"><div class="controls">
     <div><label>Scenario</label><select id="ex_scn">${scns.map(s=>`<option>${esc(s)}</option>`).join("")}</select></div>
     <div><label>Sort by</label><select id="ex_sort">
       <option value="gap">largest coupled−cold faithfulness gap</option>
       <option value="low">lowest coupled faithfulness</option>
       <option value="pid">prompt_id</option></select></div>
     <div class="muted" id="ex_count"></div></div></div>
     <div id="ex_list"></div>`;
  const draw=()=>{
    const scn=$("#ex_scn").value, sort=$("#ex_sort").value;
    let rows=M.browse.filter(b=>scn==="all"||b.scenario===scn);
    const ov=(b,side)=>{const f=b.faith&&b.faith[side];return f&&f.avg?f.avg.overall_score:null;};
    if(sort==="gap")rows.sort((a,b)=>Math.abs((ov(b,"coupled")??0)-(ov(b,"cold")??0))-Math.abs((ov(a,"coupled")??0)-(ov(a,"cold")??0)));
    else if(sort==="low")rows.sort((a,b)=>(ov(a,"coupled")??99)-(ov(b,"coupled")??99));
    else rows.sort((a,b)=>a.prompt_id<b.prompt_id?-1:1);
    $("#ex_count").textContent=`${rows.length} prompts`;
    $("#ex_list").innerHTML=rows.slice(0,40).map(b=>`
      <div class="card">
        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px">
          <div><span class="pill pc">${esc(b.scenario)}</span> <span class="muted">${esc(b.role)} · ${esc(b.condition)} · ${esc(b.prompt_id)}</span></div>
        </div>
        <div class="resp" style="max-height:120px;margin:8px 0">${esc(b.user)}</div>
        <div class="two">
          <div class="col-c"><div class="h" style="color:${COL_C}">Coupled — ${esc(b.c_name)}</div>
            ${traitRows(b.c_traits)}
            <div class="resp" style="margin-top:8px">${esc(b.c_resp)}</div>
            <details><summary>faithfulness judges</summary>${judgeBlock(b.faith&&b.faith.coupled)}</details>
          </div>
          <div class="col-s"><div class="h" style="color:${COL_S}">Solo persona → cold reply — ${esc(b.s_name)}</div>
            ${traitRows(b.s_traits)}
            <div class="resp" style="margin-top:8px">${esc(b.s_resp)}</div>
            <details><summary>faithfulness judges</summary>${judgeBlock(b.faith&&b.faith.cold)}</details>
          </div>
        </div>
      </div>`).join("");
  };
  $("#ex_scn").onchange=draw;$("#ex_sort").onchange=draw;draw();
}

/* ---------- method ---------- */
function renderMethod(){
  const stick=(x,y,c)=>`<g stroke="${c}" stroke-width="2" fill="none">
    <circle cx="${x}" cy="${y}" r="7"/><line x1="${x}" y1="${y+7}" x2="${x}" y2="${y+27}"/>
    <line x1="${x-10}" y1="${y+15}" x2="${x+10}" y2="${y+15}"/>
    <line x1="${x}" y1="${y+27}" x2="${x-9}" y2="${y+42}"/>
    <line x1="${x}" y1="${y+27}" x2="${x+9}" y2="${y+42}"/></g>`;
  const fig=`<svg id="setup-fig" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 340" width="100%" style="max-width:1040px;height:auto">
    <defs>
      <marker id="ac" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0 0 L7 3 L0 6 z" fill="${COL_C}"/></marker>
      <marker id="as" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0 0 L7 3 L0 6 z" fill="${COL_S}"/></marker>
    </defs>
    <text x="95" y="22" text-anchor="middle" font-size="14" font-weight="700" fill="#0f172a">Many different users</text>
    ${stick(55,44,"#475569")}${stick(95,44,"#0d9488")}${stick(135,44,"#d97706")}
    <text x="95" y="116" text-anchor="middle" font-size="11.5" fill="#64748b">different user-types</text>
    <rect x="22" y="132" width="146" height="50" rx="10" fill="#f1f5f9" stroke="#cbd5e1"/>
    <text x="95" y="154" text-anchor="middle" font-size="12.5" fill="#334155">one user message</text>
    <text x="95" y="172" text-anchor="middle" font-size="12" fill="#94a3b8">“…”</text>
    <text x="95" y="202" text-anchor="middle" font-size="11.5" fill="#64748b">across many scenarios</text>
    <path d="M168 157 H232" stroke="#94a3b8" stroke-width="2" fill="none"/>
    <circle cx="236" cy="157" r="4" fill="#94a3b8"/>
    <path d="M236 157 C268 157 268 86 298 86" stroke="${COL_C}" stroke-width="2" fill="none" marker-end="url(#ac)"/>
    <path d="M236 157 C268 157 268 246 298 246" stroke="${COL_S}" stroke-width="2" fill="none" marker-end="url(#as)"/>
    <rect x="300" y="42" width="716" height="92" rx="12" fill="#f5f8ff" stroke="${COL_C}" stroke-width="1.5"/>
    <text x="320" y="68" font-size="14" font-weight="700" fill="${COL_C}">1 · Coupled — one generation</text>
    <rect x="320" y="78" width="252" height="46" rx="8" fill="#fff" stroke="#bfdbfe"/>
    <text x="446" y="98" text-anchor="middle" font-size="12.5" fill="#334155">declare persona</text>
    <text x="446" y="115" text-anchor="middle" font-size="11" fill="#64748b">persona traits</text>
    <path d="M574 101 H616" stroke="${COL_C}" stroke-width="2" fill="none" marker-end="url(#ac)"/>
    <rect x="622" y="78" width="378" height="46" rx="8" fill="#fff" stroke="#bfdbfe"/>
    <text x="811" y="98" text-anchor="middle" font-size="12.5" fill="#334155">write reply — as that persona</text>
    <text x="811" y="115" text-anchor="middle" font-size="11" fill="#64748b">traits sit right next to the reply</text>
    <rect x="300" y="198" width="716" height="112" rx="12" fill="#fff7fb" stroke="${COL_S}" stroke-width="1.5"/>
    <text x="320" y="224" font-size="14" font-weight="700" fill="${COL_S}">2 · Solo — two separate calls</text>
    <rect x="320" y="236" width="252" height="52" rx="8" fill="#fff" stroke="#fbcfe8"/>
    <text x="446" y="258" text-anchor="middle" font-size="12.5" fill="#334155">declare persona — alone</text>
    <text x="446" y="275" text-anchor="middle" font-size="11" fill="#64748b">traits only, no reply</text>
    <line x1="597" y1="232" x2="597" y2="294" stroke="#f9a8d4" stroke-width="1.5" stroke-dasharray="4 4"/>
    <text x="597" y="306" text-anchor="middle" font-size="10.5" fill="${COL_S}">separate</text>
    <rect x="622" y="236" width="378" height="52" rx="8" fill="#fff" stroke="#fbcfe8"/>
    <text x="811" y="258" text-anchor="middle" font-size="12.5" fill="#334155">write reply — cold</text>
    <text x="811" y="275" text-anchor="middle" font-size="11" fill="#64748b">the persona is never shown to it</text>
  </svg>`;
  $("#tab-method").innerHTML=`<div class="card"><h3>How the experiment is set up</h3>
    <p class="sub">The same users and messages go through <b>two ways</b> of asking the model to self-report the "character" it will use.</p>
    <div class="figtools"><button class="btn" onclick="downloadFigurePDF()">⭳ Download figure (PDF)</button></div>
    ${fig}
    <div class="cap"><b>Left:</b> many different users, each sending one message. <b>Right — the two ways to self-report:</b> <span style="color:${COL_C};font-weight:700">Coupled</span> writes the persona and the reply <i>together</i> in one go; <span style="color:${COL_S};font-weight:700">Solo</span> writes the persona <i>on its own</i>, then a separate <i>cold</i> reply that never sees the persona. Everything else is identical — so any difference is down to <b>when</b> the self-report happens.</div>
  </div>
  <div class="card"><h3>Method</h3>
    <p class="sub">Experiment 4 — closed-vocabulary (Mini-Marker) replication of the Experiment-3 design comparison.</p>
    <ul style="font-size:14px;line-height:1.7">
      <li><b>Vocabulary.</b> Personas are described only with the 40-word Saucier (1994) Mini-Marker set (5 factors × ± poles), each trait tagged with an AB5C facet blend (primary/secondary) and scored 0–10, plus an optional 5-factor <code>factor_profile</code>. Prompt: <code>infer_persona_prompt_mini_marker_v1</code>.</li>
      <li><b>Coupled.</b> One generation: the model writes its <code>&lt;persona&gt;</code> then, conditioned on it, its <code>&lt;response&gt;</code>. Traits are "in-context" with the reply.</li>
      <li><b>Solo.</b> The persona is elicited alone (stop after <code>&lt;/persona&gt;</code>) and the reply is generated in a separate call with <b>no</b> self-modeling scaffold. Only one framing is used — unlike the free-form decoupled design there is no v1/v3 averaging.</li>
      <li><b>Behavioural DVs.</b> Every reply is scored by <code>anthropic/claude-sonnet-4.5</code> for warmth / formality / advice_density / primary_emotion — identical to every other Experiment-3 run.</li>
      <li><b>Faithfulness.</b> <code>judge_faithfulness_prompt_mini_marker</code> scores reply-vs-persona on fidelity / purity / overall (0–10) with per-trait verdicts (FAITHFUL/UNDER/OVER/ABSENT/INVERTED), an inversion scan and a leakage scan. Run through <b>${esc((M?"":"")+DATA.judges.join(" and "))}</b>; the numeric scores are <b>averaged</b> across the two judges.</li>
      <li><b>Pairing.</b> Coupled and solo are paired on <code>prompt_id</code> (run 0) over the same balanced role × eval_condition × scenario subsample, so trait and behaviour comparisons are within-prompt.</li>
      <li><b>Rejections.</b> A sample is a rejection when the model produced no reply or no committed Mini-Marker persona (0 scored traits). Rejected prompts are excluded from every comparison — a prompt is dropped if <b>either</b> design rejected it (counts in the Overview).</li>
    </ul>
    <div class="cap"><b>Two judges, averaged.</b> ${esc(DATA.judges.join(" + "))}. Judge-agreement stats are in the Faithfulness tab.</div>
  </div>`;
}

/* ---------- download any figure as a tightly-cropped PDF ---------- */
/* PDF page = figure's own pixel size (in pt) -> no title, no excess whitespace. */
function fallbackPrintSvg(svg){
  const w=window.open("","_blank");
  w.document.write(`<!doctype html><html><head><style>@page{margin:4mm}body{margin:0}svg{width:100%;height:auto}</style></head><body>${svg.outerHTML}</body></html>`);
  w.document.close(); w.focus(); setTimeout(()=>{w.print();},300);   // "Save as PDF"
}
async function exportSvgElement(svg,w,h,filename){
  const jsPDF=(window.jspdf&&window.jspdf.jsPDF)||window.jsPDF;
  if(jsPDF){
    try{
      const doc=new jsPDF({orientation:w>=h?"landscape":"portrait",unit:"pt",format:[w,h]});
      if(typeof doc.svg==="function"){ await doc.svg(svg,{x:0,y:0,width:w,height:h}); doc.save(filename); return; }
    }catch(e){/* fall through to print */}
  }
  fallbackPrintSvg(svg);
}
async function plotDivToPDF(gd,filename){
  try{
    try{ await Plotly.Plots.resize(gd); }catch(e){}      // sync to on-screen size
    const w=Math.round((gd._fullLayout&&gd._fullLayout.width)||gd.clientWidth||760);
    const h=Math.round((gd._fullLayout&&gd._fullLayout.height)||gd.clientHeight||400);
    const url=await Plotly.toImage(gd,{format:"svg",width:w,height:h});
    const txt=decodeURIComponent(url.replace(/^data:image\/svg\+xml,?/,""));
    const tmp=document.createElement("div"); tmp.innerHTML=txt;
    const svg=tmp.querySelector("svg"); if(!svg) throw new Error("no svg produced");
    await exportSvgElement(svg,w,h,filename);
  }catch(e){ alert("Could not export this figure: "+e.message); }
}
/* attach a small PDF button to every Plotly figure (re-runs after each render) */
function addPdfButtons(){
  document.querySelectorAll(".js-plotly-plot").forEach((gd,i)=>{
    if(gd.querySelector(":scope > .pdf-btn")) return;
    if(getComputedStyle(gd).position==="static") gd.style.position="relative";
    const b=document.createElement("button");
    b.className="pdf-btn"; b.textContent="PDF";
    b.onclick=e=>{ e.stopPropagation(); plotDivToPDF(gd,"experiment4-"+(gd.id||("figure-"+i))+".pdf"); };
    gd.appendChild(b);
  });
}
async function downloadFigurePDF(){   // the hand-built method SVG
  const svg=document.getElementById("setup-fig"); if(!svg) return;
  await exportSvgElement(svg,1040,340,"experiment4-setup-figure.pdf");
}

initModel();initNav();renderAll();
</script>
</body></html>"""


if __name__ == "__main__":
    main()
