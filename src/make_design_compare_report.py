"""Build an interactive HTML comparing the two Experiment-3 elicitation designs.

COUPLED   ("traits followed by a reply")  -> results/exp3_persona/<model>/*.jsonl
          the model writes its <persona> block and then, conditioned on it, the
          <response>, in ONE generation. Traits are "in-context" with the reply.

STANDALONE ("traits alone, avg of v1 & v3") -> results/exp3_decoupled/<model>/*.jsonl
          the persona is elicited on its own by two prompts (v1 staging, v3
          self-report), trait values AVERAGED; the reply is generated separately
          with NO self-modeling scaffold.

Both designs cover the identical 9-scenario x 12-role x 3-condition factorial on
the same prompt_ids, so we pair on prompt_id (run 0) and ask two questions:

  1. TRAITS   -- does the verbalized self-model look different when a reply is
                 going to follow (coupled) vs when it stands alone (avg v1/v3)?
  2. RESPONSE -- does the behaviour differ when the model has just written a
                 persona (in-context) vs when it replies cold (alone)?

Any model that has BOTH result dirs is included automatically and picked via a
dropdown, so adding the other 3 models later is just: run both designs for them,
re-run this script.

    python -m src.make_design_compare_report
    -> analysis_design_compare.html   (self-contained, at repo root)
"""
from __future__ import annotations

import glob
import html
import json
import math
import re
from collections import Counter, defaultdict

from . import config

OUT = config.ROOT / "analysis_design_compare.html"

# --------------------------------------------------------------------------- #
# small stats helpers (no numpy/scipy dependency)
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


def ci95(xs):
    """Half-width of the 95% CI of the mean (1.96 * SEM, sample SD). 0 if n<2."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))   # sample SD
    return 1.96 * sd / math.sqrt(n)


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


def norm_name(name) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


# --------------------------------------------------------------------------- #
# archetype classifier (coarse persona family from the name)
# --------------------------------------------------------------------------- #
_SUPPORTER = ("listener", "support", "empath", "compassion", "sympath", "warm",
              "caring", "companion", "counsel", "friend", "comfort", "gentle",
              "nurtur", "reassur")
_ADVISOR = ("advisor", "adviser", "clinical", "analyst", "expert", "guide",
            "consultant", "coach", "educator", "technical", "strateg", "cautious",
            "safety", "risk", "informant", "informer", "authorit", "pragmat",
            "realist", "financial")


def archetype(name: str) -> str:
    n = norm_name(name)
    if not n:
        return "other"
    sup = sum(1 for k in _SUPPORTER if k in n)
    adv = sum(1 for k in _ADVISOR if k in n)
    if sup > adv:
        return "supporter"
    if adv > sup:
        return "advisor"
    return "other"


DVS = ("warmth", "formality", "advice_density")


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
    """prompt_id -> record (run 0 only; first if run missing)."""
    out = {}
    for r in recs:
        if r.get("run", 0) != 0:
            continue
        out.setdefault(r["prompt_id"], r)
    return out


FAITH_KEYS = ("intensity_fidelity", "expression_fidelity", "persona_gestalt",
              "overall_faithfulness")


def _load_faith(model):
    """prompt_id -> faith record (coupled / cold v1,v3,avg), if judged."""
    p = config.EXP3_FAITH_DIR / model / "all.jsonl"
    out = {}
    if p.exists():
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["prompt_id"]] = r
    return out


def _compact_judge(j):
    """Trim a full persona-judge dict to what the Examples tab needs."""
    if not j:
        return None
    out = {k: j.get(k) for k in FAITH_KEYS}
    out["n_unlisted"] = j.get("n_unlisted")
    out["unlisted"] = (j.get("unlisted_dominant_traits") or [])[:6]
    out["notes"] = j.get("judge_notes", "")
    out["traits"] = [{"name": t.get("trait_name"), "dec": t.get("declared_value"),
                      "obs": t.get("observed_intensity"), "dev": t.get("deviation"),
                      "match": t.get("expression_match"),
                      "ev": (t.get("evidence") or [])[:2]}
                     for t in (j.get("trait_evaluations") or [])]
    return out


def _pt_factory():
    return {"n": 0, "dec": [], "obs": [], "match": Counter(), "disp": None}


def _ingest_traits(acc, judge):
    """Fold one judge's per-trait evaluations into a name-keyed accumulator."""
    if not judge:
        return
    for t in judge.get("trait_evaluations") or []:
        name = t.get("trait_name")
        key = norm_name(name) if name else ""
        if not key:
            continue
        a = acc[key]
        a["n"] += 1
        if a["disp"] is None:
            a["disp"] = name
        d, o = _num(t.get("declared_value")), _num(t.get("observed_intensity"))
        if d is not None:
            a["dec"].append(d)
        if o is not None:
            a["obs"].append(o)
        em = str(t.get("expression_match") or "").strip().lower()
        if em in ("yes", "partial", "no"):
            a["match"][em] += 1


def _pt_combined(pt_coup, pt_cold, top=14, min_n=8):
    """Per-trait declared-vs-observed table. Declared is DESIGN-SPECIFIC: the
    coupled reply's persona declares its own trait values in-context, the cold
    reply's persona (v1/v3) declares them separately -- so each design carries its
    own declared marker, observed marker, and gap (declared - observed; a positive
    gap = the trait was named/selected but under-delivered, e.g. empathy declared
    8, seen 4). Keyed by the union of traits either design named >= min_n times."""
    def _row(disp, c, k):
        dcl_c = mean(c["dec"]) if c else None
        obs_c = mean(c["obs"]) if c else None
        dcl_k = mean(k["dec"]) if k else None
        obs_k = mean(k["obs"]) if k else None
        tot_k = (sum(k["match"].values()) or 1) if k else 1
        return {
            "name": disp,
            "n_coupled": c["n"] if c else 0, "n_cold": k["n"] if k else 0,
            "declared_coupled": round(dcl_c, 2) if dcl_c is not None else None,
            "obs_coupled": round(obs_c, 2) if obs_c is not None else None,
            "gap_coupled": round(dcl_c - obs_c, 2) if dcl_c is not None and obs_c is not None else None,
            "declared_cold": round(dcl_k, 2) if dcl_k is not None else None,
            "obs_cold": round(obs_k, 2) if obs_k is not None else None,
            "gap_cold": round(dcl_k - obs_k, 2) if dcl_k is not None and obs_k is not None else None,
            "yes_cold": round(100 * k["match"]["yes"] / tot_k) if k else None,
            "partial_cold": round(100 * k["match"]["partial"] / tot_k) if k else None,
            "no_cold": round(100 * k["match"]["no"] / tot_k) if k else None,
        }
    keys = {key for key, a in pt_cold.items() if a["n"] >= min_n} | \
           {key for key, a in pt_coup.items() if a["n"] >= min_n}
    rows = [_row((pt_cold.get(key) or pt_coup.get(key))["disp"],
                 pt_coup.get(key), pt_cold.get(key)) for key in keys]
    rows.sort(key=lambda r: -(r["n_cold"] + r["n_coupled"]))
    return rows[:top]


def models_with_both():
    have = []
    for m in config.MODELS:
        c = config.EXP3_PERSONA_DIR / m
        d = config.EXP3_DECOUPLED_DIR / m
        if c.exists() and d.exists() and glob.glob(str(c / "*.jsonl")) \
                and glob.glob(str(d / "*.jsonl")):
            have.append(m)
    return have


# --------------------------------------------------------------------------- #
# per-model analysis
# --------------------------------------------------------------------------- #
def coupled_traits(rec):
    return [{"name": t.get("name", ""), "value": _num(t.get("value"))}
            for t in (rec.get("traits") or []) if str(t.get("name", "")).strip()]


def standalone_traits(rec, source="avg"):
    """source: 'avg' (merged value), 'v1', 'v3'."""
    if source == "avg":
        return [{"name": t.get("name", ""), "value": _num(t.get("value"))}
                for t in (rec.get("traits") or []) if str(t.get("name", "")).strip()]
    block = (rec.get("elicit") or {}).get(source) or {}
    return [{"name": t.get("name", ""), "value": _num(t.get("value"))}
            for t in (block.get("traits") or []) if str(t.get("name", "")).strip()]


def trait_stats(list_of_traitlists):
    counts = [len(tl) for tl in list_of_traitlists]
    allvals = [t["value"] for tl in list_of_traitlists for t in tl if t["value"] is not None]
    vocab = Counter()
    vocab_vals = defaultdict(list)
    display = {}
    for tl in list_of_traitlists:
        for t in tl:
            k = norm_name(t["name"])
            if not k:
                continue
            vocab[k] += 1
            display.setdefault(k, t["name"])
            if t["value"] is not None:
                vocab_vals[k].append(t["value"])
    top = [(display[k], k, c, round(mean(vocab_vals[k]) or 0, 2))
           for k, c in vocab.most_common(20)]
    vhist = [0] * 11
    for v in allvals:
        vhist[min(10, max(0, int(round(v))))] += 1
    chist = Counter(counts)
    return {
        "n_dp": len(list_of_traitlists),
        "mean_count": round(mean(counts) or 0, 2),
        "count_hist": {str(k): chist[k] for k in sorted(chist)},
        "value_mean": round(mean(allvals) or 0, 2),
        "value_sd": round(pstdev(allvals), 2),
        "value_hist": vhist,
        "n_trait_tokens": len(allvals),
        "top": [{"name": n, "key": k, "count": c, "mean_val": mv} for n, k, c, mv in top],
        "_vocab": vocab, "_vocab_vals": vocab_vals, "_display": display,
    }


def analyse_model(model):
    coupled = _index_run0(_load(config.EXP3_PERSONA_DIR / model))
    stand = _index_run0(_load(config.EXP3_DECOUPLED_DIR / model))
    faith_idx = _load_faith(model)
    shared = sorted(set(coupled) & set(stand))

    c_tl, s_tl, v1_tl, v3_tl = [], [], [], []
    browse = []
    dv_c = {k: [] for k in DVS}
    dv_s = {k: [] for k in DVS}
    scn_delta = defaultdict(lambda: {k: [] for k in DVS})
    cond_delta = defaultdict(lambda: {k: [] for k in DVS})
    role_delta = defaultdict(lambda: {k: [] for k in DVS})
    role_dv = defaultdict(lambda: {k: {"c": [], "s": []} for k in DVS})  # absolute means per role
    emo_c, emo_s = Counter(), Counter()
    len_c, len_s = [], []
    arch_c, arch_v1, arch_v3 = Counter(), Counter(), Counter()
    name_c_v1 = name_v1_v3 = name_n = 0
    scatter = {k: [] for k in DVS}
    # v1 vs v3 averaging scatter (traits both framings named)
    avg_scatter = []
    # faithfulness accumulators
    fc = {k: [] for k in FAITH_KEYS}         # coupled (all judged)
    fk = {k: [] for k in FAITH_KEYS}         # cold avg (all judged)
    fpair = {k: {"c": [], "s": []} for k in FAITH_KEYS}   # both present
    f_scn = defaultdict(lambda: {"c": [], "s": []})       # overall by scenario
    f_cond = defaultdict(lambda: {"c": [], "s": []})
    f_role = defaultdict(lambda: {"c": [], "s": []})
    unlisted_c, unlisted_s = [], []
    v1v3_c = []
    fhist_c, fhist_s = [0] * 11, [0] * 11
    n_fcoupled = n_fcold = 0
    pt_coup, pt_cold = defaultdict(_pt_factory), defaultdict(_pt_factory)

    for pid in shared:
        rc, rs = coupled[pid], stand[pid]
        ct = coupled_traits(rc)
        st = standalone_traits(rs, "avg")
        s1 = standalone_traits(rs, "v1")
        s3 = standalone_traits(rs, "v3")
        c_tl.append(ct); s_tl.append(st); v1_tl.append(s1); v3_tl.append(s3)

        arch_c[archetype(rc.get("persona_name", ""))] += 1
        arch_v1[archetype(rs.get("persona_name_v1", ""))] += 1
        arch_v3[archetype(rs.get("persona_name_v3", ""))] += 1
        cn, n1, n3 = (rc.get("persona_name", ""), rs.get("persona_name_v1", ""),
                      rs.get("persona_name_v3", ""))
        if n1.strip():
            name_n += 1
            if norm_name(cn) == norm_name(n1):
                name_c_v1 += 1
            if norm_name(n1) == norm_name(n3):
                name_v1_v3 += 1

        # v1/v3 per-trait agreement (from merged block: value_v1 & value_v3 present)
        for t in (rs.get("traits") or []):
            a, b = _num(t.get("value_v1")), _num(t.get("value_v3"))
            if a is not None and b is not None:
                avg_scatter.append([a, b])

        jc, js = rc.get("judge") or {}, rs.get("judge") or {}
        for k in DVS:
            a, b = _num(jc.get(k)), _num(js.get(k))
            if a is not None and b is not None:
                dv_c[k].append(a); dv_s[k].append(b)
                scn_delta[rc["scenario"]][k].append(b - a)
                cond_delta[rc["eval_condition"]][k].append(b - a)
                role_delta[rc["role"]][k].append(b - a)
                role_dv[rc["role"]][k]["c"].append(a); role_dv[rc["role"]][k]["s"].append(b)
                scatter[k].append([a, b])
        if jc.get("primary_emotion"):
            emo_c[jc["primary_emotion"]] += 1
        if js.get("primary_emotion"):
            emo_s[js["primary_emotion"]] += 1
        len_c.append(len(rc.get("response", "")))
        len_s.append(len(rs.get("response", "")))

        # faithfulness (may be absent for some prompts / not yet judged)
        fr = faith_idx.get(pid) or {}
        fcoup = (fr.get("coupled") or {}).get("judge")
        fcold = (fr.get("cold") or {}).get("avg")
        fv1 = ((fr.get("cold") or {}).get("v1") or {}).get("judge")
        fv3 = ((fr.get("cold") or {}).get("v3") or {}).get("judge")
        if fcoup:
            n_fcoupled += 1
            for k in FAITH_KEYS:
                if fcoup.get(k) is not None:
                    fc[k].append(fcoup[k])
            if fcoup.get("n_unlisted") is not None:
                unlisted_c.append(fcoup["n_unlisted"])
            if fcoup.get("overall_faithfulness") is not None:
                fhist_c[int(round(fcoup["overall_faithfulness"]))] += 1
        if fcold:
            n_fcold += 1
            for k in FAITH_KEYS:
                if fcold.get(k) is not None:
                    fk[k].append(fcold[k])
            if fcold.get("n_unlisted") is not None:
                unlisted_s.append(fcold["n_unlisted"])
            if fcold.get("overall_faithfulness") is not None:
                fhist_s[int(round(fcold["overall_faithfulness"]))] += 1
        if fcoup and fcold:
            for k in FAITH_KEYS:
                a, b = fcoup.get(k), fcold.get(k)
                if a is not None and b is not None:
                    fpair[k]["c"].append(a); fpair[k]["s"].append(b)
            oc, osd = fcoup.get("overall_faithfulness"), fcold.get("overall_faithfulness")
            if oc is not None and osd is not None:
                f_scn[rc["scenario"]]["c"].append(oc); f_scn[rc["scenario"]]["s"].append(osd)
                f_cond[rc["eval_condition"]]["c"].append(oc); f_cond[rc["eval_condition"]]["s"].append(osd)
                f_role[rc["role"]]["c"].append(oc); f_role[rc["role"]]["s"].append(osd)
        if fv1 and fv3:
            a, b = fv1.get("overall_faithfulness"), fv3.get("overall_faithfulness")
            if a is not None and b is not None:
                v1v3_c.append([a, b])
        # per-trait declared-vs-observed (cold folds both standalone framings)
        _ingest_traits(pt_coup, fcoup)
        _ingest_traits(pt_cold, fv1)
        _ingest_traits(pt_cold, fv3)

        browse.append({
            "prompt_id": pid, "scenario": rc["scenario"], "role": rc["role"],
            "condition": rc["eval_condition"], "x": rc.get("x_value"),
            "user": rc.get("user", ""),
            "c_name": rc.get("persona_name", ""),
            "c_traits": [{"name": t["name"], "value": t["value"]} for t in ct],
            "c_resp": rc.get("response", ""),
            "c_judge": {k: jc.get(k) for k in DVS + ("primary_emotion",)},
            "s_v1_name": rs.get("persona_name_v1", ""),
            "s_v3_name": rs.get("persona_name_v3", ""),
            "s_v1_traits": [{"name": t["name"], "value": t["value"]} for t in s1],
            "s_v3_traits": [{"name": t["name"], "value": t["value"]} for t in s3],
            "s_traits": [{"name": t.get("name"), "avg": _num(t.get("value")),
                          "v1": _num(t.get("value_v1")), "v3": _num(t.get("value_v3")),
                          "n": t.get("n_sources")} for t in (rs.get("traits") or [])],
            "s_resp": rs.get("response", ""),
            "s_judge": {k: js.get(k) for k in DVS + ("primary_emotion",)},
            "faith": {"coupled": _compact_judge(fcoup), "cold_avg": fcold,
                      "v1": _compact_judge(fv1), "v3": _compact_judge(fv3)} if fr else None,
        })

    cstat = trait_stats(c_tl)
    sstat = trait_stats(s_tl)
    v1stat = trait_stats(v1_tl)
    v3stat = trait_stats(v3_tl)

    # vocabulary overlap (traits appearing >=3 datapoints in a design)
    THRESH = 3
    cset = {k for k, v in cstat["_vocab"].items() if v >= THRESH}
    sset = {k for k, v in sstat["_vocab"].items() if v >= THRESH}
    inter, union = cset & sset, cset | sset
    jaccard = round(len(inter) / len(union), 3) if union else 0.0
    only_c = sorted(cset - sset, key=lambda k: -cstat["_vocab"][k])
    only_s = sorted(sset - cset, key=lambda k: -sstat["_vocab"][k])

    # shared recurrent traits: value comparison (present >=5x in BOTH)
    shared_vals = []
    for k in inter:
        cn_, sn_ = len(cstat["_vocab_vals"][k]), len(sstat["_vocab_vals"][k])
        if cn_ >= 5 and sn_ >= 5:
            cm, sm = mean(cstat["_vocab_vals"][k]), mean(sstat["_vocab_vals"][k])
            shared_vals.append({
                "name": cstat["_display"].get(k, k), "coupled_mean": round(cm, 2),
                "coupled_n": cn_, "standalone_mean": round(sm, 2), "standalone_n": sn_,
                "delta": round(sm - cm, 2)})
    shared_vals.sort(key=lambda d: -abs(d["delta"]))

    # response paired stats
    resp_dv = {}
    for k in DVS:
        deltas = [b - a for a, b in zip(dv_c[k], dv_s[k])]
        dsd = pstdev(deltas)
        resp_dv[k] = {
            "coupled_mean": round(mean(dv_c[k]) or 0, 2),
            "standalone_mean": round(mean(dv_s[k]) or 0, 2),
            "delta_mean": round(mean(deltas) or 0, 2),
            "delta_sd": round(dsd, 2),
            "d_z": round((mean(deltas) or 0) / dsd, 2) if dsd > 0 else None,
            "pct_up": round(100 * sum(1 for d in deltas if d > 0) / len(deltas), 1) if deltas else None,
            "r": round(pearson(dv_c[k], dv_s[k]) or 0, 2),
            "n": len(deltas),
        }

    by_scn = [{"scenario": s,
               **{k: round(mean(scn_delta[s][k]) or 0, 2) for k in DVS},
               "n": len(scn_delta[s][DVS[0]])}
              for s in sorted(scn_delta)]
    by_cond = [{"condition": c,
                **{k: round(mean(cond_delta[c][k]) or 0, 2) for k in DVS},
                "n": len(cond_delta[c][DVS[0]])}
               for c in sorted(cond_delta)]
    by_role = [{"role": r,
                **{k: round(mean(role_delta[r][k]) or 0, 2) for k in DVS},
                "n": len(role_delta[r][DVS[0]])}
               for r in sorted(role_delta)]
    by_role_abs = []
    for rr in sorted(role_dv):
        row = {"role": rr}
        for k in DVS:
            row[k + "_c"] = round(mean(role_dv[rr][k]["c"]) or 0, 2)
            row[k + "_s"] = round(mean(role_dv[rr][k]["s"]) or 0, 2)
        by_role_abs.append(row)

    r_v1v3 = pearson([a for a, _ in avg_scatter], [b for _, b in avg_scatter])
    mad = mean([abs(a - b) for a, b in avg_scatter])

    # ---- faithfulness aggregates ----
    def paired_stat(cs, ss):
        deltas = [b - a for a, b in zip(cs, ss)]
        dsd = pstdev(deltas)
        return {
            "coupled_mean": round(mean(cs), 2) if cs else None,
            "cold_mean": round(mean(ss), 2) if ss else None,
            "coupled_ci": round(ci95(cs), 3),   # 95% CI half-width of the mean
            "cold_ci": round(ci95(ss), 3),
            "delta_mean": round(mean(deltas), 2) if deltas else None,
            "d_z": round(mean(deltas) / dsd, 2) if dsd > 0 else None,
            "pct_cold_higher": round(100 * sum(1 for d in deltas if d > 0) / len(deltas), 1) if deltas else None,
            "r": round(pearson(cs, ss), 2) if len(cs) > 1 and pearson(cs, ss) is not None else None,
            "n": len(deltas),
        }
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
                  "n": len(f_role[r]["c"])} for r in sorted(f_role)]
    v13r = pearson([a for a, _ in v1v3_c], [b for _, b in v1v3_c])
    v13mad = mean([abs(a - b) for a, b in v1v3_c])
    faith = {
        "available": bool(faith_idx),
        "n_coupled": n_fcoupled, "n_cold": n_fcold,
        "n_paired": faith_paired["overall_faithfulness"]["n"],
        "means": faith_means, "paired": faith_paired,
        "hist": {"coupled": fhist_c, "cold": fhist_s},
        "by_scenario": f_by_scn, "by_condition": f_by_cond, "by_role": f_by_role,
        "unlisted": {"coupled": round(mean(unlisted_c), 2) if unlisted_c else None,
                     "cold": round(mean(unlisted_s), 2) if unlisted_s else None},
        "v1v3": {"r": round(v13r, 3) if v13r is not None else None,
                 "mean_abs_diff": round(v13mad, 2) if v13mad is not None else None,
                 "n": len(v1v3_c)},
        "scatter": [[a, b] for a, b in zip(fpair["overall_faithfulness"]["c"],
                                           fpair["overall_faithfulness"]["s"])],
        "per_trait": _pt_combined(pt_coup, pt_cold),
    }

    def strip(stat):
        return {k: v for k, v in stat.items() if not k.startswith("_")}

    return {
        "meta": {
            "n_pairs": len(shared),
            "scenarios": sorted({b["scenario"] for b in browse}),
            "roles": sorted({b["role"] for b in browse}),
            "conditions": sorted({b["condition"] for b in browse}),
        },
        "traits": {
            "coupled": strip(cstat), "standalone": strip(sstat),
            "standalone_v1": strip(v1stat), "standalone_v3": strip(v3stat),
            "vocab_overlap": {
                "jaccard": jaccard, "n_coupled": len(cset), "n_standalone": len(sset),
                "n_shared": len(inter),
                "only_coupled": [{"name": cstat["_display"].get(k, k),
                                  "count": cstat["_vocab"][k]} for k in only_c[:15]],
                "only_standalone": [{"name": sstat["_display"].get(k, k),
                                     "count": sstat["_vocab"][k]} for k in only_s[:15]],
            },
            "shared_values": shared_vals,
            "archetype": {"coupled": dict(arch_c), "v1": dict(arch_v1), "v3": dict(arch_v3)},
            "name_match": {
                "coupled_vs_v1": round(name_c_v1 / name_n, 3) if name_n else None,
                "v1_vs_v3": round(name_v1_v3 / name_n, 3) if name_n else None,
                "n": name_n,
            },
            "averaging": {
                "n_both": len(avg_scatter),
                "r_v1v3": round(r_v1v3, 3) if r_v1v3 is not None else None,
                "mean_abs_diff": round(mad, 2) if mad is not None else None,
                "scatter": avg_scatter[:1500],
            },
        },
        "responses": {
            "dv": resp_dv, "by_scenario": by_scn, "by_condition": by_cond,
            "by_role": by_role, "by_role_abs": by_role_abs,
            "emotion": {"coupled": dict(emo_c), "standalone": dict(emo_s)},
            "length": {
                "coupled_chars": round(mean(len_c) or 0),
                "standalone_chars": round(mean(len_s) or 0),
            },
            "scatter": scatter,
        },
        "faith": faith,
        "browse": browse,
    }


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def build_html(payload):
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE.replace("/*DATA*/", data_json)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 3 — Design Comparison: traits + reply vs traits alone</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
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
button.btn{border:1px solid var(--c);background:var(--c);color:#fff;padding:7px 13px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer}
button.btn:hover{filter:brightness(1.06)}
button.btn:disabled{opacity:.55;cursor:default}
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
.stat .n{font-size:26px;font-weight:700}.stat .l{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.stat .d{font-size:13px;color:var(--mut);margin-top:3px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 9px;border-bottom:1px solid var(--bd);text-align:left}
th{color:var(--mut);font-weight:600;cursor:pointer;user-select:none}
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
.jbadge{display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;border-radius:6px;background:#f1f5f9;font-size:12px}
details{margin-top:8px}summary{cursor:pointer;color:var(--c);font-size:13px}
.note{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:12px 14px;font-size:13px;color:#78350f;margin:14px 0}
.cap{font-size:12.5px;color:#475569;background:#f8fafc;border-left:3px solid #cbd5e1;padding:7px 11px;margin-top:10px;border-radius:0 6px 6px 0;line-height:1.5}
.cap b{color:#334155}
.defs{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:4px 0 14px}
.defs>div{background:#f8fafc;border:1px solid var(--bd);border-radius:8px;padding:9px 11px;font-size:12.5px;line-height:1.5}
.defs .t{font-weight:700;color:var(--c)}
@media(max-width:760px){.defs{grid-template-columns:1fr}}
code{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}
</style></head>
<body>
<header>
  <h1>Experiment 3 — Design Comparison</h1>
  <p>Verbalized traits <b>followed by a reply</b> (coupled) &nbsp;vs&nbsp; traits <b>elicited alone</b> (avg of v1 &amp; v3), and the reply <b>in-context</b> vs <b>alone</b>.</p>
  <div class="legend">
    <span><b style="background:#2563eb"></b>Coupled — persona + reply in one generation (traits in-context)</span>
    <span><b style="background:#db2777"></b>Standalone — persona alone (avg v1/v3); reply generated cold</span>
  </div>
</header>
<div class="wrap">
  <div class="controls">
    <div><label>Model</label><select id="model"></select></div>
    <div class="muted" id="pairinfo"></div>
    <div class="muted" style="margin-left:auto;font-size:12px">tip: hover any chart → click the ⬇ button to save just that figure as PDF</div>
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
const COL_C="#2563eb", COL_S="#db2777";
const MODELS_LIST = Object.keys(DATA.models);
const MCOLORS = ["#2563eb","#db2777","#7c3aed","#059669","#f59e0b"];
const SUMMARY_KEY = "__all__";
const isSummary = () => document.querySelector("#model").value===SUMMARY_KEY;
let M = null;

// Per-figure download button: a hover toolbar on EACH chart that exports only
// that chart to a one-page PDF (not the whole page).
const PDF_ICON = {width:24,height:24,path:"M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"};
const PDF_BTN = {name:"downloadPDF", title:"Download this figure as PDF",
  icon:PDF_ICON, click:(gd)=>exportOneFigurePDF(gd)};
const PCONF = {displaylogo:false, responsive:true, modeBarButtons:[[PDF_BTN]]};

const $ = s => document.querySelector(s);
const esc = s => (s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const fmt = (x,d=2)=> x==null?"—":(typeof x==="number"?x.toFixed(d):x);
const sign = x => x>0?("+"+x.toFixed(2)):x.toFixed(2);
const cls = x => x>0?"pos":(x<0?"neg":"");

const TABS=[["overview","Overview"],["traits","Traits: coupled vs alone"],
  ["responses","Responses: in-context vs alone"],["faith","Faithfulness: reply vs its persona"],
  ["examples","Paired examples"],["method","Method"]];

function initModel(){
  const sel=$("#model"); sel.innerHTML="";
  const o0=document.createElement("option");o0.value=SUMMARY_KEY;o0.textContent="★ All models — summary";sel.appendChild(o0);
  MODELS_LIST.forEach(m=>{const o=document.createElement("option");o.value=m;o.textContent=m;sel.appendChild(o);});
  sel.value=SUMMARY_KEY;
  sel.onchange=()=>renderAll();
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
  const summ=isSummary();
  M = summ ? DATA.models[MODELS_LIST[0]] : DATA.models[$("#model").value];  // MREF for model-agnostic Method tab
  if(summ){
    $("#pairinfo").textContent=`Cross-model summary · ${MODELS_LIST.length} models · ${M.meta.n_pairs} paired prompts each`;
    renderOverviewSummary();renderTraitsSummary();renderResponsesSummary();renderFaithSummary();renderExamplesSummary();renderMethod();
    return;
  }
  $("#pairinfo").textContent=`${M.meta.n_pairs} paired prompts · ${M.meta.scenarios.length} scenarios · ${M.meta.roles.length} roles · ${M.meta.conditions.length} conditions`;
  renderOverview();renderTraits();renderResponses();renderFaith();renderExamples();renderMethod();
}

/* ---------- overview ---------- */
function stat(n,l,d){return `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div>${d?`<div class="d">${d}</div>`:""}</div>`;}
function renderOverview(){
  const t=M.traits, r=M.responses;
  const dv=r.dv;
  let cards=`<div class="grid3">
    ${stat(M.meta.n_pairs,"paired prompts","same prompt_id, run 0, both designs")}
    ${stat(t.coupled.mean_count+" / "+t.standalone.mean_count,"traits per datapoint","coupled / standalone")}
    ${stat(t.vocab_overlap.jaccard,"trait-vocabulary overlap","Jaccard, traits used ≥3×")}
  </div>`;
  let dvrows = DVS.map(k=>`<tr><td>${k}</td>
    <td class="num">${fmt(dv[k].coupled_mean)}</td>
    <td class="num">${fmt(dv[k].standalone_mean)}</td>
    <td class="num ${cls(dv[k].delta_mean)}">${sign(dv[k].delta_mean)}</td>
    <td class="num">${fmt(dv[k].d_z)}</td>
    <td class="num">${fmt(dv[k].r)}</td></tr>`).join("");
  const headline = `<div class="card"><h3>Does writing a persona first change the behaviour?</h3>
    <p class="sub">Judged DVs on the SAME prompts. Δ = standalone − coupled (positive ⇒ the cold reply scores higher). d<sub>z</sub> = paired effect size; r = cross-design correlation.</p>
    <table><thead><tr><th>DV</th><th class="num">coupled</th><th class="num">standalone</th><th class="num">Δ mean</th><th class="num">d<sub>z</sub></th><th class="num">r</th></tr></thead>
    <tbody>${dvrows}</tbody></table></div>`;
  const tnote = `<div class="card"><h3>Does the self-model itself change when a reply will follow?</h3>
   <p class="sub">Trait value = the model's own 0–10 score. Vocabulary = distinct trait names (normalized).</p>
   <div class="grid3">
     ${stat(fmt(t.coupled.value_mean)+" / "+fmt(t.standalone.value_mean),"mean trait value","coupled / standalone")}
     ${stat(t.vocab_overlap.n_shared,"shared trait names","of "+ (t.vocab_overlap.n_coupled)+" coupled · "+(t.vocab_overlap.n_standalone)+" standalone")}
     ${stat(fmt(100*(t.name_match.coupled_vs_v1||0),0)+"%","persona name: coupled = v1","identical label, n="+t.name_match.n)}
   </div></div>`;
  let faithCard = "";
  const F = M.faith;
  if (F && F.available) {
    const o = F.paired.overall_faithfulness;
    faithCard = `<div class="card"><h3>Is the reply faithful to the persona — and does writing it first matter?</h3>
      <p class="sub">A separate persona-judge scores how strongly the reply actually embodies its declared traits (0–10). Coupled = reply vs the persona it wrote in-context; cold = the scaffold-free reply vs the standalone (v1/v3-averaged) persona. Δ = cold − coupled.</p>
      <div class="grid3">
        ${stat(fmt(o.coupled_mean)+" → "+fmt(o.cold_mean),"overall faithfulness","coupled → cold (paired, n="+o.n+")")}
        ${stat(sign(o.delta_mean),"Δ faithfulness","d_z "+fmt(o.d_z)+" · cold higher on "+fmt(o.pct_cold_higher,0)+"% of prompts")}
        ${stat(fmt(o.r),"cross-design r","do the same prompts rank alike?")}
      </div>
      <p class="sub" style="margin-top:8px">Full breakdown in the <b>Faithfulness</b> tab.</p></div>`;
  }
  $("#tab-overview").innerHTML = cards + headline + tnote + faithCard +
    `<div class="note">One model so far (<code>${$("#model").value}</code>). The dropdown lists every model that has <b>both</b> designs on disk — run the other three through both pipelines and re-run the report to add them.</div>`;
}

/* ---------- traits ---------- */
function barTwo(divid,title,cats,cVals,sVals,ytitle){
  Plotly.newPlot(divid,[
    {x:cats,y:cVals,name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:cats,y:sVals,name:"standalone",type:"bar",marker:{color:COL_S}},
  ],{title:{text:title,font:{size:14}},barmode:"group",margin:{t:36,r:10,b:60,l:50},
     height:300,legend:{orientation:"h",y:1.15},yaxis:{title:ytitle},font:{size:12}},PCONF);
}
function renderTraits(){
  const t=M.traits;
  const el=$("#tab-traits");
  el.innerHTML=`
   <div class="grid2">
     <div class="card"><h3>Trait value distribution</h3><p class="sub">How the model scores its own load-bearing traits (0–10). coupled mean ${fmt(t.coupled.value_mean)} · standalone mean ${fmt(t.standalone.value_mean)}.</p><div id="tv_val"></div>
       <div class="cap"><b>How to read:</b> each bar counts how many traits received a given 0–10 self-score, blue = coupled, pink = standalone. Both pile up on the right, so the model rates its own traits highly in either design — and the two shapes nearly overlap, meaning <b>the scores barely change when a reply is going to follow</b>.</div></div>
     <div class="card"><h3>Traits per datapoint</h3><p class="sub">How many load-bearing traits are named. coupled ${t.coupled.mean_count} · standalone ${t.standalone.mean_count} (standalone = union of v1∪v3).</p><div id="tv_cnt"></div>
       <div class="cap"><b>How to read:</b> the number of traits the model lists for one prompt. Standalone (pink) sits higher only because it pools the traits named by <b>two</b> elicitation prompts (v1 ∪ v3); coupled (blue) lists a single set in one generation.</div></div>
   </div>
   <div class="card"><h3>Most frequent traits — side by side</h3>
     <p class="sub">Top trait vocabulary in each design (count across ${M.meta.n_pairs} datapoints; value = mean self-score).</p>
     <div class="grid2">
       <div><div class="h"><span class="pill pc">coupled</span></div><div id="topc"></div></div>
       <div><div class="h"><span class="pill ps">standalone</span></div><div id="tops"></div></div>
     </div></div>
   <div class="card"><h3>Shared recurrent traits — is the score the same?</h3>
     <p class="sub">Traits used ≥5× in <b>both</b> designs. Δ = standalone − coupled mean self-score. Sorted by |Δ|.</p>
     <div id="sharedtbl"></div></div>
   <div class="grid2">
     <div class="card"><h3>Vocabulary that is design-specific</h3>
       <p class="sub">Jaccard overlap ${t.vocab_overlap.jaccard} (names used ≥3×). Names that show up in one design but not the other:</p>
       <div class="grid2">
         <div><div class="h"><span class="pill pc">only coupled</span></div><div id="onlyc"></div></div>
         <div><div class="h"><span class="pill ps">only standalone</span></div><div id="onlys"></div></div>
       </div></div>
     <div class="card"><h3>Persona archetype mix</h3>
       <p class="sub">Coarse family from the persona name. Coupled uses its single name; standalone shown for v1 &amp; v3.</p>
       <div id="arch"></div>
       <div class="cap"><b>How to read:</b> every persona name is sorted into a coarse family (advisor / supporter / other) by keyword. The three bars compare the coupled persona with the two standalone framings — <b>similar heights mean the model adopts the same kind of character regardless of design</b> (here, mostly "advisor").</div></div>
   </div>
   <div class="card"><h3>Why average v1 &amp; v3?</h3>
     <p class="sub">For traits <b>both</b> standalone framings named (n=${t.averaging.n_both} trait-pairs): v1 vs v3 self-score. r=${fmt(t.averaging.r_v1v3)}, mean |v1−v3|=${fmt(t.averaging.mean_abs_diff)} pts — averaging halves that wording noise.</p>
     <div id="avg"></div>
     <div class="cap"><b>How to read:</b> each dot is one trait that <b>both</b> standalone prompts named — its score from v1 (x-axis) against v3 (y-axis). Dots hugging the dotted <code>v1=v3</code> line mean the two framings agreed; the scatter around it is per-prompt wording noise, which taking the average of v1 and v3 cuts roughly in half.</div></div>`;

  // value hist
  barTwo("tv_val","", [...Array(11).keys()], t.coupled.value_hist, t.standalone.value_hist, "# traits");
  // count hist
  const ks=[...new Set([...Object.keys(t.coupled.count_hist),...Object.keys(t.standalone.count_hist)])].map(Number).sort((a,b)=>a-b);
  barTwo("tv_cnt", "", ks, ks.map(k=>t.coupled.count_hist[k]||0), ks.map(k=>t.standalone.count_hist[k]||0), "# datapoints");

  const topTbl = arr => `<table><thead><tr><th>trait</th><th class="num">n</th><th class="num">mean</th></tr></thead><tbody>`+
    arr.map(r=>`<tr><td>${esc(r.name)}</td><td class="num">${r.count}</td><td class="num">${fmt(r.mean_val)}</td></tr>`).join("")+`</tbody></table>`;
  $("#topc").innerHTML=topTbl(t.coupled.top);
  $("#tops").innerHTML=topTbl(t.standalone.top);

  const sv=t.shared_values;
  $("#sharedtbl").innerHTML = sv.length? `<table><thead><tr><th>trait</th><th class="num">coupled</th><th class="num">standalone</th><th class="num">Δ</th><th class="num">n(c/s)</th></tr></thead><tbody>`+
    sv.map(r=>`<tr><td>${esc(r.name)}</td><td class="num">${fmt(r.coupled_mean)}</td><td class="num">${fmt(r.standalone_mean)}</td><td class="num ${cls(r.delta)}">${sign(r.delta)}</td><td class="num muted">${r.coupled_n}/${r.standalone_n}</td></tr>`).join("")+`</tbody></table>`
    : `<p class="muted">No trait cleared ≥5 uses in both designs.</p>`;

  const lst = arr => `<table><tbody>`+arr.map(r=>`<tr><td>${esc(r.name)}</td><td class="num muted">${r.count}</td></tr>`).join("")+`</tbody></table>`;
  $("#onlyc").innerHTML=lst(t.vocab_overlap.only_coupled);
  $("#onlys").innerHTML=lst(t.vocab_overlap.only_standalone);

  // archetype grouped bar
  const fam=["advisor","supporter","other"];
  Plotly.newPlot("arch",[
    {x:fam,y:fam.map(f=>t.archetype.coupled[f]||0),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:fam,y:fam.map(f=>t.archetype.v1[f]||0),name:"standalone v1",type:"bar",marker:{color:"#f472b6"}},
    {x:fam,y:fam.map(f=>t.archetype.v3[f]||0),name:"standalone v3",type:"bar",marker:{color:"#be185d"}},
  ],{barmode:"group",margin:{t:10,r:10,b:40,l:44},height:280,legend:{orientation:"h",y:1.18},font:{size:12}},PCONF);

  // averaging scatter
  const sc=t.averaging.scatter;
  Plotly.newPlot("avg",[
    {x:sc.map(p=>p[0]),y:sc.map(p=>p[1]),mode:"markers",type:"scattergl",
     marker:{color:COL_S,size:5,opacity:.35},name:"trait"},
    {x:[0,10],y:[0,10],mode:"lines",line:{dash:"dot",color:"#94a3b8"},name:"v1=v3",hoverinfo:"skip"},
  ],{margin:{t:10,r:10,b:44,l:44},height:320,xaxis:{title:"v1 self-score",range:[-.5,10.5]},
     yaxis:{title:"v3 self-score",range:[-.5,10.5]},showlegend:false,font:{size:12}},PCONF);
}

/* ---------- responses ---------- */
function renderResponses(){
  const r=M.responses, el=$("#tab-responses");
  el.innerHTML=`
   <div class="card"><h3>Judged behaviour: in-context vs alone</h3>
     <p class="sub">Same prompts, paired. Δ = standalone − coupled. d<sub>z</sub>=paired effect size, %↑ = share of prompts where the cold reply scored higher, r = cross-design correlation.</p>
     <div id="dvbar"></div>
     <div class="cap"><b>How to read:</b> average judge score (0–10) of the reply for each behaviour, blue = coupled, pink = standalone. Where blue is taller, <b>writing a persona first produced a warmer / more formal / more advice-heavy reply</b> than replying cold. The table below adds the paired effect size (d<sub>z</sub>), the share of prompts where the cold reply scored higher (%↑), and how tightly the two designs track each other (r).</div>
     <table style="margin-top:12px"><thead><tr><th>DV</th><th class="num">coupled</th><th class="num">standalone</th><th class="num">Δ</th><th class="num">d<sub>z</sub></th><th class="num">%↑</th><th class="num">r</th></tr></thead>
     <tbody>${DVS.map(k=>`<tr><td>${k}</td><td class="num">${fmt(r.dv[k].coupled_mean)}</td><td class="num">${fmt(r.dv[k].standalone_mean)}</td><td class="num ${cls(r.dv[k].delta_mean)}">${sign(r.dv[k].delta_mean)}</td><td class="num">${fmt(r.dv[k].d_z)}</td><td class="num">${fmt(r.dv[k].pct_up,0)}</td><td class="num">${fmt(r.dv[k].r)}</td></tr>`).join("")}</tbody></table></div>
   <div class="card" style="padding-bottom:6px"><h3 style="margin-bottom:2px">Per-prompt agreement</h3>
     <p class="sub">The three plots below place each of the ${M.meta.n_pairs} prompts as one dot: its coupled score on the x-axis, standalone score on the y-axis.</p>
     <div class="cap"><b>How to read:</b> a dot on the dotted line = the two designs gave that prompt the same score; below the line = the cold (standalone) reply scored lower; above = higher. A tight diagonal cloud (high r) means the design mostly shifts every reply by a similar amount and preserves the ranking; a fuzzy cloud (low r) means the persona scaffold reshuffles which prompts score high.</div></div>
   <div class="grid3" id="scatters"></div>
   <div class="grid2">
     <div class="card"><h3>Δ by scenario</h3><p class="sub">standalone − coupled, per DV.</p><div id="byscn"></div>
       <div class="cap"><b>How to read:</b> for each scenario, the average change (standalone − coupled) in each behaviour. Bars below zero mean the cold reply scored <b>lower</b> than the persona-primed reply for that topic; this shows whether the effect is uniform or concentrated in a few scenarios.</div></div>
     <div class="card"><h3>Δ by eval-condition</h3><p class="sub">standalone − coupled, per DV.</p><div id="bycond"></div>
       <div class="cap"><b>How to read:</b> the same standalone − coupled change, split by the eval frame (deployment / eval_cue / neutral_sys). Roughly equal bars across the three mean the framing doesn't modulate the design effect.</div></div>
   </div>
   <div class="card"><h3>Δ by user role</h3><p class="sub">standalone − coupled, per DV, split by the user role in the prompt.</p><div id="byrole"></div>
     <div class="cap"><b>How to read:</b> the standalone − coupled change per DV, broken out by <b>who the user is</b> (12 roles). Bars below zero = the cold reply scored lower than the persona-primed reply for that user type. Watch for roles where the design effect is amplified — e.g. vulnerable/crisis users, where writing a persona first may add warmth that the cold reply drops.</div></div>
   <div class="card"><h3>Warmth / formality / advice by user role — coupled vs solo</h3>
     <p class="sub">Absolute judged DV (0–10), not the delta. Each role has three positions (warmth, formality, advice); within each, the <b>wide light bar is coupled</b> and the <b>narrow dark bar nested inside is solo</b> (standalone). The two share one footprint so you can read the change directly.</p>
     <div id="rolebars"></div>
     <div class="cap"><b>How to read:</b> hue = DV (blue warmth / pink formality / purple advice); the nested dark bar is the solo (cold) value, the surrounding light bar is coupled. Where the dark bar falls short of the light one, that behaviour drops when the persona isn't in context for that user role; where it overshoots, the cold reply scores higher.</div></div>
   <div class="grid2">
     <div class="card"><h3>Primary emotion mix</h3><div id="emo"></div>
       <div class="cap"><b>How to read:</b> how often the blind judge tagged the reply with each primary-emotion label, coupled vs standalone. Watch for labels that appear in only one design — e.g. cold standalone replies pick up <code>neutral</code> / <code>clinical</code> tags that the persona-primed replies almost never get.</div></div>
     <div class="card"><h3>Reply length</h3><p class="sub">Mean characters per reply.</p><div id="len"></div>
       <div class="cap"><b>How to read:</b> average characters per reply. A taller standalone bar means <b>replying cold produces longer answers</b>; the persona-primed replies are more compressed.</div></div>
   </div>`;

  Plotly.newPlot("dvbar",[
    {x:DVS,y:DVS.map(k=>r.dv[k].coupled_mean),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:DVS,y:DVS.map(k=>r.dv[k].standalone_mean),name:"standalone",type:"bar",marker:{color:COL_S}},
  ],{barmode:"group",height:300,margin:{t:14,r:10,b:40,l:44},legend:{orientation:"h",y:1.16},yaxis:{title:"0–10",range:[0,10]},font:{size:12}},PCONF);

  const sd=$("#scatters");
  DVS.forEach(k=>{const d=document.createElement("div");d.className="card";d.innerHTML=`<h3 style="font-size:14px">${k}</h3><p class="sub">coupled (x) vs standalone (y), r=${fmt(r.dv[k].r)}</p><div id="sc_${k}"></div>`;sd.appendChild(d);
    const pts=r.scatter[k];
    Plotly.newPlot("sc_"+k,[
      {x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),mode:"markers",type:"scattergl",marker:{color:"#7c3aed",size:6,opacity:.35}},
      {x:[0,10],y:[0,10],mode:"lines",line:{dash:"dot",color:"#94a3b8"},hoverinfo:"skip"},
    ],{margin:{t:6,r:8,b:36,l:36},height:230,xaxis:{range:[-.5,10.5]},yaxis:{range:[-.5,10.5]},showlegend:false,font:{size:11}},PCONF);
  });

  const deltaBars=(divid,rows,key,h)=>{
    Plotly.newPlot(divid,DVS.map((k,i)=>({x:rows.map(r=>r[key]),y:rows.map(r=>r[k]),name:k,type:"bar",
      marker:{color:[COL_C,COL_S,"#7c3aed"][i]}})),
      {barmode:"group",height:h||300,margin:{t:10,r:8,b:120,l:44},legend:{orientation:"h",y:1.16},
       yaxis:{title:"Δ"},font:{size:11}},PCONF);
  };
  deltaBars("byscn",r.by_scenario,"scenario");
  deltaBars("bycond",r.by_condition,"condition");
  deltaBars("byrole",r.by_role,"role",340);
  // absolute warmth/formality/advice per role, coupled (wide light) with solo nested (narrow dark)
  (function(){
    const rows=r.by_role_abs||[]; if(!rows.length) return;
    const roles=rows.map(x=>x.role), idx=roles.map((_,i)=>i);
    const offs=[-0.26,0,0.26], Wg=0.24;
    const LIGHT={warmth:"#93c5fd",formality:"#f9a8d4",advice_density:"#c4b5fd"};
    const DARK={warmth:"#1d4ed8",formality:"#be185d",advice_density:"#6d28d9"};
    const traces=[];
    DVS.forEach((k,di)=>{const x=idx.map(i=>i+offs[di]);
      traces.push({x,y:rows.map(v=>v[k+"_c"]),width:Wg,type:"bar",name:k+" · coupled",marker:{color:LIGHT[k]},legendgroup:k});
      traces.push({x,y:rows.map(v=>v[k+"_s"]),width:Wg*0.46,type:"bar",name:k+" · solo",marker:{color:DARK[k]},legendgroup:k});
    });
    Plotly.newPlot("rolebars",traces,{barmode:"overlay",height:440,margin:{t:10,r:10,b:132,l:46},
      legend:{orientation:"h",y:1.1},yaxis:{title:"judged DV 0–10",range:[0,10]},
      xaxis:{tickmode:"array",tickvals:idx,ticktext:roles,tickangle:-40,range:[-0.6,roles.length-0.4]},font:{size:11},bargap:0},PCONF);
  })();

  const emc=r.emotion.coupled, ems=r.emotion.standalone;
  const elabs=[...new Set([...Object.keys(emc),...Object.keys(ems)])];
  Plotly.newPlot("emo",[
    {x:elabs,y:elabs.map(e=>emc[e]||0),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:elabs,y:elabs.map(e=>ems[e]||0),name:"standalone",type:"bar",marker:{color:COL_S}},
  ],{barmode:"group",height:280,margin:{t:10,r:8,b:80,l:40},legend:{orientation:"h",y:1.18},font:{size:11}},PCONF);

  Plotly.newPlot("len",[
    {x:["coupled","standalone"],y:[r.length.coupled_chars,r.length.standalone_chars],type:"bar",
     marker:{color:[COL_C,COL_S]}},
  ],{height:280,margin:{t:10,r:8,b:40,l:50},yaxis:{title:"chars"},font:{size:12}},PCONF);
}

/* ---------- faithfulness ---------- */
const FKEYS=[["intensity_fidelity","intensity"],["expression_fidelity","expression"],
  ["persona_gestalt","gestalt"],["overall_faithfulness","overall"]];
function renderFaith(){
  const el=$("#tab-faith"), F=M.faith;
  if(!F||!F.available){
    el.innerHTML=`<div class="note">No faithfulness judging found for <code>${$("#model").value}</code>. Run <code>python -m src.run_exp3_persona_judge --models ${$("#model").value}</code> and re-build the report.</div>`;
    return;
  }
  const o=F.paired.overall_faithfulness;
  el.innerHTML=`
    <div class="card"><h3>Does the reply embody the persona it declared?</h3>
      <p class="sub">The persona-judge (same model, blind to the DV judge) reads the user message, the persona spec (traits + declared 0–10 intensity + predicted expression) and the reply, and scores how faithfully the reply <b>is</b> that persona on four dimensions. Coupled reply is judged against its in-context persona; the cold reply against the standalone v1 &amp; v3 personas (scores averaged). Coverage: coupled ${F.n_coupled}, cold ${F.n_cold}, paired ${F.n_paired} of ${M.meta.n_pairs} (the rest emitted no persona to judge).</p>
      <div class="defs">
        <div><span class="t">intensity</span> — did each trait show up as <b>strongly</b> as declared? The spec says empathy = 8/10; does the reply read like an 8, or more like a 3? (average per-trait intensity match)</div>
        <div><span class="t">expression</span> — did each trait appear the <b>way</b> predicted? Spec: "empathy = validating their feelings"; if the reply is warm but via casual jokes instead, that's only a partial match.</div>
        <div><span class="t">gestalt</span> — step back from the checklist: read as a whole, does the reply <b>sound like the named character</b>? A reply can tick individual traits yet not feel like "Cautious Clinical Advisor" — or nail the character without matching every trait. The holistic "does it read as this persona" score.</div>
        <div><span class="t">overall</span> — the judge's bottom line, weighting intensity and gestalt most heavily (never more than 2 pts above gestalt).</div>
      </div>
      <div id="f_bar"></div>
      <div class="cap"><b>How to read:</b> average judge score (0–10) on each of the four dimensions, blue = coupled, pink = cold. Taller blue everywhere means <b>a reply written with the persona in context embodies it much more closely</b> than a cold reply embodies the same model's standalone-declared persona — though the cold bars sitting well above 0 mean the self-model still describes cold behaviour partially. <b>Error bars are 95% confidence intervals of each mean</b> (1.96·SEM over the ${F.n_paired} paired prompts); they are tight because each mean pools hundreds of prompts, so the coupled–cold gap sits far outside sampling noise. The per-trait figure below opens up <i>which</i> traits drive the gap.</div>
      <table style="margin-top:12px"><thead><tr><th>dimension</th><th class="num">coupled</th><th class="num">cold</th><th class="num">Δ</th><th class="num">d<sub>z</sub></th><th class="num">cold&gt;coupled</th><th class="num">r</th></tr></thead>
      <tbody>${FKEYS.map(([k,lab])=>{const p=F.paired[k];return `<tr><td>${lab}</td><td class="num">${fmt(p.coupled_mean)}</td><td class="num">${fmt(p.cold_mean)}</td><td class="num ${cls(p.delta_mean)}">${sign(p.delta_mean)}</td><td class="num">${fmt(p.d_z)}</td><td class="num">${fmt(p.pct_cold_higher,0)}%</td><td class="num">${fmt(p.r)}</td></tr>`;}).join("")}</tbody></table></div>

    ${(F.per_trait&&F.per_trait.length)?`<div class="card"><h3>Per-trait: which declared traits actually show up?</h3>
      <p class="sub">For every trait the <b>generator</b> named in its persona (top ${F.per_trait.length} by frequency): the intensity <b>it declared for itself</b> (0–10, from its own persona block — the in-context persona for coupled, the standalone v1/v3 persona for cold) vs the intensity the <b>judge observed</b> in the reply. The judge is handed the declared number and echoes it; only the observed value is the judge's own reading. This answers the "empathy was selected but not expressed" question directly.</p>
      <div id="f_pt"></div>
      <div class="cap" id="f_pt_cap"></div>
      <p class="sub" style="margin:14px 0 4px">Declared is shown for <b>each design separately</b> — coupled reply vs its in-context persona, cold reply vs its standalone persona. gap = declared − observed (positive = declared but under-delivered).</p>
      <table><thead><tr><th>trait</th><th class="num">n C/K</th><th class="num">coupled decl→obs</th><th class="num">gap</th><th class="num">cold decl→obs</th><th class="num">gap</th><th class="num">cold expr y/p/n</th></tr></thead>
      <tbody>${F.per_trait.map(t=>`<tr><td>${esc(t.name)}</td><td class="num muted">${t.n_coupled}/${t.n_cold}</td><td class="num">${fmt(t.declared_coupled,1)} → ${fmt(t.obs_coupled,1)}</td><td class="num ${t.gap_coupled>=2?"neg":""}">${t.gap_coupled==null?"—":fmt(t.gap_coupled,1)}</td><td class="num">${fmt(t.declared_cold,1)} → ${fmt(t.obs_cold,1)}</td><td class="num ${t.gap_cold>=2?"neg":""}">${t.gap_cold==null?"—":fmt(t.gap_cold,1)}</td><td class="num muted">${t.yes_cold==null?"—":t.yes_cold+"/"+t.partial_cold+"/"+t.no_cold}</td></tr>`).join("")}</tbody></table></div>`:""}
    <div class="grid2">
      <div class="card"><h3>Distribution of overall faithfulness</h3><p class="sub">How many replies land at each 0–10 score.</p><div id="f_hist"></div>
        <div class="cap"><b>How to read:</b> coupled (blue) piles up at the high end (8–10); cold (pink) spreads lower and left. A long pink tail near 0–3 flags cold replies that <b>contradict</b> the persona the model said it would adopt.</div></div>
      <div class="card"><h3>Per-prompt: coupled vs cold</h3><p class="sub">Each dot = one prompt, coupled overall (x) vs cold overall (y), r=${fmt(o.r)}.</p><div id="f_sc"></div>
        <div class="cap"><b>How to read:</b> dots below the dotted line = the cold reply was <b>less</b> faithful than the in-context reply for that prompt (the common case). Low r means the persona that a reply embodies well in-context isn't necessarily the one the cold reply drifts toward.</div></div>
    </div>
    <div class="grid2">
      <div class="card"><h3>Overall faithfulness by scenario</h3><p class="sub">coupled vs cold, mean overall.</p><div id="f_scn"></div>
        <div class="cap"><b>How to read:</b> the coupled−cold gap per topic. A gap that persists across scenarios means the in-context advantage isn't driven by one subject area.</div></div>
      <div class="card"><h3>By eval-condition &amp; extra signals</h3><p class="sub">coupled vs cold overall, per frame.</p><div id="f_cond"></div>
        <div class="cap"><b>How to read:</b> faithfulness split by deployment / eval_cue / neutral_sys. Below: <b>undeclared dominant traits</b> the judge found (traits that shape the reply but weren't in the spec) — higher = the reply expresses a character the spec didn't mention.</div>
        <div class="grid2" style="margin-top:12px">
          ${stat(fmt(F.unlisted.coupled)+" / "+fmt(F.unlisted.cold),"undeclared dominant traits","coupled / cold (mean per reply)")}
          ${stat(fmt(F.v1v3.r)+" · "+fmt(F.v1v3.mean_abs_diff),"v1 vs v3 agreement","r · mean |Δ| on the same cold reply (n="+F.v1v3.n+")")}
        </div></div>
    </div>
    ${(F.by_role&&F.by_role.length)?`<div class="card"><h3>Overall faithfulness by user role</h3>
      <p class="sub">coupled vs cold overall faithfulness, split by the user role in the prompt.</p><div id="f_role"></div>
      <div class="cap"><b>How to read:</b> mean overall faithfulness (0–10) per user role, blue = coupled, pink = cold. The blue−pink gap per role is how much the persona had to be in context for the reply to stay in character for <b>that user type</b>. A tall gap on vulnerable/crisis users means the model's cold reply drifts furthest from the empathetic persona it declared for them.</div></div>`:""}`;

  Plotly.newPlot("f_bar",[
    {x:FKEYS.map(f=>f[1]),y:FKEYS.map(([k])=>F.paired[k].coupled_mean),name:"coupled",type:"bar",marker:{color:COL_C},
     error_y:{type:"data",array:FKEYS.map(([k])=>F.paired[k].coupled_ci),visible:true,thickness:1.3,width:4,color:"#1e3a8a"}},
    {x:FKEYS.map(f=>f[1]),y:FKEYS.map(([k])=>F.paired[k].cold_mean),name:"cold",type:"bar",marker:{color:COL_S},
     error_y:{type:"data",array:FKEYS.map(([k])=>F.paired[k].cold_ci),visible:true,thickness:1.3,width:4,color:"#9d174d"}},
  ],{barmode:"group",height:320,margin:{t:14,r:10,b:40,l:44},legend:{orientation:"h",y:1.15},yaxis:{title:"0–10",range:[0,10]},font:{size:12}},PCONF);

  if(F.per_trait&&F.per_trait.length){
    const pts=F.per_trait.slice().reverse();       // most frequent at top
    const names=pts.map(t=>t.name);
    const lineXY=(dk,ok)=>{const x=[],y=[];pts.forEach(t=>{if(t[dk]!=null&&t[ok]!=null){x.push(t[ok],t[dk],null);y.push(t.name,t.name,null);}});return{x,y};};
    const lC=lineXY("declared_coupled","obs_coupled"), lK=lineXY("declared_cold","obs_cold");
    Plotly.newPlot("f_pt",[
      {x:lC.x,y:lC.y,mode:"lines",line:{color:"#bfdbfe",width:3},hoverinfo:"skip",showlegend:false},
      {x:lK.x,y:lK.y,mode:"lines",line:{color:"#fbcfe8",width:3},hoverinfo:"skip",showlegend:false},
      {x:pts.map(t=>t.declared_coupled),y:names,mode:"markers",name:"coupled — declared ◆",marker:{color:COL_C,size:11,symbol:"diamond"}},
      {x:pts.map(t=>t.obs_coupled),y:names,mode:"markers",name:"coupled — observed ●",marker:{color:COL_C,size:10,symbol:"circle"}},
      {x:pts.map(t=>t.declared_cold),y:names,mode:"markers",name:"cold — declared ◆",marker:{color:COL_S,size:11,symbol:"diamond"}},
      {x:pts.map(t=>t.obs_cold),y:names,mode:"markers",name:"cold — observed ●",marker:{color:COL_S,size:10,symbol:"circle"}},
    ],{height:34*names.length+100,margin:{t:8,r:14,b:40,l:160},xaxis:{title:"trait intensity 0–10",range:[0,10.3],zeroline:false},yaxis:{automargin:true},legend:{orientation:"h",y:1.05},font:{size:11.5}},PCONF);
    const byGap=F.per_trait.filter(t=>t.gap_cold!=null).slice().sort((a,b)=>b.gap_cold-a.gap_cold);
    const emp=F.per_trait.find(t=>/empath|compassion|emotional/i.test(t.name)&&t.gap_cold!=null);
    const w=emp||byGap[0];
    const ex=w?`e.g. <b>${esc(w.name)}</b>: declared ${fmt(w.declared_cold,1)}/10 in the cold persona but the judge saw only ${fmt(w.obs_cold,1)}/10 in the cold reply (a ${fmt(w.gap_cold,1)}-pt drop) — while the coupled reply delivered ${fmt(w.obs_coupled,1)}/10 against its own ${fmt(w.declared_coupled,1)} declaration. `:"";
    $("#f_pt_cap").innerHTML=`<b>How to read:</b> each trait has two marker pairs, one per design. A <b>diamond ◆ = declared</b> (the generator's own 0–10 for that trait, from its persona block) and a <b>circle ● = observed</b> (the judge's reading of the reply). <span style="color:${COL_C};font-weight:700">Blue = coupled</span>, <span style="color:${COL_S};font-weight:700">pink = cold</span> — so declared is shown for <i>both</i> settings, not one. The two diamonds sit close together (the generator declares similar values whether or not a reply follows), but the pink circle drops far below its pink diamond for affective traits: <b>declared but not expressed</b> once the persona isn't in context. ${ex}The blue circle stays near its blue diamond — in-context replies deliver what they declared.`;
  }

  Plotly.newPlot("f_hist",[
    {x:[...Array(11).keys()],y:F.hist.coupled,name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:[...Array(11).keys()],y:F.hist.cold,name:"cold",type:"bar",marker:{color:COL_S}},
  ],{barmode:"group",height:300,margin:{t:10,r:8,b:40,l:44},legend:{orientation:"h",y:1.16},xaxis:{title:"overall faithfulness"},yaxis:{title:"# replies"},font:{size:12}},PCONF);

  Plotly.newPlot("f_sc",[
    {x:F.scatter.map(p=>p[0]),y:F.scatter.map(p=>p[1]),mode:"markers",type:"scattergl",marker:{color:"#7c3aed",size:6,opacity:.35}},
    {x:[0,10],y:[0,10],mode:"lines",line:{dash:"dot",color:"#94a3b8"},hoverinfo:"skip"},
  ],{height:300,margin:{t:10,r:8,b:40,l:40},xaxis:{title:"coupled",range:[-.5,10.5]},yaxis:{title:"cold",range:[-.5,10.5]},showlegend:false,font:{size:12}},PCONF);

  const scnBar=(divid,rows,key,h)=>Plotly.newPlot(divid,[
    {x:rows.map(r=>r[key]),y:rows.map(r=>r.coupled),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:rows.map(r=>r[key]),y:rows.map(r=>r.cold),name:"cold",type:"bar",marker:{color:COL_S}},
  ],{barmode:"group",height:h||300,margin:{t:10,r:8,b:120,l:44},legend:{orientation:"h",y:1.16},yaxis:{title:"0–10",range:[0,10]},font:{size:11}},PCONF);
  scnBar("f_scn",F.by_scenario,"scenario");
  scnBar("f_cond",F.by_condition,"condition");
  if(F.by_role&&F.by_role.length) scnBar("f_role",F.by_role,"role",340);
}

/* ---------- examples ---------- */
function renderExamples(){
  const el=$("#tab-examples");
  el.innerHTML=`
   <div class="card">
     <div class="controls">
       <div><label>Scenario</label><select id="ex_scn"></select></div>
       <div><label>Role</label><select id="ex_role"></select></div>
       <div><label>Condition</label><select id="ex_cond"></select></div>
     </div>
     <div id="ex_body"></div>
   </div>`;
  const fill=(id,vals)=>{const s=$(id);s.innerHTML=vals.map(v=>`<option>${esc(v)}</option>`).join("");};
  fill("#ex_scn",M.meta.scenarios);fill("#ex_role",M.meta.roles);fill("#ex_cond",M.meta.conditions);
  ["#ex_scn","#ex_role","#ex_cond"].forEach(id=>$(id).onchange=showExample);
  showExample();
}
function traitRows(arr){return arr.map(t=>`<div class="tr"><span>${esc(t.name)}</span><span class="v">${fmt(t.value,1)}</span></div>`).join("");}
function mergedRows(arr){return `<table><thead><tr><th>trait</th><th class="num">v1</th><th class="num">v3</th><th class="num">avg</th></tr></thead><tbody>`+
  arr.map(t=>`<tr><td>${esc(t.name)}${t.n===2?' <span class="pill ps" style="font-size:10px">both</span>':''}</td><td class="num muted">${fmt(t.v1,1)}</td><td class="num muted">${fmt(t.v3,1)}</td><td class="num">${fmt(t.avg,1)}</td></tr>`).join("")+`</tbody></table>`;}
function judgeBadges(j){return DVS.map(k=>`<span class="jbadge">${k}: <b>${fmt(j[k],1)}</b></span>`).join("")+`<span class="jbadge">${esc(j.primary_emotion)}</span>`;}
function faithScores(j){return FKEYS.map(([k,lab])=>`<span class="jbadge">${lab}: <b>${fmt(j[k],1)}</b></span>`).join("");}
function faithTraitTbl(traits){
  if(!traits||!traits.length)return `<p class="muted" style="font-size:12px">no per-trait detail</p>`;
  return `<table><thead><tr><th>trait</th><th class="num">decl</th><th class="num">obs</th><th class="num">Δ</th><th>expr</th></tr></thead><tbody>`+
    traits.map(t=>`<tr><td>${esc(t.name)}</td><td class="num">${fmt(t.dec,0)}</td><td class="num">${fmt(t.obs,0)}</td><td class="num ${cls(t.dev)}">${t.dev==null?"—":(t.dev>0?"+":"")+t.dev}</td><td>${esc(t.match)}</td></tr>`).join("")+`</tbody></table>`;
}
function faithSection(f){
  const cj=f.coupled, ca=f.cold_avg, v1=f.v1, v3=f.v3;
  return `<div class="card" style="margin-top:14px"><h3>Faithfulness verdict — does each reply embody its persona?</h3>
    <p class="sub">Scored by the persona-judge (0–10): observed vs declared intensity per trait, and an overall. Δ = observed − declared.</p>
    <div class="two">
      <div class="col-c"><div class="h"><span class="pill pc">coupled reply vs its in-context persona</span></div>
        ${cj?`${faithScores(cj)}${cj.n_unlisted?`<span class="jbadge">+${cj.n_unlisted} undeclared</span>`:""}
          <div style="margin-top:8px">${faithTraitTbl(cj.traits)}</div>
          ${cj.notes?`<div class="cap" style="margin-top:8px">${esc(cj.notes)}</div>`:""}`
          :`<p class="muted">no persona declared -> not judged</p>`}
      </div>
      <div class="col-s"><div class="h"><span class="pill ps">cold reply vs standalone persona (avg v1/v3)</span></div>
        ${ca?`${FKEYS.map(([k,lab])=>`<span class="jbadge">${lab}: <b>${fmt(ca[k],1)}</b></span>`).join("")}
          <details style="margin-top:8px"><summary>v1 &amp; v3 per-trait detail</summary>
            <div class="grid2" style="margin-top:8px">
              <div><div class="h">vs v1 persona</div>${v1?faithTraitTbl(v1.traits):`<p class="muted">—</p>`}</div>
              <div><div class="h">vs v3 persona</div>${v3?faithTraitTbl(v3.traits):`<p class="muted">—</p>`}</div>
            </div></details>`
          :`<p class="muted">no standalone persona -> not judged</p>`}
      </div>
    </div></div>`;
}
function showExample(){
  const scn=$("#ex_scn").value, role=$("#ex_role").value, cond=$("#ex_cond").value;
  const rec=M.browse.find(b=>b.scenario===scn&&b.role===role&&b.condition===cond);
  const body=$("#ex_body");
  if(!rec){body.innerHTML=`<p class="muted">No datapoint for that combination.</p>`;return;}
  body.innerHTML=`
    <p class="sub" style="margin-top:10px"><b>User message</b> (x=${esc(rec.x)}):</p>
    <div class="resp" style="max-height:150px">${esc(rec.user)}</div>
    <div class="two" style="margin-top:14px">
      <div class="col-c">
        <div class="h"><span class="pill pc">coupled</span> — traits written, then reply</div>
        <div><b>${esc(rec.c_name)||"<span class=muted>(no persona)</span>"}</b></div>
        ${traitRows(rec.c_traits)}
        <p class="sub" style="margin:10px 0 4px">Reply (conditioned on the persona above)</p>
        <div class="resp">${esc(rec.c_resp)}</div>
        <div style="margin-top:8px">${judgeBadges(rec.c_judge)}</div>
      </div>
      <div class="col-s">
        <div class="h"><span class="pill ps">standalone</span> — traits alone (avg v1/v3), reply cold</div>
        <div class="muted" style="font-size:12px">v1: <b>${esc(rec.s_v1_name)}</b> · v3: <b>${esc(rec.s_v3_name)}</b></div>
        ${mergedRows(rec.s_traits)}
        <p class="sub" style="margin:10px 0 4px">Reply (no persona in context)</p>
        <div class="resp">${esc(rec.s_resp)}</div>
        <div style="margin-top:8px">${judgeBadges(rec.s_judge)}</div>
      </div>
    </div>
    ${rec.faith?faithSection(rec.faith):""}
    <details><summary>Raw v1 &amp; v3 persona traits</summary>
      <div class="grid2" style="margin-top:8px">
        <div><div class="h">v1 traits</div>${traitRows(rec.s_v1_traits)}</div>
        <div><div class="h">v3 traits</div>${traitRows(rec.s_v3_traits)}</div>
      </div></details>`;
}

/* ---------- cross-model summary views ---------- */
function renderResponsesSummary(){
  $("#tab-responses").innerHTML=`
    <div class="card"><h3>What writing the persona first does to the reply — all models</h3>
      <p class="sub">The single most important Responses takeaway. For each judged DV, Δ = <b>standalone (cold reply) − coupled (in-context reply)</b>, one bar per model. Above 0 = the cold reply scored higher; below 0 = writing the persona first raised that DV.</p>
      <div id="s_resp"></div>
      <div class="cap"><b>How to read:</b> each cluster is a DV (warmth / formality / advice_density); the four coloured bars are the models. Bar height = how much that behaviour shifts between the two designs. Same-side bars across models = a consistent design effect; opposite signs = model-specific (writing a persona makes some models warmer, others cooler).</div>
      <table style="margin-top:12px"><thead><tr><th>model</th>${DVS.map(k=>`<th class="num">${k} Δ (d<sub>z</sub>)</th>`).join("")}</tr></thead>
      <tbody>${MODELS_LIST.map(m=>{const dv=DATA.models[m].responses.dv;return `<tr><td>${m}</td>${DVS.map(k=>`<td class="num ${cls(dv[k].delta_mean)}">${sign(dv[k].delta_mean)} <span class="muted">(${fmt(dv[k].d_z)})</span></td>`).join("")}</tr>`;}).join("")}</tbody></table>
    </div>`;
  Plotly.newPlot("s_resp",MODELS_LIST.map((m,i)=>({x:DVS,y:DVS.map(k=>DATA.models[m].responses.dv[k].delta_mean),name:m,type:"bar",marker:{color:MCOLORS[i]}})),
    {barmode:"group",height:380,margin:{t:14,r:10,b:44,l:54},legend:{orientation:"h",y:1.13},yaxis:{title:"Δ  (standalone − coupled)",zeroline:true,zerolinecolor:"#94a3b8"},font:{size:12}},PCONF);
}
function renderFaithSummary(){
  $("#tab-faith").innerHTML=`
    <div class="card"><h3>Persona faithfulness: in-context vs cold — all models</h3>
      <p class="sub">Does the reply embody the persona it declared (overall faithfulness, 0–10)? Coupled = reply written with the persona in context; cold = scaffold-free reply vs the standalone persona. The key Faithfulness takeaway: every model embodies its persona far better when it wrote it in context.</p>
      <div id="s_faith"></div>
      <div class="cap"><b>How to read:</b> per model, blue = coupled and pink = cold overall faithfulness; the blue−pink gap is the in-context advantage, negative (cold lower) for every model. <b>Error bars are 95% confidence intervals of each mean</b> (1.96·SEM). They are tight for the three well-covered models — the gap is far outside sampling noise — but very wide for <b>llama-3.1-8b</b>, which rests on only a handful of judged pairs (it rarely emits a persona); treat its bars as indicative only.</div>
      <table style="margin-top:12px"><thead><tr><th>model</th><th class="num">coupled</th><th class="num">cold</th><th class="num">Δ</th><th class="num">d<sub>z</sub></th><th class="num">paired n</th></tr></thead>
      <tbody>${MODELS_LIST.map(m=>{const p=DATA.models[m].faith.paired.overall_faithfulness;return `<tr><td>${m}</td><td class="num">${fmt(p.coupled_mean)}</td><td class="num">${fmt(p.cold_mean)}</td><td class="num ${cls(p.delta_mean)}">${sign(p.delta_mean)}</td><td class="num">${fmt(p.d_z)}</td><td class="num muted">${p.n}</td></tr>`;}).join("")}</tbody></table>
    </div>`;
  Plotly.newPlot("s_faith",[
    {x:MODELS_LIST,y:MODELS_LIST.map(m=>DATA.models[m].faith.paired.overall_faithfulness.coupled_mean),name:"coupled (in-context)",type:"bar",marker:{color:COL_C},
     error_y:{type:"data",array:MODELS_LIST.map(m=>DATA.models[m].faith.paired.overall_faithfulness.coupled_ci),visible:true,thickness:1.3,width:4,color:"#1e3a8a"}},
    {x:MODELS_LIST,y:MODELS_LIST.map(m=>DATA.models[m].faith.paired.overall_faithfulness.cold_mean),name:"cold",type:"bar",marker:{color:COL_S},
     error_y:{type:"data",array:MODELS_LIST.map(m=>DATA.models[m].faith.paired.overall_faithfulness.cold_ci),visible:true,thickness:1.3,width:4,color:"#9d174d"}},
  ],{barmode:"group",height:360,margin:{t:14,r:10,b:44,l:44},legend:{orientation:"h",y:1.13},yaxis:{title:"overall faithfulness 0–10",range:[0,10]},font:{size:12}},PCONF);
}
function renderTraitsSummary(){
  $("#tab-traits").innerHTML=`
    <div class="card"><h3>The self-model barely changes between designs — all models</h3>
      <p class="sub">The key Traits takeaway. Mean trait self-score (0–10, the value the generator assigns its own load-bearing traits), coupled vs standalone. Near-level pairs = whether or not a reply follows, the model reports essentially the same traits — so it is the <b>behaviour</b> that shifts (Responses / Faithfulness), not the stated self-model.</p>
      <div id="s_traits"></div>
      <div class="cap"><b>How to read:</b> per model, blue = coupled and pink = standalone mean trait value. The pairs sit almost level for every model (differences well under a point), unlike the behavioural gaps in the other tabs. (llama-3.1-8b barely emits personas, so its trait counts are near zero.)</div>
      <table style="margin-top:12px"><thead><tr><th>model</th><th class="num">coupled value</th><th class="num">standalone value</th><th class="num">traits/reply C/S</th><th class="num">vocab overlap</th></tr></thead>
      <tbody>${MODELS_LIST.map(m=>{const t=DATA.models[m].traits;return `<tr><td>${m}</td><td class="num">${fmt(t.coupled.value_mean)}</td><td class="num">${fmt(t.standalone.value_mean)}</td><td class="num muted">${t.coupled.mean_count}/${t.standalone.mean_count}</td><td class="num">${fmt(t.vocab_overlap.jaccard)}</td></tr>`;}).join("")}</tbody></table>
    </div>`;
  Plotly.newPlot("s_traits",[
    {x:MODELS_LIST,y:MODELS_LIST.map(m=>DATA.models[m].traits.coupled.value_mean),name:"coupled",type:"bar",marker:{color:COL_C}},
    {x:MODELS_LIST,y:MODELS_LIST.map(m=>DATA.models[m].traits.standalone.value_mean),name:"standalone",type:"bar",marker:{color:COL_S}},
  ],{barmode:"group",height:340,margin:{t:14,r:10,b:44,l:44},legend:{orientation:"h",y:1.13},yaxis:{title:"mean trait self-score 0–10",range:[0,10]},font:{size:12}},PCONF);
}
function renderOverviewSummary(){
  const rows=MODELS_LIST.map(m=>{const X=DATA.models[m],dv=X.responses.dv,f=X.faith.paired.overall_faithfulness;
    return `<tr><td>${m}</td><td class="num ${cls(dv.warmth.delta_mean)}">${sign(dv.warmth.delta_mean)}</td><td class="num ${cls(dv.formality.delta_mean)}">${sign(dv.formality.delta_mean)}</td><td class="num ${cls(dv.advice_density.delta_mean)}">${sign(dv.advice_density.delta_mean)}</td><td class="num">${fmt(f.coupled_mean)} → ${fmt(f.cold_mean)}</td><td class="num neg">${sign(f.delta_mean)}</td></tr>`;}).join("");
  $("#tab-overview").innerHTML=`
    <div class="card"><h3>Four models at a glance</h3>
      <p class="sub">Headline numbers per model, one row each. Response Δ = standalone − coupled (how behaviour shifts between designs); faithfulness = how well the reply embodies its declared persona, coupled → cold. Pick a specific model in the dropdown for the full breakdown of any tab.</p>
      <table><thead><tr><th>model</th><th class="num">Δ warmth</th><th class="num">Δ formality</th><th class="num">Δ advice</th><th class="num">faith C→cold</th><th class="num">faith Δ</th></tr></thead><tbody>${rows}</tbody></table></div>
    <div class="grid2">
      <div class="card"><h3>What holds across every model</h3>
        <ul style="margin:0;padding-left:18px;font-size:14px;line-height:1.7">
          <li><b>Writing the persona first makes the reply more faithful to it</b> — coupled &gt; cold overall faithfulness in all four models (faith Δ always negative).</li>
          <li><b>The stated self-model barely moves</b> between designs — the change is in delivered behaviour, not the declared traits.</li>
          <li><b>Cold replies drop the affective traits</b> (empathy / emotional attunement) while keeping the epistemic ones.</li>
        </ul></div>
      <div class="card"><h3>Where models differ</h3>
        <ul style="margin:0;padding-left:18px;font-size:14px;line-height:1.7">
          <li><b>qwen3-235b</b> — most faithful embodier; its cold reply best matches its own self-model.</li>
          <li><b>qwen3-30b</b> — most steerable; the biggest coupled→cold faithfulness drop.</li>
          <li><b>Response Δ signs flip by model</b> — a persona makes some models warmer, others cooler.</li>
          <li><b>llama-3.1-8b</b> — rarely emits a persona, so its persona/faith numbers rest on very few pairs.</li>
        </ul></div>
    </div>`;
}
function renderExamplesSummary(){
  $("#tab-examples").innerHTML=`<div class="card"><h3>Paired examples</h3><div class="note">Examples are per-datapoint (the exact persona + coupled vs cold reply for one prompt), so they aren't summarised across models. <b>Pick a specific model</b> from the dropdown above to browse them by scenario / role / condition.</div></div>`;
}

/* ---------- method ---------- */
function renderMethod(){
  $("#tab-method").innerHTML=`
   <div class="card"><h3>The two designs</h3>
    <div class="two">
      <div class="col-c"><div class="h"><span class="pill pc">coupled</span></div>
        <p>One generation: the model writes its <code>&lt;persona&gt;</code> block (name + 4–6 load-bearing traits, each 0–10) and then, <b>conditioned on it</b>, its <code>&lt;response&gt;</code>. Traits are elicited <b>in-context with the reply</b>. Source: <code>results/exp3_persona/</code>.</p></div>
      <div class="col-s"><div class="h"><span class="pill ps">standalone</span></div>
        <p>Three independent calls: two persona-<b>only</b> prompts (<code>v1</code> staging, <code>v3</code> self-report) whose per-trait 0–10 values are <b>averaged</b>, plus the reply generated separately with <b>no self-modeling scaffold</b> (only the eval-condition frame is kept). Source: <code>results/exp3_decoupled/</code>.</p></div>
    </div></div>
   <div class="card"><h3>Pairing &amp; judging</h3>
     <p class="sub">Both designs run the identical 9-scenario × 12-role × 3-condition factorial on the same <code>prompt_id</code>s. We pair on <code>prompt_id</code> using <b>run 0</b> from each (${M.meta.n_pairs} pairs). Replies are scored by the same blind LLM judge (<code>anthropic/claude-sonnet-4.5</code>) for warmth / formality / advice_density (0–10) + a primary-emotion label — it never sees the persona block, so the trait channel can't leak into the DV.</p></div>
   <div class="card"><h3>Faithfulness judging</h3>
     <p class="sub">A second judge (<code>judge_prompt_persona</code>, same model) measures whether a reply <b>embodies the persona it was paired with</b>. It receives the user message, a cleanly formatted persona spec (each trait with its declared 0–10 intensity and predicted expression) and the reply, and returns per-trait observed intensity + expression match, plus intensity / expression / gestalt / overall fidelity (0–10) and any undeclared dominant traits. The <b>coupled</b> reply is judged against the persona it wrote in-context; the <b>cold</b> reply is judged against the v1 and v3 standalone personas separately and the scores are averaged. Prompts where a framing emitted no persona are simply not scored (hence coverage &lt; ${M.meta.n_pairs}).</p></div>
   <div class="note"><b>Read with care.</b> Trait names are free-text, so cross-design vocabulary overlap is a lower bound (synonyms like “empathy”/“compassion” count as different names). The persona-name archetype is a coarse keyword bucket. Only <b>one model</b> is loaded so far; deltas are within-model and not yet shown to generalize.</p></div>`;
}

/* ---------- export ONE figure to a one-page PDF ---------- */
async function exportOneFigurePDF(gd){
  try{
    const {jsPDF}=window.jspdf;
    const title=(((gd.closest(".card")||{}).querySelector?.("h3"))||{}).textContent||"figure";
    const modelName=$("#model").value===SUMMARY_KEY?"all-models-summary":$("#model").value;
    // Capture the figure exactly as shown (it is visible when its button is
    // clicked), at natural size -- no width/height overrides, no cloning.
    const url=await Plotly.toImage(gd,{format:"png",scale:2});
    const img=await new Promise((res,rej)=>{const im=new Image();im.onload=()=>res(im);im.onerror=rej;im.src=url;});
    // Standard A4 page (landscape for wide figures), whole image fit inside with a
    // small margin -- reliably shows the complete chart with low white space.
    const land=img.width>=img.height;
    const doc=new jsPDF({unit:"pt",format:"a4",orientation:land?"landscape":"portrait"});
    const m=14, pw=doc.internal.pageSize.getWidth(), ph=doc.internal.pageSize.getHeight();
    const availW=pw-2*m, availH=ph-2*m;
    let w=availW, h=w*img.height/img.width;
    if(h>availH){ h=availH; w=h*img.width/img.height; }
    doc.addImage(url,"PNG", m+(availW-w)/2, m+(availH-h)/2, w, h);
    const slug=title.replace(/[^a-z0-9]+/gi,"-").toLowerCase().slice(0,50).replace(/^-|-$/g,"");
    doc.save(`figure_${modelName}_${slug}.pdf`);
  }catch(e){ alert("PDF export failed: "+(e&&e.message||e)); }
}

initModel();initNav();renderAll();
</script>
</body></html>"""


def main():
    models = models_with_both()
    if not models:
        raise SystemExit("no model has BOTH results/exp3_persona and results/exp3_decoupled")
    payload = {"models": {m: analyse_model(m) for m in models}}
    OUT.write_text(build_html(payload), encoding="utf-8")
    for m in models:
        meta = payload["models"][m]["meta"]
        print(f"  {m}: {meta['n_pairs']} pairs")
    size = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}  ({size:.2f} MB, models: {', '.join(models)})")


if __name__ == "__main__":
    main()
