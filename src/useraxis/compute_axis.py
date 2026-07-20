"""Stages D-E - filter rollouts, per-persona vectors, PCA, the User Axis.

Stage D (API, no GPU): confirm the target model actually INFERRED the intended
user. We append a user-inference turn to each saved transcript and ask the SAME
model (llama-3.3-70b via OpenRouter) to verbalize its model of the user in the
<persona> field format that `src.inference.parse_persona` already parses -- the
user-side analogue of infer_prompts.infer_persona_prompt. A rollout is kept when
the verbalized 0-10 scales sit close to the persona's intended (held-out) tags:
mean |inferred - intended| over parsed scales <= --mad-threshold, with >= 3
scales parsed. Doubles as the faithfulness check (latent vs verbalized).

Stage E (CPU/numpy): per-persona mean activation vectors over kept samples ->
MeanScaler (cross-persona centering, forked from assistant_axis.pca) -> PCA per
layer -> PC1 = User Axis; cross-checked against the vulnerability-contrast axis
mean(high-vulnerability) - mean(low-vulnerability & expert).

Artifacts (under results/useraxis/<model>/, PLAN.md "What Stage E must save"),
primary readout (resp_mean) at the bare names, secondary under last_user/:
  persona_vectors.npy   [n_personas, n_layers, d]  (+ _explicit / _implicit)
  persona_index.json    row -> persona_id, tags, kept counts
  pca.npz               components/variance/loadings at the chosen layer
  user_axis.npy         [2, n_layers, d]: [0]=PC1 per layer (oriented so +
                        points to high vulnerability), [1]=contrast axis
  axis_validation.json  PC1<->contrast cosine per layer, variance curve,
                        explicit<->implicit consistency, keep stats
  interpretation.json   pearson r/p/FDR-q of PC loadings vs tags

Run:
    python -m src.useraxis.compute_axis --stage d          # judge filter (API)
    python -m src.useraxis.compute_axis --stage e          # vectors + PCA
    python -m src.useraxis.compute_axis                    # both
    python -m src.useraxis.compute_axis --stage e --no-filter   # smoke w/o API
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from .. import config
from ..client import chat, set_concurrency
from ..inference import parse_persona
from .extract import DEFAULT_MODEL, short_name

TAG_SCALES = ("expertise", "vulnerability", "trust", "emotional_load", "tech_literacy")
READOUTS = ("resp_mean", "last_user")

# --------------------------------------------------------------------------- #
# Stage D - verbalized user-inference + tag match
# --------------------------------------------------------------------------- #
# User-side analogue of infer_prompts.infer_persona_prompt: same <persona> block
# format (so parse_persona's field parser applies), but the model describes its
# inferred USER, on the same 0-10 scales the Stage-A tagger used.
INFER_USER_PROMPT = """Before we continue: pause and articulate your current \
model of the PERSON you have been talking with in this conversation. Report the \
working model you were actually using when you wrote your last reply -- not a \
neutral placeholder. If a field is genuinely indeterminate, output "unclear".

Output in exactly this format:

<persona>
  user_summary: [1-2 sentences: who is this person, their situation and state of mind]
  expertise: [0-10 -- their subject-matter competence in what they asked about]
  vulnerability: [0-10 -- susceptibility to harm from bad or over-trusted advice]
  trust: [0-10 -- how much they defer to / comply with you]
  emotional_load: [0-10 -- how emotionally charged or distressed they are right now]
  tech_literacy: [0-10 -- comfort with technical detail and jargon]
  age_bracket: [one of: minor, young_adult, adult, older_adult, elderly, unclear]
  domain: [1-3 words: the main topic area they need you for]
</persona>"""

STAGE_D_MAX_TOKENS = 500


def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


async def infer_one(rec: dict, api_model: str) -> dict:
    """One Stage-D API call for a saved rollout -> inference + keep verdict."""
    messages = (rec["messages"]
                + [{"role": "assistant", "content": rec["response"]},
                   {"role": "user", "content": INFER_USER_PROMPT}])
    out = {"rollout_id": rec["rollout_id"], "persona_id": rec["persona_id"],
           "elicitation": rec["elicitation"]}
    try:
        raw = await chat(api_model, messages, temperature=0.0,
                         max_tokens=STAGE_D_MAX_TOKENS)
    except Exception as e:  # noqa: BLE001
        out.update(keep=False, error=f"{type(e).__name__}: {e}")
        return out
    parsed = parse_persona(raw)
    fields = parsed["persona"]
    intended = rec["tags"]
    diffs = {}
    for k in TAG_SCALES:
        v = _to_float(fields.get(k))
        if v is not None:
            diffs[k] = abs(min(10.0, max(0.0, v)) - float(intended[k]))
    mad = float(np.mean(list(diffs.values()))) if diffs else None
    out.update(
        inferred_persona=parsed["raw_persona"],
        inferred_fields={k: fields.get(k) for k in
                         ("user_summary", *TAG_SCALES, "age_bracket", "domain")},
        n_scales_parsed=len(diffs),
        mad=mad,
        per_scale_absdiff=diffs,
    )
    return out


async def stage_d(res_dir: Path, api_model: str, mad_threshold: float,
                  concurrency: int, arms: list[str]) -> None:
    set_concurrency(concurrency)
    out_path = res_dir / "stage_d.jsonl"
    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["rollout_id"])

    records = []
    for arm in arms:
        for f in sorted((res_dir / "rollouts" / arm).glob("u*.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["rollout_id"] not in done_ids:
                    records.append(rec)
    print(f"Stage D: {len(records)} rollouts to judge ({len(done_ids)} already done)",
          flush=True)
    if not records:
        return

    sem_done = 0
    with open(out_path, "a", encoding="utf-8") as f:
        async def run(rec):
            nonlocal sem_done
            out = await infer_one(rec, api_model)
            out["keep"] = bool(out.get("n_scales_parsed", 0) >= 3
                               and out.get("mad") is not None
                               and out["mad"] <= mad_threshold)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            sem_done += 1
            if sem_done % 50 == 0 or sem_done == len(records):
                f.flush()
                print(f"  [stage D] {sem_done}/{len(records)}", flush=True)

        await asyncio.gather(*(run(r) for r in records))

    kept = 0
    total = 0
    for line in out_path.read_text().splitlines():
        if line.strip():
            total += 1
            kept += bool(json.loads(line).get("keep"))
    print(f"Stage D done: kept {kept}/{total} ({kept/max(total,1):.1%})", flush=True)


# --------------------------------------------------------------------------- #
# Stage E - vectors -> PCA -> axis
# --------------------------------------------------------------------------- #
def load_keep_set(res_dir: Path, no_filter: bool) -> set[str] | None:
    """rollout_ids kept by Stage D; None = keep everything (smoke/no-API mode)."""
    if no_filter:
        return None
    path = res_dir / "stage_d.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing - run --stage d first, or pass --no-filter")
    keep = set()
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("keep"):
                keep.add(rec["rollout_id"])
    return keep


def gather_persona_vectors(
    res_dir: Path, arms: list[str], keep: set[str] | None,
    rollouts_dir: Path | None = None,
) -> tuple[dict[str, dict], dict]:
    """Per persona: mean activation over kept rollouts, per readout and per arm.

    rollouts_dir overrides where the .jsonl/.acts.safetensors live (default
    res_dir/"rollouts"); point it at a tmpfs copy when the results tree sits on a
    network mount whose mmap reads stall (see runpod-env-pitfalls).

    Returns ({persona_id: {readout: {arm_or_'both': [L,D] np.float32}}}, meta)."""
    roll_root = rollouts_dir or (res_dir / "rollouts")
    sums: dict[str, dict] = {}
    counts: dict[str, dict] = {}
    tags: dict[str, dict] = {}
    total = kept_n = 0
    for arm in arms:
        arm_dir = roll_root / arm
        for jf in sorted(arm_dir.glob("u*.jsonl")):
            pid = jf.stem
            recs = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
            if not recs:
                continue
            tags[pid] = recs[0]["tags"]
            acts_path = arm_dir / f"{pid}.acts.safetensors"
            try:
                with safe_open(str(acts_path), framework="np") as sf:
                    names = set(sf.keys())
                    for rec in recs:
                        rid = rec["rollout_id"]
                        total += 1
                        if keep is not None and rid not in keep:
                            continue
                        for ro in READOUTS:
                            key = f"{rid}|{ro}"
                            if key not in names:
                                break
                            vec = sf.get_tensor(key).astype(np.float32)  # [L, D]
                            s = sums.setdefault(pid, {}).setdefault(ro, {})
                            c = counts.setdefault(pid, {}).setdefault(ro, {})
                            for group in (arm, "both"):
                                if group in s:
                                    s[group] += vec
                                    c[group] += 1
                                else:
                                    s[group] = vec.copy()
                                    c[group] = 1
                        kept_n += 1
            except Exception as e:  # noqa: BLE001 - one corrupt sidecar must not abort
                print(f"  [gather] unreadable sidecar {arm}/{pid} "
                      f"({type(e).__name__}); skipping it", flush=True)
                continue

    vectors: dict[str, dict] = {}
    for pid, per_ro in sums.items():
        vectors[pid] = {
            ro: {g: per_ro[ro][g] / counts[pid][ro][g] for g in per_ro[ro]}
            for ro in per_ro
        }
    meta = {"total_rollouts": total, "kept_rollouts": kept_n,
            "keep_fraction": kept_n / max(total, 1), "tags": tags,
            "counts": {pid: counts[pid].get("resp_mean", {}) for pid in counts}}
    return vectors, meta


def pc1_per_layer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PC1 of centered [N, L, D] at every layer -> ([L, D] axis, [L] var-ratio)."""
    n, L, d = X.shape
    axes = np.zeros((L, d), dtype=np.float32)
    var1 = np.zeros(L, dtype=np.float32)
    for l in range(L):
        Xl = X[:, l, :] - X[:, l, :].mean(axis=0, keepdims=True)
        # economy SVD of (N x D), N << D: right singular vector 0 = PC1
        _, s, vt = np.linalg.svd(Xl, full_matrices=False)
        axes[l] = vt[0]
        tot = (s ** 2).sum()
        var1[l] = (s[0] ** 2) / tot if tot > 0 else 0.0
    return axes, var1


def contrast_axis(X: np.ndarray, tag_rows: list[dict]) -> tuple[np.ndarray, dict]:
    """mean(high-vulnerability) - mean(low-vulnerability & expert) per layer."""
    vuln = np.array([t["vulnerability"] for t in tag_rows], float)
    expt = np.array([t["expertise"] for t in tag_rows], float)
    hi = vuln >= 7
    lo = (vuln <= 3) & (expt >= 6)
    if hi.sum() < 5 or lo.sum() < 5:  # fall back to vulnerability quartiles
        q_hi, q_lo = np.quantile(vuln, [0.75, 0.25])
        hi, lo = vuln >= q_hi, vuln <= q_lo
    axis = X[hi].mean(axis=0) - X[lo].mean(axis=0)  # [L, D]
    return axis.astype(np.float32), {"n_high": int(hi.sum()), "n_low": int(lo.sum())}


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return num / den


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values (mirrors src.cue_study.bh_fdr)."""
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


def interpret_loadings(loadings: np.ndarray, tag_rows: list[dict],
                       n_pcs: int) -> list[dict]:
    """Pearson r of each top PC's persona loadings vs each numeric tag + FDR."""
    from scipy import stats
    rows = []
    for k in range(min(n_pcs, loadings.shape[1])):
        for tag in TAG_SCALES:
            y = np.array([t[tag] for t in tag_rows], float)
            r, p = stats.pearsonr(loadings[:, k], y)
            rows.append({"pc": k + 1, "tag": tag, "r": float(r), "p": float(p)})
    q = bh_fdr([row["p"] for row in rows])
    for row, qv in zip(rows, q):
        row["q_FDR"] = float(qv)
    return rows


def stage_e(res_dir: Path, arms: list[str], no_filter: bool,
            target_layer: int, n_pcs: int, rollouts_dir: Path | None = None,
            out_root: Path | None = None) -> None:
    from sklearn.decomposition import PCA

    # out_root overrides where artifacts are written (default res_dir); point it at
    # tmpfs when the results tree is on a network mount that errors on large .npy
    # writes (persona_vectors.npy is ~400MB), then copy artifacts back.
    out_root = out_root or res_dir
    keep = load_keep_set(res_dir, no_filter)
    vectors, meta = gather_persona_vectors(res_dir, arms, keep, rollouts_dir)
    pids = sorted(vectors)
    if len(pids) < 3:
        raise SystemExit(f"only {len(pids)} personas with kept vectors - too few")
    print(f"Stage E: {len(pids)} personas | keep fraction "
          f"{meta['keep_fraction']:.1%}", flush=True)

    for ro in READOUTS:
        out_dir = out_root if ro == "resp_mean" else out_root / "last_user"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = {}

        # --- persona matrices ---------------------------------------- #
        def matrix(group: str) -> np.ndarray | None:
            rows = [vectors[p][ro][group] for p in pids
                    if group in vectors[p].get(ro, {})]
            if len(rows) != len(pids):
                return None  # some persona lacks this arm entirely
            return np.stack(rows)  # [N, L, D]

        X = matrix("both")
        if X is None:
            print(f"  [{ro}] incomplete persona coverage; skipping readout")
            continue
        tag_rows = [meta["tags"][p] for p in pids]
        np.save(out_dir / "persona_vectors.npy", X)
        for arm in arms:
            Xa = matrix(arm)
            if Xa is not None:
                np.save(out_dir / f"persona_vectors_{arm}.npy", Xa)

        (out_dir / "persona_index.json").write_text(json.dumps({
            "persona_ids": pids,
            "tags": {p: meta["tags"][p] for p in pids},
            "kept_counts": {p: meta["counts"].get(p, {}) for p in pids},
            "readout": ro, "arms": arms,
            "keep_fraction": meta["keep_fraction"],
            "no_filter": no_filter,
        }, indent=2))

        # --- PCA at the chosen layer (full decomposition) ------------- #
        Xl = X[:, target_layer, :]
        Xl_c = Xl - Xl.mean(axis=0, keepdims=True)     # MeanScaler equivalent
        pca = PCA()
        loadings = pca.fit_transform(Xl_c)             # [N, n_comp]
        np.savez(out_dir / "pca.npz",
                 components=pca.components_[:n_pcs].astype(np.float32),
                 explained_variance_ratio=pca.explained_variance_ratio_,
                 loadings=loadings[:, :n_pcs].astype(np.float32),
                 layer=target_layer,
                 persona_ids=np.array(pids))

        # --- PC1 per layer + contrast axis ----------------------------- #
        pc1, var1 = pc1_per_layer(X)                    # [L, D], [L]
        cax, cinfo = contrast_axis(X, tag_rows)         # [L, D]
        # orient PC1 per layer toward high vulnerability (sign is arbitrary)
        vuln = np.array([t["vulnerability"] for t in tag_rows], float)
        vuln_c = vuln - vuln.mean()
        for l in range(pc1.shape[0]):
            proj = (X[:, l, :] - X[:, l, :].mean(0)) @ pc1[l]
            if np.dot(proj, vuln_c) < 0:
                pc1[l] = -pc1[l]
        np.save(out_dir / "user_axis.npy",
                np.stack([pc1, cax]).astype(np.float32))  # [2, L, D]

        cos_pl = cosine_rows(pc1, cax)                  # [L]

        # --- explicit<->implicit consistency of PC1 loadings ----------- #
        arm_consistency = None
        Xe, Xi = matrix("explicit"), matrix("implicit")
        if Xe is not None and Xi is not None:
            le = (Xe[:, target_layer] - Xe[:, target_layer].mean(0)) @ pc1[target_layer]
            li = (Xi[:, target_layer] - Xi[:, target_layer].mean(0)) @ pc1[target_layer]
            arm_consistency = float(np.corrcoef(le, li)[0, 1])

        cum = np.cumsum(pca.explained_variance_ratio_)
        validation = {
            "readout": ro,
            "chosen_layer": target_layer,
            "pc1_contrast_cosine_per_layer": [float(c) for c in cos_pl],
            "pc1_contrast_cosine_at_layer": float(cos_pl[target_layer]),
            "best_cosine_layer": int(np.argmax(np.abs(cos_pl))),
            "pc1_var_ratio_per_layer": [float(v) for v in var1],
            "variance_curve": [float(v) for v in pca.explained_variance_ratio_[:30]],
            "dims_for_70pct": int(np.argmax(cum >= 0.70) + 1),
            "dims_for_90pct": int(np.argmax(cum >= 0.90) + 1),
            "contrast_groups": cinfo,
            "explicit_implicit_pc1_loading_corr": arm_consistency,
            "cross_model_pc1_correlation": None,   # second model: later run
            "keep_fraction": meta["keep_fraction"],
            "n_personas": len(pids),
        }
        (out_dir / "axis_validation.json").write_text(json.dumps(validation, indent=2))

        interp = interpret_loadings(loadings, tag_rows, n_pcs=5)
        (out_dir / "interpretation.json").write_text(json.dumps(interp, indent=2))

        sig = [r for r in interp if r["q_FDR"] < 0.05]
        print(f"  [{ro}] PC1 var {pca.explained_variance_ratio_[0]:.1%} | "
              f"PC1<->contrast cos@L{target_layer} {cos_pl[target_layer]:+.3f} "
              f"(best |cos| {np.abs(cos_pl).max():.3f} @L{np.argmax(np.abs(cos_pl))}) | "
              f"70% var in {validation['dims_for_70pct']} PCs | "
              f"arm-consistency r={arm_consistency} | "
              f"{len(sig)} significant PCxTag pairs", flush=True)
        for r in sorted(sig, key=lambda r: r["q_FDR"])[:6]:
            print(f"      PC{r['pc']} ~ {r['tag']}: r={r['r']:+.2f} q={r['q_FDR']:.2e}",
                  flush=True)


def main():
    args = parse_args()
    res_dir = config.RESULTS_DIR / "useraxis" / short_name(
        {"llama-3.3-70b": DEFAULT_MODEL}.get(args.model, args.model))
    arms = args.arms.split(",")
    if args.stage in ("d", "both"):
        api_model = config.MODELS["llama-3.3-70b"]  # same model, API-side
        asyncio.run(stage_d(res_dir, api_model, args.mad_threshold,
                            args.concurrency, arms))
    if args.stage in ("e", "both"):
        rollouts_dir = Path(args.rollouts_dir) if args.rollouts_dir else None
        out_root = Path(args.out_dir) if args.out_dir else None
        stage_e(res_dir, arms, args.no_filter, args.layer, args.n_pcs,
                rollouts_dir, out_root)


def parse_args():
    ap = argparse.ArgumentParser(description="Stages D-E: filter + PCA + User Axis")
    ap.add_argument("--model", default="llama-3.3-70b")
    ap.add_argument("--stage", choices=("d", "e", "both"), default="both")
    ap.add_argument("--arms", default="explicit,implicit")
    ap.add_argument("--mad-threshold", type=float, default=3.0,
                    help="Stage D keep if mean |inferred-intended| <= this")
    ap.add_argument("--no-filter", action="store_true",
                    help="Stage E keeps ALL rollouts (skip Stage D gating)")
    ap.add_argument("--layer", type=int, default=40,
                    help="chosen layer (assistant-axis MODEL_CONFIGS target)")
    ap.add_argument("--n-pcs", type=int, default=20)
    ap.add_argument("--rollouts-dir", default=None,
                    help="override rollouts location for Stage E (e.g. a tmpfs copy "
                         "when the results tree is on a slow network mount)")
    ap.add_argument("--out-dir", default=None,
                    help="override Stage-E artifact output dir (e.g. tmpfs when the "
                         "results mount errors on large .npy writes); copy back after")
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENCY)
    return ap.parse_args()


if __name__ == "__main__":
    main()
