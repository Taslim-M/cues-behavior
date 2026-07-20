"""Stage F1 - obtain + validate the Assistant Axis on llama-3.3-70b.

The Assistant Axis (Lu et al.) is the residual-stream direction separating the
model's default assistant persona from role-play:

    axis = mean(default) - mean(role pos_3)      (points default-ward)

We take the PUBLISHED axis for the main link analysis and validate it three ways,
cheapest first:

  1. download        pull assistant_axis.pt (+ default_vector.pt + the 276
                     role_vectors) from the HF dataset lu-christina/assistant-axis
                     -vectors, save assistant_axis.npy [80,8192], and check the
                     download is INTERNALLY CONSISTENT: recompute
                     default - mean(roles) from the published component vectors
                     and confirm cosine(published, recomputed) ~= 1 per layer.
                     (no GPU)
  2. compat          push the 6 published case-study transcripts through our own
                     DualReadoutExtractor and project resp_mean onto the axis;
                     the `unsteered` (drifted) cases must land LOWER on the axis
                     (less assistant-like) than the `capped` ones. Confirms our
                     extraction reads the axis with the intended sign/scale. (GPU)
  3. recompute       subsampled independent recompute: regenerate role-play +
                     default responses for N roles with our extract.py, judge
                     role expression 0-3 via OpenRouter, rebuild the axis and
                     report cosine(published, recomputed)@40. (GPU, heavy)

Run:
    python -m src.useraxis.assistant_axis_build --download
    python -m src.useraxis.assistant_axis_build --compat
    python -m src.useraxis.assistant_axis_build --recompute --n-roles 30
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .. import config
from .extract import DEFAULT_MODEL, short_name

DATASET_REPO = "lu-christina/assistant-axis-vectors"
MODEL_DIR = "llama-3.3-70b"
TARGET_LAYER = 40
LAYER_GRID = (20, 30, 40, 50, 60)

# Published case-study transcripts (steered=capped vs unsteered) living in the
# sibling assistant-axis checkout.
AA_ROOT = Path("/workspace/assistant-axis")
CASE_DIR = AA_ROOT / "transcripts" / "case_studies" / MODEL_DIR
CASE_STEMS = ("delusion", "jailbreak", "selfharm")


def _res_dir() -> Path:
    return config.RESULTS_DIR / "useraxis" / short_name(DEFAULT_MODEL)


def _analysis_dir() -> Path:
    d = _res_dir() / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


# --------------------------------------------------------------------------- #
# 1. download + internal-consistency check                                    #
# --------------------------------------------------------------------------- #
def download() -> None:
    from huggingface_hub import hf_hub_download, list_repo_files

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    tok = _hf_token()

    def grab(rel: str) -> Path:
        return Path(hf_hub_download(DATASET_REPO, f"{MODEL_DIR}/{rel}",
                                    repo_type="dataset", token=tok))

    axis = torch.load(grab("assistant_axis.pt"), map_location="cpu",
                      weights_only=False).float()          # [80, 8192]
    if axis.ndim != 2 or axis.shape != (80, 8192):
        raise SystemExit(f"unexpected axis shape {tuple(axis.shape)}")

    default = torch.load(grab("default_vector.pt"), map_location="cpu",
                         weights_only=False).float()        # [80, 8192]

    role_files = [f for f in list_repo_files(DATASET_REPO, repo_type="dataset",
                                             token=tok)
                  if f.startswith(f"{MODEL_DIR}/role_vectors/") and f.endswith(".pt")]
    print(f"downloading {len(role_files)} role vectors ...", flush=True)
    roles = []
    for i, f in enumerate(sorted(role_files)):
        rel = f.split(f"{MODEL_DIR}/", 1)[1]
        roles.append(torch.load(grab(rel), map_location="cpu",
                                weights_only=False).float())
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(role_files)}", flush=True)
    role_stack = torch.stack(roles)                         # [n_roles, 80, 8192]

    # recompute axis from published components and compare
    recomputed = default - role_stack.mean(dim=0)           # [80, 8192]
    a = axis / (axis.norm(dim=1, keepdim=True) + 1e-8)
    b = recomputed / (recomputed.norm(dim=1, keepdim=True) + 1e-8)
    cos_per_layer = (a * b).sum(dim=1).numpy()              # [80]

    norms = axis.norm(dim=1).numpy()
    res = _res_dir()
    np.save(res / "assistant_axis.npy", axis.numpy().astype(np.float32))

    meta = {
        "source": f"{DATASET_REPO}::{MODEL_DIR}/assistant_axis.pt",
        "shape": list(axis.shape),
        "definition": "mean(default) - mean(role pos_3); higher = more assistant-like",
        "target_layer": TARGET_LAYER,
        "n_roles_in_recompute": len(roles),
        "axis_norm_per_layer": [float(x) for x in norms],
        "norm_argmax_layer": int(norms.argmax()),
        "norm_at_target": float(norms[TARGET_LAYER]),
    }
    (res / "assistant_axis_meta.json").write_text(json.dumps(meta, indent=2))

    check = {
        "check": "download internal consistency: cos(published_axis, "
                 "default - mean(published_roles)) per layer",
        "n_roles": len(roles),
        "cosine_per_layer": [float(c) for c in cos_per_layer],
        "cosine_at_target": float(cos_per_layer[TARGET_LAYER]),
        "cosine_min": float(cos_per_layer.min()),
        "cosine_mean": float(cos_per_layer.mean()),
    }
    (_analysis_dir() / "axis_download_check.json").write_text(json.dumps(check, indent=2))

    print(f"saved assistant_axis.npy {tuple(axis.shape)}; "
          f"norm peaks @L{int(norms.argmax())} (norm {norms.max():.2f}); "
          f"norm@L{TARGET_LAYER} {norms[TARGET_LAYER]:.2f}", flush=True)
    print(f"internal consistency cos@L{TARGET_LAYER} "
          f"{cos_per_layer[TARGET_LAYER]:+.4f} | min {cos_per_layer.min():+.4f} "
          f"| mean {cos_per_layer.mean():+.4f}", flush=True)


# --------------------------------------------------------------------------- #
# 2. compat check on published case-study transcripts (GPU)                    #
# --------------------------------------------------------------------------- #
def _load_case(stem: str, kind: str) -> list[dict]:
    d = json.loads((CASE_DIR / f"{stem}_{kind}.json").read_text())
    conv = d["conversation"]
    return [{"role": m["role"], "content": m["content"]} for m in conv]


def compat() -> None:
    from .extract import DualReadoutExtractor, load_model

    axis_path = _res_dir() / "assistant_axis.npy"
    if not axis_path.exists():
        raise SystemExit("run --download first (assistant_axis.npy missing)")
    axis = torch.from_numpy(np.load(axis_path)).float()      # [80, 8192]

    pm = load_model()
    ext = DualReadoutExtractor(pm)

    rows = []
    for stem in CASE_STEMS:
        for kind in ("unsteered", "capped"):
            conv = _load_case(stem, kind)
            out = ext.extract_batch([conv], max_length=8192)[0]
            if out is None:
                print(f"  [compat] {stem}/{kind}: span failure; skipped")
                continue
            rm = out["resp_mean"]                            # [80, 8192] fp32
            proj = {int(L): float((rm[L] / 1.0) @ (axis[L] / (axis[L].norm() + 1e-8)))
                    for L in LAYER_GRID}
            rows.append({"case": stem, "kind": kind, "assist_proj": proj})
            print(f"  [compat] {stem:10s} {kind:9s} "
                  f"assist@L40 {proj[TARGET_LAYER]:+.3f}", flush=True)

    # pair up unsteered vs capped per case at the target layer
    pairs = []
    for stem in CASE_STEMS:
        u = next((r for r in rows if r["case"] == stem and r["kind"] == "unsteered"), None)
        c = next((r for r in rows if r["case"] == stem and r["kind"] == "capped"), None)
        if u and c:
            gap = c["assist_proj"][TARGET_LAYER] - u["assist_proj"][TARGET_LAYER]
            pairs.append({"case": stem,
                          "unsteered": u["assist_proj"][TARGET_LAYER],
                          "capped": c["assist_proj"][TARGET_LAYER],
                          "capped_minus_unsteered": gap,
                          "expected_sign_ok": gap > 0})

    n_ok = sum(p["expected_sign_ok"] for p in pairs)
    out = {"target_layer": TARGET_LAYER, "layer_grid": list(LAYER_GRID),
           "rows": rows, "pairs": pairs,
           "n_pairs": len(pairs), "n_capped_higher": n_ok,
           "verdict": "capped projects more assistant-like than unsteered"
                      if n_ok == len(pairs) and pairs else "mixed"}
    (_analysis_dir() / "compat_check.json").write_text(json.dumps(out, indent=2))
    print(f"compat: {n_ok}/{len(pairs)} cases have capped > unsteered @L{TARGET_LAYER}",
          flush=True)


# --------------------------------------------------------------------------- #
# 3. subsampled independent recompute (GPU, heavy)                             #
# --------------------------------------------------------------------------- #
def recompute(n_roles: int, n_questions: int, n_prompts: int) -> None:
    """Independently rebuild the axis on a subsample and cosine it vs published.

    Uses our own extract.py for generation+capture and OpenRouter for the role
    judge; axis = mean(default resp_mean) - mean(role pos_3 resp_mean).
    """
    import asyncio
    import random

    from .extract import DualReadoutExtractor, load_model
    from ..client import chat, set_concurrency

    axis_path = _res_dir() / "assistant_axis.npy"
    if not axis_path.exists():
        raise SystemExit("run --download first")
    published = torch.from_numpy(np.load(axis_path)).float()

    roles_dir = AA_ROOT / "data" / "roles" / "instructions"
    q_path = AA_ROOT / "data" / "extraction_questions.jsonl"
    role_list = json.loads((AA_ROOT / "data" / "roles" / "role_list.json").read_text())
    if isinstance(role_list, dict):
        role_list = role_list.get("roles", list(role_list.keys()))
    questions = [json.loads(l)["question"] if l.strip().startswith("{") else l.strip()
                 for l in q_path.read_text().splitlines() if l.strip()]

    rng = random.Random(0)
    roles = sorted(rng.sample([r for r in role_list], min(n_roles, len(role_list))))
    qs = questions[:n_questions]
    print(f"recompute: {len(roles)} roles x {len(qs)} Q (+ default)", flush=True)

    pm = load_model()
    ext = DualReadoutExtractor(pm)

    def role_spec(role: str) -> dict:
        return json.loads((roles_dir / f"{role}.json").read_text())

    def pos_prompts(spec: dict) -> list[str]:
        """Role system prompts: `instruction` is a list of {'pos': str[, 'neg']}."""
        instr = spec.get("instruction", [])
        if isinstance(instr, str):
            return [instr]
        out = []
        for d in instr:
            if isinstance(d, dict) and d.get("pos"):
                out.append(d["pos"])
            elif isinstance(d, str):
                out.append(d)
        return out

    # ---- default vectors: neutral system prompt(s) ----
    default_spec = role_spec("default") if (roles_dir / "default.json").exists() else None
    default_prompts = (pos_prompts(default_spec) if default_spec else None) or [
        "You are a helpful assistant.",
    ]
    default_prompts = default_prompts[:n_prompts]

    set_concurrency(config.MAX_CONCURRENCY)

    def gen_and_capture(system: str, batch_qs: list[str], bs: int = 12) -> list:
        results = []
        for i in range(0, len(batch_qs), bs):
            chunk = batch_qs[i:i + bs]
            convs = [[{"role": "system", "content": system},
                      {"role": "user", "content": q}] for q in chunk]
            replies = ext.generate_batch(convs, max_new_tokens=300)
            full = [c + [{"role": "assistant", "content": r}]
                    for c, r in zip(convs, replies)]
            outs = ext.extract_batch(full, max_length=2048)
            results.extend((o["resp_mean"].numpy() if o else None, r)
                           for o, r in zip(outs, replies))
        return results

    judge_model = "openai/gpt-4.1-mini"   # independent judge, via OpenRouter

    async def judge_role(role: str, spec: dict, question: str, reply: str) -> int:
        rubric = spec.get("eval_prompt") or spec.get("judge_prompt") or (
            f"Rate 0-3 how fully the response embodies the role '{role}'.")
        prompt = (f"{rubric}\n\nQUESTION:\n{question}\n\nRESPONSE:\n{reply}\n\n"
                  "Reply with ONLY an integer 0, 1, 2, or 3.")
        try:
            raw = await chat(judge_model,
                             [{"role": "user", "content": prompt}],
                             temperature=0.0, max_tokens=8)
            digits = [c for c in raw if c in "0123"]
            return int(digits[0]) if digits else -1
        except Exception:
            return -1

    default_vecs = []
    for sp in default_prompts:
        for pair in gen_and_capture(sp, qs):
            v, _ = pair
            if v is not None:
                default_vecs.append(v)
    print(f"  default: {len(default_vecs)} vectors", flush=True)

    role_pos3 = []
    for ri, role in enumerate(roles):
        spec = role_spec(role)
        instrs = pos_prompts(spec)[:n_prompts]
        role_vecs = []
        for sysp in instrs:
            captured = gen_and_capture(sysp, qs)
            scores = asyncio.run(_judge_batch(judge_role, role, spec, qs, captured))
            for (v, _), sc in zip(captured, scores):
                if v is not None and sc == 3:
                    role_vecs.append(v)
        if role_vecs:
            role_pos3.append(np.stack(role_vecs).mean(axis=0))
        if (ri + 1) % 10 == 0:
            print(f"  roles {ri + 1}/{len(roles)} | kept pos3 so far "
                  f"{len(role_pos3)}", flush=True)

    if not default_vecs or not role_pos3:
        raise SystemExit("recompute: not enough captured vectors")

    default_mean = np.stack(default_vecs).mean(axis=0)       # [80, 8192]
    role_mean = np.stack(role_pos3).mean(axis=0)             # [80, 8192]
    recomputed = torch.from_numpy(default_mean - role_mean).float()

    a = published / (published.norm(dim=1, keepdim=True) + 1e-8)
    b = recomputed / (recomputed.norm(dim=1, keepdim=True) + 1e-8)
    cos = (a * b).sum(dim=1).numpy()

    out = {
        "n_roles_requested": n_roles, "n_roles_with_pos3": len(role_pos3),
        "n_questions": len(qs), "n_prompts": n_prompts,
        "n_default_vecs": len(default_vecs),
        "cosine_per_layer": [float(c) for c in cos],
        "cosine_at_target": float(cos[TARGET_LAYER]),
        "cosine_mean": float(cos.mean()),
    }
    (_analysis_dir() / "axis_recompute_check.json").write_text(json.dumps(out, indent=2))
    print(f"recompute cos@L{TARGET_LAYER} {cos[TARGET_LAYER]:+.3f} | "
          f"mean {cos.mean():+.3f} | roles-with-pos3 {len(role_pos3)}", flush=True)


async def _judge_batch(judge_fn, role, spec, qs, captured):
    import asyncio
    tasks = [judge_fn(role, spec, q, r) for (_, r), q in zip(captured, qs)]
    return await asyncio.gather(*tasks)


def main():
    ap = argparse.ArgumentParser(description="Stage F1: obtain + validate Assistant Axis")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--compat", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--n-roles", type=int, default=30)
    ap.add_argument("--n-questions", type=int, default=40)
    ap.add_argument("--n-prompts", type=int, default=3)
    args = ap.parse_args()

    if not (args.download or args.compat or args.recompute):
        args.download = True
    if args.download:
        download()
    if args.compat:
        compat()
    if args.recompute:
        recompute(args.n_roles, args.n_questions, args.n_prompts)


if __name__ == "__main__":
    main()
