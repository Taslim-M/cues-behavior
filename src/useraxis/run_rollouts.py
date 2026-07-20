"""Stage C (RunPod GPU) - rollouts + dual-readout activation capture.

For each persona x elicitation-arm x sampled probe, run the target model and
save (a) the FULL untruncated transcript and (b) activation sidecars for both
readouts (resp_mean / last_user), per PLAN.md "Stage C" + "Per-rollout schema".

Message construction (LOCKED, wiring A):
  * explicit : system = persona.explicit_system_prompts[elicit_idx]
               user   = probe                      -> read model's response
  * implicit : user#1 = persona.implicit_openers[elicit_idx]  (no system prompt)
               model replies, then user#2 = the SAME shared probe
                                                   -> read model's TURN-2 response

The probe subset is sampled ONCE (seeded, stratified across the 12 themes) and
shared by every persona and both arms, so the only thing that varies across
rollouts of a probe is *who the user is*. elicit_idx cycles j % 5 (explicit) /
j % 10 (implicit) over a persona's phrasings, giving the within-persona samples
that are averaged into its vector in Stage E.

Layout (results/useraxis/<model>/rollouts/):
  <arm>/<persona_id>.jsonl             one record per rollout, full messages+response
  <arm>/<persona_id>.acts.safetensors  "<rollout_id>|resp_mean" / "...|last_user",
                                       each fp32 [n_layers, d_model]
Resumable: a (persona, arm) pair whose .jsonl AND sidecar exist is skipped.

Run:
    python -m src.useraxis.run_rollouts --personas 8 --probes 4      # smoke
    python -m src.useraxis.run_rollouts                              # full 150 x 24
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from .. import config
from .extract import DEFAULT_MODEL, DualReadoutExtractor, load_model, short_name

PERSONAS_PATH = config.ROOT / "generate_synthetic_data" / "user_personas.jsonl"
QUESTIONS_PATH = config.ROOT / "generate_synthetic_data" / "extraction_questions.json"

GEN_TEMPERATURE = 0.7   # matches config.TEMPERATURE
GEN_TOP_P = 0.9
MAX_NEW_TOKENS = 512


def load_personas(path: Path, limit: int = 0) -> list[dict]:
    personas = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return personas[:limit] if limit else personas


def sample_shared_probes(path: Path, k: int, seed: int) -> list[dict]:
    """One shared, theme-stratified probe subset used by ALL personas/arms."""
    data = json.loads(path.read_text())
    questions = data["questions"]
    by_theme: dict[str, list[dict]] = {}
    for q in questions:
        by_theme.setdefault(q["theme"], []).append(q)
    rng = random.Random(seed)
    themes = sorted(by_theme)
    picked: list[dict] = []
    per_theme = -(-k // len(themes))  # ceil
    for t in themes:
        pool = sorted(by_theme[t], key=lambda q: q["id"])
        picked.extend(rng.sample(pool, min(per_theme, len(pool))))
    rng.shuffle(picked)
    return picked[:k]


def rollout_id(persona_id: str, arm: str, probe_id: str, elicit_idx: int) -> str:
    return f"{persona_id}_{arm}_{probe_id}_e{elicit_idx}"


# --------------------------------------------------------------------------- #
# Per (persona, arm) processing — one generation/extraction batch each
# --------------------------------------------------------------------------- #
def build_explicit(persona: dict, probes: list[dict]) -> list[dict]:
    """Rollout stubs (messages w/o final response) for the explicit arm."""
    stubs = []
    n_sys = len(persona["explicit_system_prompts"])
    for j, probe in enumerate(probes):
        e = j % n_sys
        stubs.append({
            "elicit_idx": e,
            "probe": probe,
            "messages": [
                {"role": "system", "content": persona["explicit_system_prompts"][e]},
                {"role": "user", "content": probe["text"]},
            ],
        })
    return stubs


def build_implicit_turn1(persona: dict, probes: list[dict]) -> list[dict]:
    stubs = []
    n_open = len(persona["implicit_openers"])
    for j, probe in enumerate(probes):
        e = j % n_open
        stubs.append({
            "elicit_idx": e,
            "probe": probe,
            "messages": [
                {"role": "user", "content": persona["implicit_openers"][e]},
            ],
        })
    return stubs


def process_persona_arm(
    ex: DualReadoutExtractor,
    persona: dict,
    arm: str,
    probes: list[dict],
    out_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
) -> tuple[int, int]:
    """Generate + extract all rollouts for one (persona, arm). Returns (saved, failed)."""
    pid = persona["persona_id"]
    jsonl_path = out_dir / arm / f"{pid}.jsonl"
    acts_path = out_dir / arm / f"{pid}.acts.safetensors"
    if jsonl_path.exists() and acts_path.exists():
        return (-1, 0)  # already done
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    if arm == "explicit":
        stubs = build_explicit(persona, probes)
    else:
        stubs = build_implicit_turn1(persona, probes)
        # turn 1: opener -> model reply (batched)
        for i in range(0, len(stubs), batch_size):
            chunk = stubs[i:i + batch_size]
            replies = ex.generate_batch([s["messages"] for s in chunk],
                                        max_new_tokens=MAX_NEW_TOKENS,
                                        temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P)
            for s, r in zip(chunk, replies):
                s["messages"] = s["messages"] + [
                    {"role": "assistant", "content": r},
                    {"role": "user", "content": s["probe"]["text"]},
                ]

    # readout turn: generate the response we read activations on
    for i in range(0, len(stubs), batch_size):
        chunk = stubs[i:i + batch_size]
        responses = ex.generate_batch([s["messages"] for s in chunk],
                                      max_new_tokens=MAX_NEW_TOKENS,
                                      temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P)
        for s, r in zip(chunk, responses):
            s["response"] = r
            s["full"] = s["messages"] + [{"role": "assistant", "content": r}]

    # activation pass (one forward per conversation batch, dual readout)
    tensors: dict[str, torch.Tensor] = {}
    records = []
    failed = 0
    for i in range(0, len(stubs), batch_size):
        chunk = stubs[i:i + batch_size]
        acts = ex.extract_batch([s["full"] for s in chunk], max_length=max_length)
        for s, a in zip(chunk, acts):
            rid = rollout_id(pid, arm, s["probe"]["id"], s["elicit_idx"])
            if a is None or not s["response"]:
                failed += 1
                continue
            tensors[f"{rid}|resp_mean"] = a["resp_mean"]
            tensors[f"{rid}|last_user"] = a["last_user"]
            records.append({
                "rollout_id": rid,
                "model": model_name,
                "elicitation": arm,
                "persona_id": pid,
                "tags": persona["tags"],
                "lean": persona.get("lean", ""),
                "probe_id": s["probe"]["id"],
                "probe_theme": s["probe"]["theme"],
                "elicit_idx": s["elicit_idx"],
                "messages": s["messages"],          # FULL, untruncated context
                "response": s["response"],          # FULL readout-turn response
                "gen": {"temperature": GEN_TEMPERATURE, "top_p": GEN_TOP_P,
                        "max_new_tokens": MAX_NEW_TOKENS},
            })

    # atomic-ish write: sidecar first, jsonl last (resume checks both)
    if tensors:
        save_file(tensors, str(acts_path))
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return (len(records), failed)


def main():
    args = parse_args()
    # config.MODELS keys are short names bound to OpenRouter ids; rollouts need
    # the HF id, so map the short key (or the OpenRouter id) to HF explicitly.
    hf_ids = {"llama-3.3-70b": DEFAULT_MODEL,
              "meta-llama/llama-3.3-70b-instruct": DEFAULT_MODEL}
    model_name = hf_ids.get(args.model, args.model)
    if "/" not in model_name:
        raise SystemExit(f"unknown model {args.model!r}: pass an HF id or a key in "
                         f"{sorted(hf_ids)}")

    personas = load_personas(Path(args.personas_file), args.personas)
    probes = sample_shared_probes(Path(args.questions_file), args.probes, args.seed)
    arms = args.arms.split(",")
    out_dir = config.RESULTS_DIR / "useraxis" / short_name(model_name) / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # run manifest (shared probe subset is part of the experiment definition)
    manifest = {
        "model": model_name,
        "n_personas": len(personas),
        "persona_ids": [p["persona_id"] for p in personas],
        "probes": [{"id": q["id"], "theme": q["theme"], "text": q["text"]} for q in probes],
        "arms": arms,
        "seed": args.seed,
        "gen": {"temperature": GEN_TEMPERATURE, "top_p": GEN_TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"Loading {model_name} (bf16, device_map=auto) ...", flush=True)
    t0 = time.time()
    pm = load_model(model_name)
    ex = DualReadoutExtractor(pm)
    print(f"  loaded in {time.time()-t0:.0f}s | {ex.n_layers} layers x d={ex.d_model}",
          flush=True)

    total = done = skipped = failed_total = 0
    n_jobs = len(personas) * len(arms)
    for p_i, persona in enumerate(personas):
        for arm in arms:
            t1 = time.time()
            saved, failed = process_persona_arm(
                ex, persona, arm, probes, out_dir, model_name,
                args.batch_size, args.max_length)
            total += 1
            if saved < 0:
                skipped += 1
                print(f"[{total}/{n_jobs}] {persona['persona_id']} {arm}: skip (exists)",
                      flush=True)
                continue
            done += saved
            failed_total += failed
            print(f"[{total}/{n_jobs}] {persona['persona_id']} {arm}: "
                  f"{saved} rollouts ({failed} failed) in {time.time()-t1:.0f}s",
                  flush=True)

    print(f"\nDone: {done} rollouts saved, {failed_total} failed, "
          f"{skipped} persona-arms skipped -> {out_dir}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Stage C: rollouts + activation capture")
    ap.add_argument("--model", default="llama-3.3-70b",
                    help="config.MODELS key or HF model id")
    ap.add_argument("--personas", type=int, default=0, help="first N personas (0=all)")
    ap.add_argument("--probes", type=int, default=24,
                    help="shared probe subset size (theme-stratified)")
    ap.add_argument("--arms", default="explicit,implicit")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=4096,
                    help="max tokens per conversation in the activation pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--personas-file", default=str(PERSONAS_PATH))
    ap.add_argument("--questions-file", default=str(QUESTIONS_PATH))
    return ap.parse_args()


if __name__ == "__main__":
    main()
