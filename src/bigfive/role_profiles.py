"""Big Five profiles of every Assistant-Axis role, read by our M2 probes.

For each of the 275 roles (+ the `default` Assistant), we roll out the role's
own system prompts on a shared set of extraction questions, capture the
*response*-token activation (``gen_mean`` -- the same readout the Assistant Axis
uses), and project it onto our per-layer M2 Big Five directions at each trait's
probe-optimal layer. Every response therefore yields one 5-trait reading; a role
accumulates a *distribution* of readings from which we report per-trait
min / max / mean / std / median (the within-role spread).

This fixes the prompt-vs-response caveat from the character stage: the probes were
fit on character prompt activations, but Stage 3 already showed they read
sensible Big Five on response-mean role vectors -- here we apply them to our own
rollouts at the individual-response level.

Storage is tiny: we project in-flight and keep only the 5 numbers per response,
never the raw activations.

    python -m src.bigfive.role_profiles --questions 24 --batch-size 32 \
        --out-dir results/bigfive/llama-3.3-70b/role_profiles
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.bigfive import stimuli as S
from src.bigfive.extract import BigFiveExtractor
from src.useraxis.extract import DEFAULT_MODEL, load_model, short_name

AA = Path("/workspace/assistant-axis")
RESULTS = Path("results/bigfive/llama-3.3-70b")


def load_roles() -> list[dict]:
    """[{name, system_prompts:[..5..]}] for 275 roles + default (default first)."""
    out = []
    default = json.loads((AA / "data/roles/instructions/default.json").read_text())
    out.append({"name": "default",
                "system_prompts": [list(x.values())[0] for x in default["instruction"]]})
    for f in sorted(glob.glob(str(AA / "data/roles/instructions/*.json"))):
        name = Path(f).stem
        if name == "default":
            continue
        d = json.loads(Path(f).read_text())
        out.append({"name": name,
                    "system_prompts": [x["pos"] for x in d["instruction"]]})
    return out


def load_questions(k: int) -> list[dict]:
    qs = [json.loads(l) for l in open(AA / "data/extraction_questions.jsonl")]
    return qs[:k] if k else qs


def conv(system: str, question: str) -> list[dict]:
    msgs = []
    if system and system.strip():
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--questions", type=int, default=24, help="shared questions/role")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--roles", default="", help="comma subset (smoke test)")
    ap.add_argument("--limit-roles", type=int, default=0)
    ap.add_argument("--out-dir", default=str(RESULTS / "role_profiles"))
    args = ap.parse_args()

    # ---- probes: M2 direction at each trait's probe-optimal layer ----
    bank = {t: np.load(RESULTS / "direction_bank.npz")[t] for t in S.TRAITS}
    sel = json.loads((RESULTS / "stage1_selection.json").read_text())["selection"]
    probe_layer = {t: int(sel[t]["layer"]) for t in S.TRAITS}          # e.g. EXT->30
    probe_vec = {t: torch.tensor(bank[t][probe_layer[t]], dtype=torch.float32)
                 for t in S.TRAITS}
    READ_LAYERS = sorted({*probe_layer.values(), 40})                  # +L40 for AA-comparability
    print(f"[roles] probe layers {probe_layer}; also reading L40")

    roles = load_roles()
    if args.roles:
        want = set(args.roles.split(","))
        roles = [r for r in roles if r["name"] in want]
    if args.limit_roles:
        roles = roles[:args.limit_roles]
    questions = load_questions(args.questions)
    print(f"[roles] {len(roles)} roles x {len(roles[0]['system_prompts'])} sys x "
          f"{len(questions)} Q = {len(roles)*5*len(questions)} rollouts")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in roles if not (out_dir / f"{r['name']}.jsonl").exists()]
    print(f"[roles] {len(todo)} roles to run ({len(roles)-len(todo)} already done)")
    if not todo:
        aggregate(out_dir, roles)
        return

    pm = load_model(args.model)
    ex = BigFiveExtractor(pm)

    t0 = time.time()
    done_rollouts = 0
    for ri, role in enumerate(todo):
        units = [(si, q["id"], conv(sp, q["question"]))
                 for si, sp in enumerate(role["system_prompts"])
                 for q in questions]
        rows = []
        for s in range(0, len(units), args.batch_size):
            batch = units[s:s + args.batch_size]
            acts, texts = ex.run_batch([u[2] for u in batch], generate=True,
                                       max_new_tokens=args.max_new_tokens,
                                       do_sample=args.temperature > 0,
                                       temperature=args.temperature, top_p=args.top_p)
            gm = acts["gen_mean"]                       # [B, n_layers, d]
            for bi, (si, qid, _) in enumerate(batch):
                reading = {t: float(gm[bi, probe_layer[t]] @ probe_vec[t].numpy())
                           for t in S.TRAITS}
                reading_l40 = {t: float(gm[bi, 40] @ probe_vec[t].numpy())
                               for t in S.TRAITS}
                rows.append({"role": role["name"], "sys_idx": si, "qid": qid,
                             "read": reading, "read_L40": reading_l40})
            done_rollouts += len(batch)
        (out_dir / f"{role['name']}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        if ri % 5 == 0 or ri == len(todo) - 1:
            el = time.time() - t0
            rate = done_rollouts / el * 3600
            eta = (sum(5 * len(questions) for _ in todo[ri + 1:])) / max(rate, 1) * 3600 / 3600 / 60
            print(f"  role {ri+1}/{len(todo)} {role['name']:<22} "
                  f"{rate:.0f} rollouts/hr  eta {eta:.0f}min", flush=True)

    aggregate(out_dir, roles)


def aggregate(out_dir: Path, roles: list[dict]) -> None:
    """Per-role per-trait min/max/mean/std/median (+ z vs role population)."""
    per_role = {}
    all_means = {t: [] for t in S.TRAITS}
    for r in roles:
        f = out_dir / f"{r['name']}.jsonl"
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        prof = {"n": len(recs)}
        for t in S.TRAITS:
            v = np.array([rec["read"][t] for rec in recs])
            prof[t] = {"mean": float(v.mean()), "std": float(v.std()),
                       "min": float(v.min()), "max": float(v.max()),
                       "median": float(np.median(v)),
                       "iqr": float(np.percentile(v, 75) - np.percentile(v, 25))}
            all_means[t].append((r["name"], float(v.mean())))
        per_role[r["name"]] = prof

    # z-score each role's mean against the role population
    pop = {t: np.array([m for _, m in all_means[t]]) for t in S.TRAITS}
    mu = {t: float(pop[t].mean()) for t in S.TRAITS}
    sd = {t: float(pop[t].std()) or 1.0 for t in S.TRAITS}
    for name, prof in per_role.items():
        prof["z_vs_roles"] = {t: round((prof[t]["mean"] - mu[t]) / sd[t], 3) for t in S.TRAITS}

    out = {"per_role": per_role, "population_mean": mu, "population_std": sd,
           "n_roles": len(per_role)}
    (out_dir.parent / "role_bigfive_profiles.json").write_text(json.dumps(out, indent=1))
    print(f"[agg] wrote role_bigfive_profiles.json ({len(per_role)} roles)")
    # quick face-validity print
    for t in S.TRAITS:
        ranked = sorted(all_means[t], key=lambda x: x[1])
        print(f"  {t}: low {[n for n,_ in ranked[:3]]}  high {[n for n,_ in ranked[-3:]]}")


if __name__ == "__main__":
    main()
