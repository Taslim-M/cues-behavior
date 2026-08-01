"""Track E: scenario intensity vs disposition.

Factorial: fixed users x scenario mode x intensity (normal->elevated->extreme).
Holds the USER identity fixed (in the system slot; 'generic' = none) and varies
only the scenario (the user message, from the scenario bank), to separate the
model's persona response into a *user* effect and a *scenario* effect.

Reads each response's gen_mean onto: Big Five probes, the 274 role vectors
(nearest LLM persona), and the Assistant Axis. Also saves the transcript.

    python -m src.useraxis.scenario_mapping --samples 3
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from src import config
from src.bigfive import stimuli as BF
from src.bigfive.extract import BigFiveExtractor
from src.useraxis.extract import DEFAULT_MODEL, load_model
from src.useraxis.user_persona_mapping import load_role_matrix, AA_LAYER

ROOT = config.ROOT
BANK = ROOT / "generate_synthetic_data" / "scenario_bank.json"
PERSONAS = ROOT / "generate_synthetic_data" / "user_personas.jsonl"
BF_DIR = ROOT / "results" / "bigfive" / "llama-3.3-70b"
AA_PATH = ROOT / "results" / "useraxis" / "llama-3.3-70b" / "assistant_axis.npy"

# fixed users: generic (no system) + 4 contrasting real personas
FIXED_USERS = {"generic": None, "vulnerable": "u0069", "expert": "u0005",
               "low_trust": "u0085", "calm": "u0008"}
LEVELS = ["normal", "elevated", "extreme"]


def load_fixed_users():
    P = {json.loads(l)["persona_id"]: json.loads(l) for l in open(PERSONAS)}
    out = {}
    for label, pid in FIXED_USERS.items():
        out[label] = None if pid is None else P[pid]["explicit_system_prompts"][0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "scenario_mapping" / "llama-3.3-70b"))
    args = ap.parse_args()

    bank = json.loads(BANK.read_text())
    users = load_fixed_users()
    bankcopy = {t: np.load(BF_DIR / "direction_bank.npz")[t] for t in BF.TRAITS}
    sel = json.loads((BF_DIR / "stage1_selection.json").read_text())["selection"]
    probe_layer = {t: int(sel[t]["layer"]) for t in BF.TRAITS}
    probe_vec = {t: bankcopy[t][probe_layer[t]].astype(np.float32) for t in BF.TRAITS}
    AA = np.load(AA_PATH).astype(np.float32)
    aa_unit = AA[AA_LAYER] / (np.linalg.norm(AA[AA_LAYER]) + 1e-8)
    role_names, R = load_role_matrix()

    # build all cells: (user_label, mode, level) x samples
    cells = []
    for ulabel, sysmsg in users.items():
        for mode, m in bank.items():
            for level in LEVELS:
                msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + \
                       [{"role": "user", "content": m["levels"][level]}]
                for s in range(args.samples):
                    cells.append({"user": ulabel, "mode": mode, "level": level,
                                  "sample": s, "messages": msgs})
    print(f"[scenario] {len(users)} users x {len(bank)} modes x {len(LEVELS)} levels "
          f"x {args.samples} = {len(cells)} rollouts")

    pm = load_model(args.model)
    ex = BigFiveExtractor(pm)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for s in range(0, len(cells), args.batch_size):
        batch = cells[s:s + args.batch_size]
        acts, texts = ex.run_batch([c["messages"] for c in batch], generate=True,
                                   max_new_tokens=args.max_new_tokens,
                                   do_sample=args.temperature > 0,
                                   temperature=args.temperature, top_p=1.0)
        gm = acts["gen_mean"]
        for bi, c in enumerate(batch):
            g40 = gm[bi, AA_LAYER]; g40u = g40 / (np.linalg.norm(g40) + 1e-8)
            cos = R @ g40u; top = np.argsort(-cos)[:5]
            rows.append({"user": c["user"], "mode": c["mode"], "level": c["level"],
                         "sample": c["sample"],
                         "user_msg": c["messages"][-1]["content"], "response": texts[bi].strip(),
                         "bigfive": {t: float(gm[bi, probe_layer[t]] @ probe_vec[t]) for t in BF.TRAITS},
                         "aa_proj": float(g40 @ aa_unit),
                         "top_roles": [[role_names[j], round(float(cos[j]), 4)] for j in top]})
        print(f"  {s+len(batch)}/{len(cells)}  {(s+len(batch))/(time.time()-t0)*3600:.0f}/hr", flush=True)

    (out_dir / "rollouts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"[scenario] wrote {len(rows)} rollouts -> {out_dir/'rollouts.jsonl'}")


if __name__ == "__main__":
    main()
