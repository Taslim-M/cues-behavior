"""Stage 1 §3.2 -- collect activations for the 406 characters x 10 Alpaca instructions.

Template is Listing 2: the character's `self_description` (built in Stage 0b)
goes in the system slot, one Alpaca instruction in the user slot. All three
positions and all 80 layers are captured per §2.

Also runs gate G2 (extraction round-trip) first, and can collect the adjective
stimuli (Listing 4) used for the held-out probe-generalization check in §3.4.

Usage
-----
    python -m src.bigfive.collect_activations --g2-only
    python -m src.bigfive.collect_activations --what characters --out-dir /dev/shm/bf_acts
    python -m src.bigfive.collect_activations --what adjectives --out-dir /dev/shm/bf_acts
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.bigfive import stimuli as S  # noqa: E402
from src.bigfive.extract import BigFiveExtractor, ActivationStore, round_trip_check  # noqa: E402
from src.useraxis.extract import DEFAULT_MODEL, load_model, short_name  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent.parent.parent / "results" / "bigfive"


def character_units(profiles: list[dict], instructions: list[dict]) -> list[dict]:
    return [
        {"stimulus_id": f"{p['id']}__{ins['id']}", "stage": "stage1_characters",
         "source": "character", "character_id": p["id"],
         "character": p["character"], "franchise": p["franchise"],
         "instruction_id": ins["id"], "condition": "listing2",
         "label_or_score": p["z"], "seed": 0}
        for p in profiles for ins in instructions
    ]


def adjective_units(instructions: list[dict]) -> list[dict]:
    adj = S.adjectives()
    return [
        {"stimulus_id": f"{trait}_{pol}_{a}__{ins['id']}", "stage": "stage1_adjectives",
         "source": "adjective", "trait": trait, "polarity": pol, "adjective": a,
         "instruction_id": ins["id"], "condition": "listing4",
         "label_or_score": 1 if pol == "pos" else 0, "seed": 0}
        for trait, d in adj.items() for pol in ("pos", "neg")
        for a in d[pol] for ins in instructions
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: activation collection")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--what", default="characters", choices=["characters", "adjectives"])
    ap.add_argument("--profiles", default="", help="character_profiles.json (Stage 0b)")
    ap.add_argument("--out-dir", default="", help="tmpfs strongly recommended")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-generate", action="store_true",
                    help="single prompt-only forward pass; captures last_prompt + "
                         "prompt_mean, leaves gen_mean zero. ~20x faster for the "
                         "long character self_description prompts, and prompt_mean "
                         "is the paper's own probe-optimal position.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--g2-only", action="store_true", help="run gate G2 and exit")
    args = ap.parse_args()

    pm = load_model(args.model)

    # ---------------- gate G2 ---------------- #
    print("[G2] extraction round-trip check")
    g2 = round_trip_check(pm)
    if not g2["pass"]:
        raise SystemExit("[G2] FAILED -- fix extraction before any downstream stage")
    print("[G2] PASS")
    if args.g2_only:
        out = Path(args.out_dir) if args.out_dir else OUT_ROOT / short_name(args.model)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gate_g2.json").write_text(json.dumps(g2, indent=1))
        return

    instructions = S.alpaca10()
    if args.what == "characters":
        prof_path = Path(args.profiles) if args.profiles else \
            OUT_ROOT / short_name(args.model) / "character_profiles.json"
        profiles = json.loads(Path(prof_path).read_text())
        if args.limit:
            profiles = profiles[:args.limit]
        units = character_units(profiles, instructions)
        desc = {p["id"]: p["self_description"] for p in profiles}
        build = lambda u: S.listing2_messages(desc[u["character_id"]],
                                              next(i["instruction"] for i in instructions
                                                   if i["id"] == u["instruction_id"]))
    else:
        units = adjective_units(instructions)
        if args.limit:
            units = units[:args.limit]
        build = lambda u: S.listing4_messages(
            next(i["instruction"] for i in instructions if i["id"] == u["instruction_id"]),
            u["adjective"])

    ex = BigFiveExtractor(pm)

    # Length-sort so each batch holds similar-length prompts: minimises padding
    # waste and makes peak memory predictable (no mixed short/long batches). The
    # store records each batch's units next to its activations at the same row
    # offset, so reordering does not desync the index.
    tok = ex.tok
    units.sort(key=lambda u: len(tok.apply_chat_template(
        build(u), tokenize=True, add_generation_prompt=True)))

    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / short_name(args.model)
    store_dir = out_dir / f"acts_{args.what}"
    store = ActivationStore(store_dir, len(units), ex.n_layers, ex.d_model)
    print(f"[stage1] {args.what}: {len(units)} stimuli -> {store_dir} "
          f"({len(units)*3*ex.n_layers*ex.d_model*4/2**30:.1f} GiB) "
          f"generate={not args.no_generate}")

    t0 = time.time()
    texts_out = []
    for s in range(0, len(units), args.batch_size):
        batch = units[s:s + args.batch_size]
        acts, texts = ex.run_batch([build(u) for u in batch],
                                   generate=not args.no_generate,
                                   max_new_tokens=args.max_new_tokens, do_sample=False)
        store.write(s, acts, batch)
        texts_out.extend({"stimulus_id": u["stimulus_id"], "text": t}
                         for u, t in zip(batch, texts))
        done = s + len(batch)
        if done % (args.batch_size * 10) == 0 or done >= len(units):
            el = time.time() - t0
            print(f"  {done}/{len(units)}  {done/el*3600:.0f}/hr  "
                  f"eta {(len(units)-done)/max(done/el,1e-9)/60:.1f} min", flush=True)

    store.finalize({"stage": f"stage1_{args.what}", "model": args.model,
                    "quant_config": "bf16",
                    "decoding": "none" if args.no_generate else "greedy",
                    "generated": not args.no_generate,
                    "gen_mean_valid": not args.no_generate,
                    "max_new_tokens": None if args.no_generate else args.max_new_tokens})
    (store_dir / "generations.json").write_text(json.dumps(texts_out, indent=1))
    print(f"[stage1] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
