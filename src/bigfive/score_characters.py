"""Stage 0b -- build the 406 trait-annotated character profiles.

The source repo for these profiles is gone (see stimuli.py provenance note), so
we regenerate them with the paper's own procedure: prompt Llama-3.3-70B to
answer each of the 50 IPIP items *as* each character (Listing 1), then

  * aggregate the Likert answers per trait, applying Table 2 reverse-keying,
    into a raw sum (10-50) and a z-score across the 406 characters;
  * concatenate the 50 explanations into the character's `self_description`,
    which is what Listing 2 puts in the system slot at Stage 1.

Decoding is greedy per plan §2 (this is a derivation input, so it must be
deterministic and reproducible).

Usage
-----
    python -m src.bigfive.score_characters                    # all 406
    python -m src.bigfive.score_characters --limit 4          # smoke test
    python -m src.bigfive.score_characters --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.bigfive import stimuli as S  # noqa: E402
from src.useraxis.extract import DEFAULT_MODEL, load_model, short_name  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parent.parent.parent / "results" / "bigfive"

# Longest-first so "neither agree nor disagree" wins over "agree"/"disagree",
# and "strongly disagree" wins over "disagree".
_LIKERT_ORDER = sorted(S.LIKERT, key=len, reverse=True)


def parse_response(text: str) -> tuple[str | None, str]:
    """-> (likert_level or None, explanation).

    Tolerates the observed formatting drift: capitalisation, a trailing period,
    quotes/asterisks around the level, and the level sharing a line with the
    explanation (Appendix E shows ": Agree. I tend to be...").
    """
    cleaned = text.strip().lstrip(":").strip()
    cleaned = re.sub(r"^[\s'\"*`<]+", "", cleaned)
    low = cleaned.lower()
    for lv in _LIKERT_ORDER:
        if low.startswith(lv):
            rest = cleaned[len(lv):]
            rest = re.sub(r"^[\s'\"*`>.,;:-]+", "", rest)
            return lv, rest.strip()
    return None, cleaned


def build_self_description(items: list[dict]) -> str:
    """Mirror Appendix E: 'item text: Likert. explanation' per item, in order."""
    parts = []
    for r in items:
        if r["likert"] is None:
            continue
        lv = r["likert"].capitalize()
        parts.append(f"{r['item']}: {lv}. {r['explanation']}".strip())
    return "\n".join(parts)


@torch.inference_mode()
def generate_batch(pm, prompts: list[str], max_new_tokens: int) -> list[str]:
    tok = pm.tokenizer
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=2048, add_special_tokens=False).to(pm.model.device)
    out = pm.model.generate(**enc, max_new_tokens=max_new_tokens,
                            do_sample=False, temperature=None, top_p=None,
                            pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 0b: score 406 characters on IPIP-50")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="first N characters (0=all)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out-dir", default="", help="override output dir (use tmpfs)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    chars = S.characters()
    if args.limit:
        chars = chars[:args.limit]
    items = S.ipip50()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_ROOT / short_name(args.model)
    raw_dir = out_dir / "character_items"
    raw_dir.mkdir(parents=True, exist_ok=True)

    todo = [c for c in chars
            if args.overwrite or not (raw_dir / f"{c['id']}.json").exists()]
    print(f"[stage0b] {len(chars)} characters, {len(todo)} to run, "
          f"{len(items)} items each -> {len(todo)*len(items)} generations")
    if not todo:
        print("[stage0b] nothing to generate; aggregating existing files")

    if todo:
        pm = load_model(args.model)
        tok = pm.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Flatten to (character, item) work units so batches stay full.
        work = [(c, it) for c in todo for it in items]
        results: dict[str, list[dict]] = {c["id"]: [] for c in todo}
        t0 = time.time()
        for s in range(0, len(work), args.batch_size):
            batch = work[s:s + args.batch_size]
            prompts = [
                tok.apply_chat_template(
                    S.listing1_messages(c["character"], c["franchise"], it["item"]),
                    tokenize=False, add_generation_prompt=True)
                for c, it in batch
            ]
            texts = generate_batch(pm, prompts, args.max_new_tokens)
            for (c, it), txt in zip(batch, texts):
                lv, expl = parse_response(txt)
                results[c["id"]].append({
                    "item_id": it["id"], "trait": it["trait"], "item": it["item"],
                    "keyed": it["keyed"], "likert": lv, "explanation": expl,
                    "score": S.score_item(lv, it["keyed"]) if lv else None,
                    "raw": txt if lv is None else None,   # keep only failures
                })
            # Flush each character as soon as all its items are in, so a crash
            # costs at most one character instead of the whole run.
            by_id = {c["id"]: c for c, _ in batch}
            for cid, c in by_id.items():
                if len(results[cid]) == len(items):
                    rows = sorted(results[cid], key=lambda r: r["item_id"])
                    (raw_dir / f"{cid}.json").write_text(
                        json.dumps({**c, "items": rows}, indent=1, ensure_ascii=False))
                    results[cid] = []          # release; the file is the record now

            done = s + len(batch)
            if done % (args.batch_size * 20) == 0 or done >= len(work):
                el = time.time() - t0
                print(f"  {done}/{len(work)} gens  {done/el*3600:.0f}/hr  "
                      f"eta {(len(work)-done)/max(done/el,1e-9)/60:.1f} min", flush=True)

    # ---------------- aggregate -> profiles ---------------- #
    profiles, n_bad = [], 0
    for c in chars:
        f = raw_dir / f"{c['id']}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text())
        rows = rec["items"]
        raw_scores, parsed = {}, 0
        for t in S.TRAITS:
            vals = [r["score"] for r in rows if r["trait"] == t and r["score"] is not None]
            parsed += len(vals)
            # Scale a partial scale up to the 10-item equivalent so the 10-50
            # range stays comparable when an item fails to parse.
            raw_scores[t] = round(sum(vals) / len(vals) * 10, 3) if vals else None
        n_bad += 50 - parsed
        profiles.append({
            **c,
            "raw": raw_scores,
            "n_parsed": parsed,
            "self_description": build_self_description(rows),
        })

    # z-score each trait across characters
    import statistics as st
    for t in S.TRAITS:
        vals = [p["raw"][t] for p in profiles if p["raw"][t] is not None]
        mu = st.mean(vals)
        sd = st.pstdev(vals) or 1.0
        for p in profiles:
            p.setdefault("z", {})[t] = (
                round((p["raw"][t] - mu) / sd, 4) if p["raw"][t] is not None else None)
        print(f"[trait {t}] mean={mu:.2f} sd={sd:.2f} "
              f"min={min(vals):.1f} max={max(vals):.1f} n={len(vals)}")

    out_f = out_dir / "character_profiles.json"
    out_f.write_text(json.dumps(profiles, indent=1, ensure_ascii=False))
    print(f"[stage0b] wrote {len(profiles)} profiles -> {out_f}")
    print(f"[stage0b] unparsed items: {n_bad}/{len(profiles)*50} "
          f"({n_bad/max(len(profiles)*50,1)*100:.2f}%)")


if __name__ == "__main__":
    main()
