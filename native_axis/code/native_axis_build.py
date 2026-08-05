"""Build LLM-native behavioral direction vectors at L40 (persona-vector / CAA style).

For each native factor we prompt the model with a high-pole vs low-pole system
instruction across shared neutral questions, read the response activation
gen_mean[L40], and take dir = mean(high) - mean(low) (unit-normalized). These
directions are then used (in native_axis_analysis.py) to test whether native
behavioral factors explain the Assistant Axis better than the Big Five.

    HF_HOME=/dev/shm/hf HF_HUB_OFFLINE=1 python -m src.bigfive.native_axis_build
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.bigfive.extract import BigFiveExtractor
from src.bigfive.role_profiles import load_questions, conv
from src.useraxis.extract import DEFAULT_MODEL, load_model

OUT = Path("native_axis/results")
AA_LAYER = 40

# native factor -> (high-pole system prompt, low-pole system prompt)
NATIVE = {
    "verbosity": (
        "Respond at great length, elaborate and expansive, with thorough detail and many words.",
        "Respond as tersely as possible, blunt and minimal, using the fewest words you can."),
    "deference": (
        "Be extremely deferential and agreeable: defer to the user, validate them, never push back.",
        "Be challenging and independent: push back, disagree openly, assert your own view over theirs."),
    "boldness": (
        "Be bold, assertive and confident: take strong, decisive stances with no hedging.",
        "Be timid, cautious and tentative: hedge heavily, qualify everything, avoid any commitment."),
    "guardedness": (
        "Be guarded and withholding: reveal as little as possible, deflect, stay closed off.",
        "Be open and forthcoming: share freely, be transparent, self-disclosing and candid."),
    "warmth": (
        "Be warm, caring and emotionally supportive: nurturing, gentle and kind.",
        "Be cold, detached and clinical: impersonal, unfeeling and strictly matter-of-fact."),
    "groundedness": (
        "Be grounded, literal and factual: concrete, plain and matter-of-fact, no flourish.",
        "Be mystical, abstract and dramatic: cryptic, theatrical, grandiose and poetic."),
    "formality": (
        "Be highly formal and professional: polished, proper, businesslike and precise.",
        "Be very casual and colloquial: chatty, slangy, loose and informal."),
}

# CONTROL: Big Five built at L40 with the SAME contrastive method, to isolate
# whether the native advantage is the factor choice or just the layer/method.
BIGFIVE_CONTRAST = {
    "EXT": ("You are extremely extraverted: outgoing, energetic, talkative and sociable.",
            "You are extremely introverted: reserved, quiet, solitary and withdrawn."),
    "AGR": ("You are extremely agreeable: warm, cooperative, trusting and compassionate.",
            "You are extremely disagreeable: cold, antagonistic, critical and unkind."),
    "CSN": ("You are extremely conscientious: organized, disciplined, careful and dependable.",
            "You are extremely unconscientious: disorganized, careless, impulsive and unreliable."),
    "EST": ("You are extremely emotionally stable: calm, secure, even-tempered and resilient.",
            "You are extremely neurotic: anxious, insecure, moody and easily upset."),
    "OPN": ("You are extremely open: imaginative, curious, unconventional and artistic.",
            "You are extremely closed-minded: conventional, practical, routine-bound and uncreative."),
}


def build_dirs(ex, questions, spec, aa_unit, npz_name, meta_name):
    """Build contrastive high-low direction at L40 for each factor in `spec`."""
    if (OUT / npz_name).exists():
        print(f"[skip] {npz_name} exists"); return
    stubs = []
    for fac, (hi, lo) in spec.items():
        for pole, sysp in (("high", hi), ("low", lo)):
            for q in questions:
                stubs.append((fac, pole, conv(sysp, q["question"])))
    acc = {(f, p): [] for f in spec for p in ("high", "low")}
    B = 24
    for s in range(0, len(stubs), B):
        batch = stubs[s:s + B]
        actsb, _ = ex.run_batch([b[2] for b in batch], generate=True,
                                max_new_tokens=128, do_sample=True,
                                temperature=0.8, top_p=0.9)
        gm = actsb["gen_mean"]
        for bi, (f, p, _) in enumerate(batch):
            acc[(f, p)].append(gm[bi, AA_LAYER].astype(np.float64))
        print(f"  [{npz_name}] {min(s+B,len(stubs))}/{len(stubs)} generations", flush=True)
    dirs, meta = {}, {}
    for f in spec:
        hi = np.mean(acc[(f, "high")], axis=0); lo = np.mean(acc[(f, "low")], axis=0)
        d = (hi - lo).astype(np.float32); u = d / (np.linalg.norm(d) + 1e-8)
        dirs[f] = u
        meta[f] = {"cos_with_AA": round(float(u @ aa_unit), 3),
                   "aa_proj_high": round(float(hi @ aa_unit), 3),
                   "aa_proj_low": round(float(lo @ aa_unit), 3),
                   "raw_norm": round(float(np.linalg.norm(d)), 3)}
        print(f"[dir:{npz_name}] {f:12} cos(AA)={meta[f]['cos_with_AA']:+.3f} "
              f"aa_high={meta[f]['aa_proj_high']:+.2f} aa_low={meta[f]['aa_proj_low']:+.2f}", flush=True)
    np.savez(OUT / npz_name, **dirs)
    (OUT / meta_name).write_text(json.dumps(
        {"factors": list(spec), "meta": meta, "layer": AA_LAYER}, indent=1))
    print(f"wrote {OUT}/{npz_name} ({len(dirs)} factors)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    questions = load_questions(12)
    aa = np.load("results/useraxis/llama-3.3-70b/assistant_axis.npy").astype(np.float32)
    aa_unit = aa[AA_LAYER] / (np.linalg.norm(aa[AA_LAYER]) + 1e-8)

    pm = load_model(DEFAULT_MODEL)
    ex = BigFiveExtractor(pm)
    build_dirs(ex, questions, NATIVE, aa_unit, "native_dirs.npz", "native_dirs_meta.json")
    build_dirs(ex, questions, BIGFIVE_CONTRAST, aa_unit, "bigfive_l40_dirs.npz", "bigfive_l40_meta.json")


if __name__ == "__main__":
    main()
