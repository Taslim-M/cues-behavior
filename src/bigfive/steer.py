"""Stage 1 §3.6 -- steering derivation + forced-choice evaluation.

Interventions (each uses the per-layer M2 direction bank, unit-normalised):
  S0  additive / last token / ALL layers           -- the paper's weak baseline.
       h <- h + alpha*r_L,  alpha in linspace(-0.4, 0.4, 9)
  S1  norm-scaled additive / all positions / band  -- stronger.
       h <- h + c*resid_norms[L]*r_L,  c in {+-0.05,+-0.1,+-0.2,+-0.3}
  S2  capping / all positions / band               -- clamp the projection.
       ceiling (push toward LOW trait): cap proj onto +r_L at tau_high
       floor   (push toward HIGH trait): cap proj onto -r_L at -tau_low
       tau in {10th,25th,50th} pct of the trait's per-layer projection distn.
  Layer band L*: center in {0.4,0.6,0.8}*depth, width in {8,16,24}.

Primary metric = forced-choice (Listing 3): present 10 self-statements (5 positive-
polarity, 5 negative-polarity, each mixing IPIP-seen and held-out-extended items),
model picks 5. positive_fraction = picked-positive / 5. A monotone, full-range
(0<->1) response to steering strength is the H1 signal. A coherence guard tracks
whether the model returns a valid 5-pick list (degeneration under strong steering).

This module runs the GRID for all traits and writes raw results; selection and the
confirmatory metrics live in run_steering.py / steer_confirm.py.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "/workspace/assistant-axis")
from assistant_axis.steering import ActivationSteering  # noqa: E402

from src.bigfive import stimuli as S

DATA = Path(__file__).resolve().parent.parent.parent / "data" / "bigfive"


# --------------------------------------------------------------------------- #
# forced-choice statement sets (5 positive-polarity, 5 negative-polarity)
# --------------------------------------------------------------------------- #
def build_statement_sets() -> dict:
    """Per trait: {'pos':[5 statements], 'neg':[5], 'provenance':{stmt:'ipip'|'ext'}}.

    Mixes IPIP items (provenance-seen) with held-out extended items so the
    positive_fraction can also be recomputed on the held-out subset for a
    leakage-robust check.
    """
    ipip = S.ipip50()
    ext = json.loads((DATA / "forced_choice_extended.json").read_text())
    out = {}
    for t in S.TRAITS:
        ipos = [i["item"] for i in ipip if i["trait"] == t and i["keyed"] == "+"]
        ineg = [i["item"] for i in ipip if i["trait"] == t and i["keyed"] == "-"]
        epos, eneg = ext[t]["pos"], ext[t]["neg"]
        # 2 IPIP + 3 extended per polarity where available (EST pos has only 2 ipip)
        pos = (ipos[:2] + epos[:3])[:5]
        neg = (ineg[:2] + eneg[:3])[:5]
        while len(pos) < 5:
            pos.append(epos[len(pos)])
        while len(neg) < 5:
            neg.append(eneg[len(neg)])
        prov = {}
        for s_ in pos[:5] + neg[:5]:
            prov[s_] = "ipip" if (s_ in ipos or s_ in ineg) else "ext"
        out[t] = {"pos": pos[:5], "neg": neg[:5], "provenance": prov}
    return out


def forced_choice_messages(statements: list[str]) -> list[dict]:
    return S.listing3_messages(statements)


_BULLET = re.compile(r"^\s*[-*\d.)•]+\s*")


def parse_picks(text: str, statements: list[str]) -> list[str] | None:
    """Return the subset of `statements` the model picked, or None if unparseable.

    Matches each output line to a presented statement by normalised-substring, so
    minor rewording/truncation still resolves. Requires >=4 distinct valid picks.
    """
    def norm(x):
        return re.sub(r"[^a-z0-9 ]", "", x.lower()).strip()
    nstmt = {norm(s): s for s in statements}
    picks = []
    for ln in text.splitlines():
        c = norm(_BULLET.sub("", ln))
        if not c:
            continue
        hit = None
        if c in nstmt:
            hit = nstmt[c]
        else:
            for ns, orig in nstmt.items():
                if ns and (ns in c or c in ns):
                    hit = orig
                    break
        if hit and hit not in picks:
            picks.append(hit)
    return picks if len(picks) >= 4 else None


def positive_fraction(picks: list[str], pos_set: set) -> float:
    return sum(1 for p in picks if p in pos_set) / len(picks)


# --------------------------------------------------------------------------- #
# steering config grid
# --------------------------------------------------------------------------- #
def layer_bands(n_layers: int) -> list[tuple[int, int]]:
    bands = []
    for center_frac in (0.4, 0.6, 0.8):
        for width in (8, 16, 24):
            c = int(round(center_frac * n_layers))
            lo = max(0, c - width // 2)
            hi = min(n_layers, c + width // 2)
            bands.append((lo, hi))
    return bands


def build_configs(n_layers: int) -> list[dict]:
    cfgs = []
    # S0: additive, last, all layers
    for a in np.linspace(-0.4, 0.4, 9):
        cfgs.append({"kind": "S0", "alpha": round(float(a), 3),
                     "label": f"S0_a{a:+.2f}"})
    bands = layer_bands(n_layers)
    # S1: norm-scaled additive, all positions, band
    for (lo, hi) in bands:
        for c in (-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3):
            cfgs.append({"kind": "S1", "c": c, "band": [lo, hi],
                         "label": f"S1_c{c:+.2f}_L{lo}-{hi}"})
    # S2: capping, all positions, band, both poles, 3 percentiles
    for (lo, hi) in bands:
        for pct in (10, 25, 50):
            for pole in ("ceiling", "floor"):
                cfgs.append({"kind": "S2", "pct": pct, "pole": pole,
                             "band": [lo, hi],
                             "label": f"S2_{pole}_p{pct}_L{lo}-{hi}"})
    return cfgs


# --------------------------------------------------------------------------- #
# apply a config as a steering context
# --------------------------------------------------------------------------- #
class Steerer:
    def __init__(self, model, bank: dict, resid_norms: np.ndarray,
                 proj_pct: dict):
        self.model = model
        self.bank = bank                  # trait -> [L, D] unit dirs
        self.norms = resid_norms          # [L]
        self.proj_pct = proj_pct          # trait -> {pct -> [L] percentile value}

    @contextmanager
    def apply(self, trait: str, cfg: dict):
        dirs = self.bank[trait]           # [L, D]
        if cfg["kind"] == "S0":
            layers = list(range(dirs.shape[0]))
            vecs = [torch.tensor(dirs[l]) for l in layers]
            coefs = [cfg["alpha"]] * len(layers)
            with ActivationSteering(self.model, steering_vectors=vecs,
                                    coefficients=coefs, layer_indices=layers,
                                    intervention_type="addition", positions="last"):
                yield
        elif cfg["kind"] == "S1":
            lo, hi = cfg["band"]
            layers = list(range(lo, hi))
            vecs = [torch.tensor(dirs[l]) for l in layers]
            coefs = [cfg["c"] * float(self.norms[l]) for l in layers]
            with ActivationSteering(self.model, steering_vectors=vecs,
                                    coefficients=coefs, layer_indices=layers,
                                    intervention_type="addition", positions="all"):
                yield
        elif cfg["kind"] == "S2":
            lo, hi = cfg["band"]
            layers = list(range(lo, hi))
            pct = cfg["pct"]
            if cfg["pole"] == "ceiling":
                vecs = [torch.tensor(dirs[l]) for l in layers]
                taus = [float(self.proj_pct[trait][pct][l]) for l in layers]
            else:  # floor: cap projection onto -r at -tau (using a LOW percentile)
                vecs = [torch.tensor(-dirs[l]) for l in layers]
                taus = [-float(self.proj_pct[trait][pct][l]) for l in layers]
            with ActivationSteering(self.model, steering_vectors=vecs,
                                    coefficients=[1.0] * len(layers),
                                    layer_indices=layers,
                                    intervention_type="capping", positions="all",
                                    cap_thresholds=taus):
                yield
        else:
            raise ValueError(cfg["kind"])


def compute_proj_percentiles(acts_char, bank, char_layers=None) -> dict:
    """trait -> {10/25/50 -> [L] percentile of char projections onto r_L}."""
    out = {}
    L = bank[S.TRAITS[0]].shape[0]
    for t in S.TRAITS:
        pcts = {p: np.zeros(L, dtype=np.float32) for p in (10, 25, 50)}
        for l in range(L):
            X = np.ascontiguousarray(acts_char[:, l, :])
            proj = X @ bank[t][l]
            for p in (10, 25, 50):
                pcts[p][l] = np.percentile(proj, p)
        out[t] = pcts
    return out
