"""Unified activation-extraction protocol (plan §2). Used identically by every stage.

Contract
--------
* **Hook point:** residual stream at the output of each transformer block
  ("resid_post"), captured with a forward hook on each decoder layer. This is
  the *same* convention `src/useraxis/extract.py` uses, which is what makes our
  vectors comparable with the published Assistant-Axis artifacts.
* **Layer indexing:** we store `[80, d_model]`, index `l` = output of block `l`
  (0-based over `model.model.layers`). The plan writes `hidden_states[l]` for
  `l=1..80` with `hidden_states[0]` = embeddings; those are the same tensors
  offset by one, i.e. our index `l` == plan's `hidden_states[l+1]`. Recorded
  here so downstream projections are unambiguous.
* **Positions (all three captured every pass):**
    - `last_prompt` : activation at the final prompt token
    - `prompt_mean` : mean over non-pad prompt tokens
    - `gen_mean`    : mean over generated (non-pad) tokens
* **Storage dtype:** float32.

Reduction happens *inside* the hook: a full `[B, T, 80, 8192]` tensor would be
GBs per batch, so we never materialise it.

Generation is two-phase -- generate with no hooks, then a single hooked forward
pass over the completed sequence. Hooking during generation forces a CPU sync
per decode step per layer and is ~10x slower (measured).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

POSITIONS = ("last_prompt", "prompt_mean", "gen_mean")


class BigFiveExtractor:
    """Three-position, all-layer resid_post extractor."""

    def __init__(self, pm):
        self.pm = pm
        self.tok = pm.tokenizer
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.layers = pm.get_layers()
        self.n_layers = len(self.layers)
        self.d_model = pm.hidden_size

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def run_batch(self, message_lists: list[list[dict]], *, generate: bool = True,
                  max_new_tokens: int = 256, do_sample: bool = False,
                  temperature: float = 1.0, top_p: float = 1.0,
                  max_length: int = 4096) -> tuple[dict[str, np.ndarray], list[str]]:
        """-> ({position: [B, n_layers, d]}, generated_texts).

        With `generate=False` no text is produced and `gen_mean` is zero-filled
        (used for prompt-only stimuli such as the forced-choice/adjective probes
        where the plan reads the last prompt token).
        """
        tok = self.tok
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in message_lists]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, add_special_tokens=False).to(self.pm.model.device)
        plen = enc["input_ids"].shape[1]
        pmask = enc["attention_mask"].bool()

        if generate:
            gkw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            if do_sample:
                gkw.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                gkw.update(do_sample=False, temperature=None, top_p=None)
            out = self.pm.model.generate(**enc, **gkw)
            gen_ids = out[:, plen:]
            gmask = (gen_ids != tok.pad_token_id)
            texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
            full = out
            attn = torch.cat([enc["attention_mask"], gmask.long()], dim=1)
        else:
            full = enc["input_ids"]
            attn = enc["attention_mask"]
            gmask = None
            texts = [""] * len(message_lists)

        B = full.shape[0]
        buf = {p: torch.zeros(self.n_layers, B, self.d_model, dtype=torch.float32)
               for p in POSITIONS}
        # index of the last real prompt token (left padding => always plen-1)
        last_idx = plen - 1

        def make_hook(li: int):
            def hook(_m, _inp, out):
                a = (out[0] if isinstance(out, tuple) else out).float()
                dev = a.device
                pm_ = pmask.to(dev)
                buf["last_prompt"][li] = a[:, last_idx, :].cpu()
                pa = a[:, :plen] * pm_.unsqueeze(-1)
                buf["prompt_mean"][li] = (
                    pa.sum(1) / pm_.sum(1, keepdim=True).clamp(min=1)).cpu()
                if gmask is not None:
                    gm_ = gmask.to(dev)
                    ga = a[:, plen:] * gm_.unsqueeze(-1)
                    buf["gen_mean"][li] = (
                        ga.sum(1) / gm_.sum(1, keepdim=True).clamp(min=1)).cpu()
            return hook

        handles = [self.layers[li].register_forward_hook(make_hook(li))
                   for li in range(self.n_layers)]
        try:
            self.pm.model(input_ids=full, attention_mask=attn)
        finally:
            for h in handles:
                h.remove()

        # -> [B, n_layers, d]
        return {p: buf[p].permute(1, 0, 2).contiguous().numpy() for p in POSITIONS}, texts


class ActivationStore:
    """Memmapped `[N, n_layers, d]` float32 array per position + a JSON index.

    The plan §2 asks for `index.parquet`; pandas/pyarrow are not installed in
    this environment, so the index is JSON with the same columns. Recorded as a
    deviation in the stage report.
    """

    def __init__(self, root: Path, n: int, n_layers: int, d_model: int,
                 mode: str = "w+"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shape = (n, n_layers, d_model)
        self.arr = {
            p: np.lib.format.open_memmap(self.root / f"acts_{p}.npy", mode=mode,
                                         dtype=np.float32, shape=self.shape)
            for p in POSITIONS
        }
        self.index: list[dict] = []

    def write(self, row0: int, acts: dict[str, np.ndarray], rows: list[dict]) -> None:
        n = len(rows)
        for p in POSITIONS:
            self.arr[p][row0:row0 + n] = acts[p]
        self.index.extend(rows)

    def finalize(self, meta: dict) -> None:
        for p in POSITIONS:
            self.arr[p].flush()
        (self.root / "index.json").write_text(json.dumps(self.index, indent=1))
        (self.root / "meta.json").write_text(json.dumps(
            {**meta, "shape": list(self.shape), "positions": list(POSITIONS),
             "dtype": "float32",
             "layer_convention": "index l = output of decoder block l (0-based); "
                                 "equals plan's hidden_states[l+1]"}, indent=1))


def round_trip_check(pm, verbose: bool = True) -> dict:
    """Gate G2: 5-stimulus extract -> save -> reload round trip.

    Verifies shapes, dtype, finiteness, byte-exact reload, and that the three
    positions are genuinely distinct readouts (not accidentally aliased).
    """
    import tempfile

    from src.bigfive import stimuli as S

    ex = BigFiveExtractor(pm)
    msgs = [S.listing4_messages(a["instruction"]) for a in S.alpaca10()[:5]]
    acts, _ = ex.run_batch(msgs, generate=True, max_new_tokens=32)

    res: dict = {"shapes": {}, "dtype": {}, "finite": {}}
    for p in POSITIONS:
        res["shapes"][p] = list(acts[p].shape)
        res["dtype"][p] = str(acts[p].dtype)
        res["finite"][p] = bool(np.isfinite(acts[p]).all())

    with tempfile.TemporaryDirectory() as td:
        st = ActivationStore(Path(td), 5, ex.n_layers, ex.d_model)
        st.write(0, acts, [{"stimulus_id": f"g2_{i}"} for i in range(5)])
        st.finalize({"stage": "G2"})
        reloaded = {p: np.load(Path(td) / f"acts_{p}.npy") for p in POSITIONS}
        res["roundtrip_exact"] = {
            p: bool(np.array_equal(acts[p], reloaded[p])) for p in POSITIONS}

    # positions must differ from one another
    res["positions_distinct"] = {
        f"{a}_vs_{b}": float(np.abs(acts[a] - acts[b]).mean())
        for a, b in (("last_prompt", "prompt_mean"),
                     ("last_prompt", "gen_mean"),
                     ("prompt_mean", "gen_mean"))
    }
    res["pass"] = (
        all(res["finite"].values())
        and all(res["roundtrip_exact"].values())
        and all(s == [5, ex.n_layers, ex.d_model] for s in res["shapes"].values())
        and all(v > 1e-6 for v in res["positions_distinct"].values())
    )
    if verbose:
        print(json.dumps(res, indent=1))
    return res
