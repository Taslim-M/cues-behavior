"""User-Axis activation extraction (RunPod GPU phase).

Forked from `assistant-axis` (github.com/safety-research/assistant-axis):
we reuse its `ProbingModel` (bf16 + device_map loading, layer discovery) and
`ConversationEncoder.build_turn_spans` (exact content-token spans per turn under
the chat template), but replace its ActivationExtractor with a memory-lean
dual-readout extractor.

Two readouts per rollout, captured in ONE forward pass over the full
conversation (per PLAN.md "Methodology"):

  * resp_mean  (PRIMARY)   mean of post-block residual-stream activations over
                           ALL content tokens of the FINAL assistant response
                           (the paper's exact recipe, applied to the readout turn).
  * last_user  (SECONDARY) activation at the LAST content token of the LAST user
                           turn (dashboard-style "user model" readout).

Both are [n_layers, d_model] (80 x 8192 for Llama-3.3-70B). The assistant-axis
extractor materializes (n_layers, B, seq, hidden) before reducing; at 80 layers
x 8192 dims that is GBs per batch, so our hooks reduce to the two readouts
*inside* the hook and never store the full sequence.
"""
from __future__ import annotations

from typing import Optional

import torch

from assistant_axis import MODEL_CONFIGS, get_config  # noqa: F401  (re-export)
from assistant_axis.internals import ProbingModel, ConversationEncoder

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# Short key used for results paths (matches config.MODELS key style).
SHORT_NAMES = {DEFAULT_MODEL: "llama-3.3-70b"}


def short_name(model_name: str) -> str:
    return SHORT_NAMES.get(model_name, model_name.split("/")[-1].lower())


def load_model(model_name: str = DEFAULT_MODEL,
               dtype: torch.dtype = torch.bfloat16) -> ProbingModel:
    """Load model bf16 sharded over all visible GPUs (2x80GB fits 70B)."""
    return ProbingModel(model_name, device=None, dtype=dtype)


class DualReadoutExtractor:
    """One forward pass -> (resp_mean, last_user), each [n_layers, d_model]."""

    def __init__(self, pm: ProbingModel):
        self.pm = pm
        self.encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
        self.n_layers = len(pm.get_layers())
        self.d_model = pm.hidden_size

    # ------------------------------------------------------------------ #
    # Span bookkeeping
    # ------------------------------------------------------------------ #
    def _readout_positions(self, conversation: list[dict]) -> tuple[list[int], int, int]:
        """Token ids + [start,end) of the final assistant response content and
        the index of the last user-turn content token."""
        full_ids, spans = self.encoder.build_turn_spans(conversation)
        asst = [s for s in spans if s["role"] == "assistant"]
        user = [s for s in spans if s["role"] == "user"]
        if not asst or not user:
            raise ValueError("conversation needs >=1 user and >=1 assistant turn")
        resp = asst[-1]
        if resp["end"] <= resp["start"]:
            raise ValueError("empty final assistant span")
        last_user_tok = user[-1]["end"] - 1
        if not (0 <= last_user_tok < len(full_ids)):
            raise ValueError("last-user token index out of range")
        return full_ids, (resp["start"], resp["end"]), last_user_tok

    # ------------------------------------------------------------------ #
    # Batched extraction with in-hook reduction
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def extract_batch(
        self,
        conversations: list[list[dict]],
        max_length: int = 4096,
    ) -> list[Optional[dict[str, torch.Tensor]]]:
        """Return per conversation {'resp_mean': [L,D], 'last_user': [L,D]} float32
        (None where spans could not be resolved or were fully truncated)."""
        B = len(conversations)
        metas: list[Optional[tuple]] = []
        for conv in conversations:
            try:
                metas.append(self._readout_positions(conv))
            except Exception as e:  # noqa: BLE001 - skip unresolvable, keep going
                print(f"  [extract] span failure ({type(e).__name__}: {e}); skipping one")
                metas.append(None)

        valid = [i for i, m in enumerate(metas) if m is not None]
        if not valid:
            return [None] * B

        # left-align + right-pad (positions stay = span indices)
        seq_lens = {i: min(len(metas[i][0]), max_length) for i in valid}
        max_len = max(seq_lens.values())
        pad_id = self.pm.tokenizer.pad_token_id
        input_ids = torch.full((len(valid), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(valid), max_len), dtype=torch.long)
        # per-valid-row readout indices, clipped to the truncated length
        rows = []
        for row, i in enumerate(valid):
            full_ids, (rs, re_), lu = metas[i]
            n = seq_lens[i]
            input_ids[row, :n] = torch.tensor(full_ids[:n], dtype=torch.long)
            attn[row, :n] = 1
            re_c = min(re_, n)
            if rs >= re_c or lu >= n:  # response or user token truncated away
                rows.append(None)
                if len(full_ids) > max_length:
                    print(f"  [extract] conv {i} truncated past readout span; skipping")
            else:
                rows.append((rs, re_c, lu))

        device = self.pm.device
        input_ids = input_ids.to(device)
        attn = attn.to(device)

        # accumulators on CPU: [n_layers, n_valid, d_model]
        resp_mean = torch.zeros(self.n_layers, len(valid), self.d_model, dtype=torch.float32)
        last_user = torch.zeros(self.n_layers, len(valid), self.d_model, dtype=torch.float32)

        handles = []

        def make_hook(layer_idx: int):
            def hook(module, inputs, output):
                act = output[0] if isinstance(output, tuple) else output  # (B,S,H)
                for row, pos in enumerate(rows):
                    if pos is None:
                        continue
                    rs, re_c, lu = pos
                    resp_mean[layer_idx, row] = act[row, rs:re_c].float().mean(dim=0).cpu()
                    last_user[layer_idx, row] = act[row, lu].float().cpu()
            return hook

        layers = self.pm.get_layers()
        for li in range(self.n_layers):
            handles.append(layers[li].register_forward_hook(make_hook(li)))
        try:
            _ = self.pm.model(input_ids=input_ids, attention_mask=attn)
        finally:
            for h in handles:
                h.remove()

        out: list[Optional[dict[str, torch.Tensor]]] = [None] * B
        for row, i in enumerate(valid):
            if rows[row] is None:
                continue
            out[i] = {
                "resp_mean": resp_mean[:, row].clone(),   # [L, D] fp32
                "last_user": last_user[:, row].clone(),   # [L, D] fp32
            }
        return out

    # ------------------------------------------------------------------ #
    # Generation (HF; batched, left-padded)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate_batch(
        self,
        conversations: list[list[dict]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[str]:
        """Sampled continuations for a batch of conversations (assistant next)."""
        tok = self.pm.tokenizer
        prompts = [
            tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations
        ]
        tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(self.pm.device)
        out = self.pm.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.pad_token_id,
        )
        new = out[:, enc.input_ids.shape[1]:]
        return [tok.decode(seq, skip_special_tokens=True).strip() for seq in new]
