"""
Predictor — wraps an `EncoderDecoderModel` plus tokenizer callbacks
into a single callable mapping `context -> (text, ids, mean_logprob)`.

Why a class (not a free function)? A `Predictor` carries non-trivial
per-call state that is expensive to recompute every time:

    * `model.eval()` is set ONCE in the constructor
    * special-token ids and length budgets are bound ONCE
    * encode/decode functions are bound ONCE

Domain ignorance: the predictor does not know what TypeScript is. It
takes generic `encode: str -> list[int]` / `decode: list[int] -> str`
callbacks. To use it with a different tokenizer (e.g. SentencePiece,
character-level), pass different callables.

Confidence score: we report `mean_logprob` — the per-token average of
log P(token | prefix) under the model's own teacher-forced forward.
Negative numbers, closer to 0 means more confident. Callers (e.g. an
inference filter) can threshold this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F

from .encoder_decoder_model import EncoderDecoderModel


# ══════════════════════════════════════════════════════════════════════
# Public dataclasses
# ══════════════════════════════════════════════════════════════════════

@dataclass
class PredictResult:
    """One model prediction plus its confidence score."""

    text: str               # decoded string (no <BOS>, may contain <EOS>-stripped suffix)
    ids: List[int]          # raw token ids (without <BOS>; may include <EOS> if model emitted it)
    mean_logprob: float     # average log p(token | prefix); 0 = certain, very negative = unsure


# Tokenizer callback types — kept abstract so this file has zero
# ts-type-refiner imports.
EncodeFn = Callable[[str], List[int]]
DecodeFn = Callable[[List[int]], str]


# ══════════════════════════════════════════════════════════════════════
# Predictor
# ══════════════════════════════════════════════════════════════════════

class Predictor:
    """
    Callable: `predictor(context_str) -> PredictResult`.

    Construct once per model+tokenizer pair, reuse for every candidate.

    Parameters
    ----------
    model
        Trained `EncoderDecoderModel`. Switched to `eval()` mode here.
    encode / decode
        String <-> token-id callbacks (e.g. `tok.encode`, `tok.decode`).
    bos_id, eos_id
        Special tokens passed through to `model.generate` AND used to
        rebuild the teacher-forced input for the log-prob pass.
    max_src_len, max_tgt_len
        Length budgets. Source is truncated; target generation stops
        either at `<EOS>` or after `max_tgt_len` tokens.
    device
        Device to host all input tensors on (must match `model`).
    temperature
        Sampling temperature for `model.generate`. Defaults to a very
        small value so generation is effectively greedy / deterministic
        — appropriate for inference where we want repeatable answers.
    """

    def __init__(
        self,
        model: EncoderDecoderModel,
        encode: EncodeFn,
        decode: DecodeFn,
        *,
        bos_id: int,
        eos_id: int,
        max_src_len: int,
        max_tgt_len: int,
        device: torch.device,
        temperature: float = 0.01,
    ) -> None:
        # `eval()` disables training-only behaviours (dropout, etc.).
        # The model is mutated in place — by design, so external code
        # sees `model.training == False` after constructing a Predictor.
        model.eval()

        self.model         = model
        self.encode        = encode
        self.decode        = decode
        self.bos_id        = bos_id
        self.eos_id        = eos_id
        self.max_src_len   = max_src_len
        self.max_tgt_len   = max_tgt_len
        self.device        = device
        self.temperature   = temperature

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, context: str) -> PredictResult:
        """
        Run `context -> tokens -> generated tokens -> string`, then
        compute the mean per-token log-prob via a single teacher-forced
        forward pass. Two forward passes total per call (one autoregressive
        decode + one teacher-forced rescoring).
        """

        # ── Tokenise + truncate the source ───────────────────────────
        src_ids = self.encode(context)[: self.max_src_len]
        src_tensor = torch.tensor([src_ids], device=self.device)

        # ── Autoregressive generation ────────────────────────────────
        gen = self.model.generate(
            src_tensor,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            max_new_tokens=self.max_tgt_len,
            temperature=self.temperature,
            verbose=False,
        )
        # gen: (1, gen_len) — does NOT include leading <BOS>.
        gen_ids: List[int] = gen[0].tolist()

        if len(gen_ids) == 0:
            # Model produced <EOS> immediately, or `max_tgt_len == 0`.
            return PredictResult(text="", ids=[], mean_logprob=float("-inf"))

        # ── Teacher-forced log-prob rescoring ────────────────────────
        # Build (decoder_input, decoder_target) the same way training does:
        #     in     = [BOS] + gen_ids[:-1]
        #     target =         gen_ids
        # Then sum log p(target_t | in_{0..t}) and average over t.
        dec_in  = torch.tensor([[self.bos_id] + gen_ids[:-1]], device=self.device)
        dec_tgt = torch.tensor([gen_ids], device=self.device)

        logits     = self.model(src_tensor, dec_in, verbose=False)   # (1, T, V)
        logp       = F.log_softmax(logits, dim=-1)
        token_logp = logp.gather(2, dec_tgt.unsqueeze(-1)).squeeze(-1)  # (1, T)
        mean_lp    = token_logp.mean().item()

        return PredictResult(
            text         = self.decode(gen_ids),
            ids          = gen_ids,
            mean_logprob = mean_lp,
        )
