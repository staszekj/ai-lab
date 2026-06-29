"""
Encoder-Decoder Transformer Model (T5 / BART style)
====================================================================

The canonical, production reference seq2seq Transformer for the
`core` package. Domain-agnostic: knows nothing about TypeScript,
tokenizers, training loops, checkpoints or validators — it is a pure
nn.Module that maps `(src_ids, tgt_in_ids) → logits` and supports
autoregressive generation.

Companion modules in `core`:
    - core.trainer    : pure `train(model, batches, …)` function
    - core.predictor  : `Predictor(model, encode, decode, …)` callable
    - core.checkpoint : save / load / build_model helpers

Architecture (Pre-LN variant — used by GPT-2, T5, BART, modern LLMs):

    ┌──────────────────────────────────────────────────────────────┐
    │  ENCODER STACK  (N × ManualTransformerEncoderBlock)          │
    │    - Bidirectional multi-head self-attention (no mask)       │
    │    - Feed-forward network                                    │
    │    - Two LayerNorms (Pre-LN: before each sub-layer)          │
    │    - Two residual connections                                │
    └──────────────────────────────────────────────────────────────┘
                              │
                              │ encoder_output (cached)
                              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  DECODER STACK  (N × ManualDecoderBlock)                     │
    │    - Masked multi-head self-attention (causal mask)          │
    │    - Cross-attention (Q from decoder, K/V from encoder_out)  │
    │    - Feed-forward network                                    │
    │    - Three LayerNorms                                        │
    │    - Three residual connections                              │
    └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     LM Head → logits (vocab)

Key concepts:

    TRAINING (teacher forcing):
        src_ids                      → encoder → encoder_output
        [<BOS>] + tgt_ids[:-1]       → decoder(encoder_output) → logits
        loss = cross_entropy(logits, tgt_ids + [<EOS>])

    GENERATION (autoregressive):
        encoder runs ONCE, encoder_output is cached.
        decoder runs one step at a time:
            tgt = [<BOS>]
            while not <EOS>:
                logits = decoder(tgt, encoder_output)
                tgt.append(sample(logits[:, -1]))

Key shapes (always commented inline):
    src_ids        : (batch, src_len)
    tgt_ids        : (batch, tgt_len)
    encoder_output : (batch, src_len, d_model)
    logits         : (batch, tgt_len, vocab_size)

This file is the production reference implementation — it is trained on real
degraded-types corpora on CUDA. For a hand-walked tour with a hard-coded tiny
example (`const enabled : string` → `"ON" | "OFF"`), tiny tensors and pretty
matrix prints, run:

    uv run --package core presentation-encoder-decoder

In-code comments below cross-reference the STEPs in that presentation script.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# [LoRA] LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method.
# [LoRA] Instead of updating full pretrained weights W, we freeze W and learn
# [LoRA] a low-rank update DeltaW = B @ A, so only small adapter matrices train.
# [LoRA] This keeps most model behavior intact while reducing trainable params
# [LoRA] and VRAM needs.


# ══════════════════════════════════════════════════════════════════════
# MICRO-MODEL — reference dimensions used in all inline comments
# ══════════════════════════════════════════════════════════════════════
#
# Inline comments throughout this file show concrete tensor shapes and
# matrix examples using a tiny "micro-model" identical to the one in
# presentation_encoder_decoder.py.  Every number in the comments can be
# traced back to one of these names:
#
#   vocab_size  = 12   tokens: <pad> <bos> const let var enabled : string number ON | OFF
#   d_model     =  6   embedding / hidden dimension (must equal num_heads × d_k)
#   num_heads   =  3   parallel attention heads
#   d_k         =  2   = d_model // num_heads  — dimension per head
#   d_ff        = 12   feed-forward inner dimension (here 2 × d_model; prod. uses 4 ×)
#   num_layers  =  2   encoder blocks AND decoder blocks
#   max_seq_len = 16   positional embedding table size
#   batch       =  1   always 1 in presentation examples
#   src_len     =  4   source tokens: "const", "enabled", ":", "string"
#   tgt_len     =  3   decoder-input tokens: "<bos>", "ON", "|"
#                      (decoder target "ON | OFF" is tgt_len=3 shifted by 1)
#
# Example pair traced end-to-end:
#   SOURCE  "const enabled : string"   src_ids = [[2, 5, 6, 7]]  shape (1, 4)
#   TARGET  "ON | OFF"                 tgt_ids = [[9, 10, 11]]   shape (1, 3)
#   decoder INPUT  (teacher-forced)  = [[1,  9, 10]]  ("<bos> ON |")
#   decoder TARGET (what to predict) = [[9, 10, 11]]  ("ON | OFF")
#
# Production model (ts-type-refiner) uses much larger values, e.g.:
#   vocab_size=2048, d_model=256, num_heads=8, d_k=32, d_ff=1024, num_layers=4
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# ENCODER BLOCK — bidirectional self-attention + FFN (Pre-LN)
# ══════════════════════════════════════════════════════════════════════

class ManualTransformerEncoderBlock(nn.Module):
    """
    A single Pre-LN Transformer encoder block.

    See: presentation_encoder_decoder.py — STEP 2
          (bidirectional self-attention pattern, full attn-weight matrix)

    Sub-layers:
        1. Multi-head self-attention (no mask → bidirectional)
        2. Feed-forward network

    Pre-LN formula:
        x1     = x  + Attention(LayerNorm(x))
        output = x1 + FFN(LayerNorm(x1))

    Input / output shape: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_ff = d_ff

        assert d_model == num_heads * self.d_k, (
            f"d_model ({d_model}) must equal num_heads ({num_heads}) × d_k ({self.d_k})"
        )

        # Attention projections — Q/K/V/O are all (d_model → d_model).
        # Q,K,V split into heads via reshape inside forward().
        # [LoRA] Candidate target: encoder self-attention query projection (Q).
        # [LoRA] Adapting Q changes what each source token asks from context.
        self.W_Q = nn.Linear(d_model, d_model)
        # [LoRA] Candidate target: encoder self-attention key projection (K).
        # [LoRA] Adapting K changes how source tokens expose retrievable features.
        self.W_K = nn.Linear(d_model, d_model)
        # [LoRA] Candidate target: encoder self-attention value projection (V).
        # [LoRA] Adapting V changes which source features are returned.
        self.W_V = nn.Linear(d_model, d_model)
        # [LoRA] Candidate target: encoder self-attention output projection (O).
        # [LoRA] Adapting O changes how multi-head outputs are recombined.
        self.W_O = nn.Linear(d_model, d_model)

        # Feed-forward network: expand to d_ff, GELU, compress back.
        # d_ff is conventionally 4 × d_model.
        # [LoRA] Candidate target: FFN expansion (d_model -> d_ff).
        # [LoRA] This often provides strong adaptation with low-rank updates.
        self.ffn_linear1 = nn.Linear(d_model, d_ff)
        self.ffn_gelu    = nn.GELU()
        # [LoRA] Candidate target: FFN compression (d_ff -> d_model).
        self.ffn_linear2 = nn.Linear(d_ff, d_model)

        # [LoRA] Showcase-only pseudo integration (commented, non-executable):
        # [LoRA] W_eff = W_frozen + (B @ A) * (alpha / r)
        # [LoRA] Typical first targets in this block: W_Q, W_V, ffn_linear1.

        # Pre-LN: one LayerNorm before each sub-layer.
        self.layer_norm_1 = nn.LayerNorm(d_model)  # before attention
        self.layer_norm_2 = nn.LayerNorm(d_model)  # before FFN

        # Scaling factor 1/sqrt(d_k) prevents softmax saturation when d_k is large.
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (batch, seq_len, d_model)
        attn_mask : optional (seq_len, seq_len) additive mask (0=attend, -inf=block)

        Returns
        -------
        (batch, seq_len, d_model)

        Presentation example (d_model=6, heads=3, d_k=2, src_seq=4):
            x  "const enabled : string"            (1, 4, 6)
            Q = W_Q(x)                             (1, 4, 6)
            Q split into heads                     (1, 3, 4, 2)   [batch, heads, seq, d_k]
            scores = Q @ K^T / sqrt(2)             (1, 3, 4, 4)   ← SQUARE: every src token × every src token
            weights = softmax(scores)              (1, 3, 4, 4)   ← each row sums to 1
            head_out = weights @ V                 (1, 3, 4, 2)
            concat + W_O                           (1, 4, 6)
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model

        if verbose:
            print(f"\n{'='*60}")
            print(f"EncoderBlock | x: {x.shape}")

        # ── Pre-LN + multi-head self-attention ───────────────────────
        x_norm1 = self.layer_norm_1(x)
        # x_norm1: (batch, seq_len, d_model)  e.g. (1, 4, 6)
        if verbose:
            print(f"After LayerNorm 1:                 x_norm1:  {x_norm1.shape}")

        # Project to Q, K, V — three independent (d_model → d_model) linear maps.
        # Q asks "what am I looking for?", K says "what do I have?",
        # V says "what do I return if matched?".
        Q = self.W_Q(x_norm1)    # (1, 4, 6)
        K = self.W_K(x_norm1)    # (1, 4, 6)
        V = self.W_V(x_norm1)    # (1, 4, 6)

        # Split d_model into `num_heads` independent attention subspaces.
        # (batch, seq, d_model) → (batch, seq, heads, d_k) → (batch, heads, seq, d_k)
        # Example: (1, 4, 6) → (1, 4, 3, 2) → (1, 3, 4, 2)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q,K,V: (1, 3, 4, 2)

        # scores[h, i, j] = how much token i attends to token j in head h.
        # (1, 3, 4, 2) @ (1, 3, 2, 4) = (1, 3, 4, 4) — SQUARE because Q and K
        # both come from the same source sequence (bidirectional self-attention).
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (1, 3, 4, 4)

        if attn_mask is not None:
            # Additive mask: -inf at blocked positions → softmax drives those to 0.
            scores = scores + attn_mask

        weights = torch.softmax(scores, dim=-1)
        # weights: (1, 3, 4, 4) — each row is a probability distribution over src tokens.
        # Row i sums to 1: token i's attention is fully distributed over all 4 src positions.

        # Weighted sum of values: how much of each V row to mix for each query token.
        head_out = torch.matmul(weights, V)
        # (1, 3, 4, 4) @ (1, 3, 4, 2) = (1, 3, 4, 2)

        # Merge heads: last two dims (heads, seq, d_k) → (seq, d_model).
        # (1, 3, 4, 2) → (1, 4, 3, 2) → (1, 4, 6)
        attn_out = head_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.W_O(attn_out)
        # attn_out: (1, 4, 6)

        x1 = x + attn_out   # first residual: (1, 4, 6)
        if verbose:
            print(f"After attention + residual #1:     x1:       {x1.shape}")

        # ── Pre-LN + feed-forward network ────────────────────────────
        # FFN applies the same two-layer MLP independently to each token.
        # Expand: (1, 4, 6) → (1, 4, 12)   [d_model → d_ff, typically 4×]
        # Compress: (1, 4, 12) → (1, 4, 6) [d_ff → d_model]
        x1_norm = self.layer_norm_2(x1)
        ffn_hidden = self.ffn_linear1(x1_norm)   # (1, 4, 12)
        ffn_hidden = self.ffn_gelu(ffn_hidden)
        ffn_out    = self.ffn_linear2(ffn_hidden) # (1, 4, 6)

        output = x1 + ffn_out   # second residual: (1, 4, 6)
        if verbose:
            print(f"After FFN + residual #2:           output:   {output.shape}")
            print(f"{'='*60}")

        return output


# ══════════════════════════════════════════════════════════════════════
# DECODER BLOCK — masked self-attn + cross-attn + FFN (Pre-LN)
# ══════════════════════════════════════════════════════════════════════

class ManualDecoderBlock(nn.Module):
    """
    A single Pre-LN Transformer Decoder Block with three sub-layers:

    See: presentation_encoder_decoder.py — STEP 3
          (causal mask diagram, cross-attention 3×4 weight matrix print)

        1. Masked Self-Attention   (Q,K,V from decoder; causal mask)
        2. Cross-Attention         (Q from decoder; K,V from encoder_output)
        3. Feed-Forward Network

    Pre-LN formula:
        x1     = x  + MaskedSelfAttention(LayerNorm(x), mask=causal)
        x2     = x1 + CrossAttention(LayerNorm(x1), encoder_output)
        output = x2 + FFN(LayerNorm(x2))

    Input  shape: (batch, tgt_len, d_model)
    Output shape: (batch, tgt_len, d_model)

    Cross-attention:
        Q comes from the decoder.
        K, V come from encoder_output (shape: batch, src_len, d_model).
        Score matrix shape: (batch, heads, tgt_len, src_len) — NOT square
        when tgt_len ≠ src_len.

    Gradient flow note:
        dL/dcross_W_K and dL/dcross_W_V flow back through encoder_output
        into encoder weights — the encoder is trained by decoder loss.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_ff = d_ff

        assert d_model == num_heads * self.d_k, (
            f"d_model ({d_model}) must equal num_heads ({num_heads}) × d_k ({self.d_k})"
        )
        self.scale = math.sqrt(self.d_k)

        # Sub-layer 1: masked self-attention (Q,K,V all from decoder).
        # [LoRA] Candidate target: decoder masked self-attention projections.
        self.self_W_Q = nn.Linear(d_model, d_model)
        self.self_W_K = nn.Linear(d_model, d_model)
        self.self_W_V = nn.Linear(d_model, d_model)
        self.self_W_O = nn.Linear(d_model, d_model)

        # Sub-layer 2: cross-attention. K and V come from encoder_output.
        # These are SEPARATE matrices from the self-attention ones — they
        # learn different transformations.
        # [LoRA] High-priority target: cross-attention query projection (Q).
        # [LoRA] This controls what decoder states ask from encoder memory.
        self.cross_W_Q = nn.Linear(d_model, d_model)
        # [LoRA] High-priority target: cross-attention key projection (K).
        # [LoRA] This controls how encoder memory is indexed for retrieval.
        self.cross_W_K = nn.Linear(d_model, d_model)
        # [LoRA] High-priority target: cross-attention value projection (V).
        # [LoRA] This controls which encoder content is injected into decoder.
        self.cross_W_V = nn.Linear(d_model, d_model)
        # [LoRA] Candidate target: cross-attention output projection (O).
        # [LoRA] This controls post-attention mixing into decoder hidden states.
        self.cross_W_O = nn.Linear(d_model, d_model)

        # [LoRA] Showcase-only pseudo integration (commented, non-executable):
        # [LoRA] Replace selected Linear layers with LoRA-wrapped equivalents.
        # [LoRA] Train only adapter matrices A/B; keep base W frozen.
        # [LoRA] Suggested first targets: cross_W_Q, cross_W_K, cross_W_V, cross_W_O.

        # Sub-layer 3: feed-forward network.
        # [LoRA] Candidate target: decoder FFN expansion (d_model -> d_ff).
        self.ffn_linear1 = nn.Linear(d_model, d_ff)
        self.ffn_gelu    = nn.GELU()
        # [LoRA] Candidate target: decoder FFN compression (d_ff -> d_model).
        self.ffn_linear2 = nn.Linear(d_ff, d_model)

        # Pre-LN: one LayerNorm before each sub-layer.
        self.layer_norm_1 = nn.LayerNorm(d_model)  # before masked self-attention
        self.layer_norm_2 = nn.LayerNorm(d_model)  # before cross-attention
        self.layer_norm_3 = nn.LayerNorm(d_model)  # before FFN

    # ------------------------------------------------------------------
    # Generic multi-head attention used by both self- and cross-attention
    # ------------------------------------------------------------------
    def _multi_head_attention(
        self,
        Q_src: torch.Tensor,
        K_src: torch.Tensor,
        V_src: torch.Tensor,
        W_Q: nn.Linear,
        W_K: nn.Linear,
        W_V: nn.Linear,
        W_O: nn.Linear,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Self-attention:   Q_src = K_src = V_src = decoder hidden state x.
        Cross-attention:  Q_src = decoder x; K_src = V_src = encoder_output.

        Parameters
        ----------
        Q_src     : (batch, q_len,  d_model)
        K_src     : (batch, kv_len, d_model)
        V_src     : (batch, kv_len, d_model)
        attn_mask : (q_len, kv_len) additive mask or None

        Returns
        -------
        (batch, q_len, d_model)
        """
        batch_size = Q_src.shape[0]
        q_len      = Q_src.shape[1]
        kv_len     = K_src.shape[1]

        # Project to Q, K, V; each still (batch, *, d_model).
        Q = W_Q(Q_src)
        K = W_K(K_src)
        V = W_V(V_src)

        # Split into heads. Head dim is moved before sequence dim so the
        # last two dims are (len, d_k) and matmul broadcasts naturally.
        Q = Q.view(batch_size, q_len,  self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, kv_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, kv_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q: (batch, heads, q_len,  d_k)
        # K: (batch, heads, kv_len, d_k)
        # V: (batch, heads, kv_len, d_k)

        # Scaled dot-product attention.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (batch, heads, q_len, kv_len)

        if attn_mask is not None:
            scores = scores + attn_mask

        weights = torch.softmax(scores, dim=-1)
        # weights: (batch, heads, q_len, kv_len)

        head_out = torch.matmul(weights, V)
        # head_out: (batch, heads, q_len, d_k)

        # Concatenate heads back to d_model.
        out = head_out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return W_O(out)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x              : (batch, tgt_len, d_model)
        encoder_output : (batch, src_len, d_model)
        tgt_mask       : (tgt_len, tgt_len) causal mask for self-attention
        src_mask       : (tgt_len, src_len) optional padding mask for cross-attention
        """
        batch_size, tgt_len, d_model = x.shape
        _, src_len, _ = encoder_output.shape
        assert d_model == self.d_model

        if verbose:
            print(f"\n{'='*60}")
            print(f"DecoderBlock | x: {x.shape}  encoder_output: {encoder_output.shape}")

        # ── Sub-layer 1: masked self-attention ───────────────────────
        # Decoder tokens attend to themselves and PAST decoder tokens only
        # (enforced by the causal tgt_mask added to scores).
        #
        # Presentation: tgt_len=3 ("<bos>","ON","|"), d_model=6, heads=3, d_k=2
        #   Q=K=V from x_norm1        (1, 3, 6)
        #   split into heads          (1, 3, 3, 2)   [batch, heads, tgt_len, d_k]
        #   scores = Q @ K^T          (1, 3, 3, 3)   ← SQUARE: tgt × tgt
        #   + causal mask             (3, 3)
        #
        #              <bos>    ON      |
        #   <bos>  [   0.      -inf    -inf ]
        #      ON  [   0.       0.     -inf ]
        #       |  [   0.       0.      0.  ]
        #
        #   weights after softmax     (1, 3, 3, 3)
        #   head_out = weights @ V    (1, 3, 3, 2)   → concat → (1, 3, 6)
        x_norm1 = self.layer_norm_1(x)
        self_attn_out = self._multi_head_attention(
            Q_src=x_norm1, K_src=x_norm1, V_src=x_norm1,
            W_Q=self.self_W_Q, W_K=self.self_W_K,
            W_V=self.self_W_V, W_O=self.self_W_O,
            attn_mask=tgt_mask,  # causal
        )
        x1 = x + self_attn_out   # (1, 3, 6)
        if verbose:
            print(f"After masked-self-attn + residual: x1:       {x1.shape}")

        # ── Sub-layer 2: cross-attention ─────────────────────────────
        # The decoder "reads" the encoder output here.  Q comes from the
        # decoder (what the decoder is currently thinking about generating),
        # K and V come from encoder_output (the full source context).
        # No causal mask — the decoder may attend to ALL encoder positions.
        #
        # Presentation: tgt_len=3, src_len=4, heads=3, d_k=2
        #   Q from decoder x1_norm    (1, 3, 6)  → heads → (1, 3, 3, 2)
        #   K from encoder_output     (1, 4, 6)  → heads → (1, 3, 4, 2)
        #   V from encoder_output     (1, 4, 6)  → heads → (1, 3, 4, 2)
        #
        #   scores = Q @ K^T          (1, 3, 3, 2) @ (1, 3, 2, 4) = (1, 3, 3, 4)
        #                             ← NOT SQUARE: 3 decoder rows × 4 encoder cols
        #
        #   Each decoder token's row = "how much does this decoder token attend
        #   to each of the 4 source tokens?".  After training, when generating
        #   "ON", the "<bos>→ON" row should peak strongly on "enabled" and
        #   "string" columns, because those tokens carry the type information.
        #
        #   weights after softmax     (1, 3, 3, 4)   [tgt_len × src_len per head]
        #   head_out = weights @ V    (1, 3, 3, 4) @ (1, 3, 4, 2) = (1, 3, 3, 2)
        #   concat + W_O              (1, 3, 6)
        x1_norm = self.layer_norm_2(x1)
        cross_attn_out = self._multi_head_attention(
            Q_src=x1_norm,         # Q from decoder   (1, 3, 6)
            K_src=encoder_output,  # K from encoder   (1, 4, 6)
            V_src=encoder_output,  # V from encoder   (1, 4, 6)
            W_Q=self.cross_W_Q, W_K=self.cross_W_K,
            W_V=self.cross_W_V, W_O=self.cross_W_O,
            attn_mask=src_mask,
        )
        x2 = x1 + cross_attn_out   # (1, 3, 6)
        if verbose:
            print(f"After cross-attn + residual:       x2:       {x2.shape}")

        # ── Sub-layer 3: feed-forward network ────────────────────────
        # Same MLP as in the encoder, applied independently per token position.
        # (1, 3, 6) → (1, 3, 12) → GELU → (1, 3, 6)
        x2_norm = self.layer_norm_3(x2)
        ffn_hidden = self.ffn_linear1(x2_norm)
        ffn_hidden = self.ffn_gelu(ffn_hidden)
        ffn_out    = self.ffn_linear2(ffn_hidden)

        output = x2 + ffn_out   # (1, 3, 6)
        if verbose:
            print(f"After FFN + residual:              output:   {output.shape}")
            print(f"{'='*60}")

        return output

    # ------------------------------------------------------------------
    # KV-cache helpers — used by EncoderDecoderModel.generate() only.
    # ------------------------------------------------------------------

    def precompute_cross_kv(
        self, encoder_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project encoder_output → (K, V) in multi-head format once per generate() call.

        encoder_output : (batch, src_len, d_model)
        returns        : K, V each (batch, heads, src_len, d_k)
        """
        batch_size, src_len, _ = encoder_output.shape
        K = self.cross_W_K(encoder_output).view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.cross_W_V(encoder_output).view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)
        return K, V

    def forward_cached(
        self,
        x: torch.Tensor,
        cross_kv: tuple[torch.Tensor, torch.Tensor],
        self_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step decoder forward for autoregressive inference with KV cache.

        Unlike forward(), processes ONE new token at a time. Cross K/V are
        passed in precomputed; self K/V are accumulated across steps.

        Parameters
        ----------
        x        : (batch, 1, d_model)  — embedding of the single new token
        cross_kv : (K, V) each (batch, heads, src_len, d_k) — constant across steps
        self_kv  : (K, V) each (batch, heads, past_len, d_k), or None at step 0

        Returns
        -------
        output      : (batch, 1, d_model)
        new_self_kv : updated (K, V) with current token appended
        """
        batch_size = x.shape[0]

        # ── Sub-layer 1: self-attention ──────────────────────────────
        x_norm1 = self.layer_norm_1(x)

        Q     = self.self_W_Q(x_norm1).view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)
        new_K = self.self_W_K(x_norm1).view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)
        new_V = self.self_W_V(x_norm1).view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)

        if self_kv is not None:
            full_K = torch.cat([self_kv[0], new_K], dim=2)
            full_V = torch.cat([self_kv[1], new_V], dim=2)
        else:
            full_K, full_V = new_K, new_V

        # No causal mask: Q is always the last position so attending to
        # full_K/V (all past + current) is already causal by construction.
        scores  = torch.matmul(Q, full_K.transpose(-2, -1)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        head_out = torch.matmul(weights, full_V)
        self_attn_out = self.self_W_O(
            head_out.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        )
        x1 = x + self_attn_out

        # ── Sub-layer 2: cross-attention (precomputed K, V) ──────────
        x1_norm = self.layer_norm_2(x1)
        cross_K, cross_V = cross_kv

        Q_c = self.cross_W_Q(x1_norm).view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)
        scores_c  = torch.matmul(Q_c, cross_K.transpose(-2, -1)) / self.scale
        weights_c = torch.softmax(scores_c, dim=-1)
        head_out_c = torch.matmul(weights_c, cross_V)
        cross_attn_out = self.cross_W_O(
            head_out_c.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        )
        x2 = x1 + cross_attn_out

        # ── Sub-layer 3: FFN ─────────────────────────────────────────
        x2_norm = self.layer_norm_3(x2)
        ffn_out = self.ffn_linear2(self.ffn_gelu(self.ffn_linear1(x2_norm)))
        output  = x2 + ffn_out

        return output, (full_K, full_V)


# ══════════════════════════════════════════════════════════════════════
# ENCODER-DECODER MODEL — full seq2seq Transformer
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EncoderDecoderConfig:
    """
    Hyper-parameters describing the SHAPE of the model.

    These six numbers fully determine `state_dict` shapes — they are
    the ONLY thing that needs to be persisted alongside the weights
    for a checkpoint to be reloadable. See `core.checkpoint`.
    """

    vocab_size: int = 1000
    max_seq_len: int = 128
    d_model: int = 256
    num_heads: int = 4
    d_ff: int = 1024
    num_layers: int = 4


class EncoderDecoderModel(nn.Module):
    """
    Complete Encoder-Decoder Transformer.

    Components:
        Shared token embedding      (vocab_size → d_model)
        Shared positional embedding (max_seq_len → d_model)
        Encoder: N × ManualTransformerEncoderBlock (no mask)
        Encoder final LayerNorm
        Decoder: N × ManualDecoderBlock (causal + cross-attention)
        Decoder final LayerNorm
        LM Head: d_model → vocab_size

    Source and target share the same token embedding table — standard
    when both sides use the same subword tokenizer (BPE, SentencePiece).

    The hyper-parameters live in a single `EncoderDecoderConfig` object
    so callers can build identical models from a serialized dict
    (`EncoderDecoderModel(EncoderDecoderConfig(**ckpt["model_config"]))`).
    Also exposed as `EncoderDecoderModel.Config` for ergonomic access.
    """

    Config = EncoderDecoderConfig

    def __init__(self, cfg: EncoderDecoderConfig) -> None:
        super().__init__()

        # Pull each hyper-parameter onto a friendlier short name.
        # We also keep `self.cfg` so external code (checkpointing,
        # logging) can re-serialize the exact configuration that was
        # used to instantiate this module.
        self.cfg         = cfg
        vocab_size       = cfg.vocab_size
        max_seq_len      = cfg.max_seq_len
        d_model          = cfg.d_model
        num_heads        = cfg.num_heads
        d_ff             = cfg.d_ff
        num_layers       = cfg.num_layers

        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model     = d_model
        self.num_heads   = num_heads
        self.num_layers  = num_layers

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        # Shared embeddings (used by both encoder and decoder).
        # → presentation STEP 1: vocab table + token_emb + pos_emb diagram.
        self.token_embedding      = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)

        # Encoder stack: bidirectional self-attention.
        self.encoder_blocks = nn.ModuleList([
            ManualTransformerEncoderBlock(
                d_model=d_model, num_heads=num_heads, d_ff=d_ff,
            )
            for _ in range(num_layers)
        ])
        self.encoder_final_norm = nn.LayerNorm(d_model)

        # Decoder stack: causal self-attention + cross-attention to encoder.
        self.decoder_blocks = nn.ModuleList([
            ManualDecoderBlock(
                d_model=d_model, num_heads=num_heads, d_ff=d_ff,
            )
            for _ in range(num_layers)
        ])
        self.decoder_final_norm = nn.LayerNorm(d_model)

        # Language modelling head: project decoder output to vocab logits.
        # → presentation STEP 4: (tgt_len, d_model) @ (d_model, vocab) projection.
        # [LoRA] Optional target: LM head projection.
        # [LoRA] Usually lower priority than cross-attention unless vocabulary
        # [LoRA] behavior itself must shift strongly.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Causal mask: upper-triangular -inf so position i attends only to 0..i.
    # → presentation STEP 3: 3×3 causal mask visualization with -inf cells.
    # ------------------------------------------------------------------
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Presentation example: tgt_seq=3, decoder input "<bos> ON |"
        #
        #   Added to scores before softmax — -inf cells become 0 after softmax.
        #
        #              <bos>    ON      |
        #   <bos>  [   0.      -inf    -inf ]   ← <bos> may only attend to itself
        #      ON  [   0.       0.     -inf ]   ← ON sees <bos> and ON
        #       |  [   0.       0.      0.  ]   ← | sees full past context
        #
        #   This enforces causality: when predicting token t we must not
        #   "see" tokens t+1, t+2, … which haven't been generated yet.
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1,
        )

    # ------------------------------------------------------------------
    # encode(): src_ids → encoder_output (used for training AND generation)
    # → presentation STEP 1 (embeddings) + STEP 2 (encoder block forward).
    # ------------------------------------------------------------------
    def encode(self, src_ids: torch.Tensor, verbose: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        src_ids : (batch, src_len) integer token IDs

        Returns
        -------
        (batch, src_len, d_model)

        Presentation example  src = "const enabled : string"  (vocab=12, d_model=6):

            src_ids = [[2, 5, 6, 7]]                        shape (1, 4)
                           │
            token_embedding picks rows 2,5,6,7
            from the shared (vocab × d_model) = (12 × 6) table     → (1, 4, 6)

            pos_ids = [0, 1, 2, 3]
            positional_embedding picks rows 0-3
            from the (max_seq × d_model) = (16 × 6) table          → (4, 6)

            enc_x = token_emb + pos_emb  [broadcast over batch]    → (1, 4, 6)
                ┌──────────────────────────────────────────────────┐
                │ row 0: embedding("const")  + pos(0)  = vec_const │
                │ row 1: embedding("enabled")+ pos(1)  = vec_enab  │
                │ row 2: embedding(":")      + pos(2)  = vec_colon │
                │ row 3: embedding("string") + pos(3)  = vec_str   │
                └──────────────────────────────────────────────────┘

            → 2 × EncoderBlock (bidirectional self-attn, NO mask)   (1, 4, 6)
            → encoder_final_norm                                     (1, 4, 6)
        """
        batch_size, src_len = src_ids.shape
        assert src_len <= self.max_seq_len

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# ENCODER FORWARD PASS  | src_ids: {src_ids.shape}")
            print(f"{'#'*60}")

        # Token + positional embedding lookup.
        position_ids = torch.arange(src_len, device=src_ids.device)
        token_emb    = self.token_embedding(src_ids)              # (batch, src_len, d_model)
        pos_emb      = self.positional_embedding(position_ids)    # (src_len, d_model) — broadcast
        x = token_emb + pos_emb                                   # (batch, src_len, d_model)

        # Run through every encoder block — NO mask, so every source token
        # can attend to every other source token (bidirectional).
        for i, block in enumerate(self.encoder_blocks):
            if verbose:
                print(f"\n── Encoder block {i} ──")
            x = block(x, attn_mask=None, verbose=verbose)

        encoder_output = self.encoder_final_norm(x)
        # encoder_output: (batch, src_len, d_model)  e.g. (1, 4, 6)
        # This tensor is the "memory" of the source sentence — it is passed
        # unchanged as K and V to every cross-attention layer of every decoder block.
        return encoder_output

    # ------------------------------------------------------------------
    # decode(): tgt_ids + encoder_output → logits
    # → presentation STEP 3 (decoder block forward) + STEP 4 (LM head).
    # ------------------------------------------------------------------
    def decode(
        self,
        tgt_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tgt_ids        : (batch, tgt_len) — teacher-forced target IDs
        encoder_output : (batch, src_len, d_model)

        Returns
        -------
        logits : (batch, tgt_len, vocab_size)

        Presentation example  tgt_input = "<bos> ON |"  (vocab=12, d_model=6):

            tgt_ids = [[1, 9, 10]]                          shape (1, 3)
                │
            token_embedding picks rows 1,9,10 from (12×6)        → (1, 3, 6)
            pos_emb picks rows 0,1,2 from (16×6)                  → (3, 6)
            dec_x = token_emb + pos_emb                           → (1, 3, 6)
                ┌────────────────────────────────────────────┐
                │ row 0: emb("<bos>") + pos(0) = vec_bos     │
                │ row 1: emb("ON")    + pos(1) = vec_on      │
                │ row 2: emb("|")     + pos(2) = vec_pipe    │
                └────────────────────────────────────────────┘

            tgt_mask: (3, 3)  causal — see _create_causal_mask

            → 2 × DecoderBlock (masked self-attn + cross-attn + FFN)  (1, 3, 6)
            → decoder_final_norm                                        (1, 3, 6)
            → lm_head  (d_model → vocab_size = 6 → 12)                 (1, 3, 12)

            logits[0, 0, :] = 12 scores for predicting after "<bos>"  → target "ON"
            logits[0, 1, :] = 12 scores for predicting after "ON"     → target "|"
            logits[0, 2, :] = 12 scores for predicting after "|"      → target "OFF"
        """
        _, tgt_len = tgt_ids.shape
        assert tgt_len <= self.max_seq_len

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# DECODER FORWARD PASS | tgt_ids: {tgt_ids.shape}  encoder_output: {encoder_output.shape}")
            print(f"{'#'*60}")

        # Token + positional embedding lookup — same tables shared with encoder.
        position_ids = torch.arange(tgt_len, device=tgt_ids.device)
        token_emb    = self.token_embedding(tgt_ids)    # (batch, tgt_len, d_model)
        pos_emb      = self.positional_embedding(position_ids)  # (tgt_len, d_model)
        x = token_emb + pos_emb
        # x: (batch, tgt_len, d_model)  e.g. (1, 3, 6)

        # Causal mask prevents token i from attending to tokens i+1, i+2, …
        tgt_mask = self._create_causal_mask(tgt_len, device=tgt_ids.device)
        # tgt_mask: (tgt_len, tgt_len)  e.g. (3, 3)

        # encoder_output is passed unchanged to every decoder block — same
        # (1, 4, 6) tensor is the K and V source for all cross-attention layers.
        for i, block in enumerate(self.decoder_blocks):
            if verbose:
                print(f"\n── Decoder block {i} ──")
            x = block(x, encoder_output, tgt_mask=tgt_mask, verbose=verbose)

        x = self.decoder_final_norm(x)      # (batch, tgt_len, d_model)
        logits = self.lm_head(x)            # (batch, tgt_len, vocab_size)  e.g. (1, 3, 12)
        return logits

    # ------------------------------------------------------------------
    # forward() — full pass for training (teacher forcing).
    # ------------------------------------------------------------------
    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        encoder_output = self.encode(src_ids, verbose=verbose)
        logits         = self.decode(tgt_ids, encoder_output, verbose=verbose)
        return logits

    # ------------------------------------------------------------------
    # generate() — autoregressive inference. Encoder runs ONCE.
    # → presentation STEP 8: step-by-step decode of "ON | OFF".
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src_ids        : (1, src_len)
        bos_id, eos_id : special token IDs
        max_new_tokens : hard cap on generated length
        temperature    : <1 → more peaked sampling; >1 → more random

        Returns
        -------
        generated_ids : (1, generated_len)  — does NOT include leading <BOS>
        """
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# AUTOREGRESSIVE GENERATION | src_ids: {src_ids.shape}  max_new_tokens: {max_new_tokens}")
            print(f"{'#'*60}")

        # Encoder runs ONCE; encoder_output is reused at every decoder step.
        encoder_output = self.encode(src_ids, verbose=False)

        # Pre-compute cross K/V for every decoder block — encoder_output is
        # constant for the entire generation loop, so this projection is done
        # once instead of once-per-step.
        cross_kv_list = [
            block.precompute_cross_kv(encoder_output) for block in self.decoder_blocks
        ]

        generated = torch.tensor([[bos_id]], device=src_ids.device)
        # generated: (1, 1) starting with <BOS>

        # Self-attention KV cache: one (K, V) entry per decoder layer.
        # None at step 0; grows by one token-slice each step.
        self_kv_list: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_layers

        for step in range(max_new_tokens):
            # Embed only the most-recently appended token.
            new_id = generated[:, -1:]
            pos    = torch.tensor([generated.shape[1] - 1], device=src_ids.device)
            x = self.token_embedding(new_id) + self.positional_embedding(pos)
            # x: (1, 1, d_model)

            for i, block in enumerate(self.decoder_blocks):
                x, self_kv_list[i] = block.forward_cached(x, cross_kv_list[i], self_kv_list[i])

            x      = self.decoder_final_norm(x)
            logits = self.lm_head(x)
            # logits: (1, 1, vocab_size)

            next_token_logits = logits[:, -1, :] / temperature
            probs      = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # next_token: (1, 1)

            generated = torch.cat([generated, next_token], dim=1)

            if verbose:
                print(f"  Step {step:3d}: token {next_token.item():5d}  "
                      f"(prob={probs[0, next_token.item()].item():.4f})  "
                      f"seq_len={generated.shape[1]}")

            if next_token.item() == eos_id:
                if verbose:
                    print(f"  → <EOS> reached, stopping.")
                break

        return generated[:, 1:]  # strip leading <BOS>
