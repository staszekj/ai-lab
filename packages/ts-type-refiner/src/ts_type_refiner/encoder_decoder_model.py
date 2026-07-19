"""
Encoder-Decoder Transformer Model (T5 / BART style)
====================================================================

The canonical, production reference seq2seq Transformer for the
`ts_type_refiner` package. Domain-agnostic: knows nothing about TypeScript,
tokenizers, training loops, checkpoints or validators — it is a pure
nn.Module that maps `(src_ids, tgt_in_ids) → logits` and supports
autoregressive generation.

Companion modules in `ts_type_refiner`:
    - ts_type_refiner.training.trainer   : pure `train(model, batches, …)` function
    - ts_type_refiner.inference.predictor: `Predictor(model, encode, decode, …)` callable
    - ts_type_refiner.checkpoint : save / load / build_model helpers

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
degraded-types corpora on CUDA. It also includes a tiny hard-coded
(`const enabled : string` → `"ON" | "OFF"`) walkthrough reference in comments.
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
# matrix examples using a tiny "micro-model" for reference. Every number in
# the comments can be traced back to one of these names:
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
# ══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# ENCODER BLOCK — bidirectional self-attention + FFN (Pre-LN)
# ══════════════════════════════════════════════════════════════════════

class ManualTransformerEncoderBlock(nn.Module):
    """
    A single Pre-LN Transformer encoder block.

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
    for a checkpoint to be reloadable. See `ts_type_refiner.checkpoint`.
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
        # [LoRA] Optional target: LM head projection.
        # [LoRA] Usually lower priority than cross-attention unless vocabulary
        # [LoRA] behavior itself must shift strongly.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Causal mask: upper-triangular -inf so position i attends only to 0..i.
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

    # ═══════════════════════════════════════════════════════════════════════════════════════
    # ENCODE: src_ids → encoder_output  (BIDIRECTIONAL SELF-ATTENTION - THE MEMORY BUILDER)
    # ═══════════════════════════════════════════════════════════════════════════════════════
    #
    # THE ENCODER'S JOB:
    #   Read the entire source sequence ONCE and compress it into a single "memory" tensor.
    #   This memory will be used by the decoder to understand what type we need.
    #
    # REAL EXAMPLE (from encoder_decoder_pairs.jsonl, row A):
    #   SOURCE:  "const observedElements: Map<unknown, unknown> = new Map();"
    #   STEP 1: Tokenize → [[const, observedElements, :, Map, <, unknown, >, ...]]
    #   STEP 2: Run through encoder with bidirectional self-attention (ALL see ALL)
    #   STEP 3: Output encoder_output = (batch=1, src_len=~12, d_model=256)
    #           Every source token now has rich context representation.
    #
    # THE CORE MECHANISM: MULTI-HEAD SELF-ATTENTION (NO mask = bidirectional)
    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │ INPUT: x = (batch=1, src_len=4, d_model=6)  e.g. "const enabled : string"
    # │                                                                           │
    # │ STEP 1: Project x onto Q, K, V via linear layers                         │
    # │   Q = W_Q(x) → (1, 4, 6)  "What am I looking for?"                      │
    # │   K = W_K(x) → (1, 4, 6)  "What do I have?"                             │
    # │   V = W_V(x) → (1, 4, 6)  "Here's the content"                          │
    # │                                                                           │
    # │ STEP 2: Split into num_heads=3, each with d_k=2                          │
    # │   (1, 4, 6) → reshape → (1, 4, 3, 2) → transpose → (1, 3, 4, 2)         │
    # │   [batch, seq, heads, d_k] → [batch, heads, seq, d_k]                   │
    # │                                                                           │
    # │ STEP 3: Compute attention scores: Q @ K^T / sqrt(d_k)                   │
    # │   scores = (1, 3, 4, 2) @ (1, 3, 2, 4) / sqrt(2) → (1, 3, 4, 4)          │
    # │            Q           K^T                        SQUARE MATRIX          │
    # │                                                                           │
    # │   scores[h, i, j] = attention strength: does token i want to look at j? │
    # │                                                                           │
    # │ STEP 4: Softmax on last dimension (over source positions)               │
    # │   weights = softmax(scores, dim=-1) → (1, 3, 4, 4)                       │
    # │   weights[h, i, :] = probability distribution over 4 source tokens      │
    # │   Example: [0.4, 0.1, 0.05, 0.45] - token i's attention to all src     │
    # │                                                                           │
    # │ STEP 5: Weighted sum of values: context = weights @ V                   │
    # │   (1, 3, 4, 4) @ (1, 3, 4, 2) → (1, 3, 4, 2)                             │
    # │    weights      V              head_output                               │
    # │                                                                           │
    # │   Each token i blends vectors of ALL tokens j, weighted by attention     │
    # │   Token "const" gets strongest signal from "const" itself and "string"  │
    # │                                                                           │
    # │ STEP 6: Concat multi-head outputs + project via W_O                      │
    # │   (1, 3, 4, 2) → concat → (1, 4, 6) → W_O → (1, 4, 6)                    │
    # │   All 3 heads recombined into single d_model dimension                   │
    # │                                                                           │
    # │ STEP 7: Residual connection + LayerNorm + FFN + Residual + LayerNorm   │
    # │   output = x + Attention(LayerNorm(x))                                   │
    # │   output = output + FFN(LayerNorm(output))                               │
    # │   (1, 4, 6) ready for next encoder block                                 │
    # │                                                                           │
    # │ REPEAT: Stack N encoder blocks (N=4 typically) — each refines context   │
    # │         by doing another round of multi-head self-attention             │
    # │                                                                           │
    # │ FINAL: encoder_output = (batch, src_len, d_model)  ← THE MEMORY          │
    # │        Every source token now encodes "what was in the code"            │
    # │        This tensor becomes K,V for decoder cross-attention              │
    # └─────────────────────────────────────────────────────────────────────────┘
    #
    # KEY PROPERTY: BIDIRECTIONAL
    #   • Token i attends to ALL tokens (0..n-1), no causal mask.
    #   • Unlike decoder, encoder is free to look left AND right.
    #   • Each token receives a compressed view of the ENTIRE source sequence.
    #
    # INTUITION - What encoder learns:
    #   - Token "const" learns: "I start a variable declaration with type string"
    #   - Token "enabled" learns: "I am a variable name, with type annotation"
    #   - Token ":" learns: "I separate identifier from type annotation"
    #   - Token "string" learns: "I specify the type of variable 'enabled'"
    #   - Together: "const enabled : string" = a string-typed variable declaration
    #
    # FOR ROW A REAL EXAMPLE:
    #   SOURCE: "const observedElements: Map<unknown, unknown> = new Map();"
    #   - Token "Map" cluster: learns it's a generic container type
    #   - Token "<unknown>" cluster: learns it's a type parameter
    #   - Together encoder_output captures: "Map holds unknown keys and unknown values"
    #   - Decoder will read this memory and predict better type: "Map<Measurable, Data>"
    #
    def encode(self, src_ids: torch.Tensor, verbose: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        src_ids : (batch, src_len) integer token IDs

        Returns
        -------
        (batch, src_len, d_model) — encoder output, the "memory" of source
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

    # ═══════════════════════════════════════════════════════════════════════════════════════
    # DECODE: tgt_ids + encoder_output → logits  (MASKED SELF-ATTN + CROSS-ATTENTION)
    # ═══════════════════════════════════════════════════════════════════════════════════════
    #
    # THE DECODER'S JOB:
    #   Generate a type annotation token-by-token, using TWO attention mechanisms:
    #     1. MASKED SELF-ATTENTION: what have I already generated? (no peeking ahead)
    #     2. CROSS-ATTENTION: what can encoder memory tell me? (peek at source)
    #
    # REAL EXAMPLE (from encoder_decoder_pairs.jsonl, row A):
    #   encoder_output: (1, ~12, 256) from "const observedElements: Map<unknown, ...>"
    #   target (to generate): "Map<Measurable, ObservedData>"
    #   During training: decoder gets teacher-forced target tokens one by one
    #   Output: logits (1, tgt_len, 2048) - probability of each vocab token
    #
    # ┌─────────────────────────────────────────────────────────────────────────────────┐
    # │ MECHANISM 1: MASKED SELF-ATTENTION  (causality: token i sees only tokens 0..i)  │
    # │                                                                                   │
    # │ Purpose: Prevent decoder from cheating by looking at future target tokens       │
    # │          Ensures generation is truly left-to-right (autoregressive)             │
    # │                                                                                   │
    # │ Example: Generating target "Map<Measurable, ObservedData>"                      │
    # │   When predicting token 2 ("<"), decoder has already generated: ["Map", "<"]    │
    # │   Masked self-attention: token "<" attends ONLY to ["Map", "<"]                 │
    # │   Cannot peek at future "Measurable", ",", "ObservedData" (not yet generated)   │
    # │                                                                                   │
    # │ CAUSAL MASK PATTERN (tgt_len=5):                                                │
    # │   Position:  0   1   2   3   4                                                   │
    # │          0  [✓   ✗   ✗   ✗   ✗]   ← token 0 sees only position 0               │
    # │          1  [✓   ✓   ✗   ✗   ✗]   ← token 1 sees positions 0,1                 │
    # │          2  [✓   ✓   ✓   ✗   ✗]   ← token 2 sees positions 0,1,2               │
    # │          3  [✓   ✓   ✓   ✓   ✗]   ← token 3 sees positions 0,1,2,3             │
    # │          4  [✓   ✓   ✓   ✓   ✓]   ← token 4 sees all positions 0-4             │
    # │                                                                                   │
    # │   Implementation: scores + (-inf at ✗) → softmax → [?, ?, ..., 0, 0, ...]       │
    # │   Masked positions become softmax = 0 attention (all weight on unmasked)        │
    # │                                                                                   │
    # │ MATH (Q×K^T → softmax → V pattern, all from decoder):                          │
    # │   Q = W_Q(decoder_hidden) → (batch, tgt_len, d_model)                           │
    # │   K = W_K(decoder_hidden) → (batch, tgt_len, d_model)                           │
    # │   V = W_V(decoder_hidden) → (batch, tgt_len, d_model)                           │
    # │                                                                                   │
    # │   scores = Q @ K^T / sqrt(d_k) → (batch, num_heads, tgt_len, tgt_len)           │
    # │            Q        K^T                        SQUARE MATRIX                     │
    # │                                                                                   │
    # │   scores = scores + causal_mask (mask is 0 or -inf)                             │
    # │   weights = softmax(scores, dim=-1)  → -inf positions collapse to 0             │
    # │   output = weights @ V                                                           │
    # │                                                                                   │
    # │ INTUITION:                                                                       │
    # │   Forces decoder to generate left-to-right without "cheating"                   │
    # │   Each token learns from history ("what have I seen so far?")                   │
    # │   But cannot learn from future ("what's coming next?")                          │
    # └─────────────────────────────────────────────────────────────────────────────────┘
    #
    # ┌─────────────────────────────────────────────────────────────────────────────────┐
    # │ MECHANISM 2: CROSS-ATTENTION  (Q from decoder, K/V from encoder_output)         │
    # │                                                                                   │
    # │ Purpose: Allow decoder to query encoder memory                                   │
    # │          Decoder asks "What type information is in the source code?"            │
    # │          Encoder memory answers with relevant features                           │
    # │                                                                                   │
    # │ Example: Generating "Map<Measurable, ...>"                                      │
    # │   Decoder token "Map" asks encoder: "What container info do you have?"          │
    # │   Encoder peaks on its "Map" token: "The source has a Map container!"           │
    # │   Result: Decoder learns to output "Map" (matching encoder's observation)       │
    # │                                                                                   │
    # │   Decoder token "Measurable" asks: "What should go inside this container?"      │
    # │   Encoder peaks on "<unknown>": "I saw 'unknown' type in source"                │
    # │   Result: Decoder predicts concrete type to replace 'unknown'                   │
    # │                                                                                   │
    # │ SHAPES (with src_len=12, tgt_len=5, d_model=256, num_heads=8, d_k=32):          │
    # │   encoder_output: (batch=1, src_len=12, d_model=256) ← "the memory"             │
    # │   decoder hidden: (batch=1, tgt_len=5, d_model=256)  ← "the generator"          │
    # │                                                                                   │
    # │   Q = W_Q(decoder_hidden) → (1, 5, 256)   ← what decoder needs                  │
    # │   K = W_K(encoder_output) → (1, 12, 256)  ← what encoder offers                 │
    # │   V = W_V(encoder_output) → (1, 12, 256)  ← actual content from encoder         │
    # │                                                                                   │
    # │ ATTENTION COMPUTATION (split into heads):                                        │
    # │   Q: (1, 5, 256) → reshape → (1, 8, 5, 32)   [batch, heads, tgt_seq, d_k]       │
    # │   K: (1, 12, 256) → reshape → (1, 8, 12, 32) [batch, heads, src_seq, d_k]       │
    # │   V: (1, 12, 256) → reshape → (1, 8, 12, 32) [batch, heads, src_seq, d_k]       │
    # │                                                                                   │
    # │   scores = Q @ K^T / sqrt(32)                                                    │
    # │           (1,8,5,32) @ (1,8,32,12) → (1, 8, 5, 12)  ← RECTANGULAR!             │
    # │           ^5 decoder    ^12 source       5×12 scores                             │
    # │                                                                                   │
    # │   This is DIFFERENT from self-attention (which is square).                      │
    # │   Here: 5 decoder positions, each attends to 12 source positions.               │
    # │                                                                                   │
    # │   weights = softmax(scores, dim=-1) → (1, 8, 5, 12)                              │
    # │   Each decoder position has a probability distribution over ALL source positions│
    # │                                                                                   │
    # │   Example (head 0, decoder position 0 - predicting "Map"):                       │
    # │   weights[0, 0, 0, :] = [0.01, 0.02, 0.15, 0.05, 0.35, 0.12, 0.08, ...]        │
    # │                           ↑ const ↑ enable ↑ Map  ↑ unknown ↑ <      ...        │
    # │   Decoder peaks on encoder position 4 ("Map") - attention = 0.35                 │
    # │   (Reasonable: encoder has a "Map" token, decoder should probably output "Map")  │
    # │                                                                                   │
    # │   context = weights @ V                                                          │
    # │            (1,8,5,12) @ (1,8,12,32) → (1, 8, 5, 32)                              │
    # │   Each decoder position blends all 12 encoder value vectors,                     │
    # │   weighted by attention. Result: rich context vector for each decoder position. │
    # │                                                                                   │
    # │   concat + W_O: (1, 8, 5, 32) → concat → (1, 5, 256)                             │
    # │   Multi-head outputs recombined back to d_model.                                │
    # │                                                                                   │
    # │ INTUITION:                                                                       │
    # │   Decoder grounding: predictions are rooted in what encoder observed.           │
    # │   Enables "copying" from source: if "Map<unknown>" in source,                   │
    # │   decoder learns to output something like "Map<Concrete>"                        │
    # │   Cross-attention is the BRIDGE between encoder memory and decoder generation   │
    # │                                                                                   │
    # │ FOR ROW A:                                                                       │
    # │   Source: "const observedElements: Map<unknown, unknown> = new Map();"           │
    # │   Target: "Map<Measurable, ObservedData>"                                        │
    # │                                                                                   │
    # │   Decoder "Map" → cross-attention peaks on encoder "Map" → outputs "Map"        │
    # │   Decoder "<" → cross-attention peaks on encoder "<" → outputs "<"               │
    # │   Decoder "Measurable" → cross-attention peaks on "unknown" → learns to         │
    # │     replace "unknown" with a concrete type like "Measurable"                    │
    # │   Decoder ",ObservedData" → similar pattern, learns second type parameter       │
    # └─────────────────────────────────────────────────────────────────────────────────┘
    #
    # DECODER BLOCK ARCHITECTURE (Pre-LN with 3 sub-layers):
    # ┌─────────────────────────────────────────────────────────────────────────────────┐
    # │ Input: x = (batch, tgt_len, d_model)                                             │
    # │                                                                                   │
    # │ SUB-LAYER 1: Masked Self-Attention                                              │
    # │   x_norm = LayerNorm(x)                                                          │
    # │   attn = MultiHeadAttention(                                                     │
    # │     Q, K, V all from x_norm,                                                    │
    # │     attn_mask = causal_mask  ← key: forces causality                            │
    # │   )                                                                              │
    # │   x = x + attn  ← residual                                                       │
    # │                                                                                   │
    # │ SUB-LAYER 2: Cross-Attention to Encoder                                         │
    # │   x_norm = LayerNorm(x)                                                          │
    # │   attn = MultiHeadAttention(                                                     │
    # │     Q = x_norm (from decoder),                                                  │
    # │     K, V = encoder_output (from encoder),  ← KEY: different sources!            │
    # │     attn_mask = None  ← no mask, decoder can attend to all source positions    │
    # │   )                                                                              │
    # │   x = x + attn  ← residual                                                       │
    # │                                                                                   │
    # │ SUB-LAYER 3: Feed-Forward Network                                               │
    # │   x_norm = LayerNorm(x)                                                          │
    # │   ffn = Linear(d_ff)(GELU(Linear(d_model)(x_norm)))                              │
    # │   x = x + ffn  ← residual                                                        │
    # │                                                                                   │
    # │ Output: x = (batch, tgt_len, d_model)  ← ready for next decoder block or head   │
    # └─────────────────────────────────────────────────────────────────────────────────┘
    #
    # FULL DECODER FORWARD PASS:
    #   1. Embed target tokens + position information
    #      →  (batch, tgt_len, d_model)
    #
    #   2. Run through N decoder blocks (N=4 typically)
    #      Each block: masked self-attention + cross-attention + FFN
    #      →  (batch, tgt_len, d_model) after each block
    #
    #   3. Apply final LayerNorm
    #      →  (batch, tgt_len, d_model)
    #
    #   4. Project via LM head (d_model → vocab_size)
    #      →  logits (batch, tgt_len, vocab_size=2048)
    #         One probability distribution per target position
    #
    #   5. Return logits to be compared with ground-truth targets via cross-entropy loss
    #
    # KEY INSIGHT:
    #   Without cross-attention, decoder would generate randomly, ignoring source.
    #   WITH cross-attention, decoder learns to ground type predictions in source code.
    #   Example: Seeing "Map<unknown>" in source makes decoder predict "Map<Specific>"
    #   Example: Seeing "ObservedData" in source makes decoder predict similar types
    #
    # CROSS-ATTENTION IS THE GLUE that makes encoder-decoder work as a unified system!
    #
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

    # ═══════════════════════════════════════════════════════════════════════════════════════
    # FORWARD: src_ids + tgt_ids → logits  (TRAINING-TIME: encoder-decoder with teacher forcing)
    # ═══════════════════════════════════════════════════════════════════════════════════════
    #
    # THE FORWARD PASS is the complete training pipeline:
    #   1. Encode source to memory (bidirectional self-attention)
    #   2. Decode using memory + teacher-forced target (masked self-attention + cross-attention)
    #   3. Return logits to be compared with ground truth via loss function
    #
    # REAL EXAMPLE (from encoder_decoder_pairs.jsonl, row A):
    # ┌─────────────────────────────────────────────────────────────────────────────────┐
    # │ TRAINING BATCH:                                                                  │
    # │                                                                                   │
    # │ source_code = "const observedElements: Map<unknown, unknown> = new Map();"       │
    # │ target_type = "Map<Measurable, ObservedData>"                                    │
    # │                                                                                   │
    # │ TOKENIZATION:                                                                    │
    # │ src_ids = [[const, observedElements, :, Map, <, unknown, >, comma, ...]]        │
    # │           Tokenized indices (batch=1, src_len≈12)                                │
    # │                                                                                   │
    # │ tgt_ids = [[Map, <, Measurable, comma, ObservedData, >]]                         │
    # │           Ground-truth type tokens (batch=1, tgt_len≈6)                          │
    # │                                                                                   │
    # │ TEACHER FORCING:                                                                 │
    # │ Decoder receives shifted target (prepend <BOS>, remove last token):              │
    # │ decoder_input_ids = [[<BOS>, Map, <, Measurable, comma, ObservedData]]          │
    # │ decoder_target_ids = [[Map, <, Measurable, comma, ObservedData, >]]             │
    # │                                                                                   │
    # │ TRAINING STEP (this forward() call):                                             │
    # │ logits = model.forward(src_ids, tgt_ids)                                         │
    # │ logits.shape = (batch=1, tgt_len=6, vocab_size=2048)                             │
    # │ loss = cross_entropy(logits, tgt_ids)  ← only on real tokens, not padding       │
    # │ loss.backward() → update model weights                                           │
    # └─────────────────────────────────────────────────────────────────────────────────┘
    #
    # WHAT IS TEACHER FORCING?
    # ┌─────────────────────────────────────────────────────────────────────────────────┐
    # │ DURING TRAINING: Decoder receives TRUE target tokens from the batch             │
    # │ (not its own predictions - which might be wrong early in training).             │
    # │                                                                                   │
    # │ EXAMPLE - Generating "Map<Measurable, ObservedData>" during training:            │
    # │                                                                                   │
    # │ AUTOREGRESSIVE (inference, generate() method):                                   │
    # │   Step 0: Model predicts P(token | <BOS>) → sample "Map"                         │
    # │   Step 1: Model predicts P(token | <BOS>, "Map") → sample "<"                    │
    # │   Step 2: Model predicts P(token | <BOS>, "Map", "<") → sample "Measurable"     │
    # │   ... (each step uses model's own previous prediction)                           │
    # │                                                                                   │
    # │ TEACHER FORCING (training, this forward() method):                               │
    # │   Step 0: Model sees input "<BOS>", must predict "Map"                           │
    # │   Step 1: Model sees input "<BOS>, Map", must predict "<"                        │
    # │   Step 2: Model sees input "<BOS>, Map, <", must predict "Measurable"           │
    # │   ... (each step uses GROUND TRUTH previous token, not model's guess)            │
    # │                                                                                   │
    # │ WHY TEACHER FORCING?                                                              │
    # │   • Provides strong supervision signal (model always sees correct input)         │
    # │   • Speeds up training (no compounding errors in early training)                 │
    # │   • Makes loss more stable and meaningful                                        │
    # │   • Model doesn't waste time "recovering" from its own early mistakes           │
    # │                                                                                   │
    # │ TRADE-OFF (distribution mismatch):                                               │
    # │   • During training: decoder sees perfect input from batch                       │
    # │   • During inference: decoder sees its own (imperfect) predictions               │
    # │   • Solution: some training recipes do "scheduled sampling" (blend the two)     │
    # │     But standard transformers just use teacher forcing (works well enough)      │
    # └─────────────────────────────────────────────────────────────────────────────────┘
    #
    # ATTENTION FLOW IN FORWARD PASS
    # ┌─────────────────────────────────────────────────────────────────────────────────┐
    # │ PHASE 1: ENCODING (bidirectional self-attention, full source context)            │
    # │                                                                                   │
    # │ Source: "const observedElements: Map<unknown, unknown> = new Map();"             │
    # │                                                                                   │
    # │ ENCODER BLOCK 1:                                                                 │
    # │   "const" ↔ attends to all: "observedElements", ":", "Map", "<unknown>", ...    │
    # │   "observedElements" ↔ attends to all: "const", ":", "Map", "<unknown>", ...    │
    # │   "Map" ↔ attends to all: learns it's used in type position                     │
    # │   "<unknown>" ↔ attends to all: appears after "Map<"                             │
    # │   ... (each token attends to EVERY other token)                                  │
    # │                                                                                   │
    # │ ENCODER BLOCK 2 (refines layer 1 representations):                               │
    # │   Same bidirectional pattern, but working with richer layer-1 embeddings         │
    # │   Result: encoder_output = (1, src_len, 256) ← rich contextual memory            │
    # │                                                                                   │
    # │ KEY INSIGHT: BIDIRECTIONAL ENCODING                                              │
    # │   • Every source token sees the full picture                                     │
    # │   • "Map" learns about types because it sees "const ... : Map<..."               │
    # │   • "<unknown>" learns it's a type parameter                                     │
    # │   • Final encoder_output is a compressed representation of "what's in the code" │
    # │                                                                                   │
    # │ PHASE 2: DECODING (masked self-attention + cross-attention to encoder)           │
    # │                                                                                   │
    # │ Target (teacher-forced): "<BOS>, Map, <, Measurable, comma, ObservedData"       │
    # │ decoder_output goal: predict "Map, <, Measurable, comma, ObservedData, >"       │
    # │                                                                                   │
    # │ DECODER BLOCK 1:                                                                 │
    # │                                                                                   │
    # │   SUB-LAYER 1A: Masked Self-Attention                                            │
    # │   Position 0 (token "<BOS>"):                                                    │
    # │     Masked self-attention: attends only to position 0 (itself)                   │
    # │     Learns context from just "<BOS>" (start-of-sequence signal)                  │
    # │                                                                                   │
    # │   Position 1 (token "Map"):                                                      │
    # │     Masked self-attention: attends to positions 0,1 ("<BOS>", "Map")             │
    # │     Learns "I come after start-of-sequence"                                      │
    # │     (Cannot peek at future "< Measurable ..." - they're masked)                  │
    # │                                                                                   │
    # │   Position 2 (token "<"):                                                        │
    # │     Masked self-attention: attends to positions 0,1,2 ("<BOS>", "Map", "<")      │
    # │     Learns "I come after Map in the type annotation"                             │
    # │                                                                                   │
    # │   SUB-LAYER 1B: Cross-Attention (Q from decoder, K,V from encoder)               │
    # │                                                                                   │
    # │   Position 0 (token "<BOS>"):                                                    │
    # │     Q (query): "What type info do I need?" - from "<BOS>" representation         │
    # │     Queries ENTIRE encoder memory:                                               │
    # │       encoder has: "const", "observedElements", ":", "Map", "<unknown>", ...    │
    # │     Attention peaks on "Map" (highest cross-attention weight)                    │
    # │     Learns: "The source has a Map type container"                                │
    # │                                                                                   │
    # │   Position 1 (token "Map"):                                                      │
    # │     Q (query): Stronger signal "I need type container information"               │
    # │     Queries encoder memory: peaks even more on encoder "Map"                     │
    # │     Cross-attention learns: "Output 'Map' because source has 'Map'"              │
    # │                                                                                   │
    # │   Position 2 (token "<"):                                                        │
    # │     Q (query): "What comes inside the container?"                                │
    # │     Queries encoder memory: peaks on "<unknown>"                                 │
    # │     Learns: "Source has type parameters, I should output '<'"                    │
    # │                                                                                   │
    # │   Position 3 (token "Measurable"):                                               │
    # │     Q (query): "What specific type should replace unknown?"                      │
    # │     Queries encoder memory: "<unknown>" token is strong signal                   │
    # │     Learns: "Source has unknown, I should output something more concrete"        │
    # │                                                                                   │
    # │   SUB-LAYER 1C: Feed-Forward                                                     │
    # │     FFN: Per-token transformation (d_model → d_ff → d_model)                     │
    # │     No cross-sequence interaction, just enriches each token's representation     │
    # │                                                                                   │
    # │ DECODER BLOCK 2 (refines layer 1):                                               │
    # │   Same 3-layer structure, working with richer layer-1 representations            │
    # │   Result: decoder output = (1, tgt_len, 256) ← refined predictions              │
    # │                                                                                   │
    # │ LOGITS HEAD:                                                                     │
    # │   Linear projection: (d_model=256) → (vocab_size=2048)                           │
    # │   logits = (1, tgt_len=6, vocab_size=2048)                                       │
    # │   logits[b, t, v] = raw score for token v at position t in batch b               │
    # │                                                                                   │
    # │ LOSS COMPUTATION:                                                                │
    # │   loss = cross_entropy(logits, tgt_ids, ignore_index=PAD_ID)                     │
    # │                                                                                   │
    # │   Example at position 1:                                                         │
    # │     Model outputs logits for vocab (2048 values)                                 │
    # │     Ground truth: tgt_ids[1] = MAP_TOKEN_ID                                      │
    # │     Loss includes: -log(softmax(logits[MAP_TOKEN_ID]))                           │
    # │     If logits[MAP_TOKEN_ID] is high: low loss (good!)                            │
    # │     If logits[MAP_TOKEN_ID] is low: high loss (bad, gradient flows back)        │
    # │                                                                                   │
    # │ GRADIENT FLOW:                                                                   │
    # │   loss.backward() → gradients flow back through:                                 │
    # │     logits → lm_head → decoder blocks → encoder blocks → embeddings              │
    # │   Every parameter adjusted to make loss smaller next step                        │
    # └─────────────────────────────────────────────────────────────────────────────────┘
    #
    # COMPLETE FORWARD PASS DIAGRAM
    # ┌─────────────────────────────────────────────────────────────────────────────────┐
    # │                                                                                   │
    # │ INPUTS:                                                                          │
    # │   src_ids: (1, src_len≈12)  - tokenized source code                              │
    # │   tgt_ids: (1, tgt_len≈6)   - tokenized target type (ground truth)               │
    # │                                                                                   │
    # │                              ↓                                                    │
    # │                                                                                   │
    # │ STEP 1 - ENCODE (this function calls self.encode):                               │
    # │   ┌─────────────────────────────────────────────────────────────┐                │
    # │   │ Embed: (1, 12) → (1, 12, 256)                               │                │
    # │   │ Encoder Block 1: bidirectional self-attention               │                │
    # │   │ Encoder Block 2: bidirectional self-attention               │                │
    # │   │ Output: encoder_output (1, 12, 256) ← THE MEMORY             │                │
    # │   └─────────────────────────────────────────────────────────────┘                │
    # │                              ↓                                                    │
    # │                                                                                   │
    # │ STEP 2 - DECODE (this function calls self.decode):                               │
    # │   ┌─────────────────────────────────────────────────────────────┐                │
    # │   │ Embed: (1, 6) → (1, 6, 256)                                 │                │
    # │   │ Decoder Block 1:                                             │                │
    # │   │   Sub-1a: Masked self-attention (causality)                 │                │
    # │   │   Sub-1b: Cross-attention (Q from decoder, K,V from encoder) │                │
    # │   │   Sub-1c: Feed-forward                                       │                │
    # │   │ Decoder Block 2: (same 3 sub-layers)                         │                │
    # │   │ Output: decoder_out (1, 6, 256)                              │                │
    # │   │ LM Head: (1, 6, 256) → (1, 6, 2048)                          │                │
    # │   └─────────────────────────────────────────────────────────────┘                │
    # │                              ↓                                                    │
    # │                                                                                   │
    # │ OUTPUTS:                                                                         │
    # │   logits: (1, 6, 2048) - probability of each vocab token                         │
    # │                                                                                   │
    # │   During training:                                                               │
    # │     loss = cross_entropy(logits, tgt_ids)                                        │
    # │     loss.backward()                                                              │
    # │     optimizer.step()                                                             │
    # │                                                                                   │
    # │   During inference:                                                              │
    # │     Use generate() instead (autoregressive token-by-token)                       │
    # │                                                                                   │
    # └─────────────────────────────────────────────────────────────────────────────────┘
    #
    # KEY LEARNING OBJECTIVES IN THIS FORWARD PASS:
    #
    # Encoder learns (through gradients from loss):
    #   • "How to compress source code into semantically meaningful representations"
    #   • "What patterns in source code matter for type inference"
    #   • "How to highlight relevant tokens (e.g., Map, <unknown>)"
    #
    # Decoder learns (through gradients from loss):
    #   • "How to generate type annotations that match source code patterns"
    #   • "When to reuse tokens from source (Map) vs. generate new ones (Measurable)"
    #   • "How to use encoder memory to ground predictions"
    #
    # Cross-attention specifically learns:
    #   • "Which encoder tokens are relevant to each decoder position"
    #   • "How to blend information from multiple source tokens"
    #   • "How to 'copy' concepts from source (Map→Map) vs. 'refine' them (unknown→Measurable)"
    #
    # THE MAGIC OF ATTENTION:
    #   All learning happens through weighted combinations (attention) and gradients.
    #   No explicit "copy" or "generate" rules - the model learns implicitly.
    #   This is why attention is the CORE of modern transformers.
    #
    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src_ids : (batch, src_len)
            Source side: TypeScript code with degraded type annotation.
        tgt_ids : (batch, tgt_len)
            Target side: precise type we want decoder to output (teacher forcing).
        verbose : bool
            If True, print per-block shapes and intermediate tensor stats.

        Returns
        -------
        logits : (batch, tgt_len, vocab_size)
            Raw model output — one logits vector per position.
        """
        encoder_output = self.encode(src_ids, verbose=verbose)
        logits         = self.decode(tgt_ids, encoder_output, verbose=verbose)
        return logits

    # ------------------------------------------------------------------
    # generate() — autoregressive inference. Encoder runs ONCE.
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

        # ════════════════════════════════════════════════════════════════════════
        # AUTOREGRESSIVE GENERATION LOOP — one token produced per step
        # ════════════════════════════════════════════════════════════════════════
        #
        # WHAT IS THE ENCODER?
        #   The encoder already ran above (self.encode). It read the full source
        #   prompt ONCE — e.g. "const enabled : string" — and compressed it into
        #   encoder_output: a (1, 4, 6) tensor, one vector per source token.
        #   Think of it as the model's "memory" of what it read.
        #   The encoder does NOT run again during generation.
        #
        # WHAT IS THE DECODER?
        #   The decoder generates the output token by token, left to right.
        #   At each step it looks at:
        #     1. All tokens it has generated so far (via masked self-attention)
        #     2. The encoder's memory (via cross-attention)
        #   ...and predicts: "what token comes next?"
        #
        # Example: source = "const enabled : string"  →  target = "'ON' | 'OFF'"
        #
        #   Step 0: generated = [<BOS>]
        #           decoder sees <BOS> + encoder memory
        #           → predicts "'ON'"   (logits[:, -1, :] peaks at token 9)
        #           → generated = [<BOS>, 'ON']
        #
        #   Step 1: generated = [<BOS>, 'ON']
        #           decoder sees 'ON' (new token) + encoder memory
        #           → predicts "|"    (logits[:, -1, :] peaks at token 10)
        #           → generated = [<BOS>, 'ON', '|']
        #
        #   Step 2: decoder predicts "'OFF'" → generated = [<BOS>, 'ON', '|', 'OFF']
        #   Step 3: decoder predicts <EOS> → loop stops
        #           return [9, 10, 11] = "'ON' | 'OFF'" (strip leading <BOS>)
        #
        # KEY INSIGHT: encoder runs ONCE, decoder runs N times (once per output token).
        # Cross-attention connects them: at every step, every decoder layer asks
        # "what from the encoder memory is relevant now?"
        # ════════════════════════════════════════════════════════════════════════

        for step in range(max_new_tokens):
            # Take only the LAST generated token — we embed one token at a time.
            # The KV cache (self_kv_list) already holds all past context.
            # Example step 1: generated = [[1, 9]], new_id = [[9]] (token "'ON'")
            new_id = generated[:, -1:]
            pos    = torch.tensor([generated.shape[1] - 1], device=src_ids.device)

            # token_embedding: vocab table (vocab_size × d_model), picks row new_id[0]
            # positional_embedding: position table (max_seq × d_model), picks row pos
            # Example: emb("'ON'") + pos_emb(1) = vec of shape (1, 1, 6)
            x = self.token_embedding(new_id) + self.positional_embedding(pos)
            # x: (1, 1, d_model) — a single token's representation

            # Run through all decoder blocks. Each block (forward_cached) does:
            #   1. Masked self-attention: new token attends to all past tokens (KV cache)
            #   2. Cross-attention: new token attends to ALL encoder positions (precomputed K/V)
            #   3. FFN: per-token MLP
            # self_kv_list grows by 1 token slice each step — this IS the KV cache.
            for i, block in enumerate(self.decoder_blocks):
                x, self_kv_list[i] = block.forward_cached(x, cross_kv_list[i], self_kv_list[i])

            x      = self.decoder_final_norm(x)

            # LM head: project from d_model → vocab_size.
            # (1, 1, 6) → (1, 1, 12) — 12 raw scores, one per vocab token = LOGITS
            logits = self.lm_head(x)
            # logits: (1, 1, vocab_size)

            # Divide by temperature: < 1 → more peaked, > 1 → more random
            # Example: logits[:, -1, :] = [-0.5, 8.2, -1.3, ...]  (12 values)
            next_token_logits = logits[:, -1, :] / temperature

            # Softmax → probability distribution over vocab (all values sum to 1).
            # Example: probs = [0.001, 0.93, 0.002, ...]  ← peaks at token 9 ("'ON'")
            probs      = torch.softmax(next_token_logits, dim=-1)

            # Sample one token index. With temperature=0.01 this is nearly greedy.
            # Example: samples token 9 with probability 0.93 → "'ON'"
            next_token = torch.multinomial(probs, num_samples=1)
            # next_token: (1, 1) — integer token id

            # Append to the growing sequence.
            # Example: [[1, 9]] cat [[10]] → [[1, 9, 10]] i.e. "<BOS> 'ON' |"
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
