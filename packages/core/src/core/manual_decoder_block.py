"""
Educational PyTorch: Transformer Decoder Block with Cross-Attention
====================================================================

This file implements a single Transformer Decoder Block — the building
block of seq2seq (encoder-decoder) models like T5, BART, and the
original 2017 Transformer ("Attention Is All You Need").

A decoder block has THREE sub-layers (vs TWO in the encoder block):

    1. Masked Self-Attention
       Queries, Keys, Values all come from the decoder input.
       Uses a CAUSAL MASK — each target token can only attend to
       itself and previous target tokens (not future tokens).
       This is identical to GPT's attention.

    2. Cross-Attention  ← THE NEW PART
       Queries come from the decoder (after sub-layer 1).
       Keys and Values come from the ENCODER OUTPUT.
       This is how the decoder "reads" the source sequence.
       No mask — the decoder can attend to all encoder positions.

    3. Feed-Forward Network (FFN)
       Applied independently to each token position.
       Identical to the encoder's FFN.

Architecture (Pre-LN):
    x1     = x  + MaskedSelfAttention(LayerNorm(x),   mask=causal)
    x2     = x1 + CrossAttention(LayerNorm(x1), encoder_output)
    output = x2 + FFN(LayerNorm(x2))

Shapes:
    x              : (batch, tgt_len, d_model)  — decoder input
    encoder_output : (batch, src_len, d_model)  — from encoder
    output         : (batch, tgt_len, d_model)  — decoder output

Note: tgt_len and src_len CAN be different.

Gradient flow (backward):
    dL/dself_W_Q,K,V  → only decoder parameters
    dL/dcross_W_Q     → only decoder parameters
    dL/dcross_W_K,V   → decoder parameters + flows into encoder_output
                        → encoder weights receive gradients from decoder loss!
    dL/dffn_weights   → only decoder parameters
"""

import math
import torch
import torch.nn as nn


class ManualDecoderBlock(nn.Module):
    """
    A single Pre-LN Transformer Decoder Block with:
      - Masked Multi-Head Self-Attention  (causal, tgt↔tgt)
      - Multi-Head Cross-Attention        (Q from tgt, K/V from encoder)
      - Feed-Forward Network

    Hyperparameters
    ---------------
    d_model   = 768   – width of the model (embedding dimension)
    num_heads = 12    – number of parallel attention heads
    d_k       = 64    – dimension of each head  (d_model / num_heads)
    d_ff      = 3072  – hidden size of the feed-forward network

    Input / output shape: (batch_size, tgt_len, d_model)
    Cross-attention:     encoder_output shape (batch_size, src_len, d_model)
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

        # ── Sub-layer 1: Masked Self-Attention ────────────────────────
        # Q, K, V all come from the decoder input.
        # The causal mask is passed in at forward-time (tgt_mask).
        self.self_W_Q = nn.Linear(d_model, d_model)
        self.self_W_K = nn.Linear(d_model, d_model)
        self.self_W_V = nn.Linear(d_model, d_model)
        self.self_W_O = nn.Linear(d_model, d_model)

        # ── Sub-layer 2: Cross-Attention ───────────────────────────────
        # Q comes from the decoder (after self-attention + residual).
        # K and V come from encoder_output.
        # These are SEPARATE weight matrices — they learn different
        # transformations than the self-attention projections.
        self.cross_W_Q = nn.Linear(d_model, d_model)  # Q from decoder
        self.cross_W_K = nn.Linear(d_model, d_model)  # K from encoder
        self.cross_W_V = nn.Linear(d_model, d_model)  # V from encoder
        self.cross_W_O = nn.Linear(d_model, d_model)

        # ── Sub-layer 3: Feed-Forward Network ─────────────────────────
        self.ffn_linear1 = nn.Linear(d_model, d_ff)
        self.ffn_gelu    = nn.GELU()
        self.ffn_linear2 = nn.Linear(d_ff, d_model)

        # ── Three Layer Norms (Pre-LN: one before each sub-layer) ──────
        self.layer_norm_1 = nn.LayerNorm(d_model)  # before masked self-attention
        self.layer_norm_2 = nn.LayerNorm(d_model)  # before cross-attention
        self.layer_norm_3 = nn.LayerNorm(d_model)  # before FFN

    # ------------------------------------------------------------------
    # Helper: multi-head attention
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
        attn_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """
        Generic multi-head attention used by both self- and cross-attention.

        For self-attention:   Q_src = K_src = V_src = x (same source)
        For cross-attention:  Q_src = x (decoder), K_src = V_src = encoder_output

        Parameters
        ----------
        Q_src     : (batch, q_len,  d_model) — source of queries
        K_src     : (batch, kv_len, d_model) — source of keys
        V_src     : (batch, kv_len, d_model) — source of values
        attn_mask : (q_len, kv_len) or None — additive mask (0=attend, -inf=block)

        Returns
        -------
        output : (batch, q_len, d_model)
        """
        batch_size = Q_src.shape[0]
        q_len      = Q_src.shape[1]
        kv_len     = K_src.shape[1]

        # ── Project to Q, K, V ────────────────────────────────────────
        Q = W_Q(Q_src)   # (batch, q_len,  d_model)
        K = W_K(K_src)   # (batch, kv_len, d_model)
        V = W_V(V_src)   # (batch, kv_len, d_model)

        # ── Split into heads ──────────────────────────────────────────
        # Reshape: (batch, len, d_model) → (batch, heads, len, d_k)
        Q = Q.view(batch_size, q_len,  self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, kv_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, kv_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q: (batch, heads, q_len,  d_k)
        # K: (batch, heads, kv_len, d_k)
        # V: (batch, heads, kv_len, d_k)

        # ── Scaled dot-product attention ──────────────────────────────
        # scores shape: (batch, heads, q_len, kv_len)
        # For cross-attention q_len ≠ kv_len — score matrix is NOT square.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (batch, heads, q_len, kv_len)

        if attn_mask is not None:
            scores = scores + attn_mask
            # Positions where mask=-inf → softmax gives 0.0 (attended nowhere)

        weights = torch.softmax(scores, dim=-1)
        # weights: (batch, heads, q_len, kv_len)

        # ── Weighted sum of values ────────────────────────────────────
        head_out = torch.matmul(weights, V)
        # head_out: (batch, heads, q_len, d_k)

        # ── Concatenate heads + output projection ─────────────────────
        out = head_out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        # out: (batch, q_len, d_model)

        return W_O(out)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: "torch.Tensor | None" = None,
        src_mask: "torch.Tensor | None" = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x              : (batch, tgt_len, d_model) — decoder token embeddings
        encoder_output : (batch, src_len, d_model) — output of the full encoder stack
        tgt_mask       : (tgt_len, tgt_len) or None — causal mask for self-attention
        src_mask       : (tgt_len, src_len) or None — optional padding mask for cross-attention

        Returns
        -------
        output : (batch, tgt_len, d_model)
        """
        batch_size, tgt_len, d_model = x.shape
        _, src_len, _ = encoder_output.shape
        assert d_model == self.d_model, (
            f"Input d_model ({d_model}) doesn't match expected ({self.d_model})"
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"DecoderBlock | x: {x.shape}  encoder_output: {encoder_output.shape}")

        # ==============================================================
        # STEP 5.1 — Masked Self-Attention
        # ==============================================================
        # Decoder tokens attend to each other, but ONLY to past positions.
        # Q, K, V all come from x (self-attention).
        # The causal mask (tgt_mask) blocks future target positions.
        x_norm1 = self.layer_norm_1(x)
        # x_norm1: (batch, tgt_len, d_model)
        if verbose:
            print(f"After LayerNorm 1:                x_norm1:     {x_norm1.shape}")

        self_attn_out = self._multi_head_attention(
            Q_src=x_norm1, K_src=x_norm1, V_src=x_norm1,
            W_Q=self.self_W_Q, W_K=self.self_W_K,
            W_V=self.self_W_V, W_O=self.self_W_O,
            attn_mask=tgt_mask,
        )
        # self_attn_out: (batch, tgt_len, d_model)
        if verbose:
            print(f"Masked self-attention output:     self_attn:   {self_attn_out.shape}")

        x1 = x + self_attn_out
        # x1: (batch, tgt_len, d_model)
        if verbose:
            print(f"After residual #1:                x1:          {x1.shape}")

        # ==============================================================
        # STEP 5.2 — Cross-Attention  ← THE KEY NEW STEP
        # ==============================================================
        # Q comes from x1 (decoder, after self-attention).
        # K and V come from encoder_output (encoder's final hidden states).
        #
        # This is how the decoder "reads" the source:
        #   Each decoder query asks: "which encoder positions are most
        #   relevant to what I'm generating right now?"
        #
        # Score matrix shape: (batch, heads, tgt_len, src_len)
        #   — rows  = decoder positions  (q_len  = tgt_len)
        #   — cols  = encoder positions (kv_len = src_len)
        #   — NOT square when tgt_len ≠ src_len
        #
        # No mask needed — the decoder can attend to all encoder positions.
        # The source sequence is fully processed; no autoregressive constraint.
        #
        # Gradient flow:
        #   dL/dcross_W_Q  → stays in decoder (Q projection)
        #   dL/dcross_W_K  → flows back into encoder_output → into encoder weights
        #   dL/dcross_W_V  → flows back into encoder_output → into encoder weights
        #   dL/dcross_W_O  → stays in decoder
        x1_norm = self.layer_norm_2(x1)
        # x1_norm: (batch, tgt_len, d_model)
        if verbose:
            print(f"After LayerNorm 2:                x1_norm:     {x1_norm.shape}")

        cross_attn_out = self._multi_head_attention(
            Q_src=x1_norm,         # Q from decoder
            K_src=encoder_output,  # K from encoder
            V_src=encoder_output,  # V from encoder
            W_Q=self.cross_W_Q, W_K=self.cross_W_K,
            W_V=self.cross_W_V, W_O=self.cross_W_O,
            attn_mask=src_mask,    # usually None; could mask <pad> tokens
        )
        # cross_attn_out: (batch, tgt_len, d_model)
        if verbose:
            print(f"Cross-attention output:           cross_attn:  {cross_attn_out.shape}")

        x2 = x1 + cross_attn_out
        # x2: (batch, tgt_len, d_model)
        if verbose:
            print(f"After residual #2:                x2:          {x2.shape}")

        # ==============================================================
        # STEP 5.3 — Feed-Forward Network
        # ==============================================================
        # Applied independently to each of the tgt_len positions.
        # Identical to the encoder's FFN.
        x2_norm = self.layer_norm_3(x2)
        if verbose:
            print(f"After LayerNorm 3:                x2_norm:     {x2_norm.shape}")

        ffn_hidden = self.ffn_linear1(x2_norm)
        # ffn_hidden: (batch, tgt_len, d_ff)
        ffn_hidden = self.ffn_gelu(ffn_hidden)
        ffn_out    = self.ffn_linear2(ffn_hidden)
        # ffn_out: (batch, tgt_len, d_model)
        if verbose:
            print(f"FFN output:                       ffn_out:     {ffn_out.shape}")

        output = x2 + ffn_out
        # output: (batch, tgt_len, d_model)
        if verbose:
            print(f"After residual #3 (output):       output:      {output.shape}")
            print(f"{'='*60}\n")

        return output


# ══════════════════════════════════════════════════════════════════════
# Runnable example
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)

    BATCH      = 2
    TGT_LEN    = 4
    SRC_LEN    = 6   # different from TGT_LEN — cross-attention handles this
    D_MODEL    = 12
    NUM_HEADS  = 3
    D_FF       = 48

    block  = ManualDecoderBlock(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF)

    x              = torch.randn(BATCH, TGT_LEN, D_MODEL)
    encoder_output = torch.randn(BATCH, SRC_LEN, D_MODEL)

    # Causal mask for target positions
    tgt_mask = torch.triu(
        torch.ones(TGT_LEN, TGT_LEN) * float('-inf'), diagonal=1
    )

    output = block(x, encoder_output, tgt_mask=tgt_mask)
    print(f"Input  x:              {x.shape}")
    print(f"Encoder output:        {encoder_output.shape}")
    print(f"Decoder block output:  {output.shape}")
    assert output.shape == (BATCH, TGT_LEN, D_MODEL)
    print("Shape check passed ✓")
