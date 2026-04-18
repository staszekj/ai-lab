"""
Educational PyTorch Implementation: Single Transformer Encoder Block
====================================================================

This file implements a Pre-LN Transformer encoder block from scratch,
using only low-level PyTorch primitives. Every tensor transformation
is made explicit with shape annotations and runtime logging.

Architecture (Pre-LN variant):
    x1     = x + MultiHeadSelfAttention(LayerNorm(x))
    output = x1 + FFN(LayerNorm(x1))

This is the variant used by GPT-2, GPT-3, and most modern LLMs.
Pre-LN places LayerNorm *before* attention/FFN (inside the residual branch),
which stabilises training compared to the original Post-LN design.
"""

import math
import torch
import torch.nn as nn


class ManualTransformerEncoderBlock(nn.Module):
    """
    A single Pre-LN Transformer encoder block with fully manual
    multi-head self-attention.

    Hyperparameters
    ---------------
    d_model   = 768   – width of the model (embedding dimension)
    num_heads = 12    – number of parallel attention heads
    d_k       = 64    – dimension of each head  (d_model / num_heads)
    d_ff      = 3072  – hidden size of the feed-forward network (4 × d_model)

    Input / output shape: (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
    ) -> None:
        super().__init__()

        # ── Validate that d_model splits evenly across heads ─────────
        # Each head operates on a slice of size d_k = d_model / num_heads.
        # If the division isn't exact we can't cleanly reshape the tensor.
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model == num_heads * self.d_k, (
            f"d_model ({d_model}) must equal num_heads ({num_heads}) × d_k ({self.d_k})"
        )

        self.d_ff = d_ff

        # ── Attention projection weights ─────────────────────────────
        # Each of W_Q, W_K, W_V is a (d_model, d_model) linear map.
        # They project the *same* input X into three different spaces:
        #   Q (queries)  – "what am I looking for?"
        #   K (keys)     – "what do I contain?"
        #   V (values)   – "what information do I carry?"
        self.W_Q = nn.Linear(d_model, d_model)  # (d_model → d_model)
        self.W_K = nn.Linear(d_model, d_model)  # (d_model → d_model)
        self.W_V = nn.Linear(d_model, d_model)  # (d_model → d_model)

        # Output projection applied after concatenating all heads back.
        # This lets the model mix information across heads.
        self.W_O = nn.Linear(d_model, d_model)  # (d_model → d_model)

        # ── Feed-Forward Network (FFN) ───────────────────────────────
        # Two linear layers with a GELU non-linearity in between.
        # The hidden dimension d_ff is typically 4× d_model.
        self.ffn_linear1 = nn.Linear(d_model, d_ff)   # (d_model → d_ff)
        self.ffn_gelu = nn.GELU()
        self.ffn_linear2 = nn.Linear(d_ff, d_model)   # (d_ff → d_model)

        # ── Layer Norms (Pre-LN) ─────────────────────────────────────
        # In the Pre-LN design we normalise *before* attention and FFN.
        # LayerNorm normalises across the last dimension (d_model) for
        # each token independently.
        self.layer_norm_1 = nn.LayerNorm(d_model)  # before attention
        self.layer_norm_2 = nn.LayerNorm(d_model)  # before FFN

        # ── Precompute the scaling factor for attention scores ───────
        # We divide dot-product scores by sqrt(d_k) to prevent the
        # softmax from saturating into one-hot vectors when d_k is large.
        self.scale = math.sqrt(self.d_k)  # = sqrt(64) = 8.0

    # ------------------------------------------------------------------
    # Forward pass – every step is explicit
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, verbose: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch_size, seq_len, d_model)
            The input embeddings (or the output of the previous block).
        attn_mask : optional Tensor of shape (seq_len, seq_len) or
            (batch_size, num_heads, seq_len, seq_len).
            Additive mask: 0 = attend, -inf = block.
            For causal (GPT-style) decoding, pass a upper-triangular
            matrix of -inf so tokens can't attend to future positions.

        Returns
        -------
        Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, (
            f"Input d_model ({d_model}) doesn't match expected ({self.d_model})"
        )
        if verbose:
            print(f"\n{'='*60}")
            print(f"Input X:  {x.shape}")
        # X: (batch_size, seq_len, d_model)  e.g. (2, 10, 768)

        # ==============================================================
        # STEP 5.1 – First LayerNorm (Pre-LN: normalise before attention)
        # ==============================================================
        # LayerNorm normalises each token vector (length d_model)
        # to have zero mean and unit variance, then applies a learned
        # affine transform (gamma, beta).  This keeps activations stable.
        x_norm1 = self.layer_norm_1(x)
        # x_norm1: (batch_size, seq_len, d_model)  e.g. (2, 10, 768)
        if verbose:
            print(f"After LayerNorm 1 (x_norm1):  {x_norm1.shape}")

        # ==============================================================
        # STEP 5.2 – Linear projections to Q, K, V
        # ==============================================================
        # We project the *normalised* input into three separate spaces.
        # Each projection is a full (d_model → d_model) linear layer,
        # which means Q, K, V each still have d_model=768 columns.
        # The "multi-head" split happens in the next step via reshape.
        Q = self.W_Q(x_norm1)
        # Q = x_norm1 @ W_Q.weight^T + W_Q.bias
        #   (batch, seq, d_model) @ (d_model, d_model)^T + (d_model,)
        #   (2, 10, 768)          @ (768, 768)            + (768,)     = (2, 10, 768)
        K = self.W_K(x_norm1)
        # K = x_norm1 @ W_K.weight^T + W_K.bias
        #   (2, 10, 768) @ (768, 768) + (768,) = (2, 10, 768)
        V = self.W_V(x_norm1)
        # V = x_norm1 @ W_V.weight^T + W_V.bias
        #   (2, 10, 768) @ (768, 768) + (768,) = (2, 10, 768)
        # Q: (batch_size, seq_len, d_model)  e.g. (2, 10, 768)
        # K: (batch_size, seq_len, d_model)  e.g. (2, 10, 768)
        # V: (batch_size, seq_len, d_model)  e.g. (2, 10, 768)
        if verbose:
            print(f"Q after projection:  {Q.shape}")
            print(f"K after projection:  {K.shape}")
            print(f"V after projection:  {V.shape}")

        # ==============================================================
        # STEP 5.3 – Reshape into multiple heads
        # ==============================================================
        # The 768-dim vector of each token is logically carved into
        # 12 slices of 64 dims – one slice per head.
        #
        # Reshape: (batch, seq, d_model) → (batch, seq, num_heads, d_k)
        #   e.g.   (2, 10, 768)        → (2, 10, 12, 64)
        #
        # Then we *transpose* the seq and num_heads dimensions so that
        # the head dimension comes before seq_len.  This makes the last
        # two dims (seq_len, d_k), which is the shape matmul expects
        # for batched matrix multiplication [one matmul per head].
        #
        # Permute: (batch, seq, num_heads, d_k) → (batch, num_heads, seq, d_k)
        #   e.g.   (2, 10, 12, 64)             → (2, 12, 10, 64)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
        # Q: (batch_size, seq_len, num_heads, d_k)  e.g. (2, 10, 12, 64)

        Q = Q.transpose(1, 2)
        # Q: (batch_size, num_heads, seq_len, d_k)  e.g. (2, 12, 10, 64)

        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # K: (batch_size, num_heads, seq_len, d_k)  e.g. (2, 12, 10, 64)

        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # V: (batch_size, num_heads, seq_len, d_k)  e.g. (2, 12, 10, 64)

        if verbose:
            print(f"Q after split into heads:  {Q.shape}")
            print(f"K after split into heads:  {K.shape}")
            print(f"V after split into heads:  {V.shape}")

        # Verify shapes are correct after the reshape + transpose
        assert Q.shape == (batch_size, self.num_heads, seq_len, self.d_k)
        assert K.shape == (batch_size, self.num_heads, seq_len, self.d_k)
        assert V.shape == (batch_size, self.num_heads, seq_len, self.d_k)

        # ==============================================================
        # STEP 5.4 – Transpose K for dot-product attention
        # ==============================================================
        # To compute Q @ K^T we need K with its last two dims swapped:
        #   K:   (batch, heads, seq, d_k)
        #   K^T: (batch, heads, d_k, seq)
        # The matmul then contracts over d_k giving a (seq, seq) score
        # matrix for every (batch, head) pair.
        K_T = K.transpose(-2, -1)
        # K_T: (batch_size, num_heads, d_k, seq_len)  e.g. (2, 12, 64, 10)
        if verbose:
            print(f"K transposed (K^T):  {K_T.shape}")

        # ==============================================================
        # STEP 5.5 – Raw attention scores  (Q @ K^T)
        # ==============================================================
        # Each score[i][j] measures how much query-token i should
        # attend to key-token j.  This is computed independently per
        # head and per batch element.
        #
        # Matmul broadcasting:
        #   Q:      (batch, heads, seq, d_k)
        #   K^T:    (batch, heads, d_k, seq)
        #   result: (batch, heads, seq, seq)   ← one score per token pair
        attention_scores = torch.matmul(Q, K_T)
        # Q @ K^T:
        #   (batch, heads, seq, d_k) @ (batch, heads, d_k, seq)
        #   (2, 12, 10, 64)          @ (2, 12, 64, 10)          = (2, 12, 10, 10)
        # ─── matmul contracts over the last dim of Q (d_k=64)
        #     and second-to-last dim of K^T (d_k=64).
        #     The first two dims (batch, heads) are batch dims.
        if verbose:
            print(f"Attention scores (Q @ K^T):  {attention_scores.shape}")

        # ==============================================================
        # STEP 5.6 – Scale the scores
        # ==============================================================
        # Without scaling, the variance of the dot products grows with
        # d_k.  Large values push softmax into regions with tiny
        # gradients (near 0 or 1), hurting training stability.
        # Dividing by sqrt(d_k) keeps the variance ≈ 1.
        scaled_scores = attention_scores / self.scale
        # scaled_scores: (batch_size, num_heads, seq_len, seq_len)
        #   e.g. (2, 12, 10, 10)
        if verbose:
            print(f"Scaled scores:  {scaled_scores.shape}")

        # ==============================================================
        # STEP 5.6b – Apply attention mask (optional, used for causal/GPT)
        # ==============================================================
        # For causal (autoregressive) models, we add a mask of -inf to
        # positions where a token should NOT attend (future tokens).
        # After adding -inf, softmax turns those positions into 0.0,
        # effectively blocking information flow from the future.
        #
        # The mask is ADDITIVE: scaled_scores + (-inf) = -inf → softmax(-inf) = 0
        if attn_mask is not None:
            scaled_scores = scaled_scores + attn_mask
            # scaled_scores: (batch_size, num_heads, seq_len, seq_len)
            # Positions with -inf will become 0 after softmax.
            if verbose:
                print(f"Scaled scores after mask:  {scaled_scores.shape}")

        # ==============================================================
        # STEP 5.7 – Softmax → attention weights
        # ==============================================================
        # Softmax along the *last* dimension (dim=-1) so that for each
        # query token the weights across all key tokens sum to 1.
        # This gives a probability distribution: "how much should this
        # query attend to each key?"
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        # attention_weights: (batch_size, num_heads, seq_len, seq_len)
        #   e.g. (2, 12, 10, 10)
        #   Each row sums to 1.0
        if verbose:
            print(f"Attention weights (softmax):  {attention_weights.shape}")

        # ==============================================================
        # STEP 5.8 – Weighted sum of values  (attention_weights @ V)
        # ==============================================================
        # We multiply the attention weights with V to get a weighted
        # combination of value vectors.  Each output token is a blend
        # of all value tokens, weighted by how much the query attended
        # to each key.
        #
        # Matmul broadcasting:
        #   weights: (batch, heads, seq, seq)
        #   V:       (batch, heads, seq, d_k)
        #   result:  (batch, heads, seq, d_k)   ← weighted values
        head_output = torch.matmul(attention_weights, V)
        # weights @ V:
        #   (batch, heads, seq, seq) @ (batch, heads, seq, d_k)
        #   (2, 12, 10, 10)         @ (2, 12, 10, 64)          = (2, 12, 10, 64)
        # ─── matmul contracts over the last dim of weights (seq=10)
        #     and second-to-last dim of V (seq=10).
        #     Result: each token gets a weighted blend of all 64-dim value vectors.
        if verbose:
            print(f"Attention output per head:  {head_output.shape}")

        assert head_output.shape == (batch_size, self.num_heads, seq_len, self.d_k)

        # ==============================================================
        # STEP 5.9 – Concatenate heads back into d_model dimension
        # ==============================================================
        # We now reverse the split we did in Step 3.
        #
        # First transpose to move the head dim next to d_k again:
        #   (batch, heads, seq, d_k) → (batch, seq, heads, d_k)
        #
        # Then reshape so that heads × d_k collapses back to d_model:
        #   (batch, seq, heads, d_k) → (batch, seq, d_model)
        #   e.g.  (2, 10, 12, 64)    → (2, 10, 768)
        #
        # .contiguous() is needed because transpose() returns a *view*
        # with non-contiguous memory, and view() requires contiguous data.
        concatenated = head_output.transpose(1, 2).contiguous()
        # concatenated: (batch_size, seq_len, num_heads, d_k)
        #   e.g. (2, 10, 12, 64)

        concatenated = concatenated.view(batch_size, seq_len, self.d_model)
        # concatenated: (batch_size, seq_len, d_model)  e.g. (2, 10, 768)
        if verbose:
            print(f"Concatenated heads:  {concatenated.shape}")

        assert concatenated.shape == (batch_size, seq_len, self.d_model)

        # ==============================================================
        # STEP 5.10 – Output projection
        # ==============================================================
        # A final linear layer that lets the model mix information
        # *across* heads.  Without this, each head's output would stay
        # in its own 64-dim subspace and never interact.
        attention_output = self.W_O(concatenated)
        # concatenated @ W_O.weight^T + W_O.bias
        #   (batch, seq, d_model) @ (d_model, d_model) + (d_model,)
        #   (2, 10, 768)          @ (768, 768)          + (768,)    = (2, 10, 768)
        if verbose:
            print(f"Output projection:  {attention_output.shape}")

        # ==============================================================
        # STEP 5.11 – First residual connection
        # ==============================================================
        # Add the original input (before LayerNorm) to the attention
        # output.  This is the "skip connection" that lets gradients
        # flow directly from later layers back to earlier ones,
        # preventing vanishing gradients in deep networks.
        #
        # Pre-LN formula: x1 = x + Attention(LayerNorm(x))
        x1 = x + attention_output
        # x1: (batch_size, seq_len, d_model)  e.g. (2, 10, 768)
        if verbose:
            print(f"After first residual add (x1):  {x1.shape}")

        # ==============================================================
        # STEP 5.12 – Second LayerNorm (before FFN)
        # ==============================================================
        x1_norm = self.layer_norm_2(x1)
        # x1_norm: (batch_size, seq_len, d_model)  e.g. (2, 10, 768)
        if verbose:
            print(f"After LayerNorm 2 (x1_norm):  {x1_norm.shape}")

        # ==============================================================
        # STEP 5.13 – Feed-Forward Network (FFN)
        # ==============================================================
        # The FFN is applied to each token position independently
        # (same weights, no cross-token interaction).
        #
        # Structure:
        #   Linear(768 → 3072)  – expand
        #   GELU activation     – non-linearity
        #   Linear(3072 → 768)  – compress back
        #
        # The expansion to 4× lets the network learn richer per-token
        # transformations than a single 768→768 layer could.

        # 5.13a – Expand to hidden dimension
        ffn_hidden = self.ffn_linear1(x1_norm)
        # x1_norm @ W1.weight^T + W1.bias
        #   (batch, seq, d_model) @ (d_model, d_ff) + (d_ff,)
        #   (2, 10, 768)          @ (768, 3072)      + (3072,)   = (2, 10, 3072)
        if verbose:
            print(f"FFN hidden (after linear1):  {ffn_hidden.shape}")

        # 5.13b – GELU activation
        # GELU is a smooth approximation of ReLU that allows small
        # negative values through.  It's the standard activation in
        # BERT, GPT-2, and most modern transformers.
        ffn_hidden = self.ffn_gelu(ffn_hidden)
        # ffn_hidden: (batch_size, seq_len, d_ff)  e.g. (2, 10, 3072)
        #   (shape unchanged, activation is element-wise)

        # 5.13c – Compress back to d_model
        ffn_output = self.ffn_linear2(ffn_hidden)
        # ffn_hidden @ W2.weight^T + W2.bias
        #   (batch, seq, d_ff) @ (d_ff, d_model) + (d_model,)
        #   (2, 10, 3072)      @ (3072, 768)      + (768,)    = (2, 10, 768)
        if verbose:
            print(f"FFN output (after linear2):  {ffn_output.shape}")

        # ==============================================================
        # STEP 5.14 – Second residual connection
        # ==============================================================
        # Pre-LN formula: output = x1 + FFN(LayerNorm(x1))
        output = x1 + ffn_output
        # output: (batch_size, seq_len, d_model)  e.g. (2, 10, 768)
        if verbose:
            print(f"After second residual add (output):  {output.shape}")

        # ==============================================================
        # Done – return the final output
        # ==============================================================
        if verbose:
            print(f"Final output:  {output.shape}")
            print(f"{'='*60}\n")

        return output
        # output: (batch_size, seq_len, d_model)  e.g. (2, 10, 768)


# ══════════════════════════════════════════════════════════════════════
# Runnable example
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Fixed seed for reproducibility
    torch.manual_seed(42)

    # ── Hyperparameters ──────────────────────────────────────────────
    batch_size = 2
    seq_len = 10
    d_model = 768
    num_heads = 12
    d_k = 64
    d_ff = 3072

    # Sanity check: heads × head_dim must equal model width
    assert d_model == num_heads * d_k, (
        f"d_model ({d_model}) != num_heads ({num_heads}) × d_k ({d_k})"
    )

    # ── Create a random input tensor ─────────────────────────────────
    # In a real model this would be the sum of token embeddings and
    # positional encodings.
    x = torch.randn(batch_size, seq_len, d_model)
    # x: (2, 10, 768)
    print(f"Created input tensor x: {x.shape}")
    print(f"  batch_size = {batch_size}")
    print(f"  seq_len    = {seq_len}")
    print(f"  d_model    = {d_model}")

    # ── Instantiate the encoder block ────────────────────────────────
    block = ManualTransformerEncoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    # Print parameter count for context
    total_params = sum(p.numel() for p in block.parameters())
    print(f"\nTotal parameters in one encoder block: {total_params:,}")

    # ── Forward pass ─────────────────────────────────────────────────
    output = block(x)

    # ── Verify output shape ──────────────────────────────────────────
    assert output.shape == (batch_size, seq_len, d_model), (
        f"Unexpected output shape: {output.shape}"
    )
    print(f"✓ Output shape matches input shape: {output.shape}")
