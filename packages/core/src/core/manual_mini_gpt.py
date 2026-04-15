"""
Educational PyTorch: Mini-GPT — Full Decoder-Only Transformer
=============================================================

This file builds a complete (tiny) GPT-style language model from scratch,
stacking multiple decoder blocks with causal masking.

Architecture:
    Token Embedding  (vocab_size → d_model)
    + Positional Embedding  (max_seq_len → d_model)
    → [DecoderBlock 1] → [DecoderBlock 2] → ... → [DecoderBlock N]
    → Final LayerNorm
    → LM Head  (d_model → vocab_size)
    → logits → softmax → next token prediction

Key difference from encoder (BERT):
    GPT uses a CAUSAL MASK so that each token can only attend to
    itself and tokens BEFORE it — never to future tokens.
    This is what makes it "autoregressive": it generates one token
    at a time, left to right.

    Causal mask for seq_len=5:
        [[  0, -inf, -inf, -inf, -inf],
         [  0,    0, -inf, -inf, -inf],
         [  0,    0,    0, -inf, -inf],
         [  0,    0,    0,    0, -inf],
         [  0,    0,    0,    0,    0]]

    After softmax, -inf becomes 0.0 → no attention to future tokens.

Model sizes (educational, NOT production):
    mini:  4 layers, d_model=256, 4 heads, d_ff=1024   (~5M params)
    small: 6 layers, d_model=384, 6 heads, d_ff=1536   (~15M params)

We use "mini" by default so it runs instantly on CPU.
"""

import math
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════
# Transformer Encoder Block (inlined — no external dependency)
# ══════════════════════════════════════════════════════════════════════

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

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        assert d_model == num_heads * self.d_k, (
            f"d_model ({d_model}) must equal num_heads ({num_heads}) × d_k ({self.d_k})"
        )

        self.d_ff = d_ff

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.ffn_linear1 = nn.Linear(d_model, d_ff)
        self.ffn_gelu = nn.GELU()
        self.ffn_linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x         : (batch_size, seq_len, d_model)
        attn_mask : (seq_len, seq_len) or None — additive mask (0=attend, -inf=block)

        Returns
        -------
        Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, (
            f"Input d_model ({d_model}) doesn't match expected ({self.d_model})"
        )
        print(f"\n{'='*60}")
        print(f"Input X:  {x.shape}")

        # ── STEP 1 – LayerNorm 1 ──────────────────────────────────────
        x_norm1 = self.layer_norm_1(x)
        print(f"After LayerNorm 1 (x_norm1):  {x_norm1.shape}")

        # ── STEP 2 – Project to Q, K, V ──────────────────────────────
        Q = self.W_Q(x_norm1)   # (batch, seq, d_model)
        K = self.W_K(x_norm1)   # (batch, seq, d_model)
        V = self.W_V(x_norm1)   # (batch, seq, d_model)
        print(f"Q after projection:  {Q.shape}")
        print(f"K after projection:  {K.shape}")
        print(f"V after projection:  {V.shape}")

        # ── STEP 3 – Split into heads ─────────────────────────────────
        # (batch, seq, d_model) → (batch, heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        print(f"Q after split into heads:  {Q.shape}")
        print(f"K after split into heads:  {K.shape}")
        print(f"V after split into heads:  {V.shape}")

        assert Q.shape == (batch_size, self.num_heads, seq_len, self.d_k)
        assert K.shape == (batch_size, self.num_heads, seq_len, self.d_k)
        assert V.shape == (batch_size, self.num_heads, seq_len, self.d_k)

        # ── STEP 4 – Transpose K ──────────────────────────────────────
        K_T = K.transpose(-2, -1)
        # K_T: (batch, heads, d_k, seq)
        print(f"K transposed (K^T):  {K_T.shape}")

        # ── STEP 5 – Raw attention scores (Q @ K^T) ───────────────────
        attention_scores = torch.matmul(Q, K_T)
        # (batch, heads, seq, seq)
        print(f"Attention scores (Q @ K^T):  {attention_scores.shape}")

        # ── STEP 6 – Scale ────────────────────────────────────────────
        scaled_scores = attention_scores / self.scale
        print(f"Scaled scores:  {scaled_scores.shape}")

        # ── STEP 6b – Apply mask ──────────────────────────────────────
        if attn_mask is not None:
            scaled_scores = scaled_scores + attn_mask
            print(f"Scaled scores after mask:  {scaled_scores.shape}")

        # ── STEP 7 – Softmax → attention weights ──────────────────────
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        # (batch, heads, seq, seq)
        print(f"Attention weights (softmax):  {attention_weights.shape}")

        # ── STEP 8 – Weighted sum of values ───────────────────────────
        head_output = torch.matmul(attention_weights, V)
        # (batch, heads, seq, d_k)
        print(f"Attention output per head:  {head_output.shape}")

        assert head_output.shape == (batch_size, self.num_heads, seq_len, self.d_k)

        # ── STEP 9 – Concatenate heads ────────────────────────────────
        concatenated = head_output.transpose(1, 2).contiguous()
        concatenated = concatenated.view(batch_size, seq_len, self.d_model)
        # (batch, seq, d_model)
        print(f"Concatenated heads:  {concatenated.shape}")

        assert concatenated.shape == (batch_size, seq_len, self.d_model)

        # ── STEP 10 – Output projection ───────────────────────────────
        attention_output = self.W_O(concatenated)
        print(f"Output projection:  {attention_output.shape}")

        # ── STEP 11 – First residual connection ───────────────────────
        x1 = x + attention_output
        # x1: (batch, seq, d_model)
        print(f"After first residual add (x1):  {x1.shape}")

        # ── STEP 12 – LayerNorm 2 ─────────────────────────────────────
        x1_norm = self.layer_norm_2(x1)
        print(f"After LayerNorm 2 (x1_norm):  {x1_norm.shape}")

        # ── STEP 13 – Feed-Forward Network ────────────────────────────
        ffn_hidden = self.ffn_linear1(x1_norm)
        # (batch, seq, d_ff)
        print(f"FFN hidden (after linear1):  {ffn_hidden.shape}")

        ffn_hidden = self.ffn_gelu(ffn_hidden)

        ffn_output = self.ffn_linear2(ffn_hidden)
        # (batch, seq, d_model)
        print(f"FFN output (after linear2):  {ffn_output.shape}")

        # ── STEP 14 – Second residual connection ──────────────────────
        output = x1 + ffn_output
        # output: (batch, seq, d_model)
        print(f"After second residual add (output):  {output.shape}")

        print(f"Final output:  {output.shape}")
        print(f"{'='*60}\n")

        return output


class ManualMiniGPT(nn.Module):
    """
    A complete mini-GPT language model.

    Components (in order of the forward pass):
        1. Token embedding:      integer token IDs → dense vectors
        2. Positional embedding:  position indices → dense vectors
        3. N decoder blocks:      self-attention + FFN (with causal mask)
        4. Final LayerNorm:       stabilise output (Pre-LN needs this)
        5. LM head:              project to vocabulary logits

    Parameters
    ----------
    vocab_size    : number of tokens in the vocabulary
    max_seq_len   : maximum sequence length the model can handle
    d_model       : width of the model (embedding dimension)
    num_heads     : number of attention heads per block
    d_ff          : hidden dimension of FFN
    num_layers    : number of stacked decoder blocks
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        max_seq_len: int = 128,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 1024,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        # ==============================================================
        # TOKEN EMBEDDING
        # ==============================================================
        # Maps each token ID (integer) to a dense vector of size d_model.
        # This is a lookup table with vocab_size rows, each of length d_model.
        #
        # Example: if token ID = 42, we look up row 42 in the embedding
        # matrix to get a 256-dimensional vector.
        #
        # Shape: (vocab_size, d_model)  e.g. (1000, 256)
        # Input:  (batch_size, seq_len)     — integer IDs
        # Output: (batch_size, seq_len, d_model) — dense vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # ==============================================================
        # POSITIONAL EMBEDDING
        # ==============================================================
        # Tokens are processed in parallel (not sequentially), so the
        # model has no notion of ORDER.  Positional embeddings inject
        # position information: "I am the 3rd token", "I am the 7th token".
        #
        # We use LEARNED positional embeddings (like GPT-2).
        # Alternative: sinusoidal fixed embeddings (original Transformer).
        #
        # Shape: (max_seq_len, d_model)  e.g. (128, 256)
        # Input:  position indices [0, 1, 2, ..., seq_len-1]
        # Output: (seq_len, d_model) → broadcast to (batch_size, seq_len, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)

        # ==============================================================
        # STACKED DECODER BLOCKS
        # ==============================================================
        # This is the core of the model: N identical blocks stacked on
        # top of each other.  Each block:
        #   1. Applies causal self-attention (with mask)
        #   2. Applies FFN
        #   3. Has residual connections + LayerNorm (Pre-LN)
        #
        # The output of block i becomes the input to block i+1.
        # All blocks share the same architecture but have INDEPENDENT
        # weights — each block learns different patterns.
        #
        # nn.ModuleList registers all blocks so PyTorch can find their
        # parameters for .parameters(), .to(device), state_dict, etc.
        self.blocks = nn.ModuleList([
            ManualTransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
            )
            for layer_idx in range(num_layers)
        ])
        # blocks: list of N ManualTransformerEncoderBlock instances

        # ==============================================================
        # FINAL LAYER NORM
        # ==============================================================
        # In the Pre-LN design, LayerNorm is INSIDE each block (before
        # attention/FFN).  But the very last block's output goes through
        # attention/FFN and then exits via a residual add WITHOUT a
        # trailing LayerNorm.  So we need one final LayerNorm here.
        #
        # Without this, the LM head receives unnormalised activations
        # from the last residual connection, hurting training stability.
        self.final_layer_norm = nn.LayerNorm(d_model)

        # ==============================================================
        # LANGUAGE MODEL HEAD (LM Head)
        # ==============================================================
        # Projects the d_model-dimensional hidden state of each token
        # to a vocabulary-sized vector of LOGITS.
        #
        # logits[i] = "how likely is vocabulary token i to be the NEXT token?"
        #
        # In many real models (GPT-2, LLaMA), the LM head shares weights
        # with the token embedding ("weight tying").  We keep them
        # separate here for clarity.
        #
        # Shape: (d_model, vocab_size)  e.g. (256, 1000)
        # Input:  (batch_size, seq_len, d_model)
        # Output: (batch_size, seq_len, vocab_size)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create the causal (look-ahead) mask.

        This is an upper-triangular matrix of -inf values:
            [[  0, -inf, -inf, ...],
             [  0,    0, -inf, ...],
             [  0,    0,    0, ...],
             ...
             [  0,    0,    0, ... 0]]

        When added to attention scores BEFORE softmax:
            score + 0    = score      → softmax keeps it (attend)
            score + -inf = -inf       → softmax gives 0  (block)

        This ensures token at position i can only attend to positions ≤ i.

        Returns: (seq_len, seq_len) mask
        """
        # torch.triu selects the upper triangle (above the diagonal).
        # diagonal=1 means we start one ABOVE the main diagonal,
        # so the diagonal itself stays 0 (a token can attend to itself).
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1,
        )
        # mask: (seq_len, seq_len)
        # Row i has 0s at columns [0..i] and -inf at columns [i+1..seq_len-1]
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Full forward pass of mini-GPT.

        Parameters
        ----------
        input_ids : (batch_size, seq_len) — integer token IDs
        verbose   : whether to print shape logs (disable for generation)

        Returns
        -------
        logits : (batch_size, seq_len, vocab_size)
            Raw scores for the next token at every position.
            To predict the next token after position i, use logits[:, i, :].
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, (
            f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        )

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# MINI-GPT FORWARD PASS")
            print(f"# {self.num_layers} layers, d_model={self.d_model}, "
                  f"{self.num_heads} heads")
            print(f"{'#'*60}")
            print(f"\nInput IDs:  {input_ids.shape}")
            # input_ids: (batch_size, seq_len)  e.g. (2, 20)

        # ==============================================================
        # STEP 1 — Token embedding lookup
        # ==============================================================
        # Convert integer token IDs to dense vectors.
        # Each ID indexes a row in the (vocab_size, d_model) table.
        token_emb = self.token_embedding(input_ids)
        # token_emb: (batch_size, seq_len, d_model)  e.g. (2, 20, 256)
        if verbose:
            print(f"Token embeddings:  {token_emb.shape}")

        # ==============================================================
        # STEP 2 — Positional embedding lookup
        # ==============================================================
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        # These are the same for every sample in the batch.
        position_ids = torch.arange(seq_len, device=input_ids.device)
        # position_ids: (seq_len,)  e.g. [0, 1, 2, ..., 19]

        pos_emb = self.positional_embedding(position_ids)
        # pos_emb: (seq_len, d_model)  e.g. (20, 256)
        # Broadcasting will expand this to (batch_size, seq_len, d_model)
        if verbose:
            print(f"Position embeddings:  {pos_emb.shape}")

        # ==============================================================
        # STEP 3 — Combine token + position embeddings
        # ==============================================================
        # The input to the first decoder block is the SUM of token and
        # positional embeddings.  This is how the model knows both
        # WHAT each token is and WHERE it is in the sequence.
        #
        # In GPT-2 there's also dropout here; we skip it for clarity.
        x = token_emb + pos_emb
        # x: (batch_size, seq_len, d_model)  e.g. (2, 20, 256)
        # Broadcasting: (2, 20, 256) + (20, 256) → (2, 20, 256)
        if verbose:
            print(f"Token + Position embeddings:  {x.shape}")

        # ==============================================================
        # STEP 4 — Create causal mask
        # ==============================================================
        # ┌──────────────────────────────────────────────────────────┐
        # │  THIS IS THE ONLY DIFFERENCE BETWEEN ENCODER & DECODER  │
        # │                                                         │
        # │  DECODER_ONLY (GPT):  causal mask applied               │
        # │    → each token sees only itself and past tokens        │
        # │    → triangular mask with -inf above diagonal           │
        # │    → enables autoregressive generation                  │
        # │                                                         │
        # │  ENCODER_ONLY (BERT): no mask (or set mask = None)      │
        # │    → each token sees ALL other tokens (bidirectional)   │
        # │    → full attention matrix                              │
        # │    → used for classification, not generation            │
        # │                                                         │
        # │  The transformer block itself is IDENTICAL in both      │
        # │  cases — only the mask changes.                         │
        # └──────────────────────────────────────────────────────────┘
        #
        # The causal mask prevents each token from "seeing" future tokens.
        # Without it, the model could cheat during training by looking
        # at the answer it's supposed to predict!
        causal_mask = self._create_causal_mask(seq_len, device=input_ids.device)
        # causal_mask: (seq_len, seq_len)  e.g. (20, 20)
        if verbose:
            print(f"Causal mask:  {causal_mask.shape}")
            # Show a small corner of the mask for illustration
            if seq_len <= 8:
                print(f"  Mask values (0=attend, -inf=block):")
                for row in range(min(seq_len, 6)):
                    vals = ["   0" if v == 0 else "-inf" for v in causal_mask[row].tolist()]
                    print(f"    token {row}: [{', '.join(vals)}]")

        # ==============================================================
        # STEP 5 — Pass through N decoder blocks
        # ==============================================================
        # Each block transforms x: (batch, seq, d_model) → (batch, seq, d_model)
        # The causal mask is the SAME for every block — it only depends
        # on seq_len, not on the layer.
        #
        # This is the deepest part of the model.  With 4 layers:
        #   x → Block0 → Block1 → Block2 → Block3 → x'
        #
        # Each block learns different patterns:
        #   - Early layers: local patterns, syntax, simple co-occurrences
        #   - Later layers: longer-range dependencies, semantics

        for layer_idx, block in enumerate(self.blocks):
            if verbose:
                print(f"\n{'─'*60}")
                print(f"DECODER BLOCK {layer_idx} / {self.num_layers - 1}")
                print(f"{'─'*60}")
                print(f"Block {layer_idx} input:  {x.shape}")

            # DECODER_ONLY: pass causal_mask → autoregressive attention
            # ENCODER_ONLY: pass attn_mask=None → bidirectional attention
            x = block(x, attn_mask=causal_mask)
            # x: (batch_size, seq_len, d_model) — same shape, new values
            # Each block refines the representation through attention + FFN.

            if verbose:
                print(f"Block {layer_idx} output: {x.shape}")
                print(f"  output mean: {x.mean().item():.6f}  "
                      f"std: {x.std().item():.6f}")

        # ==============================================================
        # STEP 6 — Final LayerNorm
        # ==============================================================
        # Normalise the output of the last decoder block.
        # Pre-LN architecture requires this because the last residual
        # add doesn't have a LayerNorm after it.
        x = self.final_layer_norm(x)
        # x: (batch_size, seq_len, d_model)  e.g. (2, 20, 256)
        if verbose:
            print(f"\nAfter final LayerNorm:  {x.shape}")

        # ==============================================================
        # STEP 7 — LM Head: project to vocabulary logits
        # ==============================================================
        # Each position's d_model vector is projected to vocab_size logits.
        # logits[b, t, v] = "how likely is vocab token v to be the next
        #                     token after position t in batch element b?"
        #
        # Note: we get logits for EVERY position, not just the last one.
        # During training, this lets us compute loss at all positions
        # simultaneously (teacher forcing).
        # During generation, we only use the logits at the LAST position.
        logits = self.lm_head(x)
        # logits: (batch_size, seq_len, vocab_size)  e.g. (2, 20, 1000)
        if verbose:
            print(f"LM Head logits:  {logits.shape}")
            print(f"  → for each of {seq_len} positions, scores over "
                  f"{self.vocab_size} vocabulary tokens")

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive text generation — the GPT inference loop.

        This is how ChatGPT generates text: one token at a time,
        each time feeding ALL previous tokens back into the model.

        Algorithm:
            1. Run forward pass on current tokens
            2. Take logits at the LAST position
            3. Apply temperature scaling
            4. Sample next token from the probability distribution
            5. Append the new token to the sequence
            6. Repeat from step 1

        Parameters
        ----------
        input_ids      : (1, seq_len) — starting tokens (the "prompt")
        max_new_tokens : how many tokens to generate
        temperature    : controls randomness:
                         < 1.0 → more deterministic (peaked distribution)
                         = 1.0 → standard sampling
                         > 1.0 → more random (flatter distribution)

        Returns
        -------
        generated_ids : (1, seq_len + max_new_tokens) — full sequence
        """
        print(f"\n{'#'*60}")
        print(f"# AUTOREGRESSIVE GENERATION")
        print(f"# Starting tokens: {input_ids.shape[1]}, "
              f"generating {max_new_tokens} new tokens")
        print(f"# Temperature: {temperature}")
        print(f"{'#'*60}")

        generated = input_ids.clone()
        # generated: (1, current_len) — grows by 1 each step

        for step in range(max_new_tokens):
            current_len = generated.shape[1]

            # Truncate to max_seq_len if the sequence gets too long.
            # In real models this is handled by KV-cache; we just crop.
            if current_len > self.max_seq_len:
                context = generated[:, -self.max_seq_len:]
            else:
                context = generated
            # context: (1, min(current_len, max_seq_len))

            # ── Forward pass (no shape logging during generation) ────
            logits = self.forward(context, verbose=False)
            # logits: (1, context_len, vocab_size)

            # ── Take logits at the LAST position only ────────────────
            # This is the model's prediction for "what comes next?"
            next_token_logits = logits[:, -1, :]
            # next_token_logits: (1, vocab_size)  e.g. (1, 1000)

            # ── Temperature scaling ──────────────────────────────────
            # Dividing logits by temperature before softmax:
            #   T < 1 → sharpens the distribution (more confident)
            #   T = 1 → no change
            #   T > 1 → flattens the distribution (more random)
            #
            # At T → 0: always picks the highest-scoring token (greedy)
            # At T → ∞: uniform random sampling
            scaled_logits = next_token_logits / temperature
            # scaled_logits: (1, vocab_size)

            # ── Convert to probabilities ─────────────────────────────
            probs = torch.softmax(scaled_logits, dim=-1)
            # probs: (1, vocab_size) — sums to 1.0

            # ── Sample the next token ────────────────────────────────
            # torch.multinomial draws one sample from the distribution.
            # This is SAMPLING — not greedy argmax.
            # Sampling introduces variety: the same prompt can generate
            # different continuations each time.
            next_token = torch.multinomial(probs, num_samples=1)
            # next_token: (1, 1) — a single token ID

            # ── Append to the generated sequence ─────────────────────
            generated = torch.cat([generated, next_token], dim=1)
            # generated: (1, current_len + 1)

            print(f"  Step {step:3d}: generated token {next_token.item():5d}  "
                  f"(prob={probs[0, next_token.item()].item():.4f})  "
                  f"sequence length={generated.shape[1]}")

        print(f"\nGeneration complete: {generated.shape}")
        return generated


# ══════════════════════════════════════════════════════════════════════
# Manual cross-entropy loss for language modelling
# ══════════════════════════════════════════════════════════════════════

def language_model_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute language modelling loss: predict the NEXT token at every position.

    In language modelling, the target for position i is the token at
    position i+1.  So:
        logits[:, 0, :]  should predict target_ids[:, 1]
        logits[:, 1, :]  should predict target_ids[:, 2]
        ...
        logits[:, T-2, :] should predict target_ids[:, T-1]

    We shift so that logits[:-1] aligns with targets[1:].

    Parameters
    ----------
    logits     : (batch_size, seq_len, vocab_size)
    target_ids : (batch_size, seq_len) — the full sequence (same as input)

    Returns
    -------
    loss : scalar
    """
    batch_size, seq_len, vocab_size = logits.shape

    print(f"\n{'─'*60}")
    print(f"LANGUAGE MODEL LOSS (next-token prediction)")
    print(f"{'─'*60}")
    print(f"Logits:      {logits.shape}")
    print(f"Target IDs:  {target_ids.shape}")

    # ==================================================================
    # STEP 1 — Shift logits and targets
    # ==================================================================
    # logits[:, :-1, :] → predictions for positions 0..T-2
    # target_ids[:, 1:] → ground truth at positions 1..T-1
    #
    # This "shift by 1" is the fundamental setup of autoregressive LM:
    # given tokens [A, B, C, D], the model should predict:
    #   from A → predict B
    #   from A,B → predict C
    #   from A,B,C → predict D

    shift_logits = logits[:, :-1, :]
    # shift_logits: (batch_size, seq_len - 1, vocab_size)  e.g. (2, 19, 1000)

    shift_targets = target_ids[:, 1:]
    # shift_targets: (batch_size, seq_len - 1)  e.g. (2, 19)

    print(f"Shifted logits:   {shift_logits.shape}  (predicting tokens 1..{seq_len-1})")
    print(f"Shifted targets:  {shift_targets.shape}  (ground truth tokens 1..{seq_len-1})")

    # ==================================================================
    # STEP 2 — Flatten for cross-entropy
    # ==================================================================
    # We reshape from 3D to 2D so we can apply cross-entropy over
    # all (batch × position) predictions at once.
    num_predictions = batch_size * (seq_len - 1)

    flat_logits = shift_logits.reshape(num_predictions, vocab_size)
    # flat_logits: (batch_size * (seq_len-1), vocab_size)  e.g. (38, 1000)

    flat_targets = shift_targets.reshape(num_predictions)
    # flat_targets: (batch_size * (seq_len-1),)  e.g. (38,)

    print(f"Flat logits:   {flat_logits.shape}  ({num_predictions} predictions)")
    print(f"Flat targets:  {flat_targets.shape}")

    # ==================================================================
    # STEP 3 — Manual cross-entropy (same as manual_transformer_loss_backward.py)
    # ==================================================================
    # Numerically stable log-softmax + NLL

    # 3a — Subtract max for numerical stability
    max_logits = flat_logits.max(dim=-1, keepdim=True).values
    shifted = flat_logits - max_logits
    # shifted: (num_predictions, vocab_size)

    # 3b — Log-softmax
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
    log_probs = shifted - log_sum_exp
    # log_probs: (num_predictions, vocab_size)

    # 3c — Pick the log-prob of the correct next token
    target_log_probs = torch.gather(
        log_probs, dim=1, index=flat_targets.unsqueeze(1)
    ).squeeze(1)
    # target_log_probs: (num_predictions,)

    # 3d — Negate and average
    loss = -target_log_probs.mean()
    # loss: scalar []

    print(f"Loss (mean NLL):  {loss.item():.6f}")

    # What's the expected loss for a random model?
    # With vocab_size=1000, random guessing gives -log(1/1000) = 6.91
    random_loss = math.log(vocab_size)
    print(f"Expected random loss: -log(1/{vocab_size}) = {random_loss:.4f}")
    print(f"  (our loss should start near this value for an untrained model)")
    print(f"{'─'*60}\n")

    return loss


# ══════════════════════════════════════════════════════════════════════
# Runnable example
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)

    # ── Model hyperparameters (mini-GPT) ─────────────────────────────
    vocab_size = 1000      # small vocabulary (real GPT-2: 50257)
    max_seq_len = 128      # max context window (real GPT-2: 1024)
    d_model = 256          # embedding dimension (real GPT-2: 768)
    num_heads = 4          # attention heads (real GPT-2: 12)
    d_ff = 1024            # FFN hidden dim (real GPT-2: 3072)
    num_layers = 4         # decoder blocks (real GPT-2: 12)

    batch_size = 2
    seq_len = 20           # tokens in our example sequences

    print(f"{'='*60}")
    print(f"MINI-GPT CONFIGURATION")
    print(f"{'='*60}")
    print(f"  vocab_size   = {vocab_size}")
    print(f"  max_seq_len  = {max_seq_len}")
    print(f"  d_model      = {d_model}")
    print(f"  num_heads    = {num_heads}")
    print(f"  d_k          = {d_model // num_heads}")
    print(f"  d_ff         = {d_ff}")
    print(f"  num_layers   = {num_layers}")
    print(f"  batch_size   = {batch_size}")
    print(f"  seq_len      = {seq_len}")

    # ── Instantiate the model ────────────────────────────────────────
    model = ManualMiniGPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  ≈ {total_params / 1e6:.1f}M parameters")

    # Break down parameter count by component
    emb_params = sum(p.numel() for p in model.token_embedding.parameters()) + \
                 sum(p.numel() for p in model.positional_embedding.parameters())
    block_params = sum(p.numel() for p in model.blocks.parameters())
    head_params = sum(p.numel() for p in model.lm_head.parameters()) + \
                  sum(p.numel() for p in model.final_layer_norm.parameters())
    print(f"\n  Parameter breakdown:")
    print(f"    Token + Position embeddings:  {emb_params:>10,}  "
          f"({emb_params/total_params*100:.1f}%)")
    print(f"    {num_layers} Decoder blocks:              "
          f"{block_params:>10,}  ({block_params/total_params*100:.1f}%)")
    print(f"    Final LayerNorm + LM head:    {head_params:>10,}  "
          f"({head_params/total_params*100:.1f}%)")

    # ── Create fake input sequences ──────────────────────────────────
    # In a real model these would be tokenised text.
    # Here we use random token IDs.
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # input_ids: (2, 20)
    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"  Sample: {input_ids[0, :10].tolist()} ...")

    # ==================================================================
    # PHASE 1 — FORWARD PASS  (STEP 1 → 7 run inside model.forward())
    # ==================================================================
    logits = model(input_ids, verbose=True)
    # logits: (batch_size, seq_len, vocab_size)  e.g. (2, 20, 1000)

    print(f"\n{'='*60}")
    print(f"FORWARD PASS COMPLETE  (STEP 1-7)")
    print(f"  Input:  {input_ids.shape}  (integer token IDs)")
    print(f"  Output: {logits.shape}  (logits over vocabulary)")
    print(f"{'='*60}")

    # ==================================================================
    # STEP 8 — Softmax → next-token probabilities
    # ==================================================================
    # For each position, softmax turns the vocab_size logits into a
    # probability distribution over the vocabulary.
    # At position t, probs[t] = P(next token | tokens 0..t).
    print(f"\n{'#'*60}")
    print(f"# STEP 8 — SOFTMAX → NEXT-TOKEN PROBABILITIES")
    print(f"{'#'*60}")

    probs = torch.softmax(logits, dim=-1)
    # probs: (batch_size, seq_len, vocab_size) — each row sums to 1.0
    print(f"\nprobs: {probs.shape}  (each row sums to 1.0)")

    # Show top-5 predictions for the LAST position of the first sample
    last_probs = probs[0, -1]   # (vocab_size,)
    top5 = last_probs.topk(5)
    print(f"  Top-5 next-token predictions after position {seq_len - 1}:")
    for prob, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"    token {idx:>5}: {prob:.4f}")
    print(f"  (Untrained model — essentially random over {vocab_size} tokens)")

    # ==================================================================
    # STEP 9 — Language model loss  (next-token prediction)
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# STEP 9 — LOSS COMPUTATION")
    print(f"{'#'*60}")

    loss = language_model_loss(logits, input_ids)

    # ==================================================================
    # STEP 10 — Backward pass (gradient flow)
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# STEP 10 — BACKWARD PASS")
    print(f"{'#'*60}")

    loss.backward()

    # ── Gradient flow through layers ─────────────────────────────────
    print(f"\nGradient norms per layer (should be similar = healthy flow):")
    print(f"{'─'*60}")

    # Embeddings
    print(f"  token_embedding.weight:      "
          f"grad_norm={model.token_embedding.weight.grad.norm().item():.6f}")
    print(f"  positional_embedding.weight: "
          f"grad_norm={model.positional_embedding.weight.grad.norm().item():.6f}")

    # Each decoder block
    for i, block in enumerate(model.blocks):
        # Aggregate gradient norm for the whole block
        block_grad_norm = sum(
            p.grad.norm().item() ** 2 for p in block.parameters() if p.grad is not None
        ) ** 0.5
        # Also show individual key layers
        q_norm = block.W_Q.weight.grad.norm().item()
        v_norm = block.W_V.weight.grad.norm().item()
        ffn1_norm = block.ffn_linear1.weight.grad.norm().item()
        print(f"  block[{i}]: total_grad_norm={block_grad_norm:.6f}  "
              f"(W_Q={q_norm:.4f}, W_V={v_norm:.4f}, FFN1={ffn1_norm:.4f})")

    # LM head
    print(f"  final_layer_norm.weight:     "
          f"grad_norm={model.final_layer_norm.weight.grad.norm().item():.6f}")
    print(f"  lm_head.weight:              "
          f"grad_norm={model.lm_head.weight.grad.norm().item():.6f}")

    # ── Key observation ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"KEY OBSERVATION: Gradient norms across blocks")
    print(f"{'─'*60}")
    print(f"  If norms are similar across block[0]..block[{num_layers-1}],")
    print(f"  gradients flow well → residual connections are working!")
    print(f"  If block[0] norms were much smaller than block[{num_layers-1}],")
    print(f"  that would indicate vanishing gradients.")

    # ==================================================================
    # PHASE 4: GENERATION BEFORE TRAINING (baseline — random output)
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# PHASE 4: GENERATION BEFORE TRAINING (untrained model)")
    print(f"{'#'*60}")

    # We'll use the FIRST sample from our batch as the "memorisation target".
    # The model should learn to reproduce this exact sequence.
    target_sequence = input_ids[0:1]  # (1, 20)
    # Use the first 5 tokens as prompt, the remaining 15 as expected output.
    prompt_len = 5
    prompt = target_sequence[:, :prompt_len]              # (1, 5)
    expected_continuation = target_sequence[0, prompt_len:]  # (15,)

    print(f"\n  Target sequence:        {target_sequence[0].tolist()}")
    print(f"  Prompt (first {prompt_len}):       {prompt[0].tolist()}")
    print(f"  Expected continuation:  {expected_continuation.tolist()}")

    # Generate from the untrained model — should be random garbage
    generated_before = model.generate(
        prompt.clone(),
        max_new_tokens=seq_len - prompt_len,
        temperature=0.8,
    )
    gen_before_tokens = generated_before[0, prompt_len:].tolist()
    expected_tokens = expected_continuation.tolist()

    # Count how many tokens match
    matches_before = sum(g == e for g, e in zip(gen_before_tokens, expected_tokens))
    print(f"\n  Generated (untrained):  {gen_before_tokens}")
    print(f"  Expected:               {expected_tokens}")
    print(f"  Matches: {matches_before}/{len(expected_tokens)}")
    print(f"  (Random chance ≈ {1/vocab_size:.4f} per token = ~0 matches)")

    # ==================================================================
    # STEP 11 — Training loop  (teach model to memorise the sequence)
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# STEP 11 — TRAINING LOOP (AdamW, memorise one sequence)")
    print(f"{'#'*60}")

    # ── Set up the optimizer ─────────────────────────────────────────
    # AdamW is the standard optimizer for transformers.
    # lr=1e-3 is aggressive but fine for memorisation of a tiny batch.
    lr = 1e-3
    weight_decay = 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"\n  Optimizer:    AdamW (lr={lr}, weight_decay={weight_decay})")
    print(f"  Training data: {input_ids.shape}  ({batch_size} sequences of {seq_len} tokens)")
    print(f"  Goal: memorise these sequences (overfit on purpose)")

    # ── Training loop ────────────────────────────────────────────────
    # Each step:
    #   1. Forward:  model(input_ids) → logits
    #   2. Loss:     cross-entropy(logits shifted by 1, targets)
    #   3. Backward: loss.backward() → compute all gradients
    #   4. Step:     optimizer.step() → update all weights
    #   5. Zero:     optimizer.zero_grad() → reset for next iteration
    #
    # We train on the SAME batch every step — this is OVERFITTING.
    # That's intentional: we want to prove the model CAN learn.
    # In real training you'd use different batches each step.

    num_steps = 50
    initial_loss = None

    print(f"  Training for {num_steps} steps...\n")

    for step in range(num_steps):
        # ── 1. Forward pass (verbose=False to suppress shape logs) ───
        logits = model(input_ids, verbose=False)
        # logits: (batch_size, seq_len, vocab_size)

        # ── 2. Loss (next-token prediction) ──────────────────────────
        # We use PyTorch's fused CrossEntropyLoss for speed.
        # It does exactly what our manual language_model_loss() does:
        #   shift logits by 1, flatten, cross-entropy.
        shift_logits = logits[:, :-1, :].reshape(-1, vocab_size)
        shift_targets = input_ids[:, 1:].reshape(-1)
        step_loss = nn.CrossEntropyLoss()(shift_logits, shift_targets)

        if initial_loss is None:
            initial_loss = step_loss.item()

        # ── 3. Backward ──────────────────────────────────────────────
        step_loss.backward()

        # ── 4. Optimizer step ────────────────────────────────────────
        optimizer.step()

        # ── 5. Zero gradients ────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)

        # ── Compute accuracy (how many next tokens predicted correctly) ──
        with torch.no_grad():
            predictions = logits[:, :-1, :].argmax(dim=-1)  # (batch, seq-1)
            targets_check = input_ids[:, 1:]                 # (batch, seq-1)
            correct = (predictions == targets_check).sum().item()
            total = targets_check.numel()
            accuracy = correct / total

        # Log every 5 steps + first and last
        if step % 5 == 0 or step == num_steps - 1:
            print(f"  Step {step+1:>3}/{num_steps}  "
                  f"loss={step_loss.item():.4f}  "
                  f"accuracy={accuracy:.1%}  "
                  f"({correct}/{total} tokens correct)")

    print(f"\n  ── Training summary ──")
    print(f"  Initial loss: {initial_loss:.4f}  "
          f"(expected random: {math.log(vocab_size):.4f})")
    print(f"  Final loss:   {step_loss.item():.4f}")
    print(f"  Final accuracy: {accuracy:.1%}")

    if accuracy == 1.0:
        print(f"  ✓ 100% accuracy! The model memorised all {total} "
              f"next-token predictions.")
    elif accuracy > 0.9:
        print(f"  ✓ Nearly there — {accuracy:.1%} accuracy.")
    else:
        print(f"  ⚠ Accuracy is {accuracy:.1%} — may need more training steps.")

    # ==================================================================
    # PHASE 6: GENERATION AFTER TRAINING — the proof!
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# PHASE 6: GENERATION AFTER TRAINING")
    print(f"{'#'*60}")

    # Now we give the model the same prompt (first 5 tokens) and see
    # if it can reproduce the rest of the sequence from memory.
    #
    # This is the KEY test:
    #   BEFORE training: random tokens (prob ≈ 0.001 per token)
    #   AFTER training:  exact reproduction (prob → 1.0 per token)

    print(f"\n  Prompt:    {prompt[0].tolist()}")
    print(f"  Expected:  {expected_tokens}")

    # Use temperature=0.01 (nearly greedy) to get the most likely tokens.
    # With a well-trained model, the correct next token should have
    # probability ~1.0, so temperature barely matters.
    generated_after = model.generate(
        prompt.clone(),
        max_new_tokens=seq_len - prompt_len,
        temperature=0.01,
    )
    gen_after_tokens = generated_after[0, prompt_len:].tolist()

    matches_after = sum(g == e for g, e in zip(gen_after_tokens, expected_tokens))

    print(f"\n  ── Comparison ──")
    print(f"  Before training: {gen_before_tokens}")
    print(f"  After training:  {gen_after_tokens}")
    print(f"  Expected:        {expected_tokens}")
    print(f"")
    print(f"  Matches BEFORE training: {matches_before}/{len(expected_tokens)}")
    print(f"  Matches AFTER training:  {matches_after}/{len(expected_tokens)}")

    if matches_after == len(expected_tokens):
        print(f"\n  ✓ PERFECT! The model reproduces the exact sequence!")
        print(f"    This proves the full pipeline works end-to-end:")
        print(f"    Token Embedding → Positional Embedding → "
              f"{num_layers} Decoder Blocks")
        print(f"    → LayerNorm → LM Head → CrossEntropy → "
              f"backward() → AdamW.step()")
    elif matches_after > matches_before:
        print(f"\n  ✓ The model learned! {matches_after} matches "
              f"(up from {matches_before}).")
        print(f"    More training steps or lower temperature may reach 100%.")
    else:
        print(f"\n  ⚠ Generation didn't improve — try more training steps.")

    # ==================================================================
    # COMPLETE
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# COMPLETE — FULL MINI-GPT PIPELINE")
    print(f"{'#'*60}")
    print(f"\n  Model:        ManualMiniGPT ({total_params:,} params)")
    print(f"  Architecture: {num_layers} decoder blocks, {num_heads} heads, "
          f"d_model={d_model}")
    print(f"  Vocab:        {vocab_size} tokens")
    print(f"  Training:     {num_steps} steps, AdamW (lr={lr})")
    print(f"  Loss:         {initial_loss:.4f} → {step_loss.item():.4f}")
    print(f"  Accuracy:     0% → {accuracy:.0%}")
    print(f"  Generation:   {matches_before}/{len(expected_tokens)} → "
          f"{matches_after}/{len(expected_tokens)} tokens correct")
