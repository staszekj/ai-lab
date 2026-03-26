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

# Reuse our hand-built transformer block (now with mask support)
from core.manual_transformer_block import ManualTransformerEncoderBlock


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
    # PHASE 1: FORWARD PASS (through all 4 layers)
    # ==================================================================
    logits = model(input_ids, verbose=True)
    # logits: (batch_size, seq_len, vocab_size)  e.g. (2, 20, 1000)

    print(f"\n{'='*60}")
    print(f"FORWARD PASS COMPLETE")
    print(f"  Input:  {input_ids.shape}  (integer token IDs)")
    print(f"  Output: {logits.shape}  (logits over vocabulary)")
    print(f"{'='*60}")

    # ==================================================================
    # PHASE 2: LOSS COMPUTATION
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# PHASE 2: LOSS COMPUTATION")
    print(f"{'#'*60}")

    loss = language_model_loss(logits, input_ids)

    # ==================================================================
    # PHASE 3: BACKWARD PASS
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# PHASE 3: BACKWARD PASS")
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
    # PHASE 4: AUTOREGRESSIVE GENERATION
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# PHASE 4: AUTOREGRESSIVE GENERATION")
    print(f"{'#'*60}")

    # Start with a short "prompt" of 5 tokens
    prompt = torch.randint(0, vocab_size, (1, 5))
    print(f"\nPrompt: {prompt[0].tolist()}")
    print(f"  (random IDs — in a real model these would be tokenised text)")

    # Generate 15 new tokens
    generated = model.generate(
        prompt,
        max_new_tokens=15,
        temperature=0.8,
    )

    print(f"\nFull generated sequence: {generated[0].tolist()}")
    print(f"  Prompt tokens (given):    {prompt[0].tolist()}")
    print(f"  Generated tokens (new):   {generated[0, 5:].tolist()}")
    print(f"\n  Note: This is an UNTRAINED model, so the generated tokens")
    print(f"  are essentially random. After training on real text, the")
    print(f"  model would generate coherent continuations.")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"# SUMMARY: MINI-GPT ARCHITECTURE")
    print(f"{'#'*60}")
    print(f"""
    Input token IDs: (batch, seq_len)
           │
           ▼
    ┌─────────────────────────────┐
    │   Token Embedding           │  (vocab_size, d_model)
    │ + Positional Embedding      │  (max_seq_len, d_model)
    └─────────────┬───────────────┘
                  │  (batch, seq, {d_model})
                  ▼
    ┌─────────────────────────────┐
    │   Decoder Block 0           │  causal self-attention + FFN
    │   (Pre-LN + residual)       │
    └─────────────┬───────────────┘
                  │  (batch, seq, {d_model})
                  ▼
    ┌─────────────────────────────┐
    │   Decoder Block 1           │  same architecture, different weights
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │   Decoder Block 2           │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │   Decoder Block 3           │
    └─────────────┬───────────────┘
                  │  (batch, seq, {d_model})
                  ▼
    ┌─────────────────────────────┐
    │   Final LayerNorm           │
    └─────────────┬───────────────┘
                  │  (batch, seq, {d_model})
                  ▼
    ┌─────────────────────────────┐
    │   LM Head (Linear)          │  (d_model → vocab_size)
    └─────────────┬───────────────┘
                  │  (batch, seq, {vocab_size})
                  ▼
           Output logits
    """)
