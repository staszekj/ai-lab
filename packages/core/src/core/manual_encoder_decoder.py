"""
Educational PyTorch: Encoder-Decoder Transformer (T5 / BART style)
====================================================================

Builds a complete seq2seq model by stacking:
    1. N encoder blocks  (ManualTransformerEncoderBlock, bidirectional)
    2. N decoder blocks  (ManualDecoderBlock, causal + cross-attention)

Key concepts:

    TRAINING (teacher forcing):
        src_ids  →  encoder  →  encoder_output
        tgt_ids[:-1]  →  decoder (with encoder_output)  →  logits
        loss = cross_entropy(logits, tgt_ids[1:])

        The decoder receives the ground-truth target tokens shifted by one
        ("teacher forcing").  At each position t it predicts token t+1.

    GENERATION (autoregressive):
        src_ids  →  encoder  →  encoder_output   (computed ONCE)
        tgt = [<BOS>]
        while not <EOS>:
            logits = decoder(tgt, encoder_output)
            next_token = sample(logits[:, -1, :])
            tgt.append(next_token)

        The encoder runs ONCE; its output is cached and reused at every
        decoder step.  Only the decoder needs to run repeatedly.

Key shapes:
    src_ids        : (batch, src_len)
    tgt_ids        : (batch, tgt_len)
    encoder_output : (batch, src_len, d_model)
    logits         : (batch, tgt_len, vocab_size)

Comparison with ManualMiniGPT (decoder-only):
    ManualMiniGPT:         one sequence → causal self-attention only
    ManualEncoderDecoder:  two sequences → encoder + decoder + cross-attention
    ManualMiniGPT uses ManualTransformerEncoderBlock with causal mask.
    ManualEncoderDecoder uses the same block for the encoder (no mask)
    and adds ManualDecoderBlock (with cross-attention) for the decoder.
"""

import math
import torch
import torch.nn as nn

# Reuse the encoder block from manual_transformer_block.py — unchanged.
from core.manual_transformer_block import ManualTransformerEncoderBlock
# New decoder block with cross-attention.
from core.manual_decoder_block import ManualDecoderBlock


class ManualEncoderDecoder(nn.Module):
    """
    A complete Encoder-Decoder Transformer.

    Components:
        Shared token embedding     (vocab_size → d_model)
        Shared positional embedding (max_seq_len → d_model)
        Encoder: N × ManualTransformerEncoderBlock (no mask)
        Encoder final LayerNorm
        Decoder: N × ManualDecoderBlock (causal mask + cross-attention)
        Decoder final LayerNorm
        LM Head: d_model → vocab_size

    Parameters
    ----------
    vocab_size    : size of the shared vocabulary
    max_seq_len   : maximum sequence length (for both src and tgt)
    d_model       : embedding / hidden dimension
    num_heads     : number of attention heads per block
    d_ff          : FFN hidden dimension
    num_layers    : number of encoder layers AND decoder layers
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

        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model     = d_model
        self.num_heads   = num_heads
        self.num_layers  = num_layers

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        # ==============================================================
        # SHARED EMBEDDINGS
        # ==============================================================
        # Source and target share the same token embedding table.
        # This is standard in models with a shared vocabulary (e.g. subword
        # tokenisation with SentencePiece / BPE), and reduces parameter count.
        #
        # Shape: (vocab_size, d_model)
        self.token_embedding      = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)

        # ==============================================================
        # ENCODER STACK
        # ==============================================================
        # N identical ManualTransformerEncoderBlock instances.
        # No mask is passed → full bidirectional attention over the source.
        # This is the same block used in ManualMiniGPT, only without a mask.
        self.encoder_blocks = nn.ModuleList([
            ManualTransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
            )
            for _ in range(num_layers)
        ])
        self.encoder_final_norm = nn.LayerNorm(d_model)

        # ==============================================================
        # DECODER STACK
        # ==============================================================
        # N ManualDecoderBlock instances.
        # Each block receives:
        #   - x: current decoder hidden states  (batch, tgt_len, d_model)
        #   - encoder_output: full encoder output (batch, src_len, d_model)
        # The cross-attention inside each block uses K, V from encoder_output.
        self.decoder_blocks = nn.ModuleList([
            ManualDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
            )
            for _ in range(num_layers)
        ])
        self.decoder_final_norm = nn.LayerNorm(d_model)

        # ==============================================================
        # LM HEAD
        # ==============================================================
        # Projects the decoder's d_model-dimensional output to vocabulary logits.
        # Shape: (d_model, vocab_size)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Causal mask — identical to ManualMiniGPT._create_causal_mask
    # ------------------------------------------------------------------
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular matrix of -inf values.
        Token at position i can only attend to positions 0..i.

        Returns: (seq_len, seq_len)
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1,
        )

    # ------------------------------------------------------------------
    # encode()  — run the source sequence through all encoder blocks
    # ------------------------------------------------------------------
    def encode(
        self,
        src_ids: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Encode the source sequence.

        Parameters
        ----------
        src_ids : (batch, src_len) — integer token IDs

        Returns
        -------
        encoder_output : (batch, src_len, d_model)
        """
        batch_size, src_len = src_ids.shape
        assert src_len <= self.max_seq_len

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# ENCODER FORWARD PASS")
            print(f"# src_ids: {src_ids.shape}")
            print(f"{'#'*60}")

        # ==============================================================
        # STEP 1 — Token embedding lookup
        # ==============================================================
        position_ids = torch.arange(src_len, device=src_ids.device)
        token_emb    = self.token_embedding(src_ids)         # (batch, src_len, d_model)

        # ==============================================================
        # STEP 2 — Positional embedding lookup
        # ==============================================================
        pos_emb      = self.positional_embedding(position_ids)  # (src_len, d_model)

        # ==============================================================
        # STEP 3 — Combine token + position embeddings
        # ==============================================================
        x = token_emb + pos_emb                              # (batch, src_len, d_model)

        if verbose:
            print(f"Token + position embeddings: {x.shape}")

        # ==============================================================
        # STEP 5 — Pass through encoder blocks (no causal mask — bidirectional)
        # ==============================================================
        # (STEPs 1-3 done above; STEP 4 causal mask is decoder-only)
        for i, block in enumerate(self.encoder_blocks):
            if verbose:
                print(f"\n{'─'*60}")
                print(f"ENCODER BLOCK {i}")
                print(f"Input:  {x.shape}")
            x = block(x, attn_mask=None)
            if verbose:
                print(f"Output: {x.shape}  "
                      f"mean={x.mean().item():+.4f}  std={x.std().item():.4f}")

        # ==============================================================
        # STEP 6 — Final LayerNorm
        # ==============================================================
        encoder_output = self.encoder_final_norm(x)
        # encoder_output: (batch, src_len, d_model)

        if verbose:
            print(f"\nEncoder output (after final LayerNorm): {encoder_output.shape}")

        return encoder_output

    # ------------------------------------------------------------------
    # decode()  — run the target sequence through all decoder blocks
    # ------------------------------------------------------------------
    def decode(
        self,
        tgt_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Decode with cross-attention to encoder_output.

        Parameters
        ----------
        tgt_ids        : (batch, tgt_len) — target token IDs (teacher-forced)
        encoder_output : (batch, src_len, d_model) — from encode()

        Returns
        -------
        logits : (batch, tgt_len, vocab_size)
        """
        batch_size, tgt_len = tgt_ids.shape
        assert tgt_len <= self.max_seq_len

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# DECODER FORWARD PASS")
            print(f"# tgt_ids: {tgt_ids.shape}  encoder_output: {encoder_output.shape}")
            print(f"{'#'*60}")

        # ==============================================================
        # STEP 1 — Token embedding lookup
        # ==============================================================
        position_ids = torch.arange(tgt_len, device=tgt_ids.device)
        token_emb    = self.token_embedding(tgt_ids)          # (batch, tgt_len, d_model)

        # ==============================================================
        # STEP 2 — Positional embedding lookup
        # ==============================================================
        pos_emb      = self.positional_embedding(position_ids)  # (tgt_len, d_model)

        # ==============================================================
        # STEP 3 — Combine token + position embeddings
        # ==============================================================
        x = token_emb + pos_emb                               # (batch, tgt_len, d_model)

        if verbose:
            print(f"Token + position embeddings: {x.shape}")

        # ==============================================================
        # STEP 4 — Causal mask (each target token attends only to past tokens)
        # ==============================================================
        tgt_mask = self._create_causal_mask(tgt_len, device=tgt_ids.device)
        # tgt_mask: (tgt_len, tgt_len)

        if verbose:
            print(f"Causal mask (tgt): {tgt_mask.shape}")

        # ==============================================================
        # STEP 5 — Pass through decoder blocks (causal self-attn + cross-attn)
        # ==============================================================
        # encoder_output is passed into EVERY decoder block (same tensor reused).
        for i, block in enumerate(self.decoder_blocks):
            if verbose:
                print(f"\n{'─'*60}")
                print(f"DECODER BLOCK {i}")
                print(f"x: {x.shape}  encoder_output: {encoder_output.shape}")
            x = block(x, encoder_output, tgt_mask=tgt_mask)
            if verbose:
                print(f"Output: {x.shape}  "
                      f"mean={x.mean().item():+.4f}  std={x.std().item():.4f}")

        # ==============================================================
        # STEP 6 — Final LayerNorm
        # ==============================================================
        x = self.decoder_final_norm(x)

        # ==============================================================
        # STEP 7 — LM Head → vocabulary logits
        # ==============================================================
        logits = self.lm_head(x)
        # logits: (batch, tgt_len, vocab_size)

        if verbose:
            print(f"\nAfter decoder final LayerNorm: {x.shape}")
            print(f"LM Head logits: {logits.shape}")
            print(f"  → scores over {self.vocab_size} vocabulary tokens "
                  f"at each of {tgt_len} target positions")

        return logits

    # ------------------------------------------------------------------
    # forward()  — full pass used during training (teacher forcing)
    # ------------------------------------------------------------------
    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Full forward pass for training.

        Parameters
        ----------
        src_ids : (batch, src_len) — source token IDs
        tgt_ids : (batch, tgt_len) — target token IDs (teacher-forced)

        Returns
        -------
        logits : (batch, tgt_len, vocab_size)
        """
        encoder_output = self.encode(src_ids, verbose=verbose)
        logits         = self.decode(tgt_ids, encoder_output, verbose=verbose)
        return logits

    # ------------------------------------------------------------------
    # generate()  — autoregressive inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Seq2seq autoregressive generation.

        The encoder runs ONCE.  The decoder runs one step at a time,
        each time consuming the growing target sequence.

        Parameters
        ----------
        src_ids        : (1, src_len) — source tokens
        bos_id         : beginning-of-sequence token ID
        eos_id         : end-of-sequence token ID (stops generation)
        max_new_tokens : maximum number of tokens to generate
        temperature    : < 1.0 → more peaked; > 1.0 → more random

        Returns
        -------
        generated_ids : (1, generated_len)  — excludes <BOS>
        """
        print(f"\n{'#'*60}")
        print(f"# AUTOREGRESSIVE GENERATION  (seq2seq)")
        print(f"# src_ids: {src_ids.shape}  max_new_tokens: {max_new_tokens}")
        print(f"{'#'*60}")

        # ── Encode source ONCE ────────────────────────────────────────
        # This is the key advantage of encoder-decoder over decoder-only:
        # the encoder runs a single time and its output is cached.
        # In GPT-style models, every generation step re-processes the entire
        # growing context.  Here the source processing cost is fixed.
        encoder_output = self.encode(src_ids, verbose=False)
        # encoder_output: (1, src_len, d_model)
        print(f"\nEncoder output cached: {encoder_output.shape}")
        print(f"Decoding autoregressively ({max_new_tokens} max tokens)...")

        generated = torch.tensor([[bos_id]], device=src_ids.device)
        # generated: (1, 1) — starts with <BOS>

        for step in range(max_new_tokens):
            logits = self.decode(generated, encoder_output, verbose=False)
            # logits: (1, current_tgt_len, vocab_size)

            # Take only the last position's logits — the prediction for "what's next"
            next_token_logits = logits[:, -1, :] / temperature
            # next_token_logits: (1, vocab_size)

            probs      = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            # next_token: (1, 1)

            generated = torch.cat([generated, next_token], dim=1)
            # generated: (1, step + 2)

            print(f"  Step {step:3d}: token {next_token.item():5d}  "
                  f"(prob={probs[0, next_token.item()].item():.4f})  "
                  f"seq_len={generated.shape[1]}")

            if next_token.item() == eos_id:
                print(f"  → <EOS> reached, stopping.")
                break

        # Return without the leading <BOS>
        return generated[:, 1:]


# ══════════════════════════════════════════════════════════════════════
# Runnable example
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)

    VOCAB      = 50
    MAX_SEQ    = 32
    D_MODEL    = 24
    NUM_HEADS  = 4
    D_FF       = 96
    NUM_LAYERS = 2
    BATCH      = 2
    SRC_LEN    = 6
    TGT_LEN    = 5

    model = ManualEncoderDecoder(
        vocab_size=VOCAB,
        max_seq_len=MAX_SEQ,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
    )

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

    src_ids = torch.randint(0, VOCAB, (BATCH, SRC_LEN))
    tgt_ids = torch.randint(0, VOCAB, (BATCH, TGT_LEN))

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        logits = model.forward(src_ids, tgt_ids, verbose=False)

    print(f"src_ids:  {src_ids.shape}")
    print(f"tgt_ids:  {tgt_ids.shape}")
    print(f"logits:   {logits.shape}")
    assert logits.shape == (BATCH, TGT_LEN, VOCAB)
    print("Shape check passed ✓")

    # ==============================================================
    # STEP 9 — Loss
    # ==============================================================
    loss = nn.CrossEntropyLoss()(
        logits.reshape(-1, VOCAB),
        tgt_ids.reshape(-1),
    )
    print(f"Loss: {loss.item():.4f}")

    # ==============================================================
    # STEP 10 — Backward
    # ==============================================================
    loss.backward()
    print(f"Encoder block 0 W_Q grad norm: "
          f"{model.encoder_blocks[0].W_Q.weight.grad.norm().item():.4f}")
    print(f"Decoder block 0 cross_W_K grad norm: "
          f"{model.decoder_blocks[0].cross_W_K.weight.grad.norm().item():.4f}")
    print("Backward pass check passed ✓")
