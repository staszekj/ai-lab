"""
Educational PyTorch: Encoder-Decoder Transformer (T5 / BART style)
====================================================================

A complete, self-contained seq2seq Transformer in ONE file.

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

This file is the production reference implementation — it will be trained on real
degraded-types corpora on CUDA. For a hand-walked tour with a hard-coded tiny
example (`const enabled : string` → `"ON" | "OFF"`), tiny tensors and pretty
matrix prints, run:

    uv run --package core presentation-encoder-decoder

In-code comments below cross-reference the STEPs in that presentation script.
"""

import math
import torch
import torch.nn as nn


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
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Feed-forward network: expand to d_ff, GELU, compress back.
        # d_ff is conventionally 4 × d_model.
        self.ffn_linear1 = nn.Linear(d_model, d_ff)
        self.ffn_gelu    = nn.GELU()
        self.ffn_linear2 = nn.Linear(d_ff, d_model)

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
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model

        if verbose:
            print(f"\n{'='*60}")
            print(f"EncoderBlock | x: {x.shape}")

        # ── Pre-LN + multi-head self-attention ───────────────────────
        x_norm1 = self.layer_norm_1(x)
        # x_norm1: (batch, seq_len, d_model)
        if verbose:
            print(f"After LayerNorm 1:                 x_norm1:  {x_norm1.shape}")

        # Project to Q, K, V (each still d_model wide; head split is a reshape).
        Q = self.W_Q(x_norm1)
        K = self.W_K(x_norm1)
        V = self.W_V(x_norm1)
        # Q,K,V: (batch, seq_len, d_model)

        # Reshape into heads: (batch, seq, d_model) → (batch, heads, seq, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q,K,V: (batch, heads, seq_len, d_k)

        # Scaled dot-product attention.
        # scores = Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (batch, heads, seq_len, seq_len)

        if attn_mask is not None:
            # Additive mask: -inf at blocked positions → softmax → 0.
            scores = scores + attn_mask

        weights = torch.softmax(scores, dim=-1)
        # weights: (batch, heads, seq_len, seq_len)  [each row sums to 1]

        head_out = torch.matmul(weights, V)
        # head_out: (batch, heads, seq_len, d_k)

        # Concatenate heads: (batch, heads, seq, d_k) → (batch, seq, d_model)
        attn_out = head_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.W_O(attn_out)
        # attn_out: (batch, seq_len, d_model)

        # First residual connection.
        x1 = x + attn_out
        # x1: (batch, seq_len, d_model)
        if verbose:
            print(f"After attention + residual #1:     x1:       {x1.shape}")

        # ── Pre-LN + feed-forward network ────────────────────────────
        x1_norm = self.layer_norm_2(x1)
        ffn_hidden = self.ffn_linear1(x1_norm)         # (batch, seq, d_ff)
        ffn_hidden = self.ffn_gelu(ffn_hidden)
        ffn_out    = self.ffn_linear2(ffn_hidden)      # (batch, seq, d_model)

        output = x1 + ffn_out
        # output: (batch, seq_len, d_model)
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
        self.self_W_Q = nn.Linear(d_model, d_model)
        self.self_W_K = nn.Linear(d_model, d_model)
        self.self_W_V = nn.Linear(d_model, d_model)
        self.self_W_O = nn.Linear(d_model, d_model)

        # Sub-layer 2: cross-attention. K and V come from encoder_output.
        # These are SEPARATE matrices from the self-attention ones — they
        # learn different transformations.
        self.cross_W_Q = nn.Linear(d_model, d_model)
        self.cross_W_K = nn.Linear(d_model, d_model)
        self.cross_W_V = nn.Linear(d_model, d_model)
        self.cross_W_O = nn.Linear(d_model, d_model)

        # Sub-layer 3: feed-forward network.
        self.ffn_linear1 = nn.Linear(d_model, d_ff)
        self.ffn_gelu    = nn.GELU()
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
        # Decoder tokens attend to themselves and PAST decoder tokens only.
        x_norm1 = self.layer_norm_1(x)
        self_attn_out = self._multi_head_attention(
            Q_src=x_norm1, K_src=x_norm1, V_src=x_norm1,
            W_Q=self.self_W_Q, W_K=self.self_W_K,
            W_V=self.self_W_V, W_O=self.self_W_O,
            attn_mask=tgt_mask,  # causal
        )
        x1 = x + self_attn_out
        # x1: (batch, tgt_len, d_model)
        if verbose:
            print(f"After masked-self-attn + residual: x1:       {x1.shape}")

        # ── Sub-layer 2: cross-attention ─────────────────────────────
        # Each decoder query asks: "which encoder positions matter for
        # what I'm generating right now?" K and V come from encoder_output.
        # No causal mask: the decoder may attend to ALL encoder positions.
        x1_norm = self.layer_norm_2(x1)
        cross_attn_out = self._multi_head_attention(
            Q_src=x1_norm,         # Q from decoder
            K_src=encoder_output,  # K from encoder
            V_src=encoder_output,  # V from encoder
            W_Q=self.cross_W_Q, W_K=self.cross_W_K,
            W_V=self.cross_W_V, W_O=self.cross_W_O,
            attn_mask=src_mask,
        )
        x2 = x1 + cross_attn_out
        # x2: (batch, tgt_len, d_model)
        if verbose:
            print(f"After cross-attn + residual:       x2:       {x2.shape}")

        # ── Sub-layer 3: feed-forward network ────────────────────────
        x2_norm = self.layer_norm_3(x2)
        ffn_hidden = self.ffn_linear1(x2_norm)
        ffn_hidden = self.ffn_gelu(ffn_hidden)
        ffn_out    = self.ffn_linear2(ffn_hidden)

        output = x2 + ffn_out
        # output: (batch, tgt_len, d_model)
        if verbose:
            print(f"After FFN + residual:              output:   {output.shape}")
            print(f"{'='*60}")

        return output


# ══════════════════════════════════════════════════════════════════════
# ENCODER-DECODER MODEL — full seq2seq Transformer
# ══════════════════════════════════════════════════════════════════════

class ManualEncoderDecoder(nn.Module):
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
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    # ------------------------------------------------------------------
    # Causal mask: upper-triangular -inf so position i attends only to 0..i.
    # → presentation STEP 3: 3×3 causal mask visualization with -inf cells.
    # ------------------------------------------------------------------
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
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
        pos_emb      = self.positional_embedding(position_ids)    # (src_len, d_model)
        x = token_emb + pos_emb                                   # (batch, src_len, d_model)

        # Run through every encoder block (no mask → bidirectional).
        for i, block in enumerate(self.encoder_blocks):
            if verbose:
                print(f"\n── Encoder block {i} ──")
            x = block(x, attn_mask=None, verbose=verbose)

        encoder_output = self.encoder_final_norm(x)
        # encoder_output: (batch, src_len, d_model)
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
        """
        batch_size, tgt_len = tgt_ids.shape
        assert tgt_len <= self.max_seq_len

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# DECODER FORWARD PASS | tgt_ids: {tgt_ids.shape}  encoder_output: {encoder_output.shape}")
            print(f"{'#'*60}")

        # Token + positional embedding lookup.
        position_ids = torch.arange(tgt_len, device=tgt_ids.device)
        token_emb    = self.token_embedding(tgt_ids)
        pos_emb      = self.positional_embedding(position_ids)
        x = token_emb + pos_emb
        # x: (batch, tgt_len, d_model)

        # Causal mask blocks the decoder from attending to future tokens.
        tgt_mask = self._create_causal_mask(tgt_len, device=tgt_ids.device)
        # tgt_mask: (tgt_len, tgt_len)

        # Run through every decoder block. encoder_output is reused as-is
        # in every block's cross-attention sub-layer.
        for i, block in enumerate(self.decoder_blocks):
            if verbose:
                print(f"\n── Decoder block {i} ──")
            x = block(x, encoder_output, tgt_mask=tgt_mask, verbose=verbose)

        x = self.decoder_final_norm(x)
        logits = self.lm_head(x)
        # logits: (batch, tgt_len, vocab_size)
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
        # This is the key efficiency advantage of encoder-decoder over
        # decoder-only models, which re-process the entire growing context.
        encoder_output = self.encode(src_ids, verbose=False)

        generated = torch.tensor([[bos_id]], device=src_ids.device)
        # generated: (1, 1) starting with <BOS>

        for step in range(max_new_tokens):
            logits = self.decode(generated, encoder_output, verbose=False)
            # logits: (1, current_tgt_len, vocab_size)

            # Take only the LAST position's logits (prediction for next token).
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
