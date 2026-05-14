"""
ASCII Art Presentation: Encoder-Decoder Transformer
====================================================

ONE hard-coded training example, traced step-by-step through every layer:

    SOURCE  (encoder input):    const enabled : string
    TARGET  (decoder output):   ON | OFF

    Goal: teach the model that for the variable named `enabled` annotated
    with the wide type `string`, the correct refined type is the literal
    union `"ON" | "OFF"`.

Training set size = 1.  Vocabulary size = 12.
Every STEP below assumes this exact pair — no other examples exist.

Steps in this file (matching encoder_decoder_model.py):
    STEP 1 — Tokenization & embeddings  (source side)
    STEP 2 — Encoder forward pass       (bidirectional self-attention)
    STEP 3 — Decoder forward pass       (causal self-attn + cross-attn + FFN)
    STEP 4 — LM Head & logits           (project to vocabulary)
    STEP 5 — Cross-entropy loss
    STEP 6 — Backward pass              (gradient flow through cross-attn)
    STEP 7 — Training loop              (200 steps on the single example)
    STEP 8 — Generation                 (autoregressive decode after training)

Run:  uv run --package core presentation-encoder-decoder
"""

import math
import torch
import torch.nn as nn
import io
import contextlib


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def fmt(val: float, width: int = 7) -> str:
    s = f"{val:+.2f}"
    return s.rjust(width)


def print_matrix(name: str, t: torch.Tensor, indent: int = 4):
    pad = " " * indent
    rows, cols = t.shape
    print(f"{pad}{name}  ({rows}×{cols})")
    print(f"{pad}┌{'─' * (8 * cols + 1)}┐")
    for r in range(rows):
        vals = " ".join(fmt(t[r, c].item()) for c in range(cols))
        print(f"{pad}│ {vals} │")
    print(f"{pad}└{'─' * (8 * cols + 1)}┘")


def section(title: str):
    print(f"\n{'━' * 72}")
    print(f"  {title}")
    print(f"{'━' * 72}")


def step_header(num: str, title: str):
    width = max(56 - len(num) - len(title), 2)
    print(f"\n  ┌─── STEP {num} {'─' * width} {title} ───┐")


# ══════════════════════════════════════════════════════════════════════
# Configuration (HARD-CODED for "const enabled : string" → "ON | OFF")
# ══════════════════════════════════════════════════════════════════════

VOCAB      = 12
D_MODEL    = 6
HEADS      = 3
D_K        = D_MODEL // HEADS   # = 2
D_FF       = 12
NUM_LAYERS = 2
MAX_SEQ    = 16
BATCH      = 1
SRC_SEQ    = 4   # const enabled : string
TGT_SEQ    = 3   # decoder INPUT length (predicts 3 next tokens)

torch.manual_seed(42)

# 12-word vocabulary. The TS literal types `"ON"` / `"OFF"` are stored as
# bare words; we add quotes only when printing them as TS source code.
VOCAB_WORDS = [
    "<pad>", "<bos>",                    # 0  1
    "const", "let", "var",               # 2  3  4
    "enabled", ":", "string", "number",  # 5  6  7  8
    "ON", "|", "OFF",                    # 9  10 11
]
word2id = {w: i for i, w in enumerate(VOCAB_WORDS)}
id2word = VOCAB_WORDS

SRC_SENTENCE = "const enabled : string"
TGT_SENTENCE = "<bos> ON | OFF"


# ══════════════════════════════════════════════════════════════════════
# Architecture overview
# ══════════════════════════════════════════════════════════════════════

section("ARCHITECTURE — Encoder-Decoder Transformer")
print(f"""
    The whole presentation traces ONE pair end-to-end:

        ┌───────────────────────────────┐              ┌─────────────────────┐
        │  SOURCE                       │   train      │   TARGET            │
        │  const enabled : string       │ ===========> │   ON | OFF          │
        │  (degraded TS code, 4 tokens) │              │  (refined TS type)  │
        └───────────────────────────────┘              └─────────────────────┘

    Teacher forcing splits the target into INPUT and TARGET halves so the
    decoder is trained to predict the next token at every position:

        full target sequence:     <bos>   ON    |     OFF
        decoder INPUT  tgt[:-1]:  <bos>   ON    |             ← model sees these
        decoder TARGET tgt[1:]:           ON    |     OFF     ← model predicts these

    ┌──────────────────────────────────────────────────────────────────────┐
    │                        ENCODER                                       │
    │                                                                      │
    │  "const enabled : string"  →  Embedding + Pos                        │
    │    →  {NUM_LAYERS}×EncoderBlock (bidirectional, NO mask)                       │
    │    →  encoder_output  shape: (1, 4, {D_MODEL})                                │
    └──────────────────────────────────────┬───────────────────────────────┘
                                           │
                       encoder_output (the "memory" of the source)
                                           │
                              ┌────────────┴────────────┐
                              │   K, V for cross-attn   │
                              │   (same tensor fed to   │
                              │    EVERY decoder block) │
                              └────────────┬────────────┘
                                           │
    ┌──────────────────────────────────────┼───────────────────────────────┐
    │                        DECODER       │                               │
    │                                      │                               │
    │  "<bos> ON |"  →  Embedding + Pos    │                               │
    │    →  {NUM_LAYERS}×DecoderBlock                  │                               │
    │        - causal self-attn            │                               │
    │        - cross-attn  ←───────────────┘                               │
    │        - FFN                                                         │
    │    →  LM Head  →  logits  →  predict "ON", "|", "OFF"                │
    └──────────────────────────────────────────────────────────────────────┘

    Tiny by design — every tensor fits on screen:
      d_model={D_MODEL}, num_heads={HEADS}, d_k={D_K}, d_ff={D_FF}, num_layers={NUM_LAYERS}
""")


# ══════════════════════════════════════════════════════════════════════
# Build the model
# ══════════════════════════════════════════════════════════════════════

from core.encoder_decoder_model import EncoderDecoderModel

model = EncoderDecoderModel(EncoderDecoderModel.Config(
    vocab_size=VOCAB,
    max_seq_len=MAX_SEQ,
    d_model=D_MODEL,
    num_heads=HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
))

total_params = sum(p.numel() for p in model.parameters())
enc_block_p  = sum(p.numel() for p in model.encoder_blocks[0].parameters())
dec_block_p  = sum(p.numel() for p in model.decoder_blocks[0].parameters())

section("MODEL PARAMETERS")
print(f"""
    Total parameters: {total_params:,}

    Per-component breakdown:
      Shared token embedding     ({VOCAB}×{D_MODEL}):           {model.token_embedding.weight.numel():>5,}
      Shared position embedding  ({MAX_SEQ}×{D_MODEL}):           {model.positional_embedding.weight.numel():>5,}
      Encoder blocks ({NUM_LAYERS} × {enc_block_p}):                 {NUM_LAYERS * enc_block_p:>5,}
      Encoder final LayerNorm:                       {sum(p.numel() for p in model.encoder_final_norm.parameters()):>5,}
      Decoder blocks ({NUM_LAYERS} × {dec_block_p}):                {NUM_LAYERS * dec_block_p:>5,}
      Decoder final LayerNorm:                       {sum(p.numel() for p in model.decoder_final_norm.parameters()):>5,}
      LM Head ({D_MODEL}→{VOCAB}):                              {model.lm_head.weight.numel():>5,}

    Each DecoderBlock has ~{dec_block_p / enc_block_p:.2f}× the parameters of an EncoderBlock
    because it carries TWO sets of attention matrices (self + cross) vs ONE.
""")


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Tokenization & embeddings (source side)
# ══════════════════════════════════════════════════════════════════════

src_ids    = torch.tensor([[word2id[w] for w in SRC_SENTENCE.split()]])  # (1, 4)
tgt_ids    = torch.tensor([[word2id[w] for w in TGT_SENTENCE.split()]])  # (1, 4)
tgt_input  = tgt_ids[:, :-1]   # "<bos> ON |"   (1, 3) — decoder sees this
tgt_target = tgt_ids[:, 1:]    # "ON | OFF"     (1, 3) — decoder predicts this

step_header("1", "Tokenization & embeddings (source side)")
print(f"""
    SOURCE STRING:    "const enabled : string"

    1a. Split into tokens (whitespace-tokenized for clarity):

           position:    0        1         2     3
           token:    "const" "enabled"   ":"  "string"

    1b. Look each token up in the 12-word vocabulary:

           ┌─────────┬────┐ ┌─────────┬────┐ ┌─────────┬────┐
           │ <pad>   │  0 │ │ enabled │  5 │ │ ON      │  9 │
           │ <bos>   │  1 │ │ :       │  6 │ │ |       │ 10 │
           │ const   │  2 │ │ string  │  7 │ │ OFF     │ 11 │
           │ let     │  3 │ │ number  │  8 │ │         │    │
           │ var     │  4 │ │         │    │ │         │    │
           └─────────┴────┘ └─────────┴────┘ └─────────┴────┘

    1c. Result: src_ids = [[2, 5, 6, 7]]   shape: {tuple(src_ids.shape)}

    1d. Each ID picks one row from the {VOCAB}×{D_MODEL} token_embedding table.
        We also build pos_ids = [0,1,2,3] and pick rows from positional_embedding.
        Final encoder input  enc_x = token_emb + pos_emb,  shape (1, 4, {D_MODEL}).

           src_ids   pos_ids        token_emb         pos_emb         enc_x
           ┌───┐     ┌───┐         ┌──────────┐     ┌──────────┐    ┌──────────┐
           │ 2 │     │ 0 │   →     │ row[2]   │  +  │ row[0]   │  = │ vec_0    │
           │ 5 │     │ 1 │         │ row[5]   │     │ row[1]   │    │ vec_1    │
           │ 6 │     │ 2 │         │ row[6]   │     │ row[2]   │    │ vec_2    │
           │ 7 │     │ 3 │         │ row[7]   │     │ row[3]   │    │ vec_3    │
           └───┘     └───┘         └──────────┘     └──────────┘    └──────────┘
                                       (4×6)            (4×6)           (4×6)
""")

src_pos_ids   = torch.arange(SRC_SEQ)
src_token_emb = model.token_embedding(src_ids)           # (1, 4, 6)
src_pos_emb   = model.positional_embedding(src_pos_ids)   # (4, 6)
enc_x         = src_token_emb + src_pos_emb               # (1, 4, 6)

print(f"    src_ids:        {src_ids.tolist()}   shape={tuple(src_ids.shape)}")
print(f"    pos_ids:        {src_pos_ids.tolist()}     shape={tuple(src_pos_ids.shape)}")
print(f"    enc_x = token_emb + pos_emb,  shape={tuple(enc_x.shape)}\n")
print_matrix('    enc_x[batch=0]  (rows = "const","enabled",":","string")', enc_x[0])


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Encoder forward pass (bidirectional self-attention)
# ══════════════════════════════════════════════════════════════════════

step_header("2", "Encoder — bidirectional attention over source")
print(f"""
    Now we run enc_x through {NUM_LAYERS} EncoderBlocks.  Inside each block:

        ┌─────────────────────────────────────────────────────────────────┐
        │  EncoderBlock  (Pre-LN, two sub-layers)                         │
        │                                                                 │
        │   x  ──┬─►  LayerNorm ─►  Multi-Head Self-Attention  ──►  +  ──┐│
        │       │                       (NO causal mask)                 ││
        │       └────────────── residual ───────────────────────────────►┤│
        │                                                                ▼│
        │   x' ──┬─►  LayerNorm ─►  FFN (Linear → GELU → Linear)  ──►   +│
        │       │                                                        ││
        │       └────────────── residual ───────────────────────────────►┘│
        │                                                                 │
        │   shape preserved everywhere: (1, 4, {D_MODEL})                          │
        └─────────────────────────────────────────────────────────────────┘

    NO MASK means every source token can see every other source token.
    For example, when computing attention for "string" at position 3,
    the model can directly look at "enabled" at position 1 — that's how
    it can learn to associate the identifier with its type annotation.

    Attention pattern (4×4, no mask — every cell allowed):

                   const  enabled   :    string
           const  [   ✓      ✓      ✓      ✓   ]
         enabled  [   ✓      ✓      ✓      ✓   ]
              :   [   ✓      ✓      ✓      ✓   ]
          string  [   ✓      ✓      ✓      ✓   ]   ← can attend to "enabled"
""")

buf = io.StringIO()
for i, enc_block in enumerate(model.encoder_blocks):
    with contextlib.redirect_stdout(buf):
        enc_x = enc_block(enc_x, attn_mask=None)
    print(f"    After EncoderBlock {i}:  shape={tuple(enc_x.shape)}  "
          f"mean={enc_x.mean().item():+.4f}  std={enc_x.std().item():.4f}")

encoder_output = model.encoder_final_norm(enc_x)
print(f"    After final LayerNorm — encoder_output:  shape={tuple(encoder_output.shape)}\n")
print_matrix('    encoder_output[batch=0]  (one row per source token)', encoder_output[0])
print(f"""
    This tensor is the "memory" of "const enabled : string".
    Each of the 4 rows is a {D_MODEL}-dim contextualized vector — row 1
    ("enabled") has been mixed with information from "const", ":" and
    "string" via self-attention.

    encoder_output is the SAME tensor that gets passed (as K and V) to
    every cross-attention sub-layer in every decoder block.
""")


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Decoder forward pass
# ══════════════════════════════════════════════════════════════════════

step_header("3", "Decoder — causal self-attn + cross-attn + FFN")
print(f"""
    Decoder input (teacher-forced):  "<bos> ON |"     shape: {tuple(tgt_input.shape)}
    Decoder target:                  "ON | OFF"       shape: {tuple(tgt_target.shape)}

    The decoder runs the same embedding lookup as STEP 1, but on the
    target side:

           tgt_input    pos_ids     token_emb       pos_emb       dec_x
           ┌───┐        ┌───┐       ┌─────────┐    ┌─────────┐   ┌─────────┐
           │ 1 │ <bos>  │ 0 │   →   │ row[1]  │ +  │ row[0]  │ = │ vec_0   │
           │ 9 │ ON     │ 1 │       │ row[9]  │    │ row[1]  │   │ vec_1   │
           │10 │ |      │ 2 │       │ row[10] │    │ row[2]  │   │ vec_2   │
           └───┘        └───┘       └─────────┘    └─────────┘   └─────────┘
                                       (3×6)         (3×6)          (3×6)

    Each of the {NUM_LAYERS} DecoderBlocks then runs THREE sub-layers:

        ┌───────────────────────────────────────────────────────────────────┐
        │  DecoderBlock  (Pre-LN, three sub-layers)                         │
        │                                                                   │
        │  x  ──┬─►  LN ─►  CAUSAL self-attn (Q,K,V from x)  ──► +  ──┐    │
        │      │             — uses {TGT_SEQ}×{TGT_SEQ} causal mask                  │    │
        │      └─── residual ─────────────────────────────────────────►┤    │
        │                                                              ▼    │
        │  x' ──┬─►  LN ─►  CROSS-ATTN                       ──► +  ──┐    │
        │      │              Q from x',                               │    │
        │      │              K and V from encoder_output  ◄──── feed  │    │
        │      │              attention shape: ({TGT_SEQ}×{SRC_SEQ}, NOT square!)        │    │
        │      └─── residual ─────────────────────────────────────────►┤    │
        │                                                              ▼    │
        │  x'' ─┬─►  LN ─►  FFN (Linear → GELU → Linear)     ──► +         │
        │      │                                                            │
        │      └─── residual ───────────────────────────────────────►       │
        └───────────────────────────────────────────────────────────────────┘

    Causal mask ({TGT_SEQ}×{TGT_SEQ}) for self-attention — upper triangle is -inf
    (token at row i can attend only to tokens at columns 0..i):

                    <bos>    ON     |
           <bos>  [   0    -inf  -inf ]   ← <bos> sees only <bos>
              ON  [   0      0   -inf ]   ← ON sees <bos>, ON
              |   [   0      0     0  ]   ← | sees <bos>, ON, |

    Cross-attention is NOT square — rows are 3 decoder tokens, cols are 4
    source tokens.  When generating "ON" the decoder looks up info about
    "enabled" by attending to the encoder_output row for "enabled".
""")

# Embeddings for decoder input
tgt_pos_ids   = torch.arange(tgt_input.shape[1])
tgt_token_emb = model.token_embedding(tgt_input)          # (1, 3, 6)
tgt_pos_emb   = model.positional_embedding(tgt_pos_ids)    # (3, 6)
dec_x         = tgt_token_emb + tgt_pos_emb                # (1, 3, 6)

tgt_len      = tgt_input.shape[1]
tgt_mask     = model._create_causal_mask(tgt_len, device=tgt_input.device)
tgt_words_in = ["<bos>", "ON", "|"]
src_words    = ["const", "enabled", ":", "string"]

print(f"    dec_x = token_emb + pos_emb,  shape={tuple(dec_x.shape)}\n")

dec_x_cur = dec_x.clone()
for i, dec_block in enumerate(model.decoder_blocks):
    print(f"    {'═' * 60}")
    print(f"    DECODER BLOCK {i}")
    print(f"      x:              shape={tuple(dec_x_cur.shape)}")
    print(f"      encoder_output: shape={tuple(encoder_output.shape)}")

    if i == 0:
        # Compute cross-attention weights manually so we can show them
        with torch.no_grad():
            # 1) self-attention sub-layer (so x1_d feeds cross-attn)
            x_n1 = dec_block.layer_norm_1(dec_x_cur)
            Q_s  = dec_block.self_W_Q(x_n1).view(BATCH, tgt_len, HEADS, D_K).transpose(1, 2)
            K_s  = dec_block.self_W_K(x_n1).view(BATCH, tgt_len, HEADS, D_K).transpose(1, 2)
            V_s  = dec_block.self_W_V(x_n1).view(BATCH, tgt_len, HEADS, D_K).transpose(1, 2)
            sc_s = torch.matmul(Q_s, K_s.transpose(-2, -1)) / math.sqrt(D_K) + tgt_mask
            wt_s = torch.softmax(sc_s, dim=-1)
            ho_s = torch.matmul(wt_s, V_s)
            c_s  = ho_s.transpose(1, 2).contiguous().view(BATCH, tgt_len, D_MODEL)
            x1_d = dec_x_cur + dec_block.self_W_O(c_s)

            # 2) cross-attention sub-layer (the interesting part)
            x1_n = dec_block.layer_norm_2(x1_d)
            Q_c  = dec_block.cross_W_Q(x1_n).view(BATCH, tgt_len, HEADS, D_K).transpose(1, 2)
            K_c  = dec_block.cross_W_K(encoder_output).view(BATCH, SRC_SEQ, HEADS, D_K).transpose(1, 2)
            sc_c = torch.matmul(Q_c, K_c.transpose(-2, -1)) / math.sqrt(D_K)
            wt_c = torch.softmax(sc_c, dim=-1)

        print(f"\n      Cross-attention weights, head 0  ({tgt_len}×{SRC_SEQ}):")
        print(f"        rows = decoder tokens: {tgt_words_in}")
        print(f"        cols = source tokens:  {src_words}")
        print_matrix("        cross_weights[h=0]", wt_c[0, 0], indent=8)

        print(f"\n      Decoder token → most attended source token (UNTRAINED — random):")
        for t, tw in enumerate(tgt_words_in):
            top = wt_c[0, 0, t].argmax().item()
            print(f"        \"{tw:<5}\" → \"{src_words[top]:<7}\"  "
                  f"(weight={wt_c[0,0,t,top].item():.3f})")
        print()

    with contextlib.redirect_stdout(buf):
        dec_x_cur = dec_block(dec_x_cur, encoder_output, tgt_mask=tgt_mask)
    print(f"      output:         shape={tuple(dec_x_cur.shape)}  "
          f"mean={dec_x_cur.mean().item():+.4f}  std={dec_x_cur.std().item():.4f}\n")


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — LM Head & logits
# ══════════════════════════════════════════════════════════════════════

step_header("4", "LM Head — project decoder hidden state to vocabulary")
print(f"""
    The decoder's last hidden state has shape (1, {TGT_SEQ}, {D_MODEL}).
    The LM Head is a single linear layer that turns each {D_MODEL}-dim vector
    into a {VOCAB}-dim logit vector — one score per vocabulary word:

        decoder_final_norm(x)        LM Head            logits
        ┌────────────┐               (Linear            ┌──────────────┐
        │  3 × {D_MODEL}     │  ──── multiplied ─────► │  3 × {VOCAB}      │
        │            │      by weight ({D_MODEL}×{VOCAB}))     │              │
        └────────────┘                                  └──────────────┘

    Then we read predictions row by row:

        position 0 (<bos>):  argmax over {VOCAB} logits  →  predicted next token
        position 1 (ON):     argmax over {VOCAB} logits  →  predicted next token
        position 2 (|):      argmax over {VOCAB} logits  →  predicted next token

    The TARGETS for these positions are: ["ON", "|", "OFF"].
""")

dec_final = model.decoder_final_norm(dec_x_cur)
logits    = model.lm_head(dec_final)
print(f"    dec_final shape: {tuple(dec_final.shape)}   logits shape: {tuple(logits.shape)}\n")

target_words = ["ON", "|", "OFF"]
for t in range(TGT_SEQ):
    probs     = torch.softmax(logits[0, t], dim=-1)
    target_id = tgt_target[0, t].item()
    target_p  = probs[target_id].item()
    pred_id   = probs.argmax().item()
    correct   = "✓" if pred_id == target_id else "✗"
    print(f"      pos {t} (\"{tgt_words_in[t]:<5}\")  "
          f"target=\"{target_words[t]:<3}\"  "
          f"P(target)={target_p:.4f}  "
          f"argmax=\"{id2word[pred_id]:<7}\"  {correct}")

print(f"\n    The model is UNTRAINED — predictions are essentially random.")
print(f"    Expected accuracy ≈ 1/{VOCAB} = {100/VOCAB:.1f}% per position.")


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Cross-entropy loss
# ══════════════════════════════════════════════════════════════════════

step_header("5", "Cross-entropy loss")
print(f"""
    For each of the {TGT_SEQ} target positions we compute:

        loss_t = -log P(target_t | context_t)

    and average over all positions:

        ┌────────────────────────────────────────────────────────────────┐
        │  position 0:  context "<bos>"      → target "ON"   → -log p_0  │
        │  position 1:  context "<bos> ON"   → target "|"    → -log p_1  │
        │  position 2:  context "<bos> ON |" → target "OFF"  → -log p_2  │
        │                                                                │
        │  loss = (-log p_0  +  -log p_1  +  -log p_2) / 3               │
        └────────────────────────────────────────────────────────────────┘

    Reshape trick used by nn.CrossEntropyLoss:
        logits      (1, {TGT_SEQ}, {VOCAB})  →  reshape  →  ({TGT_SEQ}, {VOCAB})
        tgt_target  (1, {TGT_SEQ})       →  reshape  →  ({TGT_SEQ},)

    Random baseline:  log({VOCAB}) ≈ {math.log(VOCAB):.4f}
""")

loss = nn.CrossEntropyLoss()(
    logits.reshape(-1, VOCAB),
    tgt_target.reshape(-1),
)
random_loss = math.log(VOCAB)
print(f"    Loss:             {loss.item():.6f}")
print(f"    Random baseline:  {random_loss:.6f}")
print(f"    Ratio:            {loss.item() / random_loss:.3f}  (≈ 1.0 for an untrained model)")


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — Backward pass (gradient flow through cross-attention)
# ══════════════════════════════════════════════════════════════════════

step_header("6", "Backward pass — gradient flow through cross-attention")
print(f"""
    A single loss.backward() call differentiates through the ENTIRE graph.
    The crucial path: the encoder has NO loss of its own, but the decoder's
    cross_W_K and cross_W_V matrices multiply encoder_output, so gradients
    automatically propagate from the loss back into the encoder.

        loss
         └─► LM head weight
              └─► decoder_final_norm
                   └─► DecoderBlock {NUM_LAYERS - 1}
                        ├─► FFN weights
                        ├─► cross_W_O / cross_W_Q
                        └─► cross_W_K, cross_W_V       ← multiplied with K,V
                             └─► encoder_output         ← gradients flow BACK
                                  └─► encoder_final_norm
                                       └─► EncoderBlock {NUM_LAYERS - 1}
                                            └─► EncoderBlock 0
                                                 └─► token_embedding,
                                                     positional_embedding

    So that ONE backward() trains both the encoder and the decoder.
""")

model.zero_grad()
enc_out2 = model.encode(src_ids, verbose=False)
logits2  = model.decode(tgt_input, enc_out2, verbose=False)
loss2    = nn.CrossEntropyLoss()(
    logits2.reshape(-1, VOCAB),
    tgt_target.reshape(-1),
)
loss2.backward()
print(f"    Loss (recomputed for backward): {loss2.item():.6f}\n")
print(f"    Gradient norms after loss.backward():\n")

param_groups = [
    ("Shared embeddings", [
        ("token_embedding.weight",      model.token_embedding.weight),
        ("positional_embedding.weight", model.positional_embedding.weight),
    ]),
    ("Encoder Block 0  ← reached via cross-attn", [
        ("enc[0].W_Q.weight", model.encoder_blocks[0].W_Q.weight),
        ("enc[0].W_K.weight", model.encoder_blocks[0].W_K.weight),
        ("enc[0].W_V.weight", model.encoder_blocks[0].W_V.weight),
    ]),
    ("Decoder Block 0 — Self-Attention", [
        ("dec[0].self_W_Q.weight", model.decoder_blocks[0].self_W_Q.weight),
        ("dec[0].self_W_K.weight", model.decoder_blocks[0].self_W_K.weight),
        ("dec[0].self_W_V.weight", model.decoder_blocks[0].self_W_V.weight),
    ]),
    ("Decoder Block 0 — Cross-Attention  ← bridge to encoder", [
        ("dec[0].cross_W_Q.weight", model.decoder_blocks[0].cross_W_Q.weight),
        ("dec[0].cross_W_K.weight", model.decoder_blocks[0].cross_W_K.weight),
        ("dec[0].cross_W_V.weight", model.decoder_blocks[0].cross_W_V.weight),
    ]),
    ("LM Head", [
        ("lm_head.weight", model.lm_head.weight),
    ]),
]
for group_name, params in param_groups:
    print(f"      {group_name}:")
    for name, param in params:
        g = param.grad
        if g is None:
            print(f"        {name:<40} grad=None")
        else:
            print(f"        {name:<40} norm={g.norm().item():.4f}  max={g.abs().max().item():.4f}")
    print()

print(f"""    KEY OBSERVATION:
      enc[0].W_Q / W_K / W_V all have NON-ZERO gradients even though the
      encoder has no separate loss term.  The backward path is:

         loss → decoder → cross_W_K/cross_W_V → encoder_output → encoder

      One backward() call trains BOTH the encoder and the decoder.
""")


# ══════════════════════════════════════════════════════════════════════
# STEP 7 — Training loop on the SINGLE example
# ══════════════════════════════════════════════════════════════════════

step_header("7", "Training — overfit the ONE example (200 steps)")
print(f"""
    Training set size: 1 (a hard-coded constant for this presentation):

        SOURCE:         "const enabled : string"   IDs: [2, 5, 6, 7]
        DECODER INPUT:  "<bos> ON |"               IDs: [1, 9, 10]
        DECODER TARGET: "ON | OFF"                 IDs: [9, 10, 11]

    We repeat the SAME forward + backward + optimizer.step() 200 times.
    The model has no other examples to confuse it, so it memorizes the
    mapping completely.

        ┌─────────────────────────────────────────────────────────────┐
        │  for step in 1..200:                                        │
        │      enc_out = encode(src_ids)              # 4 source toks │
        │      logits  = decode(tgt_input, enc_out)   # predict 3     │
        │      loss    = CE(logits, tgt_target)                       │
        │      loss.backward()                                        │
        │      optimizer.step()                                       │
        │      optimizer.zero_grad()                                  │
        └─────────────────────────────────────────────────────────────┘

    Optimizer: AdamW (lr=0.01).
""")

torch.manual_seed(42)
train_model = EncoderDecoderModel(EncoderDecoderModel.Config(
    vocab_size=VOCAB,
    max_seq_len=MAX_SEQ,
    d_model=D_MODEL,
    num_heads=HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
))
optimizer_train = torch.optim.AdamW(train_model.parameters(), lr=0.01)

num_steps    = 200
initial_loss = None
_silent      = io.StringIO()

for step in range(num_steps):
    with contextlib.redirect_stdout(_silent):
        enc_out_tr = train_model.encode(src_ids, verbose=False)
        logits_tr  = train_model.decode(tgt_input, enc_out_tr, verbose=False)
    step_loss = nn.CrossEntropyLoss()(
        logits_tr.reshape(-1, VOCAB),
        tgt_target.reshape(-1),
    )
    if initial_loss is None:
        initial_loss = step_loss.item()
    step_loss.backward()
    optimizer_train.step()
    optimizer_train.zero_grad(set_to_none=True)

    with torch.no_grad():
        preds   = logits_tr.argmax(dim=-1)
        correct = (preds == tgt_target).sum().item()
        acc     = correct / tgt_target.numel()

    if step % 20 == 0 or step == num_steps - 1:
        pred_words = [id2word[i] for i in preds[0].tolist()]
        print(f"    step {step+1:>3}/{num_steps}  "
              f"loss={step_loss.item():.4f}  "
              f"acc={acc:.0%}  "
              f"predicted={pred_words}  "
              f"target={target_words}")

print(f"""
    Initial loss: {initial_loss:.4f}    (random baseline ≈ {random_loss:.4f})
    Final loss:   {step_loss.item():.4f}
    Final accuracy: {acc:.0%}
""")


# ══════════════════════════════════════════════════════════════════════
# STEP 8 — Generation (autoregressive decode after training)
# ══════════════════════════════════════════════════════════════════════

step_header("8", 'Generation — encode "const enabled : string" → decode "ON | OFF"')
print(f"""
    Inference is different from training:

      TRAINING (teacher forcing):
        We feed the WHOLE decoder input "<bos> ON |" at once and predict
        all 3 next tokens in parallel — because we already KNOW the target.

      INFERENCE (autoregressive):
        We don't know the target.  We start with just "<bos>" and grow
        the sequence one token at a time, feeding our own predictions
        back in:

          step 1:  decoder context = [<bos>]              → predict "ON"
          step 2:  decoder context = [<bos>, ON]          → predict "|"
          step 3:  decoder context = [<bos>, ON, |]       → predict "OFF"

      The encoder runs ONCE for "const enabled : string" — its output is
      cached and reused at every decoder step.  This is the efficiency
      advantage of encoder-decoder models over decoder-only models like
      GPT, which re-process the entire growing context every step.

    Real-world parallel: this is exactly what Phase 2 of the type-refiner
    pipeline does — encode the surrounding TS context, then autoregressively
    decode a refined type string.
""")

with torch.no_grad():
    with contextlib.redirect_stdout(_silent):
        train_enc_out = train_model.encode(src_ids, verbose=False)

    print(f"    Encoder ran ONCE for \"const enabled : string\" — output cached: "
          f"shape={tuple(train_enc_out.shape)}\n")
    print(f"    Decoding step by step:\n")

    generated_train = torch.tensor([[word2id["<bos>"]]])
    expected_words  = ["ON", "|", "OFF"]
    EOS_ID_TRAIN    = word2id["OFF"]

    for gen_step in range(TGT_SEQ + 1):
        with contextlib.redirect_stdout(_silent):
            logits_g = train_model.decode(generated_train, train_enc_out, verbose=False)
        next_logits = logits_g[:, -1, :] / 0.01   # near-greedy
        probs_g     = torch.softmax(next_logits, dim=-1)
        next_tok    = probs_g.argmax(dim=-1, keepdim=True)

        ctx_words  = [id2word[i] for i in generated_train[0].tolist()]
        next_word  = id2word[next_tok.item()]
        expected_w = expected_words[gen_step] if gen_step < len(expected_words) else "—"
        match      = "✓" if next_word == expected_w else "✗"

        generated_train = torch.cat([generated_train, next_tok], dim=1)

        print(f"    step {gen_step+1}: decoder context = {ctx_words}")
        print(f"             → predicted \"{next_word}\"  "
              f"(expected \"{expected_w}\")  {match}\n")

        if next_tok.item() == EOS_ID_TRAIN:
            break

result_words = [id2word[i] for i in generated_train[0, 1:].tolist()]   # skip <bos>
result_str   = " ".join(result_words)
expected_str = " ".join(expected_words)
match_all    = result_words == expected_words

print(f"    ┌{'─' * 60}┐")
print(f"    │  Source:    \"const enabled : string\"".ljust(63) + "│")
print(f"    │  Expected:  \"{expected_str}\"".ljust(63) + "│")
print(f"    │  Generated: \"{result_str}\"".ljust(63) + "│")
print(f"    │  Result:    {'✓ CORRECT — model recovered the literal union type' if match_all else '✗ mismatch'}".ljust(63) + "│")
print(f"    └{'─' * 60}┘")

print(f"""
    What just happened:
      • The model saw exactly ONE training example, 200 times.
      • It learned: when the source contains identifier "enabled" with
        type annotation "string", emit the literal union "ON | OFF".
      • This is a degenerate "training set of size 1" — useful only as a
        sanity check that the architecture (encoder + decoder + cross-attn
        + LM head + cross-entropy + AdamW) can learn ANY mapping.
      • To handle MANY identifiers with MANY types, we need many training
        pairs — that's what packages/ts-type-refiner does for real.
""")
