"""
ASCII Art Presentation: Encoder-Decoder Transformer — Full Forward + Backward
==============================================================================

Runs a REAL forward + backward pass through ManualEncoderDecoder with tiny
tensors so every value fits on screen.

    Source: "the cat climbs a tree"  (5 tokens) — input to encoder
    Target: "the dog eats a fish"    (5 tokens) — output to predict

    vocab_size  = 16   d_model = 6    num_heads  = 3
    d_ff        = 12   num_layers = 2  max_seq_len = 16

    batch_size  = 1    src_len  = 5   tgt_len     = 4  (teacher-forced)

Run:  uv run python3 -m core.presentation_encoder_decoder
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


def print_tensor_3d(name: str, t: torch.Tensor, dim_labels=None, indent: int = 4):
    d0, d1, d2 = t.shape
    labels = dim_labels or [f"[{i}]" for i in range(d0)]
    for i in range(d0):
        print_matrix(f"{name}{labels[i]}", t[i], indent=indent)


def section(title: str):
    print(f"\n{'━' * 72}")
    print(f"  {title}")
    print(f"{'━' * 72}")


def step_header(num: str, title: str):
    width = max(56 - len(num) - len(title), 2)
    print(f"\n  ┌─── STEP {num} {'─' * width} {title} ───┐")


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

VOCAB      = 16
D_MODEL    = 6
HEADS      = 3
D_K        = D_MODEL // HEADS   # = 2
D_FF       = 12
NUM_LAYERS = 2
MAX_SEQ    = 16
BATCH      = 1
SRC_SEQ    = 5
TGT_SEQ    = 4   # decoder sees tgt[:-1], predicts tgt[1:]

torch.manual_seed(42)

VOCAB_WORDS = [
    "<pad>", "the", "cat", "dog", "climbs", "sits", "a", "tree",
    "house", "on", "big", "small", "runs", "eats", "fish", "bird",
]
word2id = {w: i for i, w in enumerate(VOCAB_WORDS)}
id2word  = VOCAB_WORDS

SRC_SENTENCE = "the cat climbs a tree"
TGT_SENTENCE = "the dog eats a fish"    # target we want to generate

# ══════════════════════════════════════════════════════════════════════
# Architecture overview
# ══════════════════════════════════════════════════════════════════════

section("ARCHITECTURE — Encoder-Decoder Transformer")
print(f"""
    Source (input  to model): "{SRC_SENTENCE}"   ({BATCH}, {SRC_SEQ})
    Target (output of model): "{TGT_SENTENCE}"  ({BATCH}, {TGT_SEQ + 1})

    During training (teacher forcing):
      Decoder INPUT:  "{' '.join(TGT_SENTENCE.split()[:-1])}"   tgt[:-1]  ({BATCH}, {TGT_SEQ})
      Decoder TARGET: "{' '.join(TGT_SENTENCE.split()[1:])}"     tgt[1:]   ({BATCH}, {TGT_SEQ})

    ┌──────────────────────────────────────────────────────────────────┐
    │                        ENCODER                                  │
    │                                                                 │
    │  src_ids  →  Embedding + Pos  →  {NUM_LAYERS}×EncoderBlock  →  encoder_out │
    │                                    (bidirectional, no mask)     │
    └──────────────────────────────────────┬──────────────────────────┘
                                 encoder_output ({BATCH}, {SRC_SEQ}, {D_MODEL})
                                           │
                              ┌────────────┴────────────┐
                              │   K, V for cross-attn   │
                              │   (same tensor fed to   │
                              │    EVERY decoder block) │
                              └────────────┬────────────┘
                                           │
    ┌──────────────────────────────────────┼──────────────────────────┐
    │                        DECODER       │                          │
    │                                      │                          │
    │  tgt_ids  →  Embedding + Pos  →  {NUM_LAYERS}×DecoderBlock  →  logits   │
    │                              (causal self-attn                 │
    │                               + cross-attn ←──────────────────┤
    │                               + FFN)                           │
    └──────────────────────────────────────────────────────────────────┘

    Model (tiny, fits on screen):
      d_model={D_MODEL}, num_heads={HEADS}, d_ff={D_FF}, num_layers={NUM_LAYERS}
""")

# ══════════════════════════════════════════════════════════════════════
# Build the model
# ══════════════════════════════════════════════════════════════════════

from core.manual_encoder_decoder import ManualEncoderDecoder

model = ManualEncoderDecoder(
    vocab_size=VOCAB,
    max_seq_len=MAX_SEQ,
    d_model=D_MODEL,
    num_heads=HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
)

enc_params   = sum(p.numel() for p in model.encoder_blocks.parameters())
dec_params   = sum(p.numel() for p in model.decoder_blocks.parameters())
total_params = sum(p.numel() for p in model.parameters())

section("MODEL PARAMETERS")
enc_block_p = sum(p.numel() for p in model.encoder_blocks[0].parameters())
dec_block_p = sum(p.numel() for p in model.decoder_blocks[0].parameters())
print(f"""
    Total parameters: {total_params:,}

    Breakdown:
      Shared token embedding     ({VOCAB}×{D_MODEL}):         {model.token_embedding.weight.numel():>5,}
      Shared position embedding  ({MAX_SEQ}×{D_MODEL}):        {model.positional_embedding.weight.numel():>5,}
      Encoder blocks ({NUM_LAYERS} × {enc_block_p} params):    {enc_params:>5,}
      Encoder final LayerNorm:                     {sum(p.numel() for p in model.encoder_final_norm.parameters()):>5,}
      Decoder blocks ({NUM_LAYERS} × {dec_block_p} params):    {dec_params:>5,}
      Decoder final LayerNorm:                     {sum(p.numel() for p in model.decoder_final_norm.parameters()):>5,}
      LM Head ({D_MODEL}→{VOCAB}):                             {model.lm_head.weight.numel():>5,}

    Note: each DecoderBlock has ~{dec_block_p // enc_block_p}× the parameters of EncoderBlock
    because it has TWO sets of attention matrices (self + cross) vs ONE.
""")

# ══════════════════════════════════════════════════════════════════════
# Inputs
# ══════════════════════════════════════════════════════════════════════

src_ids    = torch.tensor([[word2id[w] for w in SRC_SENTENCE.split()]])  # (1, 5)
tgt_ids    = torch.tensor([[word2id[w] for w in TGT_SENTENCE.split()]])  # (1, 5)
tgt_input  = tgt_ids[:, :-1]   # "the dog eats a"   (1, 4) — decoder sees this
tgt_target = tgt_ids[:, 1:]    # "dog eats a fish"  (1, 4) — decoder predicts this

section("INPUT SEQUENCES")
print(f"""
    Source (encoder):
      "{SRC_SENTENCE}"
      IDs:   {src_ids[0].tolist()}
      Shape: {tuple(src_ids.shape)}

    Target — full sequence:
      "{TGT_SENTENCE}"
      IDs:   {tgt_ids[0].tolist()}

    Target — teacher forcing split:
      Decoder INPUT  (tgt[:-1]):  "{' '.join(TGT_SENTENCE.split()[:-1])}"
        IDs: {tgt_input[0].tolist()},  shape: {tuple(tgt_input.shape)}

      Decoder TARGET (tgt[1:]):   "{' '.join(TGT_SENTENCE.split()[1:])}"
        IDs: {tgt_target[0].tolist()},  shape: {tuple(tgt_target.shape)}

    Training objective:
      At position 0, seeing "{TGT_SENTENCE.split()[0]}"  → predict "{TGT_SENTENCE.split()[1]}"
      At position 1, seeing "{TGT_SENTENCE.split()[1]}"  → predict "{TGT_SENTENCE.split()[2]}"
      At position 2, seeing "{TGT_SENTENCE.split()[2]}"  → predict "{TGT_SENTENCE.split()[3]}"
      At position 3, seeing "{TGT_SENTENCE.split()[3]}"  → predict "{TGT_SENTENCE.split()[4]}"
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Encoder forward pass
# ══════════════════════════════════════════════════════════════════════

step_header("1", "Encoder — bidirectional attention over source")
print(f"""
    All {SRC_SEQ} source tokens are processed in one pass.
    No causal mask — every token can attend to every other token.
    The result is the "memory" of the source sequence.

    Shape stays ({BATCH}, {SRC_SEQ}, {D_MODEL}) through all {NUM_LAYERS} encoder blocks.
""")

buf = io.StringIO()

# Embeddings
src_pos_ids   = torch.arange(SRC_SEQ)
src_token_emb = model.token_embedding(src_ids)          # (1, 5, 6)
src_pos_emb   = model.positional_embedding(src_pos_ids)  # (5, 6)
enc_x = src_token_emb + src_pos_emb

print(f"    src_ids → token_emb:  {tuple(src_token_emb.shape)}")
print(f"    position_ids → pos_emb: {tuple(src_pos_emb.shape)}")
print(f"    enc_x = token_emb + pos_emb: {tuple(enc_x.shape)}")
print()
print_tensor_3d("enc_x", enc_x, [f"[batch 0]"])

# Encoder blocks (suppress the block's verbose print)
for i, enc_block in enumerate(model.encoder_blocks):
    with contextlib.redirect_stdout(buf):
        enc_x = enc_block(enc_x, attn_mask=None)
    print(f"\n    After EncoderBlock {i}:  {tuple(enc_x.shape)}  "
          f"mean={enc_x.mean().item():+.4f}  std={enc_x.std().item():.4f}")

encoder_output = model.encoder_final_norm(enc_x)
print(f"\n    After final LayerNorm — encoder_output:  {tuple(encoder_output.shape)}")
print()
print_tensor_3d("  encoder_output", encoder_output, [f"[batch 0]"])
print(f"""
    This is the "memory" of "{SRC_SENTENCE}".
    It encodes what each source token means in the context of all others.
    Shape: ({BATCH}, {SRC_SEQ}, {D_MODEL}) — one {D_MODEL}-dim vector per source token.
    This tensor is passed to EVERY decoder block as K and V in cross-attention.
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Decoder forward pass
# ══════════════════════════════════════════════════════════════════════

step_header("2", "Decoder — causal self-attention + cross-attention")
print(f"""
    Decoder input (teacher-forced): "{' '.join(TGT_SENTENCE.split()[:-1])}"
    Shape: {tuple(tgt_input.shape)}

    Each of the {NUM_LAYERS} decoder blocks:
      (a) Attends to previous target tokens  [causal self-attention, ({TGT_SEQ}×{TGT_SEQ})]
      (b) Attends to ALL encoder positions   [cross-attention, ({TGT_SEQ}×{SRC_SEQ})]
      (c) FFN transformation
""")

# Embeddings
tgt_pos_ids   = torch.arange(tgt_input.shape[1])
tgt_token_emb = model.token_embedding(tgt_input)          # (1, 4, 6)
tgt_pos_emb   = model.positional_embedding(tgt_pos_ids)    # (4, 6)
dec_x = tgt_token_emb + tgt_pos_emb

print(f"    tgt_input → token_emb:  {tuple(tgt_token_emb.shape)}")
print(f"    dec_x = token_emb + pos_emb:  {tuple(dec_x.shape)}")

tgt_len  = tgt_input.shape[1]
tgt_mask = model._create_causal_mask(tgt_len, device=tgt_input.device)
tgt_words_in = TGT_SENTENCE.split()[:-1]
src_words    = SRC_SENTENCE.split()

print(f"\n    Causal mask ({tgt_len}×{tgt_len}) for target self-attention:")
header = "            " + "   ".join(f'"{w[:4]}"' for w in tgt_words_in)
print(f"    {header}")
for r in range(tgt_len):
    vals = ["  0  " if tgt_mask[r, c] == 0 else " -inf" for c in range(tgt_len)]
    print(f'    "{tgt_words_in[r][:4]}"   {"   ".join(vals)}')

dec_x_cur = dec_x.clone()
for i, dec_block in enumerate(model.decoder_blocks):
    print(f"\n    {'═' * 60}")
    print(f"    DECODER BLOCK {i}")
    print(f"    x: {tuple(dec_x_cur.shape)}   encoder_output: {tuple(encoder_output.shape)}")

    # ── Show cross-attention weights for block 0 ─────────────────
    if i == 0:
        print(f"\n    Cross-attention scores (Head 0)  —  ({tgt_len}×{SRC_SEQ}, NOT square):")
        print(f"    Rows = target tokens: {tgt_words_in}")
        print(f"    Cols = source tokens: {src_words}")

        # Compute cross-attention weights manually for display
        with torch.no_grad():
            # self-attention part
            x_n1   = dec_block.layer_norm_1(dec_x_cur)
            Q_s    = dec_block.self_W_Q(x_n1).view(BATCH, tgt_len, HEADS, D_K).transpose(1, 2)
            K_s    = dec_block.self_W_K(x_n1).view(BATCH, tgt_len, HEADS, D_K).transpose(1, 2)
            V_s    = dec_block.self_W_V(x_n1).view(BATCH, tgt_len, HEADS, D_K).transpose(1, 2)
            sc_s   = torch.matmul(Q_s, K_s.transpose(-2, -1)) / math.sqrt(D_K) + tgt_mask
            wt_s   = torch.softmax(sc_s, dim=-1)
            ho_s   = torch.matmul(wt_s, V_s)
            c_s    = ho_s.transpose(1, 2).contiguous().view(BATCH, tgt_len, D_MODEL)
            so     = dec_block.self_W_O(c_s)
            x1_d   = dec_x_cur + so

            # cross-attention part
            x1_n   = dec_block.layer_norm_2(x1_d)
            Q_c    = dec_block.cross_W_Q(x1_n).view(BATCH, tgt_len, HEADS, D_K).transpose(1, 2)
            K_c    = dec_block.cross_W_K(encoder_output).view(BATCH, SRC_SEQ, HEADS, D_K).transpose(1, 2)
            sc_c   = torch.matmul(Q_c, K_c.transpose(-2, -1)) / math.sqrt(D_K)
            wt_c   = torch.softmax(sc_c, dim=-1)

        print_matrix("    cross_weights[b=0,h=0]", wt_c[0, 0], indent=4)

        print(f"\n    Decoder token → most attended source token:")
        for t, tw in enumerate(tgt_words_in):
            top  = wt_c[0, 0, t].argmax().item()
            top2 = wt_c[0, 0, t].argsort(descending=True)[1].item()
            print(f"      \"{tw:<7}\" → \"{src_words[top]:<7}\" "
                  f"({wt_c[0,0,t,top].item():.3f})  "
                  f"2nd: \"{src_words[top2]:<7}\" ({wt_c[0,0,t,top2].item():.3f})")

    with contextlib.redirect_stdout(buf):
        dec_x_cur = dec_block(dec_x_cur, encoder_output, tgt_mask=tgt_mask)

    print(f"\n    Output: {tuple(dec_x_cur.shape)}  "
          f"mean={dec_x_cur.mean().item():+.4f}  std={dec_x_cur.std().item():.4f}")

dec_final = model.decoder_final_norm(dec_x_cur)
logits    = model.lm_head(dec_final)
# logits: (1, 4, 16)

print(f"\n    After final LayerNorm + LM Head:  logits {tuple(logits.shape)}")

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Predictions
# ══════════════════════════════════════════════════════════════════════

step_header("3", "Predictions at each target position")
print(f"""
    For each of the {tgt_len} target positions, the model scores all {VOCAB} vocabulary words.
    tgt_input  = "{' '.join(tgt_words_in)}"
    tgt_target = "{' '.join(TGT_SENTENCE.split()[1:])}"
""")

tgt_target_words = TGT_SENTENCE.split()[1:]
for t in range(tgt_len):
    probs     = torch.softmax(logits[0, t], dim=-1)
    target_id = tgt_target[0, t].item()
    target_p  = probs[target_id].item()
    pred_id   = probs.argmax().item()
    correct   = "✓" if pred_id == target_id else "✗"
    print(f"    \"{tgt_words_in[t]:<7}\" → target=\"{id2word[target_id]:<7}\"  "
          f"P(target)={target_p:.4f}  "
          f"predicted=\"{id2word[pred_id]:<7}\"  {correct}")

print(f"\n    (Untrained model → essentially random predictions.  Loss ≈ log({VOCAB}).)")

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Loss
# ══════════════════════════════════════════════════════════════════════

step_header("4", "Cross-entropy loss")
print(f"""
    loss = CrossEntropyLoss(logits.reshape(-1, vocab), tgt_target.reshape(-1))

    logits     {tuple(logits.shape)}   →  reshape  →  ({tgt_len * BATCH}, {VOCAB})
    tgt_target {tuple(tgt_target.shape)}      →  reshape  →  ({tgt_len * BATCH},)
""")

loss = nn.CrossEntropyLoss()(
    logits.reshape(-1, VOCAB),
    tgt_target.reshape(-1),
)
random_loss = math.log(VOCAB)
print(f"    Loss:             {loss.item():.6f}")
print(f"    Random baseline:  {random_loss:.6f}  (= log({VOCAB}), expected for random model)")
print(f"    Ratio:            {loss.item() / random_loss:.3f}  (≈ 1.0 for untrained model)")

# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Backward pass
# ══════════════════════════════════════════════════════════════════════

step_header("5", "Backward pass — gradient flow through cross-attention")
print(f"""
    A single loss.backward() differentiates through the ENTIRE graph:

    loss
     └─► LM head weight
          └─► decoder_final_norm
               └─► DecoderBlock {NUM_LAYERS - 1}
                    ├─► FFN weights          (decoder only)
                    ├─► cross_W_O / cross_W_Q  (decoder only)
                    └─► cross_W_K, cross_W_V  ← Q contracted against K/V
                         └─► encoder_output  ← these gradients flow BACK
                              └─► encoder_final_norm
                                   └─► EncoderBlock {NUM_LAYERS - 1}
                                        └─► EncoderBlock 0
                                             └─► token_embedding, pos_embedding

    The encoder has NO separate loss.  It gets gradients ONLY because
    the decoder's cross_W_K and cross_W_V operate on encoder_output.
    PyTorch's autograd traces this path automatically.
""")

# Recompute with gradient tracking enabled
model.zero_grad()
enc_out2   = model.encode(src_ids, verbose=False)
logits2    = model.decode(tgt_input, enc_out2, verbose=False)
loss2      = nn.CrossEntropyLoss()(
    logits2.reshape(-1, VOCAB),
    tgt_target.reshape(-1),
)
loss2.backward()
print(f"    Loss (recomputed for backward): {loss2.item():.6f}")

print(f"\n    Gradient norms after loss.backward():\n")

param_groups = [
    ("Shared embeddings",              [
        ("token_embedding.weight",      model.token_embedding.weight),
        ("positional_embedding.weight", model.positional_embedding.weight),
    ]),
    ("Encoder Block 0",                [
        ("enc[0].W_Q.weight",           model.encoder_blocks[0].W_Q.weight),
        ("enc[0].W_K.weight",           model.encoder_blocks[0].W_K.weight),
        ("enc[0].W_V.weight",           model.encoder_blocks[0].W_V.weight),
        ("enc[0].ffn_linear1.weight",   model.encoder_blocks[0].ffn_linear1.weight),
    ]),
    ("Decoder Block 0 — Self-Attention", [
        ("dec[0].self_W_Q.weight",      model.decoder_blocks[0].self_W_Q.weight),
        ("dec[0].self_W_K.weight",      model.decoder_blocks[0].self_W_K.weight),
        ("dec[0].self_W_V.weight",      model.decoder_blocks[0].self_W_V.weight),
    ]),
    ("Decoder Block 0 — Cross-Attention ← key", [
        ("dec[0].cross_W_Q.weight",     model.decoder_blocks[0].cross_W_Q.weight),
        ("dec[0].cross_W_K.weight",     model.decoder_blocks[0].cross_W_K.weight),
        ("dec[0].cross_W_V.weight",     model.decoder_blocks[0].cross_W_V.weight),
    ]),
    ("LM Head",                        [
        ("lm_head.weight",              model.lm_head.weight),
    ]),
]

for group_name, params in param_groups:
    print(f"    {group_name}:")
    for name, param in params:
        if param.grad is not None:
            g = param.grad
            flag = "  ← gradients reach encoder!" if name.startswith("enc[0]") else ""
            print(f"      {name:<40}  norm={g.norm().item():.4f}  "
                  f"max={g.abs().max().item():.4f}{flag}")
        else:
            print(f"      {name:<40}  grad=None")
    print()

print(f"""    KEY OBSERVATION:
      enc[0].W_Q, W_K, W_V all have non-zero gradients, even though the
      encoder has no separate loss.  The path is:

        loss → decoder → cross_W_K/cross_W_V → encoder_output → encoder blocks

      This single backward() call trains both encoder and decoder together.
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 6 — Autoregressive generation
# ══════════════════════════════════════════════════════════════════════

step_header("6", "Autoregressive generation (encoder runs once)")
print(f"""
    During inference (no teacher forcing):

      encoder_output = encode(src_ids)        ← computed ONCE

      tgt = [<BOS>]
      while not <EOS>:
          logits     = decode(tgt, encoder_output)
          next_token = sample(logits[:, -1, :])
          tgt.append(next_token)

    The encoder cost is O(src_len²) — paid once.
    The decoder cost is O(tgt_len²) per step — grows as we generate.

    Compare with GPT (decoder-only):
      GPT would process the source tokens by prepending them to the target.
      It re-reads ALL tokens (source + generated so far) at every step.
      Encoder-decoder separates source processing from generation entirely.

    Generating up to 6 tokens from "{SRC_SENTENCE}":
""")

BOS_ID = word2id["<pad>"]
EOS_ID = word2id["bird"]    # using "bird" as a stand-in EOS for this demo

with torch.no_grad():
    generated = model.generate(
        src_ids=src_ids,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        max_new_tokens=6,
        temperature=1.0,
    )

generated_words = [id2word[i] for i in generated[0].tolist()]
print(f"\n    Generated: {generated_words}")
print(f"    (Untrained — random predictions.  Train the model to get meaningful output.)")

# ══════════════════════════════════════════════════════════════════════
# STEP 7 — Training: learn ONE mapping "the cat" → "climbs a tree"
# ══════════════════════════════════════════════════════════════════════
#
# Mirror of presentation_mini_gpt.py STEP 11, but framed as seq2seq:
#
#   GPT (decoder-only):
#     input:   "the cat climbs a tree"  (one sequence, causal mask)
#     learns:  P(climbs | the cat)  P(a | the cat climbs) ...
#
#   Encoder-Decoder:
#     source:  "the cat"         → encoder (bidirectional)
#     target:  "climbs a tree"   → decoder (causal + cross-attention)
#     learns:  GIVEN "the cat",  generate "climbs a tree"
#     The task is EXPLICIT: source ≠ target, two different sequences.
# ══════════════════════════════════════════════════════════════════════

step_header("7", "Training — learn \"the cat\" → \"climbs a tree\"")

TRAIN_SRC     = "the cat"
TRAIN_TGT     = "climbs a tree"    # what the decoder should produce
BOS_WORD      = "<pad>"            # reuse <pad> as BOS for this demo
BOS_ID_TRAIN  = word2id[BOS_WORD]

train_src_ids = torch.tensor([[word2id[w] for w in TRAIN_SRC.split()]])      # (1, 2)
train_tgt_ids = torch.tensor([[word2id[w] for w in TRAIN_TGT.split()]])      # (1, 3)

# Teacher-forcing split:
#   decoder INPUT  = [<BOS>, "climbs", "a"]     → model sees these
#   decoder TARGET = ["climbs", "a", "tree"]    → model predicts these
train_tgt_input  = torch.cat([
    torch.tensor([[BOS_ID_TRAIN]]),
    train_tgt_ids[:, :-1],
], dim=1)   # (1, 3): [<pad>, climbs, a]

train_tgt_target = train_tgt_ids   # (1, 3): [climbs, a, tree]

print(f"""
    Task framing (seq2seq):
      Source (encoder): "{TRAIN_SRC}"
        IDs: {train_src_ids[0].tolist()}

      Decoder input  (teacher-forced): "{BOS_WORD} {' '.join(TRAIN_TGT.split()[:-1])}"
        IDs: {train_tgt_input[0].tolist()}

      Decoder target (what to predict):  "{TRAIN_TGT}"
        IDs: {train_tgt_target[0].tolist()}

    At each decoder step:
      [<BOS>]               → predict → "climbs"
      [<BOS>, "climbs"]     → predict → "a"
      [<BOS>, "climbs","a"] → predict → "tree"

    Compare with GPT (one sequence, causal mask):
      ["the", "cat", "climbs", "a"]  → predict → ["cat", "climbs", "a", "tree"]
      The source tokens ("the cat") were PART of the same sequence.
      Here they are SEPARATED — the encoder processes them independently.

    Optimizer: AdamW (lr=0.01)   Steps: 200
""")

# Reset model so training starts from scratch (same seed as earlier models)
torch.manual_seed(42)
train_model = ManualEncoderDecoder(
    vocab_size=VOCAB,
    max_seq_len=MAX_SEQ,
    d_model=D_MODEL,
    num_heads=HEADS,
    d_ff=D_FF,
    num_layers=NUM_LAYERS,
)
optimizer_train = torch.optim.AdamW(train_model.parameters(), lr=0.01)

num_steps    = 200
initial_loss = None

_silent = io.StringIO()
for step in range(num_steps):
    with contextlib.redirect_stdout(_silent):
        enc_out_tr = train_model.encode(train_src_ids, verbose=False)
        logits_tr  = train_model.decode(train_tgt_input, enc_out_tr, verbose=False)

    step_loss = nn.CrossEntropyLoss()(
        logits_tr.reshape(-1, VOCAB),
        train_tgt_target.reshape(-1),
    )

    if initial_loss is None:
        initial_loss = step_loss.item()

    step_loss.backward()
    optimizer_train.step()
    optimizer_train.zero_grad(set_to_none=True)

    with torch.no_grad():
        preds   = logits_tr[:, :, :].argmax(dim=-1)   # (1, 3)
        correct = (preds == train_tgt_target).sum().item()
        total   = train_tgt_target.numel()
        acc     = correct / total

    if step % 20 == 0 or step == num_steps - 1:
        pred_words = [id2word[i] for i in preds[0].tolist()]
        print(f"    Step {step+1:>3}/{num_steps}  "
              f"loss={step_loss.item():.4f}  "
              f"accuracy={acc:.0%}  "
              f"predicted={pred_words}")

random_loss_val = math.log(VOCAB)
print(f"""
    Initial loss: {initial_loss:.4f}  (random baseline ≈ {random_loss_val:.4f})
    Final loss:   {step_loss.item():.4f}
    Final accuracy: {acc:.0%}
""")

# ══════════════════════════════════════════════════════════════════════
# STEP 8 — Generation: encode "the cat" once, decode autoregressively
# ══════════════════════════════════════════════════════════════════════

step_header("8", "Generation — prompt \"the cat\" → generates \"climbs a tree\"")
print(f"""
    Now we use the TRAINED model in inference mode.

    GPT equivalent (from presentation_mini_gpt.py STEP 12):
      Prompt:    "the cat"
      Generated: "climbs a tree"
      Method:    feed all tokens into one stream, causal mask

    Encoder-Decoder equivalent:
      Encoder input:  "the cat"          (processed ONCE, bidirectionally)
      Decoder starts: [<BOS>]
      Decoder grows:  [<BOS>] → [<BOS>, "climbs"] → [<BOS>, "climbs", "a"] → ...

    The key difference: the encoder has NO causal mask. It can attend to
    BOTH "the" and "cat" simultaneously when building representations.
    GPT's "the" can only attend to itself, "cat" can see ["the","cat"].
""")

EOS_ID_TRAIN = word2id["tree"]   # treat "tree" as the stop signal

with torch.no_grad():
    with contextlib.redirect_stdout(_silent):
        train_enc_out = train_model.encode(train_src_ids, verbose=False)

    generated_train = torch.tensor([[BOS_ID_TRAIN]])
    print(f"    Encoder ran ONCE for \"{TRAIN_SRC}\" — output cached: {train_enc_out.shape}\n")
    print(f"    Decoding step by step:\n")

    for gen_step in range(5):
        with contextlib.redirect_stdout(_silent):
            logits_g = train_model.decode(generated_train, train_enc_out, verbose=False)

        next_logits = logits_g[:, -1, :] / 0.01   # near-greedy temperature
        probs_g     = torch.softmax(next_logits, dim=-1)
        next_tok    = probs_g.argmax(dim=-1, keepdim=True)

        ctx_words  = [id2word[i] for i in generated_train[0].tolist()]
        next_word  = id2word[next_tok.item()]
        expected_i = gen_step if gen_step < len(TRAIN_TGT.split()) else None
        expected_w = TRAIN_TGT.split()[gen_step] if expected_i is not None else "—"
        match      = "✓" if next_word == expected_w else "✗"

        generated_train = torch.cat([generated_train, next_tok], dim=1)

        print(f"    Step {gen_step+1}: decoder context = {ctx_words}")
        print(f"             → predicted \"{next_word}\"  "
              f"(expected \"{expected_w}\")  {match}\n")

        if next_tok.item() == EOS_ID_TRAIN:
            break

result_words = [id2word[i] for i in generated_train[0, 1:].tolist()]  # skip BOS
print(f"    ┌─────────────────────────────────────────────────────────┐")
print(f"    │  Source:    \"{TRAIN_SRC}\"                               │")
print(f"    │  Generated: \"{' '.join(result_words)}\"                   │")
print(f"    │  Expected:  \"{TRAIN_TGT}\"                        │")
match_all = result_words == TRAIN_TGT.split()
print(f"    │  Result:    {'✓ CORRECT' if match_all else '✗ partial'}                                  │")
print(f"    └─────────────────────────────────────────────────────────┘")

print(f"""
    Side-by-side comparison with GPT (from presentation_mini_gpt.py):

    ┌──────────────────┬──────────────────────────┬──────────────────────────┐
    │                  │  GPT (decoder-only)      │  Enc-Dec                 │
    ├──────────────────┼──────────────────────────┼──────────────────────────┤
    │ What model sees  │ "the cat climbs a tree"  │ src: "the cat"           │
    │                  │  (one stream)            │ tgt: "<BOS> climbs a"    │
    │                  │                          │  (two separate streams)  │
    ├──────────────────┼──────────────────────────┼──────────────────────────┤
    │ "the" attends to │ only itself (causal)      │ "the" AND "cat" (bidir) │
    ├──────────────────┼──────────────────────────┼──────────────────────────┤
    │ "climbs" attends │ "the", "cat", itself     │ via cross-attention:     │
    │ to source via    │ (same sequence, causal)  │ ALL encoder positions    │
    ├──────────────────┼──────────────────────────┼──────────────────────────┤
    │ Prompt:          │ "the cat"                │ encoder input: "the cat" │
    │ Generated:       │ "climbs a tree"          │ decoder output: same     │
    └──────────────────┴──────────────────────────┴──────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════
# Final comparison table
# ══════════════════════════════════════════════════════════════════════

section("SUMMARY — Three Transformer Architectures")
print(f"""
    ┌──────────────────┬─────────────────┬──────────────────┬──────────────────┐
    │                  │  Encoder-only   │  Decoder-only    │ Encoder-Decoder  │
    │                  │  (BERT)         │  (GPT)           │ (T5 / BART)      │
    ├──────────────────┼─────────────────┼──────────────────┼──────────────────┤
    │ Attention mask   │ None (bidir.)   │ Causal           │ Enc: none        │
    │                  │                 │                  │ Dec: causal +    │
    │                  │                 │                  │      cross-attn  │
    ├──────────────────┼─────────────────┼──────────────────┼──────────────────┤
    │ Cross-attention  │ No              │ No               │ Yes              │
    │                  │                 │                  │ Q from dec,      │
    │                  │                 │                  │ K/V from enc     │
    ├──────────────────┼─────────────────┼──────────────────┼──────────────────┤
    │ Input sequences  │ one             │ one              │ two (src + tgt)  │
    ├──────────────────┼─────────────────┼──────────────────┼──────────────────┤
    │ Model output     │ contextualized  │ next-token       │ target sequence  │
    │                  │ embeddings      │ logits           │ logits           │
    ├──────────────────┼─────────────────┼──────────────────┼──────────────────┤
    │ Typical tasks    │ classification  │ open-ended gen.  │ translation,     │
    │                  │ NER, QA         │ code completion  │ summarization,   │
    │                  │                 │                  │ type generation  │
    ├──────────────────┼─────────────────┼──────────────────┼──────────────────┤
    │ Changed files    │ 0 (attn_mask    │ 0 (already       │ manual_decoder   │
    │                  │  =None works)   │  implemented)    │  _block.py  +    │
    │                  │                 │                  │ manual_encoder   │
    │                  │                 │                  │  _decoder.py     │
    └──────────────────┴─────────────────┴──────────────────┴──────────────────┘

    Existing files (ManualTransformerEncoderBlock, ManualMiniGPT)
    are UNCHANGED — reused as-is inside ManualEncoderDecoder.
""")
