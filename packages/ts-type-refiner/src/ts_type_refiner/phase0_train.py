"""
Phase 0 — Proof-of-concept: train ManualEncoderDecoder on hardcoded TS type pairs.

Goal: verify the encoder-decoder architecture can learn to map
      TypeScript code context → precise type string.

Usage:
    uv run --package ts-type-refiner phase0-train
"""

import torch
import torch.nn as nn

from core.manual_encoder_decoder import ManualEncoderDecoder
from ts_type_refiner.mini_tokenizer import MiniTokenizer
from ts_type_refiner.phase0_data import TRAINING_PAIRS


def pad_sequence(seqs: list[list[int]], pad_id: int) -> torch.Tensor:
    """Pad a list of variable-length sequences to the same length."""
    max_len = max(len(s) for s in seqs)
    padded = [s + [pad_id] * (max_len - len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long)


def main() -> None:
    torch.manual_seed(42)

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────
    tok = MiniTokenizer()
    print(f"Tokenizer: {tok}")

    # ── Prepare training data ─────────────────────────────────────────
    # Encoder input: source context (no special tokens needed)
    # Decoder input: <bos> + target tokens  (teacher forcing)
    # Decoder target: target tokens + <eos>  (what we predict)

    src_seqs: list[list[int]] = []
    tgt_input_seqs: list[list[int]] = []
    tgt_target_seqs: list[list[int]] = []

    print(f"\n{'─'*60}")
    print(f"Training pairs ({len(TRAINING_PAIRS)}):")
    print(f"{'─'*60}")

    for src_text, tgt_text in TRAINING_PAIRS:
        src_ids = tok.encode(src_text)
        tgt_ids = tok.encode(tgt_text)

        # Teacher forcing: decoder sees <bos> + target[:-1], predicts target + <eos>
        tgt_input = [tok.bos_id] + tgt_ids
        tgt_target = tgt_ids + [tok.eos_id]

        src_seqs.append(src_ids)
        tgt_input_seqs.append(tgt_input)
        tgt_target_seqs.append(tgt_target)

        print(f"  src: {src_text}")
        print(f"       ids={src_ids}")
        print(f"  tgt: {tgt_text}")
        print(f"       input={tgt_input}  target={tgt_target}")
        print()

    # Pad and move to device
    src_batch = pad_sequence(src_seqs, tok.pad_id).to(device)
    tgt_input_batch = pad_sequence(tgt_input_seqs, tok.pad_id).to(device)
    tgt_target_batch = pad_sequence(tgt_target_seqs, tok.pad_id).to(device)

    print(f"src_batch:        {src_batch.shape}")
    print(f"tgt_input_batch:  {tgt_input_batch.shape}")
    print(f"tgt_target_batch: {tgt_target_batch.shape}")

    # ── Model ─────────────────────────────────────────────────────────
    max_seq_len = max(src_batch.shape[1], tgt_input_batch.shape[1]) + 4
    model = ManualEncoderDecoder(
        vocab_size=tok.vocab_size,
        max_seq_len=max_seq_len,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_layers=2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: ManualEncoderDecoder")
    print(f"  vocab_size={tok.vocab_size}, d_model=64, heads=4, layers=2")
    print(f"  Parameters: {total_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)

    # ── Training loop ─────────────────────────────────────────────────
    num_steps = 300
    print(f"\n{'='*60}")
    print(f"TRAINING — {num_steps} steps")
    print(f"{'='*60}")

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad()

        # Forward: encoder reads src, decoder reads tgt_input (teacher forcing)
        logits = model(src_batch, tgt_input_batch, verbose=False)
        # logits: (batch, tgt_len, vocab_size)

        # Loss: compare logits with tgt_target (shifted by 1)
        batch_size, tgt_len, vocab_size = logits.shape
        loss = loss_fn(
            logits.reshape(batch_size * tgt_len, vocab_size),
            tgt_target_batch.reshape(batch_size * tgt_len),
        )

        loss.backward()
        optimizer.step()

        # ── Accuracy ──────────────────────────────────────────────────
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # (batch, tgt_len)
            # Mask: only count non-pad positions in target
            mask = tgt_target_batch != tok.pad_id
            correct = ((preds == tgt_target_batch) & mask).sum().item()
            total = mask.sum().item()
            accuracy = correct / total if total > 0 else 0.0

        if step == 1 or step % 20 == 0 or step == num_steps:
            print(f"  Step {step:3d}/{num_steps}  "
                  f"loss={loss.item():.4f}  "
                  f"accuracy={accuracy:.0%}  ({correct}/{total})")

    # ── Generation test ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"GENERATION TEST")
    print(f"{'='*60}")

    model.eval()
    all_correct = 0

    for src_text, expected_tgt in TRAINING_PAIRS:
        src_ids = tok.encode(src_text)
        src_tensor = torch.tensor([src_ids], device=device)

        generated = model.generate(
            src_tensor,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            max_new_tokens=20,
            temperature=0.01,
        )

        gen_ids = generated[0].tolist()
        gen_text = tok.decode(gen_ids)

        match = gen_text == expected_tgt
        if match:
            all_correct += 1
        status = "✓" if match else "✗"

        print(f"\n  {status} src:      {src_text}")
        print(f"    expected: {expected_tgt}")
        print(f"    got:      {gen_text}")

    print(f"\n{'─'*60}")
    print(f"Result: {all_correct}/{len(TRAINING_PAIRS)} correct")
    if all_correct == len(TRAINING_PAIRS):
        print("Phase 0 PASSED — architecture works!")
    else:
        print("Phase 0 PARTIAL — some pairs failed.")
    print(f"{'─'*60}")
