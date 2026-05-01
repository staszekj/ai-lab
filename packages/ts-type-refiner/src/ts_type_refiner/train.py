"""
Phase 1 — Train ManualEncoderDecoder on real TypeScript type pairs.

Loads training_pairs.jsonl, builds BPE tokenizer, trains on CUDA,
evaluates generation accuracy.

Usage:
    uv run --package ts-type-refiner phase1-train
"""

import time

import torch
import torch.nn as nn

from core.manual_encoder_decoder import ManualEncoderDecoder
from ts_type_refiner.tokenizer import build_from_jsonl, TSTokenizer
from ts_type_refiner.dataset import TypeRefinerDataset, train_val_split

DATA_PATH = "packages/ts-type-extractor/data/training_pairs.jsonl"
TOKENIZER_PATH = "packages/ts-type-refiner/tokenizer.json"


def evaluate(
    model: ManualEncoderDecoder,
    dataset: TypeRefinerDataset,
    indices: list[int],
    device: torch.device,
    max_print: int = 10,
) -> float:
    """Run autoregressive generation on given indices, return exact-match accuracy."""
    model.eval()
    tok = dataset.tokenizer
    correct = 0

    for i, idx in enumerate(indices):
        src_text, expected_tgt = dataset.pairs[idx]
        src_ids = tok.encode(src_text)[:dataset.max_src_len]
        src_tensor = torch.tensor([src_ids], device=device)

        generated = model.generate(
            src_tensor,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            max_new_tokens=dataset.max_tgt_len,
            temperature=0.01,
            verbose=False,
        )

        gen_text = tok.decode(generated[0].tolist())
        match = gen_text.strip() == expected_tgt.strip()
        if match:
            correct += 1

        if i < max_print:
            status = "✓" if match else "✗"
            print(f"  {status} expected: {expected_tgt}")
            print(f"    got:      {gen_text}")

    accuracy = correct / len(indices) if indices else 0.0
    return accuracy


def main() -> None:
    torch.manual_seed(42)

    # ── Device ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────
    print(f"Building BPE tokenizer from {DATA_PATH}...")
    tok = build_from_jsonl(DATA_PATH, vocab_size=2048)
    tok.save(TOKENIZER_PATH)
    print(f"  {tok}  saved to {TOKENIZER_PATH}")

    # ── Dataset ───────────────────────────────────────────────────────
    ds = TypeRefinerDataset(DATA_PATH, tok, max_src_len=256, max_tgt_len=64)
    train_idx, val_idx = train_val_split(ds, val_ratio=0.15)
    print(f"  Pairs: {len(ds)}  train={len(train_idx)}  val={len(val_idx)}")

    # Check sequence lengths
    src_lens = [len(tok.encode(ds.pairs[i][0])[:256]) for i in range(len(ds))]
    tgt_lens = [len(tok.encode(ds.pairs[i][1])[:64]) for i in range(len(ds))]
    print(f"  Src tokens: min={min(src_lens)} max={max(src_lens)} avg={sum(src_lens)//len(src_lens)}")
    print(f"  Tgt tokens: min={min(tgt_lens)} max={max(tgt_lens)} avg={sum(tgt_lens)//len(tgt_lens)}")

    # ── Model ─────────────────────────────────────────────────────────
    max_seq = max(max(src_lens), max(tgt_lens)) + 4
    model = ManualEncoderDecoder(
        vocab_size=tok.vocab_size,
        max_seq_len=max_seq,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=4,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: ManualEncoderDecoder")
    print(f"  vocab={tok.vocab_size} d_model=256 heads=8 layers=4 max_seq={max_seq}")
    print(f"  Parameters: {total_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────
    lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)

    # ── Training ──────────────────────────────────────────────────────
    num_epochs = 50
    batch_size = 96
    max_grad_norm = 1.0
    eval_every = 10
    eval_samples = 200  # autoregressive eval is slow, limit samples
    print(f"\n{'='*60}")
    print(f"TRAINING — {num_epochs} epochs, batch_size={batch_size}, lr={lr}")
    print(f"{'='*60}")

    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0
        t_epoch = time.time()

        for src, tgt_in, tgt_tgt in ds.iter_batches(
            batch_size, device=device, shuffle=True, indices=train_idx
        ):
            optimizer.zero_grad()
            logits = model(src, tgt_in, verbose=False)
            b, t, v = logits.shape
            loss = loss_fn(logits.reshape(b * t, v), tgt_tgt.reshape(b * t))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = tgt_tgt != tok.pad_id
                epoch_correct += ((preds == tgt_tgt) & mask).sum().item()
                epoch_total += mask.sum().item()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        elapsed = time.time() - t_epoch

        if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
            print(f"  Epoch {epoch:3d}/{num_epochs}  "
                  f"loss={avg_loss:.4f}  "
                  f"tf_acc={accuracy:.0%}  "
                  f"{elapsed:.1f}s")

        # Periodic eval with autoregressive generation
        if epoch % eval_every == 0 or epoch == num_epochs:
            eval_idx = val_idx[:eval_samples]
            val_acc = evaluate(model, ds, eval_idx, device, max_print=5)
            print(f"    → val exact-match: {val_acc:.1%} ({int(val_acc*len(eval_idx))}/{len(eval_idx)})")
            if val_acc > best_val_acc:
                best_val_acc = val_acc

    # ── Final evaluation ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION — generation on {len(val_idx)} held-out pairs")
    print(f"{'='*60}")

    val_acc = evaluate(model, ds, val_idx, device, max_print=15)
    print(f"\nValidation exact-match: {val_acc:.1%} ({int(val_acc*len(val_idx))}/{len(val_idx)})")

    train_acc = evaluate(model, ds, train_idx[:50], device, max_print=5)
    print(f"Train sample exact-match: {train_acc:.1%}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:      {total_params:,} params on {device}")
    print(f"  Data:       {len(train_idx)} train / {len(val_idx)} val")
    print(f"  Training:   {num_epochs} epochs, final loss={avg_loss:.4f}, tf_acc={accuracy:.0%}")
    print(f"  Val exact-match:   {val_acc:.1%}")
    print(f"  Best val:          {best_val_acc:.1%}")
    print(f"  Train exact-match: {train_acc:.1%}")

    # ── Save checkpoint ───────────────────────────────────────────────
    from pathlib import Path
    ckpt_dir = Path("packages/ts-type-refiner/checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "phase1_model.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": num_epochs,
        "loss": avg_loss,
        "val_accuracy": val_acc,
        "model_config": {
            "vocab_size": tok.vocab_size,
            "max_seq_len": max_seq,
            "d_model": 256,
            "num_heads": 8,
            "d_ff": 1024,
            "num_layers": 4,
        },
    }, ckpt_path)

    print(f"\n  Checkpoint saved: {ckpt_path}")
    print(f"  Tokenizer saved:  {TOKENIZER_PATH}")
