"""
Train the EncoderDecoderModel on real TypeScript type pairs.

This file is a THIN ORCHESTRATOR. It owns only the things that are
specific to the ts-type-refiner domain:

    1. Building / saving the BPE tokenizer
    2. Building the dataset and a train/val split
    3. Choosing model hyper-parameters from the data
    4. Defining the eval semantics (autoregressive exact-match accuracy)
    5. Wiring the above into `core.trainer.train(...)` and
       `core.checkpoint.save(...)`

Everything else (forward/backward, optimizer, grad-clip, log-prob,
state_dict serialization) lives in `core` and stays domain-agnostic.

Usage:
    uv run --package ts-type-refiner refiner-train
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

from core.checkpoint import save as save_checkpoint
from core.encoder_decoder_model import EncoderDecoderModel
from core.trainer import EpochStats, TrainConfig, train

from ts_type_refiner.dataset import TypeRefinerDataset, train_val_split
from ts_type_refiner.tokenizer import build_from_jsonl


DATA_PATH       = "packages/ts-type-extractor/data/encoder_decoder_pairs.jsonl"
TOKENIZER_PATH  = "packages/ts-type-refiner/tokenizer.json"
CHECKPOINT_DIR  = Path("packages/ts-type-refiner/checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "refiner.pt"


# ══════════════════════════════════════════════════════════════════════
# Eval — autoregressive exact-match accuracy on a held-out slice
# ──────────────────────────────────────────────────────────────────────
# The trainer doesn't know what "good" means for our task. We define
# it here: greedily generate a target string for each src, compare to
# the expected target with whitespace-stripped equality.
# ══════════════════════════════════════════════════════════════════════

def evaluate_exact_match(
    model: EncoderDecoderModel,
    dataset: TypeRefinerDataset,
    indices: list[int],
    device: torch.device,
    max_print: int = 5,
) -> tuple[float, dict[str, tuple[int, int]], float]:
    model.eval()
    tok = dataset.tokenizer
    correct = 0
    by_rule: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for i, idx in enumerate(indices):
        src_text, expected_tgt = dataset.pairs[idx]
        src_ids = tok.encode(src_text)[: dataset.max_src_len]
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
        rule = dataset.rules[idx] if idx < len(dataset.rules) else "<unknown>"
        by_rule[rule][1] += 1
        if match:
            correct += 1
            by_rule[rule][0] += 1

        if i < max_print:
            status = "✓" if match else "✗"
            print(f"  {status} expected: {expected_tgt}")
            print(f"    got:      {gen_text}")

    if not indices:
        return 0.0, {}, 0.0

    macro_vals = [c / n for c, n in by_rule.values() if n > 0]
    macro_acc = sum(macro_vals) / len(macro_vals) if macro_vals else 0.0
    by_rule_final = {k: (v[0], v[1]) for k, v in by_rule.items()}
    return correct / len(indices), by_rule_final, macro_acc


def print_rule_breakdown(by_rule: dict[str, tuple[int, int]], top_n: int = 12) -> None:
    if not by_rule:
        return

    rows = sorted(by_rule.items(), key=lambda item: item[1][1], reverse=True)
    print("    per-rule exact-match (top by support):")
    for rule, (hit, total) in rows[:top_n]:
        acc = hit / total if total else 0.0
        print(f"      {rule:<50} {acc:>6.1%}  ({hit:>4}/{total:<4})")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
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

    # Inspect token lengths so we can size `max_seq_len` to fit the data
    # exactly — saves embedding-table memory vs. a generous default.
    src_lens = [len(tok.encode(ds.pairs[i][0])[:ds.max_src_len]) for i in range(len(ds))]
    tgt_lens = [len(tok.encode(ds.pairs[i][1])[:ds.max_tgt_len]) for i in range(len(ds))]
    print(f"  Src tokens: min={min(src_lens)} max={max(src_lens)} avg={sum(src_lens)//len(src_lens)}")
    print(f"  Tgt tokens: min={min(tgt_lens)} max={max(tgt_lens)} avg={sum(tgt_lens)//len(tgt_lens)}")

    # ── Model ─────────────────────────────────────────────────────────
    max_seq = max(max(src_lens), max(tgt_lens)) + 4
    model_config = dict(
        vocab_size  = tok.vocab_size,
        max_seq_len = max_seq,
        d_model     = 256,
        num_heads   = 8,
        d_ff        = 1024,
        num_layers  = 4,
    )
    model = EncoderDecoderModel(EncoderDecoderModel.Config(**model_config)).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: EncoderDecoderModel")
    print(f"  vocab={tok.vocab_size} d_model=256 heads=8 layers=4 max_seq={max_seq}")
    print(f"  Parameters: {total_params:,}")

    # ── Training config ───────────────────────────────────────────────
    train_cfg = TrainConfig(
        epochs            = 50,
        lr                = 3e-4,
        weight_decay      = 0.01,
        max_grad_norm     = 1.0,
        eval_every        = 10,
        log_every_batches = 25,   # heartbeat — slow GPUs need to show signs of life mid-epoch
        seed              = 42,
    )
    batch_size   = 96
    eval_samples = 200   # autoregressive eval is slow — limit per-eval cost

    print(f"\n{'='*60}")
    print(f"TRAINING — {train_cfg.epochs} epochs, batch_size={batch_size}, lr={train_cfg.lr}")
    print(f"{'='*60}")

    # ── Wire callbacks for the pure-function trainer ──────────────────
    # `train_batches` MUST be a factory: a fresh iterable every epoch
    # so that `shuffle=True` actually re-shuffles each pass.
    def train_batches():
        return ds.iter_batches(batch_size, device=device, shuffle=True, indices=train_idx)

    def eval_fn(m: EncoderDecoderModel) -> float:
        eval_idx = val_idx[:eval_samples]
        acc, by_rule, macro_acc = evaluate_exact_match(m, ds, eval_idx, device, max_print=5)
        print(f"    → val exact-match: {acc:.1%} ({int(acc*len(eval_idx))}/{len(eval_idx)})")
        print(f"    → val macro exact-match (per-rule): {macro_acc:.1%}")
        print_rule_breakdown(by_rule, top_n=10)
        return acc

    # State captured across epochs by closures — no globals.
    progress: dict = {"best_val": 0.0, "last_loss": 0.0, "last_acc": 0.0}

    def on_epoch_end(stats: EpochStats) -> None:
        progress["last_loss"] = stats.train_loss
        progress["last_acc"]  = stats.train_tf_acc
        # Always print every epoch — on weak GPUs (no tensor cores) one
        # epoch can take minutes and "every 5" is too sparse for the user
        # to see the run is making progress.
        print(f"  Epoch {stats.epoch:3d}/{train_cfg.epochs}  "
              f"loss={stats.train_loss:.4f}  "
              f"tf_acc={stats.train_tf_acc:.0%}  "
              f"{stats.elapsed_s:.1f}s", flush=True)
        if stats.val_metric is not None and stats.val_metric > progress["best_val"]:
            progress["best_val"] = stats.val_metric

    # ── Train ─────────────────────────────────────────────────────────
    train(
        model         = model,
        train_batches = train_batches,
        pad_id        = tok.pad_id,
        cfg           = train_cfg,
        eval_fn       = eval_fn,
        on_epoch_end  = on_epoch_end,
    )

    # ── Final evaluation ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION — generation on {len(val_idx)} held-out pairs")
    print(f"{'='*60}")

    val_acc, val_by_rule, val_macro_acc = evaluate_exact_match(model, ds, val_idx, device, max_print=15)
    print(f"\nValidation exact-match: {val_acc:.1%} ({int(val_acc*len(val_idx))}/{len(val_idx)})")
    print(f"Validation macro exact-match (per-rule): {val_macro_acc:.1%}")
    print_rule_breakdown(val_by_rule, top_n=30)

    train_acc, _, _ = evaluate_exact_match(model, ds, train_idx[:50], device, max_print=5)
    print(f"Train sample exact-match: {train_acc:.1%}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:      {total_params:,} params on {device}")
    print(f"  Data:       {len(train_idx)} train / {len(val_idx)} val")
    print(f"  Training:   {train_cfg.epochs} epochs, "
          f"final loss={progress['last_loss']:.4f}, "
          f"tf_acc={progress['last_acc']:.0%}")
    print(f"  Val exact-match:   {val_acc:.1%}")
    print(f"  Best val:          {progress['best_val']:.1%}")
    print(f"  Train exact-match: {train_acc:.1%}")

    # ── Save checkpoint ───────────────────────────────────────────────
    save_checkpoint(
        model,
        CHECKPOINT_PATH,
        model_config = model_config,
        epoch        = train_cfg.epochs,
        loss         = progress["last_loss"],
        val_accuracy = val_acc,
    )

    print(f"\n  Checkpoint saved: {CHECKPOINT_PATH}")
    print(f"  Tokenizer saved:  {TOKENIZER_PATH}")
