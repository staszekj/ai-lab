"""
Train the EncoderDecoderModel on real TypeScript type pairs.

This file is a THIN ORCHESTRATOR. It owns only the things that are
specific to the ts-type-refiner domain:

    1. Building / saving the BPE tokenizer
    2. Building the dataset and a train/val split
    3. Choosing model hyper-parameters from the data
    4. Defining the eval semantics (autoregressive exact-match accuracy)
     5. Wiring the above into `ts_type_refiner.trainer.train(...)` and
         `ts_type_refiner.checkpoint.save(...)`

Everything else (forward/backward, optimizer, grad-clip, log-prob,
state_dict serialization) lives in shared local modules in this package.

Usage:
    uv run --package ts-type-refiner refiner-train
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

from ts_type_refiner.checkpoint import load as load_checkpoint, save as save_checkpoint
from ts_type_refiner.encoder_decoder_model import EncoderDecoderModel
from ts_type_refiner.trainer import EpochStats, TrainConfig, train

from ts_type_refiner.dataset import TypeRefinerDataset, train_val_split
from ts_type_refiner.tokenizer import build_from_jsonl
from ts_type_refiner.prompt import PROMPT_VERSION
from ts_type_refiner.validators import VALIDATORS


DATA_PATH            = "packages/ts-type-extractor/data/encoder_decoder_pairs.jsonl"
TOKENIZER_PATH       = "packages/ts-type-refiner/tokenizer.json"
CHECKPOINT_DIR       = Path("packages/ts-type-refiner/checkpoints")
CHECKPOINT_PATH      = CHECKPOINT_DIR / "refiner.pt"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "refiner_best.pt"


# ══════════════════════════════════════════════════════════════════════
# Eval — autoregressive accuracy on a held-out slice
# ──────────────────────────────────────────────────────────────────────
# Three metrics per pair, two semantics depending on `isNegative`:
#
#   exact_match    : prediction == target (whitespace-stripped). Works for both
#                    positives and negatives — a "correct preserve" on a negative
#                    pair has target == degraded, so EM still captures it.
#
#   validator_pass : VALIDATORS[rule](prediction) returns True. Meaningful only
#                    on positives — on a negative, the correct answer is the
#                    *degraded* form, which validators reject by design.
#
#   acceptable     : negative-aware composite —
#                       isNegative: prediction == degraded (preserve).
#                       positive  : VALIDATORS[rule](prediction) passes.
#                    This is what Step 2.4 will use as the "good output" signal.
#
# We report macro-acc as the average per-rule rate (avoids common rules
# dominating). Heatmap dump per eval lets us see which rules collapse first.
# ══════════════════════════════════════════════════════════════════════


def evaluate_exact_match(
    model: EncoderDecoderModel,
    dataset: TypeRefinerDataset,
    indices: list[int],
    device: torch.device,
    max_print: int = 5,
) -> tuple[float, dict[str, dict[str, int]], float]:
    """
    Returns:
        em_acc   — micro exact-match accuracy (correct / total)
        by_rule  — {rule: {em, vp, acc, n, n_neg, n_neg_correct}}
        macro_acc — average of per-rule `acceptable` rates
    """
    model.eval()
    tok = dataset.tokenizer

    by_rule: dict[str, dict[str, int]] = defaultdict(
        lambda: {"em": 0, "vp": 0, "acc": 0, "n": 0, "n_neg": 0, "n_neg_correct": 0}
    )
    correct_em = 0

    for i, idx in enumerate(indices):
        src_text, expected_tgt = dataset.pairs[idx]
        rule = dataset.rules[idx] if idx < len(dataset.rules) else "<unknown>"
        is_neg = dataset.is_negative[idx] if idx < len(dataset.is_negative) else False

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
        gen_text = tok.decode(generated[0].tolist()).strip()
        expected = expected_tgt.strip()
        em = gen_text == expected

        # validator_pass: only meaningful on positives.
        validator = VALIDATORS.get(rule)
        vp = bool(validator and validator(gen_text)[0]) if not is_neg else False

        # acceptable: negative-aware composite.
        if is_neg:
            acceptable = em  # correctly preserved degraded form
        else:
            acceptable = vp

        stats = by_rule[rule]
        stats["n"] += 1
        if em:
            stats["em"] += 1
            correct_em += 1
        if vp:
            stats["vp"] += 1
        if acceptable:
            stats["acc"] += 1
        if is_neg:
            stats["n_neg"] += 1
            if em:
                stats["n_neg_correct"] += 1

        if i < max_print:
            status = "✓" if em else "✗"
            tag = " [NEG]" if is_neg else ""
            print(f"  {status}{tag} expected: {expected_tgt}")
            print(f"    got:        {gen_text}")

    if not indices:
        return 0.0, {}, 0.0

    macro_acc_vals = [s["acc"] / s["n"] for s in by_rule.values() if s["n"] > 0]
    macro_acc = sum(macro_acc_vals) / len(macro_acc_vals) if macro_acc_vals else 0.0
    em_acc = correct_em / len(indices)
    return em_acc, dict(by_rule), macro_acc


def print_rule_breakdown(by_rule: dict[str, dict[str, int]], top_n: int = 12) -> None:
    if not by_rule:
        return

    rows = sorted(by_rule.items(), key=lambda kv: kv[1]["n"], reverse=True)
    print(
        "    per-rule:  "
        + f"{'rule':<48} {'em':>6} {'vp':>6} {'acc':>6}  (n,neg)"
    )
    for rule, s in rows[:top_n]:
        n = s["n"] or 1
        em_pct = s["em"] / n
        vp_pct = s["vp"] / (n - s["n_neg"]) if n > s["n_neg"] else 0.0
        acc_pct = s["acc"] / n
        print(
            f"      {rule:<48} {em_pct:>5.0%} {vp_pct:>5.0%} {acc_pct:>5.0%}  "
            f"({s['n']:>4},{s['n_neg']:>3})"
        )


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── CLI args ──────────────────────────────────────────────────────
    # This script is intentionally small at the CLI level: it mostly wires
    # together the dataset, model, evaluation logic, and checkpoint policy.
    #
    # Example:
    #   uv run --package ts-type-refiner refiner-train --epochs 50 --patience 3
    #
    # Meaning:
    #   - train for at most 50 epochs
    #   - run validation every `eval_every` epochs from TrainConfig
    #   - stop early if validation does not improve for 3 eval rounds
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--balance-rules",
        action="store_true",
        help="Use 1/sqrt(rule_count) weighted sampling (Step 2.2).",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=400,
        help="Mid-training eval slice size (val set is full at end of run).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping: evals without improvement before stopping. 0 = disabled.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["none", "cosine", "plateau"],
        default="none",
        help="LR scheduler: 'cosine' (CosineAnnealingLR) or 'plateau' (ReduceLROnPlateau on train loss).",
    )
    cli = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────
    # Training runs on CUDA when available, otherwise on CPU.
    # Nothing else in this file depends on the device choice.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────
    # We build a BPE tokenizer directly from the training JSONL.
    # It learns TypeScript-specific subword patterns from BOTH:
    #   - encoder inputs (prompts like [REFINE rule=...])
    #   - decoder targets (precise type strings)
    #
    # Example target fragments that become tokenizable units:
    #   Map<Measurable, ObservedData>
    #   React.Dispatch<React.SetStateAction<boolean>>
    #   'realClick' | 'realTouch'
    print(f"Building BPE tokenizer from {DATA_PATH}...")
    tok = build_from_jsonl(DATA_PATH, vocab_size=2048)
    tok.save(TOKENIZER_PATH)
    print(f"  {tok}  saved to {TOKENIZER_PATH}")

    # ── Dataset ───────────────────────────────────────────────────────
    # The dataset reads ready-made encoder/decoder pairs from JSONL.
    # Each row already contains:
    #   input  = full refine prompt
    #   target = expected precise type text
    #
    # Example row shape:
    #   input  = "[REFINE rule=map→unknown | kind=variable | ...]\n---\nconst x: Map<unknown, unknown> = new Map();"
    #   target = "Map<Measurable, ObservedData>"
    #
    # train_val_split prefers the precomputed `split` field from JSONL so that
    # identical prompts cannot leak across train/validation.
    ds = TypeRefinerDataset(DATA_PATH, tok, max_src_len=256, max_tgt_len=64)
    train_idx, val_idx = train_val_split(ds, val_ratio=0.15)
    print(f"  Pairs: {len(ds)}  train={len(train_idx)}  val={len(val_idx)}")

    # Inspect token lengths before building the model.
    # We want a `max_seq_len` large enough for real prompts, but not absurdly
    # oversized. The actual tensors the model sees are token IDs after BPE, so
    # raw string length is not what matters here.
    src_lens = [len(tok.encode(ds.pairs[i][0])[:ds.max_src_len]) for i in range(len(ds))]
    tgt_lens = [len(tok.encode(ds.pairs[i][1])[:ds.max_tgt_len]) for i in range(len(ds))]
    print(f"  Src tokens: min={min(src_lens)} max={max(src_lens)} avg={sum(src_lens)//len(src_lens)}")
    print(f"  Tgt tokens: min={min(tgt_lens)} max={max(tgt_lens)} avg={sum(tgt_lens)//len(tgt_lens)}")

    # ── Model ─────────────────────────────────────────────────────────
    # The model is a small encoder-decoder transformer.
    # `max_seq_len` is chosen from observed token lengths, but with a safety
    # floor of 256 so tiny demo datasets do not produce a checkpoint that later
    # crashes on realistic inference prompts.
    #
    # Example failure we are protecting against:
    #   - tiny toy dataset yields observed max source length ~40 tokens
    #   - checkpoint is trained with max_seq_len=44
    #   - real prompt at inference is 90 tokens long
    #   - model asserts because source length exceeds model capacity
    max_seq = max(256, max(max(src_lens), max(tgt_lens)) + 4)
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
    print("\nModel: EncoderDecoderModel")
    print(f"  vocab={tok.vocab_size} d_model=256 heads=8 layers=4 max_seq={max_seq}")
    print(f"  Parameters: {total_params:,}")

    # ── Training config ───────────────────────────────────────────────
    # This config governs the OPTIMIZATION LOOP, not the transformer shape.
    #
    # Important values here:
    #   - eval_every=10: run generation-based validation every 10 epochs
    #   - log_every_batches=25: mid-epoch heartbeat so long runs show progress
    #   - balance_rules: optional sampling strategy from the CLI, wired below
    train_cfg = TrainConfig(
        epochs            = cli.epochs,
        lr                = 3e-4,
        weight_decay      = 0.01,
        max_grad_norm     = 1.0,
        eval_every        = 10,
        log_every_batches = 25,   # heartbeat — slow GPUs need to show signs of life mid-epoch
        seed              = 42,
        lr_schedule       = cli.lr_schedule,
    )
    batch_size   = 96
    eval_samples = cli.eval_samples

    print(f"\n{'='*60}")
    print(f"TRAINING — {train_cfg.epochs} epochs, batch_size={batch_size}, lr={train_cfg.lr}"
          f"  balance_rules={cli.balance_rules}")
    print(f"{'='*60}")

    # ── Wire callbacks for the pure-function trainer ──────────────────
    # `train_batches` is a FACTORY, not a one-shot generator.
    # The shared trainer calls it once per epoch to get a fresh iterable.
    # That matters because:
    #   - normal training wants a new shuffle every epoch
    #   - balanced training wants a new weighted sample every epoch
    #
    # Two modes:
    #   - default: uniform batches over train_idx
    #   - --balance-rules: oversample rare rules with 1/sqrt(count(rule))
    def train_batches():
        if cli.balance_rules:
            return ds.iter_balanced_batches(
                batch_size,
                indices=train_idx,
                epoch_size=len(train_idx),
                device=device,
            )
        return ds.iter_batches(batch_size, device=device, shuffle=True, indices=train_idx)

    def eval_fn(m: EncoderDecoderModel) -> float:
        # Mid-training validation runs on a capped prefix of the validation set
        # for speed. This is generation-time evaluation, not teacher-forced loss.
        #
        # What happens here for each validation example:
        #   1. build/gather the source prompt from dataset.pairs[idx]
        #   2. autoregressively generate a predicted target type
        #   3. score it with the rule-aware evaluation semantics
        #
        # Example positive pair:
        #   input:  [REFINE ... degraded=Map<unknown, unknown>] ...
        #   target: Map<Measurable, ObservedData>
        #   good output: validator accepts the generated type
        #
        # Example negative pair:
        #   input:  [REFINE ... degraded=Map<unknown, unknown>] ...
        #   target: Map<unknown, unknown>
        #   good output: model preserves the degraded form exactly
        eval_idx = val_idx[:eval_samples]
        em_acc, by_rule, macro_acc = evaluate_exact_match(m, ds, eval_idx, device, max_print=5)
        print(f"    → val exact-match (micro): {em_acc:.1%} ({int(em_acc*len(eval_idx))}/{len(eval_idx)})")
        print(f"    → val macro acceptable (per-rule): {macro_acc:.1%}")
        print_rule_breakdown(by_rule, top_n=12)
        # We track macro acceptable as the optimization target — it captures
        # both shape correctness on positives and preservation on negatives.
        return macro_acc

    # State captured across epochs by closures — no globals.
    # This is the mutable training progress used for:
    #   - best-checkpoint tracking
    #   - patience / early stopping
    #   - summary reporting after the loop exits
    progress: dict = {
        "best_val": 0.0,
        "best_val_epoch": 0,
        "patience_counter": 0,
        "last_epoch": 0,
        "last_loss": 0.0,
        "last_acc": 0.0,
        "stopped_early": False,
    }

    def on_epoch_end(stats: EpochStats) -> bool | None:
        # This callback is where TRAINING POLICY lives.
        # The shared trainer knows only how to optimize batches; it does not know
        # what "best" means for this domain. Here we decide:
        #   - when validation improved enough to save best checkpoint
        #   - when patience is exhausted and training should stop early
        #
        # In other words:
        #   ts_type_refiner.trainer.train(...) = mechanics of optimization
        #   on_epoch_end(...)       = ts-type-refiner-specific control policy
        progress["last_loss"]  = stats.train_loss
        progress["last_acc"]   = stats.train_tf_acc
        progress["last_epoch"] = stats.epoch
        # Always print every epoch: on slower hardware a single epoch can take
        # minutes, so sparse logging makes the run look frozen.
        print(f"  Epoch {stats.epoch:3d}/{train_cfg.epochs}  "
              f"loss={stats.train_loss:.4f}  "
              f"tf_acc={stats.train_tf_acc:.0%}  "
              f"{stats.elapsed_s:.1f}s", flush=True)
        if stats.val_metric is not None:
            if stats.val_metric > progress["best_val"]:
                # Validation improved -> remember this epoch and save a separate
                # best checkpoint immediately. This file can be overwritten many
                # times during one run as the model improves.
                progress["best_val"]         = stats.val_metric
                progress["best_val_epoch"]   = stats.epoch
                progress["patience_counter"] = 0
                save_checkpoint(
                    model, BEST_CHECKPOINT_PATH,
                    model_config   = model_config,
                    epoch          = stats.epoch,
                    loss           = stats.train_loss,
                    val_accuracy   = stats.val_metric,
                    prompt_version = PROMPT_VERSION,
                )
                print(f"    ✓ new best {stats.val_metric:.1%} @ epoch {stats.epoch}"
                      f" → {BEST_CHECKPOINT_PATH.name}", flush=True)
            elif cli.patience > 0:
                # Validation did not improve -> count one patience strike.
                # When the strike count reaches the configured patience, ask the
                # trainer to break out of the epoch loop.
                progress["patience_counter"] += 1
                remaining = cli.patience - progress["patience_counter"]
                print(f"    no improvement "
                      f"({progress['patience_counter']}/{cli.patience},"
                      f" {remaining} evals left)", flush=True)
                if progress["patience_counter"] >= cli.patience:
                    print(f"  *** Early stopping at epoch {stats.epoch} ***",
                          flush=True)
                    progress["stopped_early"] = True
                    return True
        return None

    # ── Train ─────────────────────────────────────────────────────────
    # This is the hand-off point into the generic optimization loop.
    # From here, the trainer will:
    #   1. iterate epochs
    #   2. consume batches from train_batches()
    #   3. compute loss/backprop/update
    #   4. occasionally call eval_fn(model)
    #   5. call on_epoch_end(stats), which may request early stop
    train(
        model         = model,
        train_batches = train_batches,
        pad_id        = tok.pad_id,
        cfg           = train_cfg,
        eval_fn       = eval_fn,
        on_epoch_end  = on_epoch_end,
    )

    # ── Restore best weights ──────────────────────────────────────────
    # After the loop ends, the model in memory may correspond to the LAST epoch,
    # not the BEST epoch. We therefore reload the best checkpoint before running
    # the full validation pass and before saving the final public checkpoint.
    if BEST_CHECKPOINT_PATH.exists():
        ckpt = load_checkpoint(BEST_CHECKPOINT_PATH, device)
        model.load_state_dict(ckpt.state_dict)
        print(f"\nRestored best model: epoch {progress['best_val_epoch']},"
              f" val={progress['best_val']:.1%}")

    # ── Final evaluation ──────────────────────────────────────────────
    # Mid-training eval used only a slice for speed. Here we run generation on
    # the FULL validation set so the summary reflects the restored best model
    # against all held-out data.
    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION — generation on {len(val_idx)} held-out pairs (full set)")
    print(f"{'='*60}")

    val_em, val_by_rule, val_macro_acc = evaluate_exact_match(model, ds, val_idx, device, max_print=15)
    print(f"\nValidation exact-match (micro): {val_em:.1%} ({int(val_em*len(val_idx))}/{len(val_idx)})")
    print(f"Validation macro acceptable (per-rule): {val_macro_acc:.1%}")
    print_rule_breakdown(val_by_rule, top_n=40)

    train_em, _, _ = evaluate_exact_match(model, ds, train_idx[:50], device, max_print=5)
    print(f"Train sample exact-match: {train_em:.1%}")

    # ── Summary ───────────────────────────────────────────────────────
    # Final report combines:
    #   - dataset size
    #   - how many epochs actually ran
    #   - final train-side diagnostics
    #   - best validation numbers used for checkpoint selection
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:      {total_params:,} params on {device}")
    print(f"  Data:       {len(train_idx)} train / {len(val_idx)} val")
    print(f"  Training:   {train_cfg.epochs} epochs max, "
          f"ran {progress['last_epoch']} epochs, "
          f"lr_schedule={cli.lr_schedule}")
    print(f"  Final loss: {progress['last_loss']:.4f}  "
          f"tf_acc={progress['last_acc']:.0%}")
    if progress["stopped_early"]:
        print(f"  Stopped:    early (patience={cli.patience} evals)")
    print(f"  Best epoch:                {progress['best_val_epoch']}")
    print(f"  Val exact-match (micro):   {val_em:.1%}")
    print(f"  Val macro acceptable:      {val_macro_acc:.1%}")
    print(f"  Best val (during train):   {progress['best_val']:.1%}")
    print(f"  Train exact-match:         {train_em:.1%}")

    # ── Save checkpoint ───────────────────────────────────────────────
    # Public output checkpoint. At this point `model` already contains the
    # restored BEST weights, so refiner.pt becomes the artifact used later by
    # refiner-infer in the playground/runtime pipeline.
    save_checkpoint(
        model,
        CHECKPOINT_PATH,
        model_config   = model_config,
        epoch          = progress["best_val_epoch"],
        loss           = progress["last_loss"],
        val_accuracy   = val_em,
        prompt_version = PROMPT_VERSION,
    )

    print(f"\n  Checkpoint saved: {CHECKPOINT_PATH}")
    print(f"  Tokenizer saved:  {TOKENIZER_PATH}")
