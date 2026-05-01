"""
Phase 2 — Inference: predict precise types for refiner candidates.

Reads candidates JSONL produced by `refiner-locate.ts`, runs the
trained ManualEncoderDecoder, validates each suggestion against the
candidate's rule, and emits an edit JSONL ready for `refiner-apply.ts`.

Input  (one candidate per line):
    {id, file, line, start, end, kind, name, context, degradedType, rule}

Output (one prediction per line):
    {id, file, start, end, degradedType, suggestion, accepted, reason,
     logprob, ruleValidatorPassed}

Usage:
    uv run --package ts-type-refiner phase2-infer \\
        --input  candidates.jsonl \\
        --output edits.jsonl \\
        --checkpoint packages/ts-type-refiner/checkpoints/phase1_model.pt \\
        --tokenizer  packages/ts-type-refiner/tokenizer.json \\
        [--min-logprob -8.0]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from core.manual_encoder_decoder import ManualEncoderDecoder
from ts_type_refiner.tokenizer import TSTokenizer


# ══════════════════════════════════════════════════════════════════════
# Per-rule output validators
# ──────────────────────────────────────────────────────────────────────
# A validator returns (ok, reason). It is the SAFETY NET: even if the
# model outputs garbage, we never propose an edit unless it matches the
# expected shape for the rule.
# ══════════════════════════════════════════════════════════════════════

# Accept both single- and double-quoted literals to mirror degrade.ts
# rule 11, which uses /^["']/ when generating training pairs.
_STR_LIT = r"(?:'[^']*'|\"[^\"]*\")"
_STR_LIT_UNION_RE = re.compile(
    rf"^\s*{_STR_LIT}(\s*\|\s*{_STR_LIT})+\s*$"
)


def validate_string_literal_union(suggestion: str) -> tuple[bool, str]:
    s = suggestion.strip()
    if not _STR_LIT_UNION_RE.match(s):
        return False, "not a string-literal union (expect 'a' | 'b' | ...)"
    return True, "ok"


VALIDATORS = {
    "string_literal_union→string": validate_string_literal_union,
}


# ══════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_with_logprob(
    model: ManualEncoderDecoder,
    tok: TSTokenizer,
    src_text: str,
    max_src_len: int,
    max_tgt_len: int,
    device: torch.device,
) -> tuple[str, float]:
    """
    Run autoregressive generation, then compute the sequence log-prob
    in a single teacher-forced forward pass.

    Returns (decoded_text, mean_token_logprob).
    """
    src_ids = tok.encode(src_text)[:max_src_len]
    src_tensor = torch.tensor([src_ids], device=device)

    gen = model.generate(
        src_tensor,
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
        max_new_tokens=max_tgt_len,
        temperature=0.01,
        verbose=False,
    )
    # gen: (1, gen_len) without BOS.
    gen_ids = gen[0].tolist()

    # ── Log-prob (post-hoc) ──────────────────────────────────────────
    # Build (decoder_input, decoder_target) the same way training does:
    #   in     = [BOS] + gen_ids[:-1]
    #   target =         gen_ids
    # Then sum log p(target_t | in_{0..t}).
    if len(gen_ids) == 0:
        return "", float("-inf")

    dec_in  = torch.tensor([[tok.bos_id] + gen_ids[:-1]], device=device)
    dec_tgt = torch.tensor([gen_ids], device=device)

    logits = model(src_tensor, dec_in, verbose=False)  # (1, T, V)
    logp   = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(2, dec_tgt.unsqueeze(-1)).squeeze(-1)  # (1, T)
    mean_lp = token_logp.mean().item()

    text = tok.decode(gen_ids)
    return text, mean_lp


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ts-type-refiner Phase 2 inference")
    p.add_argument("--input",      required=True, help="candidates.jsonl from refiner-locate")
    p.add_argument("--output",     required=True, help="edits.jsonl for refiner-apply")
    p.add_argument("--checkpoint", default="packages/ts-type-refiner/checkpoints/phase1_model.pt")
    p.add_argument("--tokenizer",  default="packages/ts-type-refiner/tokenizer.json")
    p.add_argument("--min-logprob", type=float, default=float("-inf"),
                   help="reject suggestions with mean token log-prob below this")
    p.add_argument("--max-src-len", type=int, default=256)
    p.add_argument("--max-tgt-len", type=int, default=64)
    p.add_argument("--limit", type=int, default=0,
                   help="process at most N candidates (0 = all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load tokenizer ────────────────────────────────────────────────
    tok = TSTokenizer.from_file(args.tokenizer)
    print(f"Tokenizer: {tok}")

    # ── Load checkpoint ───────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = ManualEncoderDecoder(
        vocab_size=cfg["vocab_size"],
        max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        num_layers=cfg["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model:     {n_params:,} params  (val_acc={ckpt.get('val_accuracy', '?')})")

    # ── Load candidates ───────────────────────────────────────────────
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates: list[dict] = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    if args.limit > 0:
        candidates = candidates[: args.limit]
    print(f"Candidates: {len(candidates)}  → {out_path}")

    # ── Inference loop ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    n_accepted = 0
    n_rejected_validator = 0
    n_rejected_logprob = 0
    t0 = time.time()
    last_log = t0

    with out_path.open("w") as fout:
        for i, c in enumerate(candidates, 1):
            rule = c["rule"]
            validator = VALIDATORS.get(rule)
            if validator is None:
                edit = {
                    **{k: c[k] for k in ("id", "file", "start", "end", "degradedType")},
                    "suggestion": "",
                    "accepted": False,
                    "reason": f"no validator for rule {rule}",
                    "logprob": None,
                    "ruleValidatorPassed": False,
                }
                fout.write(json.dumps(edit) + "\n")
                continue

            suggestion, logp = predict_with_logprob(
                model, tok, c["context"],
                max_src_len=args.max_src_len,
                max_tgt_len=args.max_tgt_len,
                device=device,
            )
            ok, reason = validator(suggestion)
            accepted = ok and logp >= args.min_logprob
            if not ok:
                n_rejected_validator += 1
            elif not accepted:
                n_rejected_logprob += 1
                reason = f"logprob {logp:.2f} < {args.min_logprob}"
            if accepted:
                n_accepted += 1

            edit = {
                "id": c["id"],
                "file": c["file"],
                "start": c["start"],
                "end": c["end"],
                "degradedType": c["degradedType"],
                "suggestion": suggestion.strip(),
                "accepted": accepted,
                "reason": reason,
                "logprob": logp,
                "ruleValidatorPassed": ok,
            }
            fout.write(json.dumps(edit) + "\n")

            now = time.time()
            if now - last_log >= 2.0 or i == len(candidates):
                pct = i / len(candidates) * 100
                rate = i / (now - t0)
                print(f"  {i:5d}/{len(candidates)} ({pct:.0f}%)  "
                      f"accepted={n_accepted}  "
                      f"rej_validator={n_rejected_validator}  "
                      f"rej_logprob={n_rejected_logprob}  "
                      f"{rate:.1f}/s")
                last_log = now

    elapsed = time.time() - t0
    print(f"\n{'─'*60}")
    print(f"  Total:                {len(candidates)}")
    print(f"  Accepted:             {n_accepted}")
    print(f"  Rejected (validator): {n_rejected_validator}")
    print(f"  Rejected (logprob):   {n_rejected_logprob}")
    print(f"  Elapsed:              {elapsed:.1f}s")
    print(f"  Output:               {out_path}")


if __name__ == "__main__":
    main()
