"""
Inference: predict precise types for refiner candidates.

Reads candidates JSONL produced by `refiner-locate.ts`, runs the
trained EncoderDecoderModel through `core.predictor.Predictor`,
validates each suggestion against the candidate's rule, and emits an
edit JSONL ready for `refiner-apply.ts`.

This file is a THIN ORCHESTRATOR: load checkpoint → build predictor →
loop over candidates → route through validator → write output. No
torch tensors are constructed here; that's the predictor's job.

Input  (one candidate per line):
    {id, file, line, start, end, kind, name, context, degradedType, rule}

Output (one prediction per line):
    {id, file, start, end, degradedType, suggestion, accepted, reason,
     logprob, ruleValidatorPassed}

Usage:
    uv run --package ts-type-refiner refiner-infer \\
        --input  candidates.jsonl \\
        --output edits.jsonl \\
        --checkpoint packages/ts-type-refiner/checkpoints/refiner.pt \\
        --tokenizer  packages/ts-type-refiner/tokenizer.json \\
        [--min-logprob -8.0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from core.checkpoint import build_model, load as load_checkpoint
from core.predictor import Predictor

from ts_type_refiner.prompt import build_refine_prompt
from ts_type_refiner.tokenizer import TSTokenizer
from ts_type_refiner.validators import VALIDATORS


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ts-type-refiner Phase 2 inference")
    p.add_argument("--input",      required=True, help="candidates.jsonl from refiner-locate")
    p.add_argument("--output",     required=True, help="edits.jsonl for refiner-apply")
    p.add_argument("--checkpoint", default="packages/ts-type-refiner/checkpoints/refiner.pt")
    p.add_argument("--tokenizer",  default="packages/ts-type-refiner/tokenizer.json")
    p.add_argument("--min-logprob", type=float, default=float("-inf"),
                   help="reject suggestions with mean token log-prob below this")
    p.add_argument("--max-src-len", type=int, default=256)
    p.add_argument("--max-tgt-len", type=int, default=64)
    p.add_argument("--limit", type=int, default=0,
                   help="process at most N candidates (0 = all)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load tokenizer ────────────────────────────────────────────────
    tok = TSTokenizer.from_file(args.tokenizer)
    print(f"Tokenizer: {tok}")

    # ── Load checkpoint + rebuild model ───────────────────────────────
    ckpt  = load_checkpoint(args.checkpoint, device=device)
    model = build_model(ckpt.model_config, device=device)
    model.load_state_dict(ckpt.state_dict)

    n_params = sum(p.numel() for p in model.parameters())
    val_acc  = ckpt.extras.get("val_accuracy", "?")
    print(f"Model:     {n_params:,} params  (val_acc={val_acc})")

    # ── Build predictor (model.eval() happens inside) ─────────────────
    predict = Predictor(
        model,
        encode      = tok.encode,
        decode      = tok.decode,
        bos_id      = tok.bos_id,
        eos_id      = tok.eos_id,
        max_src_len = args.max_src_len,
        max_tgt_len = args.max_tgt_len,
        device      = device,
    )

    # ── Load candidates ───────────────────────────────────────────────
    in_path  = Path(args.input)
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
    n_accepted           = 0
    n_rejected_validator = 0
    n_rejected_logprob   = 0
    n_merged_ids         = 0
    t0       = time.time()
    last_log = t0

    best_by_id: dict[str, dict] = {}

    def choose_better(curr: dict | None, cand: dict) -> dict:
        if curr is None:
            return cand
        curr_ok = bool(curr.get("accepted", False))
        cand_ok = bool(cand.get("accepted", False))
        if cand_ok and not curr_ok:
            return cand
        if curr_ok and not cand_ok:
            return curr
        curr_lp = curr.get("logprob")
        cand_lp = cand.get("logprob")
        curr_lp = float("-inf") if curr_lp is None else curr_lp
        cand_lp = float("-inf") if cand_lp is None else cand_lp
        return cand if cand_lp > curr_lp else curr

    for i, c in enumerate(candidates, 1):
        rule      = c["rule"]
        validator = VALIDATORS.get(rule)

        # No validator registered → emit a record but never edit.
        # This is the SAFETY NET requirement: an unknown rule
        # cannot accidentally produce an `accepted=true` row.
        if validator is None:
            edit = {
                **{k: c[k] for k in ("id", "file", "start", "end", "degradedType")},
                "suggestion": "",
                "accepted": False,
                "reason":   f"no validator for rule {rule}",
                "logprob":  None,
                "ruleValidatorPassed": False,
                "rule": rule,
            }
            best_by_id[c["id"]] = choose_better(best_by_id.get(c["id"]), edit)
            continue

        # Predict → (text, ids, mean_logprob).
        prompt = build_refine_prompt(
            context=c["context"],
            name=c.get("name", "<unknown>"),
            kind=c.get("kind", "<unknown>"),
            rule=rule,
            degraded_type=c.get("degradedType", "<unknown>"),
            file=c.get("file"),
            line=c.get("line"),
        )
        result   = predict(prompt)
        ok, reason = validator(result.text)

        # Apply two independent gates: shape regex AND log-prob.
        # Gate ordering matters for the diagnostic `reason`:
        #   1. validator failure  →  "not a string-literal union ..."
        #   2. logprob too low    →  "logprob -9.30 < -8.00"
        accepted = ok and result.mean_logprob >= args.min_logprob
        if not ok:
            n_rejected_validator += 1
        elif not accepted:
            n_rejected_logprob += 1
            reason = f"logprob {result.mean_logprob:.2f} < {args.min_logprob}"
        if accepted:
            n_accepted += 1

        edit = {
            **{k: c[k] for k in ("id", "file", "start", "end", "degradedType")},
            "suggestion":          result.text,
            "accepted":            accepted,
            "reason":              reason,
            "logprob":             result.mean_logprob,
            "ruleValidatorPassed": ok,
            "rule":                rule,
        }
        prev = best_by_id.get(c["id"])
        if prev is not None:
            n_merged_ids += 1
        best_by_id[c["id"]] = choose_better(prev, edit)

        now = time.time()
        if now - last_log >= 2.0 or i == len(candidates):
            rate = i / max(now - t0, 1e-6)
            print(f"  {i}/{len(candidates)}  "
                  f"accepted={n_accepted}  "
                  f"rej_validator={n_rejected_validator}  "
                  f"rej_logprob={n_rejected_logprob}  "
                  f"({rate:.1f} cand/s)")
            last_log = now

    with out_path.open("w") as fout:
        for edit in best_by_id.values():
            fout.write(json.dumps(edit) + "\n")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Candidates:           {len(candidates)}")
    print(f"  Unique candidate IDs: {len(best_by_id)}")
    print(f"  Merged hypotheses:    {n_merged_ids}")
    print(f"  Accepted:             {n_accepted}")
    print(f"  Rejected (validator): {n_rejected_validator}")
    print(f"  Rejected (logprob):   {n_rejected_logprob}")
    print(f"  Output:               {out_path}")
