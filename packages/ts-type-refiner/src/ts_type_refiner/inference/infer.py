"""
Inference: predict precise types for refiner candidates.

Reads candidates JSONL produced by `refiner-locate.ts`, runs the
trained EncoderDecoderModel through `ts_type_refiner.inference.predictor.Predictor`,
validates each suggestion against the candidate's rule, and emits an
edit JSONL ready for `refiner-apply.ts`.

This file is a THIN ORCHESTRATOR: load checkpoint → build predictor →
loop over candidates → route through validator → write output. No
torch tensors are constructed here; that's the predictor's job.

Input  (one candidate per line):
    {id, file, line, start, end, kind, name, context, degradedType, rule}

Output (one prediction per line):
    {id, file, start, end, degradedType, suggestion, accepted, reason,
    logprob, ruleValidatorPassed, proposals}

Usage:
    uv run --package ts-type-refiner refiner-infer \\
        --input  candidates.jsonl \\
        --output edits.jsonl \\
        --checkpoint packages/ts-type-refiner/checkpoints/refiner.pt \\
        --tokenizer  packages/ts-type-refiner/tokenizer.json \\
        [--min-logprob -8.0]

Worked example:
    Candidate row from refiner-locate:
        {
            "file": "samples/Foo.tsx",
            "kind": "variable",
            "name": "observedElements",
            "context": "const observedElements: Map<unknown, unknown> = new Map();",
            "degradedType": "Map<unknown, unknown>",
            "rule": "map→unknown",
            "siblings": ""
        }

    Prompt built for the model:
        [REFINE rule=map→unknown | kind=variable | name=observedElements | degraded=Map<unknown, unknown>]
        ---
        const observedElements: Map<unknown, unknown> = new Map();

    Possible model outputs:
        - Map<Measurable, ObservedData>
        - Map<string, boolean>
        - Map<any, any>

    Validation logic:
        - the proposal must satisfy the validator for rule "map→unknown"
        - the proposal must pass the min-logprob threshold
        - if multiple rule hypotheses exist for one source span, the best
          accepted one wins for that span

    Example accepted edit row:
        {
            "file": "samples/Foo.tsx",
            "start": 24,
            "end": 45,
            "degradedType": "Map<unknown, unknown>",
            "suggestion": "Map<Measurable, ObservedData>",
            "accepted": true,
            "reason": "ok"
        }
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from ts_type_refiner.checkpoint import build_model, load as load_checkpoint
from ts_type_refiner.inference.predictor import Predictor

from ts_type_refiner.prompt import build_refine_prompt, PROMPT_VERSION
from ts_type_refiner.tokenizer import TSTokenizer
from ts_type_refiner.rules.validators import VALIDATORS


# ══════════════════════════════════════════════════════════════════════
# Identifier-overlap analysis (Step 2.4)
# ──────────────────────────────────────────────────────────────────────
# When *every* proposal fails the rule validator we want to distinguish
# two failure modes:
#
#   (a) shape failure — the proposal has the right identifiers but the
#       wrong structure (e.g. wrapped `HTMLDivElement` in extra `<>`).
#   (b) hallucinated identifier — the proposal references a type name
#       that does not appear anywhere in the prompt's siblings. The
#       model invented it, and the user almost never wants that.
#
# Recoverability of the target identifier from siblings is currently
# only ~18% (see Step 1.7), so we DO NOT use the overlap as a hard
# filter — that would reject too many correct predictions. We only use
# it as a diagnostic label and per-proposal annotation.
# ══════════════════════════════════════════════════════════════════════

_TRIVIAL_IDENTS = frozenset({
    "string", "number", "boolean", "unknown", "any", "never",
    "void", "null", "undefined", "object", "true", "false",
    "Array", "Promise", "Record", "Partial", "Required", "Readonly",
    "Pick", "Omit", "Exclude", "Extract", "NonNullable", "ReturnType",
    "Parameters", "InstanceType", "Awaited", "Map", "Set", "Date",
    "React", "JSX",
})

_IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")


def _non_trivial_idents(s: str) -> set[str]:
    return {m.group(0) for m in _IDENT_RE.finditer(s or "")} - _TRIVIAL_IDENTS


def _ident_overlap(proposal_text: str, siblings: str) -> tuple[int, int]:
    """Return (overlap, total) for non-trivial idents in proposal vs siblings."""
    prop_idents = _non_trivial_idents(proposal_text)
    if not prop_idents:
        return 0, 0
    sib_idents = _non_trivial_idents(siblings)
    return len(prop_idents & sib_idents), len(prop_idents)


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
    p.add_argument("--num-candidates", type=int, default=5,
                   help="number of decoder proposals to consider per rule hypothesis")
    p.add_argument("--candidate-attempts", type=int, default=0,
                   help="sampling attempts to collect unique candidates (0=auto)")
    p.add_argument("--candidate-temperature", type=float, default=0.7,
                   help="sampling temperature used when collecting candidate list")
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load tokenizer ────────────────────────────────────────────────
    tok = TSTokenizer.from_file(args.tokenizer)
    print(f"Tokenizer: {tok}")

    # ── Load checkpoint + rebuild model ───────────────────────────────
    ckpt  = load_checkpoint(args.checkpoint, device=device)
    ckpt_prompt_version = ckpt.extras.get("prompt_version")
    if ckpt_prompt_version != PROMPT_VERSION:
        print(
            f"WARNING: checkpoint prompt_version={ckpt_prompt_version!r} "
            f"!= current PROMPT_VERSION={PROMPT_VERSION}. "
            "Predictions may be degraded; retrain after format changes."
        )
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
                "proposals": [],
            }
            best_by_id[c["id"]] = choose_better(best_by_id.get(c["id"]), edit)
            continue

        # Predict top-N proposals; pick first validator+logprob pass.
        prompt = build_refine_prompt(
            context=c["context"],
            name=c.get("name", "<unknown>"),
            kind=c.get("kind", "<unknown>"),
            rule=rule,
            degraded_type=c.get("degradedType", "<unknown>"),
            siblings=c.get("siblings"),
        )
        proposals = predict.predict_n(
            prompt,
            n=args.num_candidates,
            attempts=args.candidate_attempts,
            temperature=args.candidate_temperature,
        )

        siblings_str = c.get("siblings", "") or ""

        selected = None
        validator_passed_any = False
        any_proposal_grounded = False  # at least one proposal's idents overlap with siblings
        best_validator_reason = "no valid proposal"
        best_valid_below_threshold_lp = float("-inf")
        best_valid_below_threshold_text = ""

        for prop in proposals:
            overlap, total_idents = _ident_overlap(prop.text, siblings_str)
            if total_idents > 0 and overlap > 0:
                any_proposal_grounded = True

            ok, reason = validator(prop.text)
            if not ok:
                best_validator_reason = reason
                continue

            validator_passed_any = True
            if prop.mean_logprob < args.min_logprob:
                if prop.mean_logprob > best_valid_below_threshold_lp:
                    best_valid_below_threshold_lp = prop.mean_logprob
                    best_valid_below_threshold_text = prop.text
                continue

            selected = prop
            best_validator_reason = reason
            break

        accepted = selected is not None
        if not validator_passed_any:
            n_rejected_validator += 1
            # Diagnostic refinement (Step 2.4): label hallucinated-identifier
            # failures explicitly. This does not affect acceptance — it only
            # makes the reason field actionable downstream.
            if not any_proposal_grounded and any(
                _ident_overlap(p.text, siblings_str)[1] > 0 for p in proposals
            ):
                best_validator_reason = (
                    f"hallucinated_identifier: {best_validator_reason}"
                )
        elif not accepted:
            n_rejected_logprob += 1
        if accepted:
            n_accepted += 1

        if accepted:
            suggestion = selected.text
            logprob = selected.mean_logprob
            reason = best_validator_reason
        elif validator_passed_any:
            suggestion = best_valid_below_threshold_text
            logprob = best_valid_below_threshold_lp
            reason = f"logprob {best_valid_below_threshold_lp:.2f} < {args.min_logprob}"
        else:
            suggestion = proposals[0].text if proposals else ""
            logprob = proposals[0].mean_logprob if proposals else None
            reason = best_validator_reason

        edit = {
            **{k: c[k] for k in ("id", "file", "start", "end", "degradedType")},
            "suggestion":          suggestion,
            "accepted":            accepted,
            "reason":              reason,
            "logprob":             logprob,
            "ruleValidatorPassed": validator_passed_any,
            "rule":                rule,
            "proposals": [
                {
                    "text": p.text,
                    "logprob": p.mean_logprob,
                    "normalized_prob": p.normalized_prob,
                    "identsInSiblings": _ident_overlap(p.text, siblings_str)[0],
                    "nonTrivialIdents": _ident_overlap(p.text, siblings_str)[1],
                }
                for p in proposals
            ],
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
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Candidates:           {len(candidates)}")
    print(f"  Unique candidate IDs: {len(best_by_id)}")
    print(f"  Merged hypotheses:    {n_merged_ids}")
    print(f"  Accepted:             {n_accepted}")
    print(f"  Rejected (validator): {n_rejected_validator}")
    print(f"  Rejected (logprob):   {n_rejected_logprob}")
    print(f"  Output:               {out_path}")
