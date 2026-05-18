"""
Data pipeline for ts-type-refiner training.

Loads encoder_decoder_pairs.jsonl → tokenized (src_ids, tgt_ids) tensors
ready for EncoderDecoderModel training.

The input file is produced by `degrade.ts`. Each row has:
    {"input": <encoder prompt>, "target": <decoder target>, "rule": <rule name>,
     ...diagnostics ignored at training time}
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch

from ts_type_refiner.tokenizer import TSTokenizer


class TypeRefinerDataset:
    """
    Loads materialized JSONL pairs and tokenizes them.

    Each row has explicit `input` / `target` fields, so no prompt
    formatting happens here — see `prompt.py::build_refine_prompt`
    (mirrored in `degrade.ts`) for the encoder input format.

    Provides batched tensors with padding.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: TSTokenizer,
        max_src_len: int = 128,
        max_tgt_len: int = 32,
        split: str | None = None,
    ) -> None:
        """
        Load encoder/decoder pairs.

        If `split` is given ("train" or "val"), only rows whose `split` field
        matches are kept. Rows without a `split` field are kept unconditionally
        (back-compat with pre-Step-1.5 datasets).
        """
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.pairs: list[tuple[str, str]] = []
        self.rules: list[str] = []
        self.splits: list[str | None] = []
        self.is_negative: list[bool] = []
        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                if split is not None:
                    row_split = row.get("split")
                    if row_split is not None and row_split != split:
                        continue
                self.pairs.append((row["input"], row["target"]))
                self.rules.append(row.get("rule", "<unknown>"))
                self.splits.append(row.get("split"))
                self.is_negative.append(bool(row.get("isNegative", False)))

    def __len__(self) -> int:
        return len(self.pairs)

    def encode_pair(self, idx: int) -> tuple[list[int], list[int], list[int]]:
        """
        Encode one pair → (src_ids, tgt_input_ids, tgt_target_ids).

        tgt_input  = [<bos>] + target_tokens   (decoder input, teacher forcing)
        tgt_target = target_tokens + [<eos>]    (what we predict)
        """
        src_text, tgt_text = self.pairs[idx]
        tok = self.tokenizer

        src_ids = tok.encode(src_text)[:self.max_src_len]
        tgt_ids = tok.encode(tgt_text)[:self.max_tgt_len]

        tgt_input = [tok.bos_id] + tgt_ids
        tgt_target = tgt_ids + [tok.eos_id]

        return src_ids, tgt_input, tgt_target

    def get_batch(
        self,
        indices: list[int],
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a padded batch from the given indices.

        Returns (src_batch, tgt_input_batch, tgt_target_batch), all on device.
        """
        pad_id = self.tokenizer.pad_id
        src_seqs, tgt_in_seqs, tgt_tgt_seqs = [], [], []

        for idx in indices:
            s, ti, tt = self.encode_pair(idx)
            src_seqs.append(s)
            tgt_in_seqs.append(ti)
            tgt_tgt_seqs.append(tt)

        def pad(seqs: list[list[int]]) -> torch.Tensor:
            max_len = max(len(s) for s in seqs)
            padded = [s + [pad_id] * (max_len - len(s)) for s in seqs]
            return torch.tensor(padded, dtype=torch.long, device=device)

        return pad(src_seqs), pad(tgt_in_seqs), pad(tgt_tgt_seqs)

    def iter_batches(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        shuffle: bool = True,
        indices: list[int] | None = None,
    ):
        """Yield (src, tgt_input, tgt_target) batches over given indices."""
        idx_list = list(indices) if indices is not None else list(range(len(self.pairs)))
        if shuffle:
            random.shuffle(idx_list)

        for start in range(0, len(idx_list), batch_size):
            batch_idx = idx_list[start : start + batch_size]
            yield self.get_batch(batch_idx, device=device)

    def iter_balanced_batches(
        self,
        batch_size: int,
        indices: list[int],
        epoch_size: int,
        device: torch.device | str = "cpu",
        rng: random.Random | None = None,
    ):
        """
        Rule-balanced sampler (Step 2.2).

        For each rule R let `count(R)` be the number of pairs with that rule
        in `indices`. We sample each pair with weight proportional to
        `1 / sqrt(count(rule(pair)))`, normalized so that all weights sum to 1.

        This levels per-batch rule coverage: tail rules get ~sqrt-amplified
        sampling frequency, common rules get ~sqrt-suppressed frequency.
        `epoch_size` defines how many samples constitute one "epoch" — usually
        the same as `len(indices)` so wall-clock per epoch matches uniform
        sampling.
        """
        rng = rng or random
        # Per-rule frequency over the *given* index set.
        counts: dict[str, int] = {}
        for i in indices:
            counts[self.rules[i]] = counts.get(self.rules[i], 0) + 1

        # Weight per sample = 1 / sqrt(count(rule)).
        weights = [1.0 / math.sqrt(counts[self.rules[i]]) for i in indices]
        total = sum(weights)
        weights = [w / total for w in weights]

        # `random.choices` is O(n) per draw. For our scale (~25k indices,
        # ~25k draws per epoch) it's fine; pre-compute cumulative for speed.
        cum: list[float] = []
        acc = 0.0
        for w in weights:
            acc += w
            cum.append(acc)

        def draw_one() -> int:
            r = rng.random()
            # Binary search for the first cumulative weight ≥ r.
            lo, hi = 0, len(cum) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cum[mid] < r:
                    lo = mid + 1
                else:
                    hi = mid
            return indices[lo]

        drawn: list[int] = [draw_one() for _ in range(epoch_size)]
        for start in range(0, len(drawn), batch_size):
            batch_idx = drawn[start : start + batch_size]
            yield self.get_batch(batch_idx, device=device)
            yield self.get_batch(batch_idx, device=device)


def train_val_split(
    dataset: TypeRefinerDataset,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Split dataset indices into train/val.

    Preferred path (Step 1.5): if every row carries a `split` field, honor it
    deterministically. Falls back to random shuffle-and-cut only for legacy
    datasets that lack the field.
    """
    if dataset.splits and all(s is not None for s in dataset.splits):
        train_idx = [i for i, s in enumerate(dataset.splits) if s == "train"]
        val_idx = [i for i, s in enumerate(dataset.splits) if s == "val"]
        return train_idx, val_idx

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    return indices[val_size:], indices[:val_size]
