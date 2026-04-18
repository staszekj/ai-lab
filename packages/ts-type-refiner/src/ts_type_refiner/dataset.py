"""
Data pipeline for ts-type-refiner Phase 1.

Loads training_pairs.jsonl → tokenized (src_ids, tgt_ids) tensors
ready for ManualEncoderDecoder training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from ts_type_refiner.tokenizer import TSTokenizer


class TypeRefinerDataset:
    """
    Loads JSONL training pairs and tokenizes them.

    Each pair:
        src = context (TypeScript code with degraded type)
        tgt = preciseType (the target type to generate)

    Provides batched tensors with padding.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: TSTokenizer,
        max_src_len: int = 128,
        max_tgt_len: int = 32,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.pairs: list[tuple[str, str]] = []
        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                self.pairs.append((row["context"], row["preciseType"]))

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


def train_val_split(
    dataset: TypeRefinerDataset,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split dataset indices into train/val."""
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    return indices[val_size:], indices[:val_size]
