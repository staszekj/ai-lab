"""
BPE Tokenizer for ts-type-refiner.

Trains a Byte-Pair Encoding tokenizer on TypeScript code from training data.
Handles arbitrary TS code without <unk> tokens.

Usage:
    # Train and save:
    tok = build_tokenizer(corpus_texts)
    tok.save("tokenizer.json")

    # Load:
    tok = TSTokenizer.from_file("tokenizer.json")
    ids = tok.encode("let x: string = 'on'")
    text = tok.decode(ids)
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


# ── Special tokens ───────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class TSTokenizer:
    """
    Wrapper around HuggingFace tokenizers.Tokenizer with convenience
    methods and special token IDs.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer
        self.pad_id = tokenizer.token_to_id(PAD_TOKEN)
        self.bos_id = tokenizer.token_to_id(BOS_TOKEN)
        self.eos_id = tokenizer.token_to_id(EOS_TOKEN)
        self.unk_id = tokenizer.token_to_id(UNK_TOKEN)
        self.vocab_size = tokenizer.get_vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Text → list of token IDs."""
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """List of token IDs → text."""
        if skip_special:
            special = {self.pad_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in special]
        return self._tok.decode(ids)

    def save(self, path: str | Path) -> None:
        self._tok.save(str(path))

    @classmethod
    def from_file(cls, path: str | Path) -> TSTokenizer:
        tok = Tokenizer.from_file(str(path))
        return cls(tok)

    def __repr__(self) -> str:
        return f"TSTokenizer(vocab_size={self.vocab_size})"


def build_tokenizer(
    texts: list[str],
    vocab_size: int = 512,
) -> TSTokenizer:
    """
    Train a BPE tokenizer on the given texts.

    Parameters
    ----------
    texts      : list of text strings (contexts + types from training data)
    vocab_size : target vocabulary size (512 is good for small corpus)

    Returns
    -------
    TSTokenizer ready to encode/decode
    """
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Enable padding
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id(PAD_TOKEN),
        pad_token=PAD_TOKEN,
    )

    return TSTokenizer(tokenizer)


def build_from_jsonl(jsonl_path: str | Path, vocab_size: int = 512) -> TSTokenizer:
    """
    Build tokenizer from training_pairs.jsonl.
    Uses context, preciseType, and degradedType fields.
    """
    texts: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["context"])
            texts.append(row["preciseType"])
            texts.append(row["degradedType"])

    return build_tokenizer(texts, vocab_size=vocab_size)


if __name__ == "__main__":
    import sys

    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "packages/ts-type-extractor/data/training_pairs.jsonl"
    print(f"Building BPE tokenizer from {jsonl_path}...")

    tok = build_from_jsonl(jsonl_path, vocab_size=512)
    print(tok)
    print(f"  pad_id={tok.pad_id}  bos_id={tok.bos_id}  eos_id={tok.eos_id}")
    print()

    # Test on sample texts
    examples = [
        "let enabled : string = 'on'",
        "'realClick' | 'realTouch'",
        "HTMLInputElement",
        "(value: string) => void",
        "const input = event.target as HTMLInputElement;",
    ]
    for text in examples:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        print(f"  input:   {text}")
        print(f"  tokens:  {len(ids)} ids")
        print(f"  decoded: {decoded}")
        print()

    # Save
    out_path = "packages/ts-type-refiner/tokenizer.json"
    tok.save(out_path)
    print(f"Saved to {out_path}")

    # Reload test
    tok2 = TSTokenizer.from_file(out_path)
    test_ids = tok2.encode("HTMLButtonElement | null")
    print(f"Reload test: {tok2.decode(test_ids)}")
