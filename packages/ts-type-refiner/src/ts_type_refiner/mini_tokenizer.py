"""
Mini tokenizer for Phase 0 of ts-type-refiner.

A hand-built vocabulary covering basic TypeScript type patterns.
Tokens are split on whitespace and punctuation boundaries.
Special tokens: <pad>, <bos>, <eos>, <unk>.
"""

import re


# ── Special tokens ───────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# ── TypeScript vocabulary ────────────────────────────────────────────
# Keywords and identifiers common in type contexts
TS_TOKENS = [
    # keywords
    "let", "const", "var", "function", "type",
    # type keywords
    "string", "number", "boolean", "any", "void", "null", "undefined",
    # punctuation & operators
    ":", ";", "=", "|", "&", ",", ".", "?",
    "(", ")", "{", "}", "[", "]", "<", ">",
    "=>",
    # common string literals (for Phase 0 examples)
    "'on'", "'off'",
    "'a'", "'b'", "'c'",
    "'ltr'", "'rtl'",
    "'click'", "'hover'",
    "'red'", "'green'", "'blue'",
    "'sm'", "'md'", "'lg'",
    "'realClick'", "'realTouch'",
    # common identifiers (for Phase 0 examples)
    "enabled", "disabled", "direction", "size", "color", "action",
    "f", "x", "y", "name", "value", "data", "event", "input",
    "Array", "Record", "Promise",
]


class MiniTokenizer:
    """
    Simple word-level tokenizer for Phase 0.

    Splits on whitespace and common TS punctuation, then maps each
    token to an integer ID via a fixed vocabulary.

    Usage:
        tok = MiniTokenizer()
        ids = tok.encode("let enabled : string = 'on'")
        text = tok.decode(ids)
    """

    def __init__(self, extra_tokens: list[str] | None = None):
        vocab_list = list(SPECIAL_TOKENS)
        seen = set(vocab_list)
        for t in TS_TOKENS:
            if t not in seen:
                vocab_list.append(t)
                seen.add(t)
        if extra_tokens:
            for t in extra_tokens:
                if t not in seen:
                    vocab_list.append(t)
                    seen.add(t)

        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(vocab_list)}
        self.id_to_token: dict[int, str] = {i: t for i, t in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)

        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.bos_id = self.token_to_id[BOS_TOKEN]
        self.eos_id = self.token_to_id[EOS_TOKEN]
        self.unk_id = self.token_to_id[UNK_TOKEN]

    # ── Tokenization regex ───────────────────────────────────────────
    # Matches:  quoted strings ('...'), =>, multi-char or single-char punctuation, words
    _SPLIT_RE = re.compile(
        r"'[^']*'"        # single-quoted string literals
        r"|=>"            # arrow
        r"|[:<>;=|&,.\?\(\)\{\}\[\]]"  # single-char punctuation
        r"|[A-Za-z_][A-Za-z0-9_]*"     # identifiers / keywords
        r"|[0-9]+"        # numbers
    )

    def tokenize(self, text: str) -> list[str]:
        """Split text into token strings."""
        return self._SPLIT_RE.findall(text)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Text → list of token IDs."""
        tokens = self.tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """List of token IDs → text."""
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, UNK_TOKEN)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def __repr__(self) -> str:
        return f"MiniTokenizer(vocab_size={self.vocab_size})"


if __name__ == "__main__":
    tok = MiniTokenizer()
    print(tok)
    print(f"Vocab: {tok.vocab_size} tokens")
    print()

    # Test encode/decode round-trip
    examples = [
        "let enabled : string = 'on'",
        "'on' | 'off'",
        "function f ( x : string )",
        "'realClick' | 'realTouch'",
        "const size : 'sm' | 'md' | 'lg' = 'md'",
    ]
    for text in examples:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        tokens = tok.tokenize(text)
        print(f"  input:    {text}")
        print(f"  tokens:   {tokens}")
        print(f"  ids:      {ids}")
        print(f"  decoded:  {decoded}")
        unk_count = sum(1 for i in ids if i == tok.unk_id)
        if unk_count:
            print(f"  WARNING: {unk_count} unknown tokens!")
        print()
