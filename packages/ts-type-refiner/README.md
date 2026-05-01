# ts-type-refiner

Trains and runs the encoder–decoder model that **inverts type degradation**:
given a TypeScript snippet with a loose type (e.g. `string`, `unknown`,
`Promise<unknown>`), it suggests a precise replacement (e.g.
`'on' | 'off'`, `ReturnType<typeof foo>`, `Promise<User>`).

The companion package `ts-type-extractor` produces both the training data
(`extract.ts` + `degrade.ts`) and the inference inputs/outputs
(`refiner-locate.ts` + `refiner-apply.ts`).

---

## Module map

| Module | Purpose |
|---|---|
| `ts_type_refiner.tokenizer` | BPE tokenizer (HuggingFace `tokenizers`) with `<pad>` `<bos>` `<eos>` `<unk>` |
| `ts_type_refiner.dataset` | Loads `training_pairs.jsonl`, builds padded teacher-forced batches |
| `ts_type_refiner.validators` | 24 shape-regex validators, one per degradation rule, plus `VALIDATORS` dict |
| `ts_type_refiner.train` | Orchestrator: tokenizer → dataset → model → `core.trainer.train` → checkpoint |
| `ts_type_refiner.infer` | Orchestrator: load checkpoint → `core.predictor` → validate → write edits |

---

## CLI

| Command | Reads | Writes |
|---|---|---|
| `refiner-train` | `packages/ts-type-extractor/data/training_pairs.jsonl` | `checkpoints/refiner.pt`, `tokenizer.json` |
| `refiner-infer --input candidates.jsonl --output edits.jsonl` | candidates from `refiner-locate.ts` | edits for `refiner-apply.ts` |

Both run via `uv run --package ts-type-refiner <name>`.

`refiner-infer` flags:
- `--checkpoint` (default `checkpoints/refiner.pt`)
- `--tokenizer`  (default `tokenizer.json`)
- `--min-logprob` reject suggestions below threshold (default `-inf`)
- `--max-src-len 256`, `--max-tgt-len 64`
- `--limit N` process at most N candidates (0 = all)

---

## API

### `tokenizer.TSTokenizer`

```python
from ts_type_refiner.tokenizer import TSTokenizer, build_from_jsonl

tok = build_from_jsonl("training_pairs.jsonl", vocab_size=2048)
tok.save("tokenizer.json")
tok = TSTokenizer.from_file("tokenizer.json")

ids  = tok.encode("string")          # add_bos / add_eos kwargs available
text = tok.decode(ids)               # skip_special=True by default
tok.pad_id  tok.bos_id  tok.eos_id  tok.unk_id  tok.vocab_size
```

### `dataset.TypeRefinerDataset`

```python
from ts_type_refiner.dataset import TypeRefinerDataset, train_val_split

ds = TypeRefinerDataset("training_pairs.jsonl", tok, max_src_len=256, max_tgt_len=64)
train_idx, val_idx = train_val_split(ds, val_ratio=0.15, seed=42)

for src, tgt_in, tgt_target in ds.iter_batches(batch_size=96, device=device,
                                                shuffle=True, indices=train_idx):
    ...
```

`ds.pairs[i]` → `(context, preciseType)` tuple. `iter_batches` yields padded
tensors already on `device`.

### `validators.VALIDATORS`

```python
from ts_type_refiner.validators import VALIDATORS

validate = VALIDATORS["string_literal_union→string"]
ok, reason = validate("'on' | 'off'")        # (True, "ok")
ok, reason = validate("string")              # (False, "expected 'a' | 'b' | …")
```

Keys match the `rule` field emitted by `refiner-locate.ts` and recorded in
`training_pairs.jsonl`. 24 keys total. Both `returntype→unknown` and
`utility_type→unknown` route to the same combined validator (they're
syntactically indistinguishable at locator time).

Each validator returns `(bool, reason_str)` — `infer.py` ANDs this with the
log-prob threshold to decide `accepted`.

### `train.main` / `infer.main`

Thin orchestrators. Read the docstrings; you don't import them — invoke via
the CLI scripts above. Hyper-parameters (epochs, batch size, model size) are
hard-coded constants at the top of each file.

---

## Pipeline overview

```
ts-type-extractor          ts-type-refiner          ts-type-extractor
─────────────────          ───────────────          ─────────────────
extract.ts ──► extracted_types.jsonl
                  │
                  ▼
              degrade.ts ──► training_pairs.jsonl ──► refiner-train ──► refiner.pt
                                                                            │
refiner-locate.ts ──► candidates.jsonl ──► refiner-infer ──► edits.jsonl ──► refiner-apply.ts ──► patched .ts files
                                                  ▲
                                                  └── uses refiner.pt + tokenizer.json + VALIDATORS
```

Trójkąt synchroniczny: the rule names in `degrade.ts`, `refiner-locate.ts`
and `validators.py` MUST stay aligned — the model only knows the rules it
was trained on.
