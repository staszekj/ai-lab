# core

Domain-agnostic building blocks for sequence-to-sequence ML in the **ai-lab** monorepo.

Knows nothing about TypeScript, BPE, or any specific dataset. All TS-specific
glue lives in `ts-type-refiner`; this package only exposes the generic pieces:
the model, a pure-function trainer, an inference wrapper, and checkpoint I/O.

---

## Module map

| Module | Purpose |
|---|---|
| `core.encoder_decoder_model` | Pre-LN T5/BART-style encoder–decoder transformer (`nn.Module`) |
| `core.trainer` | Pure `train(model, batches, …)` function — no class, no globals |
| `core.predictor` | `Predictor(model, encode, decode, …)` callable returning text + log-prob |
| `core.checkpoint` | `save` / `load` / `build_model` — the **only** file that touches `torch.save/load` |
| `core.qdrant` | Qdrant client factory with mTLS + API key (separate concern, used by bible-* packages) |

---

## API

### `core.encoder_decoder_model`

```python
from core.encoder_decoder_model import EncoderDecoderConfig, EncoderDecoderModel

cfg = EncoderDecoderConfig(
    vocab_size=2048, max_seq_len=128, d_model=256,
    num_heads=8, d_ff=1024, num_layers=4,
)
model = EncoderDecoderModel(cfg)            # also: EncoderDecoderModel.Config(...)

logits = model(src_ids, tgt_in_ids)         # (batch, tgt_len, vocab) — teacher-forced
ids    = model.generate(                    # autoregressive greedy/sampled
    src_ids,
    bos_id=BOS, eos_id=EOS,
    max_new_tokens=64, temperature=0.01,
)
```

The six hyper-parameters in `EncoderDecoderConfig` are sufficient to rebuild
`state_dict` shapes — that's all `core.checkpoint` needs to persist.

### `core.trainer`

Pure function. Caller wires in batches + callbacks; the trainer owns only
forward → cross-entropy → backward → grad-clip → step.

```python
from core.trainer import TrainConfig, train, EpochStats

cfg = TrainConfig(
    epochs=50, lr=3e-4, weight_decay=0.01,
    max_grad_norm=1.0,
    eval_every=10,             # 0 disables eval_fn invocations
    log_every_batches=25,      # 0 disables per-batch heartbeat
    seed=42,
)

train(
    model         = model,
    train_batches = lambda: ds.iter_batches(...),   # MUST be a factory (re-shuffles per epoch)
    pad_id        = tokenizer.pad_id,
    cfg           = cfg,
    eval_fn       = lambda m: my_metric(m),         # optional, returns float
    on_epoch_end  = lambda stats: print(stats),     # optional, gets EpochStats
)
```

`EpochStats` fields: `epoch`, `train_loss`, `train_tf_acc`, `val_metric` (or
`None`), `elapsed_s`.

### `core.predictor`

```python
from core.predictor import Predictor, PredictResult

predict = Predictor(
    model,
    encode      = tok.encode,
    decode      = tok.decode,
    bos_id      = tok.bos_id,
    eos_id      = tok.eos_id,
    max_src_len = 256,
    max_tgt_len = 64,
    device      = device,
)

r: PredictResult = predict("…some context…")
r.text          # decoded suggestion (no <BOS>, <EOS>-stripped)
r.ids           # raw token ids
r.mean_logprob  # average log p(token | prefix); 0 = certain, very negative = unsure
```

`model.eval()` is set once in the constructor — reuse the same `Predictor`
for every candidate.

### `core.checkpoint`

```python
from core.checkpoint import save, load, build_model, LoadedCheckpoint

save(model, "ckpt.pt", model_config=cfg.__dict__,
     epoch=50, val_accuracy=0.48)        # extras kwargs persisted verbatim

ckpt: LoadedCheckpoint = load("ckpt.pt", device=device)
ckpt.state_dict        # weights
ckpt.model_config      # dict — EncoderDecoderConfig fields
ckpt.extras            # everything else passed to save()

model = build_model(ckpt.model_config, device=device)
model.load_state_dict(ckpt.state_dict)
```

---

## Qdrant connection

Used by `bible-*` packages, unrelated to the ML stack above.

```python
from core.qdrant import get_client
client = get_client()                        # mTLS to https://qdrant.eltrue
```

Required files in `~/.secrets/`: `eltrue-ca.crt`, `eltrue-client.crt`,
`eltrue-client.key`, `qdrant-api-key`.
