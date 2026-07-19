# AI Symposium: TypeScript Type Refiner Workshop

## Part 0: Setup

- [ ] See [SETUP.md](../../SETUP.md) for environment configuration and checkpoint download
- [ ] See [math-refresher.md](./math-refresher.md) for a quick math recap (matrix multiplication, derivatives, Jacobian)



## Part 1: Practical Pipeline (30 min)

### refiner-playground (orchestrator) — `packages/refiner-playground/src/run.ts`
- [ ] In `packages/refiner-playground/src/run.ts` `main()` (approx. lines 120+) orchestrates `refiner-locate` which scans source files using hand-written rules to identify which type annotations look degraded and are worth sending to the model
- [ ] In `packages/ts-type-refiner/src/ts_type_refiner/inference/infer.py` `main()` (approx. lines 174+) loads the checkpoint and proposes more precise types for each degraded candidate, validating proposals against rule-specific validators and log-probability thresholds
- [ ] In `packages/ts-type-extractor/src/ts-data/refiner-apply.ts` the applier deterministically rewrites source files by splicing accepted suggestions back into the original byte ranges

### Inference — `packages/ts-type-refiner/src/ts_type_refiner/inference/infer.py`
- [ ] In `main()` (approx. lines 174+) the model loads checkpoint and tokenizer, then loops over candidates
- [ ] For each candidate the predictor generates top-N proposals with log-probabilities (approx. lines 290+)
- [ ] Each proposal is validated against rule-specific validators (e.g. `map→unknown` must have concrete type params, not `unknown`) (approx. lines 300+)
- [ ] A proposal is accepted only if it passes the validator AND has mean log-probability above threshold (approx. lines 303+)
- [ ] The first accepted proposal becomes the suggestion; if none accepted, the best rejected one is recorded with diagnostic reason (approx. lines 323+)

### Predictor — `packages/ts-type-refiner/src/ts_type_refiner/inference/predictor.py`
- [ ] In `predict_n()` (approx. lines 140+) the model is called multiple times for the same prompt — because temperature > 0, each call samples differently and can produce a different sequence
- [ ] Each unique proposal is scored via teacher-forced rescoring (`_score_generated`, approx. lines 220+): the generated sequence is fed back token-by-token to measure how confident the model was at each step — result is `mean_logprob` (0 = certain, very negative = unsure)
- [ ] Proposals are sorted by `mean_logprob`; infer.py picks the first one that also passes the rule validator

### Generator — `packages/ts-type-refiner/src/ts_type_refiner/encoder_decoder_model.py`
- [ ] In `EncoderDecoderModel.generate()` (approx. lines 900+) the model generates one token per step — each step embeds the last token, runs it through decoder blocks (self-attention + cross-attention + FFN), then samples the next token from the resulting probability distribution over vocabulary

## Part 2: Transformers Deep Dive (60 min)

### Tokenization
- [ ] How tokens are created

### Attention Mechanism
- [ ] What attention is

### The Core Flow: QK^T → Softmax → V → Logits
- [ ] Query-Key interaction (QK^T)
- [ ] Normalization (Softmax)
- [ ] Value projection (V)
- [ ] Output logits (Logits)

### Training Mechanics — `packages/ts-type-refiner/src/ts_type_refiner/training/trainer.py`
- [ ] In `train()` (approx. lines 120+) the training loop runs for `cfg.epochs` epochs, each epoch loops over mini-batches yielded by `train_batches()` (which rescatters the dataset each epoch)
- [ ] **Forward pass** `logits = model(src, tgt_in)` (approx. line 200): encoder-decoder transformer does one forward pass with teacher forcing — decoder sees TRUE previous tokens, not its own predictions. Returns shape (batch, tgt_len, vocab_size), one logits vector per decoder position
- [ ] **Loss computation** (approx. line 206): cross-entropy loss compares logits against true target tokens. Internally: softmax(logits) → -log(softmax[target_id]) at each position → mean over non-pad positions. Loss MINIMIZED when model peaks at correct tokens
- [ ] **Backward + step** (approx. lines 212-214): `loss.backward()` computes dL/dw for all model weights via autograd, `clip_grad_norm_()` caps gradient magnitude to prevent exploding gradients (transformer hygiene), `optimizer.step()` updates all w ← w - lr*dL/dw + L2, `scheduler.step()` adjusts learning rate (cosine or ReduceLROnPlateau)
- [ ] **Evaluation** `val_metric = eval_fn(model)` (approx. line 233): optional caller-provided callback runs model on validation set every `cfg.eval_every` epochs (not used by trainer itself — caller decides early-stopping)
