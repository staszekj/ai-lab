# AI Symposium: TypeScript Type Refiner Workshop

## Part 0: Setup

- [ ] See [SETUP.md](../../SETUP.md) for environment configuration and checkpoint download
- [ ] See [math-refresher.md](./math-refresher.md) for a quick math recap (matrix multiplication, derivatives, Jacobian)



## Part 1: Practical Pipeline (30 min)

### Before the orchestrator — `packages/refiner-playground/src/run.ts`
- [ ] where the training data comes from: the source of truth is `packages/ts-type-extractor/data/encoder_decoder_pairs.jsonl`, which is built from real usage examples and refined type pairs; show `packages/ts-type-extractor/data/extracted_usage.example.jsonc`, `packages/ts-type-extractor/data/encoder_decoder_pairs.example.jsonc`, and `packages/ts-type-extractor/data/encoder_decoder_pairs.report.md`
- [ ] explain what tokens are: the model does not read raw characters or whole words, but BPE subword tokens produced by `packages/ts-type-refiner/src/ts_type_refiner/tokenizer.py`; examples include `Map`, `<`, `unknown`, `,`, `>`, `React`, `.`, `Dispatch`, `HTMLInputElement`, and fragments like `'realClick'` or `SetStateAction`, so a type such as `React.Dispatch<React.SetStateAction<boolean>>` is split into reusable pieces rather than memorized as one giant string. Yes, the tokenizer uses BPE, so it can learn frequent TypeScript-specific fragments and recombine them into novel types at inference time

### refiner-playground (orchestrator) — `packages/refiner-playground/src/run.ts`
- [ ] The ruleset is bidirectional (`packages/ts-type-extractor/src/rules/`): `degrade.ts` synthesizes training pairs (precise → degraded for training), `refiner-locate.ts` finds candidates to refine in real code, and `validators.py` (ts-type-refiner) validates proposals during inference — the same rules enforce consistency across the entire pipeline
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
- [ ] In `EncoderDecoderModel.generate()` (approx. lines 1419+) the model generates one token per step — each step embeds the last token, runs it through decoder blocks (self-attention + cross-attention + FFN), then samples the next token from the resulting probability distribution over vocabulary

## Part 2: Transformers Deep Dive (60 min)

### Training Mechanics — `packages/ts-type-refiner/src/ts_type_refiner/training/trainer.py`
- [ ] In `train()` (approx. lines 120+) the training loop runs for `cfg.epochs` epochs, each epoch loops over mini-batches yielded by `train_batches()` (which rescatters the dataset each epoch)
- [ ] **Forward pass** `logits = model(src, tgt_in)` (approx. line 200): encoder-decoder transformer does one forward pass with teacher forcing — decoder sees TRUE previous tokens, not its own predictions. Returns shape (batch, tgt_len, vocab_size), one logits vector per decoder position
- [ ] **Loss computation** (approx. line 206): cross-entropy loss compares logits against true target tokens. Internally: softmax(logits) → -log(softmax[target_id]) at each position → mean over non-pad positions. Loss MINIMIZED when model peaks at correct tokens
- [ ] **Backward + step** (approx. lines 212-214): `loss.backward()` computes dL/dw for all model weights via autograd, `clip_grad_norm_()` caps gradient magnitude to prevent exploding gradients (transformer hygiene), `optimizer.step()` updates all w ← w - lr*dL/dw + L2, `scheduler.step()` adjusts learning rate (cosine or ReduceLROnPlateau)
- [ ] **Evaluation** `val_metric = eval_fn(model)` (approx. line 233): optional caller-provided callback runs model on validation set every `cfg.eval_every` epochs (not used by trainer itself — caller decides early-stopping)

### Forward Pass — `packages/ts-type-refiner/src/ts_type_refiner/encoder_decoder_model.py`
- [ ] In `EncoderDecoderModel.forward()` (approx. lines 1390+) the complete forward pass chains encoder and decoder: source code with degraded type goes through encoder (bidirectional self-attention), encoder_output is memory of the source, target type tokens go through decoder (masked self-attention + cross-attention to encoder), final output is logits (batch, tgt_len, vocab_size) — one probability distribution per position for what the next token should be. This is the training-time version using teacher forcing; inference uses `generate()` instead.

#### Encode: Reading the source — `EncoderDecoderModel.encode()`
- [ ] In `encode()` (approx. lines 898+) source token IDs are embedded then positional-encoded. Pass through N encoder blocks (default 4), each with bidirectional self-attention — every source token can attend to every other source position, no mask. Output is encoder_output (batch, src_len, d_model), the "memory" of what the source code said.

#### Decode: Writing the target — `EncoderDecoderModel.decode()`
- [ ] In `decode()` (approx. lines 1114+) target token IDs (under teacher forcing) are embedded then positional-encoded. Causal mask prevents token i from attending to positions i+1, i+2, … (can't peek at future). Pass through N decoder blocks, each with: (1) masked self-attention to past target tokens, (2) cross-attention to encoder_output, (3) FFN. Output is logits (batch, tgt_len, vocab_size) — the model's raw prediction for each position.


