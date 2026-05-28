# Training 1 — Using the Trained Model
### 2 days · 7 h each

**Repository:** [github.com/staszekj/ai-lab](https://github.com/staszekj/ai-lab)

## Abstract

This training uses a concrete, working system as a vehicle for understanding how a trained sequence-to-sequence model is actually put to use. The domain is TypeScript type refinement — a model that takes a widened type annotation such as `string` or `unknown` and recovers the precise type the developer originally intended (`"primary" | "secondary"`, `HTMLInputElement`, and so on). **TypeScript type refinement is the example, not the subject.** You do not need to know TypeScript. You will not be asked to understand the TypeScript compiler, the AST traversal code, or the details of any particular type system rule — those live in isolated modules (`ts-type-extractor`, `refiner-locate.ts`, `refiner-apply.ts`) that we treat as black boxes.

What we *will* cover in depth are two ideas that transfer to any sequence-to-sequence application:

- **How to build training pairs through programmatic degradation.** We have a corpus of real, precise outputs (TypeScript type annotations from open-source repositories). We systematically corrupt each one into a plausible but weaker form. Every corruption is reversible, and the reversal rule is the definition of a correct prediction. This pattern — corrupt → train to recover — applies far beyond type systems.

- **How to filter model outputs using the structure of the degradation.** Because we know exactly how each input was degraded, we can write a structural validator that checks whether the model's output has the right shape. This is our first line of defence against hallucinations, and it is entirely independent of any downstream task logic.

The second day ends with hands-on time: you will extend the sample TypeScript file, run the full pipeline, and read the raw inference output yourself. No TypeScript expertise is assumed or required at any point.

**Prerequisites:** basic Python, some exposure to PyTorch tensors, comfort with reading code.

---

## Day 1 — Mathematics, Data Pipeline, Tokenisation

### Math Refresher *(90 min)*

- **Matrix multiplication — shape is everything** *(30 min)*
  - Dot product of two vectors: each pair multiplied, summed → one number
  - Matrix × matrix: inner dimensions must match, outer dimensions survive
    ```
    (3,4) @ (4,5) = (3,5)
          ↑↑
      must match — "disappear"
      outer dims survive: 3 and 5
    ```
  - Worked example on the board: `(2,3) @ (3,4)` — draw the grid, fill one cell together
  - Matrix × vector: `(3,4) @ (4,) = (3,)` — a vector is a matrix with one column
  - Vector × matrix: `(4,) @ (4,5) = (5,)`
  - Transpose: `(4,5).T = (5,4)` — swap rows and columns; in code: `K.transpose(-2, -1)`
  - Why this matters: `scores = Q @ Kᵀ / √d_k` is the core of the entire attention mechanism

- **Tensor notation** *(15 min)*
  - A tensor is a generalisation of a matrix to more dimensions
  - Reading left to right: `(1, 3, 4, 2)`
    ```
    (1,    3,    4,  2)
     ↑     ↑     └──┘
    batch  head   matrix (4×2)
           (opt.)
    ```
  - Batch: many examples processed simultaneously — they never see each other
  - Head: parallel independent copies of the attention mechanism
  - The last two dimensions are the "real" matrix being operated on
  - Practical examples from the codebase: `enc_x (1,4,6)`, `Q (1,3,4,2)`, `scores (1,3,4,4)`

- **Derivative and gradient — intuition only** *(15 min)*
  - Derivative at a point = slope of the function at that point
  - Large slope → small change in input causes large change in output
  - Partial derivative `∂L/∂w`: "how much does the loss change if I nudge only this one weight, holding everything else fixed"
  - Gradient = vector of all partial derivatives — points in the direction of steepest increase; the optimiser steps the opposite way

- **Softmax** *(20 min)*
  - Turns arbitrary numbers (logits) into a probability distribution
  - Properties: all values ≥ 0, sum = 1
  - Formula: `softmax(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)`
  - Example: `[1.0, 2.0, 3.0]` → `[0.09, 0.24, 0.67]` — live in terminal
  - Large spread in input → one dominant value (peaked)
  - Small spread → nearly uniform distribution (flat)
  - In attention: each row of the scores matrix passes through softmax → row sums to 1

- **GELU** *(10 min)*
  - Smooth non-linearity used in the Feed-Forward Network
  - Similar to ReLU (`max(0,x)`) but with no sharp corner at zero
  - Why any non-linearity at all: without it the entire model collapses into a single large matrix multiply — stacking linear layers adds no expressive power

- **Jacobian of softmax and partial derivatives** *(10 min)*
  - Softmax is a vector function: `(n,)` → `(n,)`
  - Every output `yᵢ` depends on ALL inputs `x₁…xₙ`, not just `xᵢ`
  - Jacobian: an `(n×n)` matrix of all partial derivatives `∂yᵢ/∂xⱼ`
  - Consequence for backprop: the gradient through softmax mixes all positions simultaneously — attention weights influence each other during learning

---

☕ **Break** *(15 min)*

---

### The Problem: Degraded Types in TypeScript *(20 min)*

- What "degraded" means: a type widened to a valid supertype
  - `"primary" | "secondary" | "danger"` → `string`
  - `HTMLInputElement` → `HTMLElement`
  - `React.MouseEvent<HTMLButtonElement>` → `React.SyntheticEvent`
  - `Promise<{ id: number; name: string }>` → `Promise<unknown>`
- Code still compiles — every supertype assignment is safe in TypeScript
- The cost: lost information, weaker autocomplete, no refinement at call sites
- Goal: automatically recover the precise type from surrounding code context

---

### Data Pipeline: From Real Repositories to Training Pairs *(105 min)*

- **Step 1 — `extract.ts`: reading real TypeScript code** *(25 min)*
  - `ts-morph` parses TS/TSX files into a full AST — no regex, no heuristics
  - Six kinds of annotations extracted: `parameter`, `variable`, `return_type`, `property`, `type_assertion`, `generic_argument`
  - `contextRadius = 0` by default — single line only
    - Why: multi-line context with multiple annotations on nearby lines makes the prompt ambiguous (the model cannot tell which `string` it is being asked to refine); a one-line window forces the model to rely on the identifier name, which is almost always on the same line
  - `MAX_TYPE_LENGTH = 400` — longer types are auto-generated noise, not hand-written annotations
  - Skip patterns: `_generated`, `.generated.`, `__generated__` — machine-made types are useless for training

- **Step 2 — `siblings.ts`: enriching the context** *(20 min)*
  - Siblings = other type annotations that share the same declaring scope as the target annotation
  - Parameter → sibling params + return type: `[value: string, index: number, -> void]`
  - Property → peer properties in the same interface or class
  - Return type → function parameters
  - `@in:ContainingDecl` boost: prepended for rules where the target identifier is typically declared elsewhere (indexed access, utility types, `SetStateAction`) — `shouldBoost()` in `siblings.ts`
  - Limits: ≤12 entries, ≤250 characters — everything beyond is silently dropped
  - Training and inference MUST produce identical siblings for the same code position — any divergence is a train/inference distribution mismatch

- **Step 3 — `degrade.ts`: creating training pairs** *(25 min)*
  - 30+ degradation rules, each a function `typeText → { degraded, rule } | null`
  - First matching rule wins; `MUTED_RULES` suppresses low-support rules from pair generation
  - AST-targeted replace: uses `typeStart` / `typeEnd` offsets from `extract.ts` to replace exactly the right span — avoids accidentally replacing a peer occurrence of the same type text on the same line (e.g. `Record<string, string>`)
  - Prompt format is built identically in `degrade.ts` and `prompt.py` — `PROMPT_VERSION = 2` stamped into the checkpoint; inference refuses to run against a mismatched version
  - Dedup by SHA1(input + "\t" + target): copy-pasted props across repositories produce identical pairs
  - Content-hash train/val split: SHA1(input) % 100 — identical prompts can never straddle the train/val boundary

- **Step 4 — `negatives.ts`: teaching the model to preserve** *(20 min)*
  - Problem: the model sees only degraded → precise examples; it might always try to change something even when the type is already correct
  - Hard negatives: annotations whose `typeText` already matches the rule's degraded shape → `(degraded = typeText, target = typeText, isNegative = true)`
  - Example: for `string_literal_union→string`, a genuine bare `string` annotation becomes a negative pair — the model must learn to output `string` unchanged
  - `negativeRatio = 0.25` → 25% as many negatives as positives per rule
  - The pools are disjoint by construction: degradation rules require `typeText ≠ degraded`, so the same annotation cannot be both a positive and a negative

- **Step 5 — `dataset.py`: from pairs to tensors** *(15 min)*
  - `encode_pair`: `tgt_input = [<bos>] + tgt_ids`, `tgt_target = tgt_ids + [<eos>]` — the shift that enables teacher forcing
  - Padding: variable-length sequences padded to the longest in the batch; `CrossEntropyLoss(ignore_index=pad_id)` ignores padded positions
  - Rule-balanced sampler (`iter_balanced_batches`): weight per sample = `1 / sqrt(count(rule))` — tail rules get amplified, dominant rules get suppressed

---

☕ **Break** *(15 min)*

---

### Tokenisation *(60 min)*

- **Token ≠ word split by whitespace** *(20 min)*
  - Live demo: `MouseEventHandler<MouseEvent>` — count tokens together
  - `HTMLButtonElement | null` — count tokens
  - `React.Dispatch<React.SetStateAction<boolean>>` — count tokens
  - BPE (Byte-Pair Encoding): start with individual bytes, iteratively merge the most frequent adjacent pair into one new token
  - The vocabulary is built from the training corpus — TypeScript-specific patterns get dedicated tokens
  - Vocabulary size 2048: small by LLM standards but sufficient for TypeScript type syntax

- **`TSTokenizer` — the wrapper** *(15 min)*
  - `encode(text) → list[int]`: text to token IDs
  - `decode(ids) → str`: token IDs back to text
  - Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>` — always ids 0–3
  - `build_from_jsonl`: builds BPE tokenizer from the `input` and `target` fields of `encoder_decoder_pairs.jsonl`

- **The prompt format** *(25 min)*
  - Full format (PROMPT_VERSION=2):
    ```
    [REFINE rule=... | kind=... | name=... | degraded=... | siblings=...]
    ---
    <code context line>
    ```
  - Why metadata before code: two identical code lines can belong to different rules; the header disambiguates
  - The `---` separator: prevents the parser from being confused when the code starts with `[`
  - Live example: build a prompt for `variant: string` from `SampleComponent.tsx`

---

### End-to-End Demo *(45 min)*

- Read `SampleComponent.tsx` together — identify every degraded annotation and its expected rule
- Run `pnpm --filter refiner-playground run`
- Inspect `candidates.jsonl`: `id`, `file`, `line`, `start`, `end`, `kind`, `name`, `context`, `degradedType`, `rule`, `siblings`
- Inspect `edits.jsonl`: `suggestion`, `accepted`, `reason`, `logprob`, `ruleValidatorPassed`, `proposals` array
- Pipeline as three processes: `refiner-locate.ts` → `uv refiner-infer` → `refiner-apply.ts`

---

## Day 2 — Model, Scoring, Validators, Inference Pipeline

### The Model as a Black Box — First Look Inside *(90 min)*

- **Embeddings** *(20 min)*
  - Every token ID maps to one row in the embedding table `(vocab_size × d_model)`
  - Positional embedding: a second table `(max_seq_len × d_model)` — same token at position 0 and position 3 gets different vectors; we add both together
  - The micro-model: instantiate with `d_model=6, vocab=12` — print `enc_x (4×6)` on screen; every number visible
  - `src_ids = [[2, 5, 6, 7]]` → rows 2, 5, 6, 7 of the table → shape `(1, 4, 6)`

- **Q, K, V and self-attention** *(35 min)*
  - Three independent linear projections of the same vector
    - Q = "what am I looking for"
    - K = "what do I represent"
    - V = "what I return if selected"
  - `scores = Q @ Kᵀ / √d_k` — every token asks every other token "how relevant are you to me?" → shape `(1,3,4,4)` in the micro-model
  - Scaling by `1/√d_k`: without it, large `d_k` drives scores into regions where softmax gradient vanishes
  - `weights = softmax(scores)` — print the `(4×4)` weight matrix; read each row: "token 0 distributes its attention as..."
  - No mask in the encoder: full grid, every token sees every other token — bidirectional
  - `head_out = weights @ V` — weighted mixture of value vectors → back to `(4×6)`

- **Multi-head and projection** *(20 min)*
  - Split `d_model` into `num_heads` parallel subspaces of size `d_k = d_model / num_heads`
  - `(1,4,6)` → `(1,4,3,2)` → transpose → `(1,3,4,2)` — each head is a `(4,2)` matrix
  - Each head learns to attend along a different "dimension of meaning" simultaneously
  - Merge heads: `(1,3,4,2)` → transpose → `(1,4,3,2)` → reshape → `(1,4,6)` → multiply by `W_O`
  - Residual connection: `x1 = x + attn_out` — the input bypasses the sub-layer entirely, preserving gradient flow to early layers

- **LM head: logits as a distribution** *(15 min)*
  - After the decoder: final `(1,3,6)` tensor passes through `decoder_final_norm` then `lm_head`
  - `lm_head`: linear `(d_model → vocab_size)` = `(6 → 12)` in the micro-model
  - Output `(1,3,12)` — one row per decoder position, 12 scores over the vocabulary
  - Softmax → probability distribution; `argmax` → most likely next token

---

☕ **Break** *(15 min)*

---

### Autoregressive Generation *(45 min)*

- **Teacher forcing vs autoregressive — the key distinction** *(20 min)*
  - Training (teacher forcing): feed the whole decoder input `[<bos>, ON, |]` at once, predict `[ON, |, OFF]` in parallel — we already know the target
  - Inference (autoregressive): we don't know the target; start with `[<bos>]`, grow the sequence one token at a time, feeding our own predictions back in
    ```
    step 1: decoder context = [<bos>]        → predict "ON"
    step 2: decoder context = [<bos>, ON]    → predict "|"
    step 3: decoder context = [<bos>, ON, |] → predict "OFF"
    ```
  - Encoder runs once; its `encoder_output` is cached and reused at every decoder step

- **Temperature** *(10 min)*
  - `next_token_logits / temperature` before softmax
  - Temperature < 1: more peaked — near-greedy, deterministic
  - Temperature > 1: flatter — more diverse, more random
  - `temperature=0.01` in `Predictor.__init__` default: effectively greedy for repeatable inference
  - `temperature=0.7` for `predict_n`: introduces diversity so multiple unique candidates are sampled

- **KV-cache** *(15 min)*
  - `precompute_cross_kv`: projects `encoder_output` to `(K, V)` once before generation loop
  - `forward_cached`: single-step decoder, accumulates self-attention `(K, V)` across steps
  - Without cache: would recompute the full growing decoder sequence at every step — quadratic cost

---

### Scoring: mean\_logprob *(45 min)*

- **Why confidence matters** *(10 min)*
  - The model can generate plausible-looking but wrong types — we need a way to rank proposals
  - Teacher-forced rescoring: after generating a candidate, run a second forward pass with the known output, measure how likely the model considers its own generation

- **`_score_generated` in `predictor.py`** *(20 min)*
  - `dec_in  = [<bos>] + gen_ids[:-1]` — teacher-forced input
  - `dec_tgt = gen_ids` — targets
  - `logits = model(src_tensor, dec_in)` — one forward pass `(1, T, V)`
  - `logp = log_softmax(logits)` — log probabilities
  - `token_logp = logp.gather(2, dec_tgt)` — pick the log-prob of the actually generated token at each position
  - `mean_logprob = token_logp.mean()` — average over all T positions

- **Reading the number** *(15 min)*
  - Always negative: log of a probability ≤ 1 is always ≤ 0
  - Closer to zero = model was more confident
  - `-0.5`: model was very confident — each token had probability ≈ 0.6 on average
  - `-8.0`: model was very uncertain — rare tokens or hallucinated content
  - Random baseline: `log(1/vocab_size) = log(1/2048) ≈ -7.6`
  - `predict_n`: runs `attempts` samples, deduplicates, sorts by `mean_logprob`, normalises to `normalized_prob` via softmax over the candidate set

---

☕ **Break** *(15 min)*

---

### Validators: Degradation as a Correctness Test *(60 min)*

- **Why validators exist** *(15 min)*
  - The model can produce syntactically valid TypeScript types that are not the right answer for the given rule
  - Without a post-filter, any plausible-looking output would be written to source code
  - The degradation rule defines exactly what shape the recovered type must have — the validator is the inverse

- **Worked examples from `validators.py`** *(25 min)*
  - `html_specific_element→html_element`: regex `^HTML\w+Element$` + rejects `HTMLElement` itself — still degraded
  - `string_literal_union→string`: regex `^"..."(|"...")+$` — must have at least two string literals
  - `react_event→synthetic`: regex `^React\.\w+Event(?:<.+>)?$` + rejects `React.SyntheticEvent` — still generic
  - `promise→unknown`: `^Promise<(.+)>$` + rejects inner type in `_SIMPLE_TYPES` — `Promise<string>` is still too generic
  - `MUTED_RULES`: rules with too few training examples are disabled both in `degrade.ts` and `validators.py` — same set in both files

- **Hallucination detection** *(20 min)*
  - `_non_trivial_idents`: extract identifiers from the suggestion, subtract common TypeScript builtins
  - `_ident_overlap(proposal_text, siblings_str)`: how many non-trivial identifiers in the suggestion appear in the siblings string
  - When every proposal fails the validator AND no proposal's identifiers overlap with siblings → label reason as `hallucinated_identifier`
  - This is diagnostic only — does not affect the `accepted` field, but makes the `reason` field actionable

---

### The Full Inference Pipeline *(60 min)*

- **Loading and building** *(15 min)*
  - `load_checkpoint` in `checkpoint.py`: reads `model_state_dict` + `model_config` + extras from `.pt` file
  - `PROMPT_VERSION` check: checkpoint stores the version; inference warns if it doesn't match current `prompt.py`
  - `build_model(ckpt.model_config)`: reconstructs the model from the six config numbers — `state_dict` shapes alone are not enough (`num_heads`, `num_layers`, `max_seq_len` are not encoded in tensor shapes)
  - `model.load_state_dict(ckpt.state_dict)`: load weights

- **The inference loop in `infer.py`** *(30 min)*
  - For each candidate: look up validator by `rule` — if none registered → `accepted=False`, reason `"no validator for rule"` — safety net, an unknown rule can never accidentally produce an accepted edit
  - `build_refine_prompt`: assembles the encoder input from candidate fields
  - `predict.predict_n(prompt, n=5, temperature=0.7)`: up to 5 unique proposals
  - For each proposal: check validator → check `min_logprob` → first pass = `selected`
  - `choose_better`: merges duplicate candidate IDs — prefers accepted over rejected, then higher `logprob`

- **Reading `edits.jsonl`** *(15 min)*
  - `accepted=true`: validator passed and `logprob >= min_logprob`
  - `accepted=false` + `ruleValidatorPassed=true`: validator passed but confidence too low
  - `accepted=false` + `ruleValidatorPassed=false`: no proposal passed the validator
  - `proposals` array: all sampled candidates with their individual `logprob`, `normalized_prob`, `identsInSiblings`

---

### Hands-on *(60 min)*

- Add at least three new degraded type annotations to `SampleComponent.tsx` — choose rules from different families (React, DOM, literal union)
- Run the playground; read `candidates.jsonl` to verify locate found them
- Read `edits.jsonl` — identify which were accepted, which were rejected and why
- Experiment: lower `--min-logprob` to `-10.0` and observe which rejected proposals become accepted — are they correct?
- Experiment: set `--num-candidates 1` — does accuracy drop noticeably?
