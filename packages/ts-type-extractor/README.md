# ts-type-extractor

TypeScript-side of the type-refinement pipeline. Pure TS/AST tooling — no ML.
Uses [ts-morph](https://ts-morph.com/) for parsing.

Two roles:

1. **Training data generation** (offline): `extract.ts` walks a repo,
   `degrade.ts` produces (context, degraded, precise) training pairs.
2. **Inference plumbing** (online): `refiner-locate.ts` finds candidates in a
   user's project, `refiner-apply.ts` rewrites the source after the model
   speaks.

---

## File map

| File | Role | Reads | Writes |
|---|---|---|---|
| `extract.ts` | Walk AST, dump every type annotation with surrounding context | `*.ts` / `*.tsx` files | `extracted_types.jsonl` |
| `degrade.ts` | Apply 27 degradation rules → produce supervised pairs | `extracted_types.jsonl` | `training_pairs.jsonl` |
| `refiner-locate.ts` | Find type nodes whose text matches a known **degraded** shape | `*.ts` / `*.tsx` files | `candidates.jsonl` |
| `refiner-apply.ts` | Splice model suggestions back into source files | `edits.jsonl` (from `refiner-infer`) | mutated `*.ts` files |

The 24 trained rules in `refiner-locate.ts` are a **subset** of the 27 in
`degrade.ts` — three rules produced too few pairs to train on. Validators on
the Python side mirror exactly the locator's 24 rules.

---

## CLI

All commands run via `npx tsx src/<name>.ts`. The npm script
`pipeline` (in `package.json`) chains `extract` + `degrade` over all 14
training repos.

### `extract.ts`

```
npx tsx src/extract.ts <path...> [--context 0] [--output out.jsonl]
```

Default context radius is `0` (single line containing the annotation).
**Must match `refiner-locate.ts`** — model was trained on this radius;
mismatched windows = OOD prompts.

Emitted record (one per JSONL line):
```ts
{ file, line, kind, name, typeText, context, parentName }
```
`kind` ∈ `parameter | variable | return_type | property | type_assertion | generic_argument`.

### `degrade.ts`

```
npx tsx src/degrade.ts <extracted.jsonl> [--output pairs.jsonl]
```

Applies 27 degradation rules sequentially (first match wins). For each match:
replaces the precise type WITH the degraded type IN the context (so the model
can't cheat by copying the answer from surrounding code).

Emitted record:
```ts
{ context, name, kind, degradedType, preciseType, rule, file, line }
```

Progress: prints `i/total (pct%) pairs=N rate ETA` every 2 s or 10 000 rows.

### `refiner-locate.ts`

```
npx tsx src/refiner-locate.ts <path...> [--context 0] \
    [--output candidates.jsonl] [--rule <name|all>]
```

Default `--rule all` scans for all 24 trained rules. Use a single rule key
(e.g. `string_literal_union`) or the full rule name (e.g.
`string_literal_union→string`) to scope.

Emitted candidate:
```ts
{
  id,            // "<file>:<start>" — stable across runs
  file, line,
  start, end,    // 0-based char offsets — APPLY uses these directly
  kind, name,
  context,       // ±radius lines around the annotation
  degradedType,  // verbatim text currently at [start, end)
  rule,          // matches degrade.ts rule names
}
```

Has a `RULES` table (matchers on degraded text) and a `RULE_ORDER` list for
deterministic first-match-wins routing. Comments mark high-FP rules
(`tuple→array`, `dom_object→object`, anything matching bare `string` /
`number` / `unknown`).

### `refiner-apply.ts`

```
npx tsx src/refiner-apply.ts --input edits.jsonl [--dry-run]
```

Reads `edits.jsonl` from `refiner-infer`, sorts edits **descending by start
offset** per file (so earlier offsets aren't shifted by later splices),
re-reads each file as a guard against stale offsets, writes back in place.
`--dry-run` prints the diff without touching disk.

---

## Output format reference

| File | Producer | Consumer |
|---|---|---|
| `extracted_types.jsonl` | `extract.ts` | `degrade.ts` |
| `training_pairs.jsonl` | `degrade.ts` | `ts-type-refiner` `refiner-train` |
| `candidates.jsonl` | `refiner-locate.ts` | `ts-type-refiner` `refiner-infer` |
| `edits.jsonl` | `refiner-infer` | `refiner-apply.ts` |

Each file is line-delimited JSON. See per-file CLI sections above for record
shapes.

---

## Adding a new degradation rule

1. Add the rule to `DEGRADATION_RULES` in `degrade.ts`.
2. Re-run the `pipeline` script — confirm the new rule produces enough pairs
   (≥ ~100 in `training_pairs.jsonl` rule histogram).
3. Add the inverse matcher to `RULES` in `refiner-locate.ts` and append the
   key to `RULE_ORDER`.
4. Add a `validate_*` function and a `VALIDATORS[...]` entry in
   `packages/ts-type-refiner/src/ts_type_refiner/validators.py`.
5. Retrain the model (`refiner-train`).

All four steps are required. Skipping any one breaks the trójkąt synchroniczny.
