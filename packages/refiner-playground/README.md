# refiner-playground

End-to-end testbed for the trained `ts-type-refiner` model.

Bundles:

- `samples/` — TypeScript files to refine (start with `SampleComponent.tsx`)
- `src/run.ts` — one-shot orchestrator: locate → infer → apply

 No ML code lives here. The orchestrator shells out to:
 
- `packages/ts-type-extractor/src/rules/refiner-locate.ts`
 - `uv run --package ts-type-refiner refiner-infer`
 - `packages/ts-type-extractor/src/ts-data/refiner-apply.ts`

Intermediate artifacts (`candidates.jsonl`, `edits.jsonl`) land in
`./tmp/` so you can inspect them after a run.

---

## CLI

```bash
# All samples, apply in place
pnpm --filter refiner-playground run

# Dry-run (diff preview, no writes)
pnpm --filter refiner-playground run:dry

# One file, only a specific rule
npx tsx src/run.ts samples/SampleComponent.tsx --rule string_literal_union --dry-run

# With log-prob threshold (reject low-confidence proposals)
npx tsx src/run.ts --min-logprob -0.2
```

Flags:

| Flag | Meaning |
|---|---|
| `--target <file>` (or positional) | Target file. Repeatable. Defaults to every `*.ts`/`*.tsx` in `samples/`. |
| `--dry-run` | Pass through to `refiner-apply` — print diff, write nothing. |
| `--min-logprob N` | Forwarded to `refiner-infer`. |
| `--rule <name>` | Forwarded to `refiner-locate`. |

---

## Workflow

1. Train (or reuse) a model so `packages/ts-type-refiner/checkpoints/refiner.pt` exists.
2. Add a `.tsx` to `samples/`.
3. `pnpm --filter refiner-playground run:dry` — preview suggestions.
4. Tweak `--min-logprob` / `--rule` until the diff looks sane.
5. Drop `--dry-run` to actually write.
6. `git diff -- packages/refiner-playground/samples/` to inspect.
7. `git checkout -- packages/refiner-playground/samples/` to revert.

---

## Why a separate package?

`ts-type-extractor` is *infrastructure* (locate/apply/extract/degrade).
`ts-type-refiner` is *the model* (train/infer).
Playing with the trained model on hand-crafted code is a third concern.
Splitting it keeps each package's surface small and the dependency
direction clean (`refiner-playground → ts-type-extractor + ts-type-refiner`).
