# ts-type-extractor

TypeScript-side of the type-refinement pipeline. Pure TS/AST tooling — no ML.
Uses [ts-morph](https://ts-morph.com/) for parsing.

Two roles:

1. **Training data generation** (offline): `extract.ts` walks a repo,
   `degrade.ts` produces (context, degraded, precise) training pairs.
2. **Inference plumbing** (online): `refiner-locate.ts` finds candidates in a
   user's project, `refiner-apply.ts` rewrites the source after the model
   speaks.

Repository sourcing is split into two groups:

1. **type-defs**: canonical type-definition sources used to design degradation
   rules:
   - native browser definitions (`microsoft/TypeScript` -> `src/lib`)
   - React definitions (`DefinitelyTyped` -> `types/react`)
   - Astro definitions (`withastro/astro` -> `packages/astro/types`)
   - TanStack Query (`TanStack/query`)
   - TanStack Router (`TanStack/router`)
2. **usage**: ~20 component-library repositories (React, Astro,
   custom-elements/web-components) that use at least one ecosystem above.

Group definitions live in `src/repo-groups.ts`.
Repository URL/path manifest lives in `src/repo-manifest.ts`.

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

All commands run via `npx tsx src/<name>.ts`.

Grouped npm pipelines:

- `npm run pipeline:type-defs` -> extract+degrade for group 1
- `npm run pipeline:usage` -> extract+degrade for group 2
- `npm run pipeline:all` -> run both groups sequentially
- `npm run pipeline` -> alias of `pipeline:usage`

Repository fetch commands (no degradation / no training):

- `npm run repos:fetch` -> clone both groups
- `npm run repos:fetch:type-defs` -> clone only type-definition sources
- `npm run repos:fetch:usage` -> clone only usage repositories

### `extract.ts`

```
npx tsx src/extract.ts <path...> [--context 0] [--output out.jsonl]
npx tsx src/extract.ts <path...> [--include-dts]
```

Default context radius is `0` (single line containing the annotation).
**Must match `refiner-locate.ts`** — model was trained on this radius;
mismatched windows = OOD prompts.

By default `.d.ts` files are skipped. Use `--include-dts` for type-definition
sources.

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
2. Re-run grouped pipeline(s) (`pipeline:type-defs` and/or
   `pipeline:usage`) — confirm the new rule produces enough pairs
   (≥ ~100 in `training_pairs.jsonl` rule histogram).
3. Add the inverse matcher to `RULES` in `refiner-locate.ts` and append the
   key to `RULE_ORDER`.
4. Add a `validate_*` function and a `VALIDATORS[...]` entry in
   `packages/ts-type-refiner/src/ts_type_refiner/validators.py`.
5. Retrain the model (`refiner-train`).

All four steps are required. Skipping any one breaks the trójkąt synchroniczny.

---

## Dziennik zmian (2026-05-02)

### 1) Przebudowa źródeł danych

- Wprowadzono dwa niezależne zbiory repozytoriów:
   - `type-defs` (źródła definicji typów do projektowania reguł),
   - `usage` (repozytoria użycia do budowy par treningowych).
- Dodano manifest i grupowanie źródeł:
   - `src/repo-manifest.ts`
   - `src/repo-groups.ts`
- Dodano narzędzia do fetch/pipeline per grupa:
   - `src/fetch-repos.ts`
   - `src/pipeline-group.ts`

### 2) Rozszerzenie zakresu typów i reguł

- Rozszerzono źródła `type-defs` o kolejne ekosystemy/browserowe definicje
   (React DOM, Astro language-tools, TypeScript DOM lib generator, Lit, FAST).
- Rozszerzono reguły degradacji i ich odpowiedniki locator/validator dla:
   - React,
   - TanStack,
   - Astro,
   - natywnych typów browser/custom-elements.
- Locator emituje wielohipotezowo kandydatów dla niejednoznacznych kształtów
   (szczególnie `unknown`).

### 3) Śledzenie pochodzenia i raportowanie

- Dodano propagację pola `repo` do rekordów extract/degrade.
- Dodano raport wkładu repozytoriów:
   - `src/repo-contribution-report.ts`
   - skrypt npm: `report:repos:usage`
- Dodano raport pokrycia reguł:
   - `src/rule-coverage-report.ts`
   - skrypty npm: `report:coverage:type-defs`, `report:coverage:usage`

### 4) Wyciszenie reguł DROP-FOR-NOW (bez usuwania ich z kodu)

- Reguły o niskim wsparciu zostały zachowane w implementacji, ale są obecnie
   wyciszone dla treningu i inferencji, aby model nie był oceniany na regułach,
   których nie trenowaliśmy w tej iteracji.
- Stan wyciszenia:
   - `degrade.ts`: reguły wyciszone nie tworzą par treningowych,
   - `refiner-locate.ts`: reguły wyciszone nie emitują kandydatów,
   - `ts-type-refiner/validators.py`: reguły wyciszone nie trafiają do aktywnego
      `VALIDATORS`.
- Reguły pozostają w kodzie i można je łatwo odciszyć w kolejnej iteracji.
