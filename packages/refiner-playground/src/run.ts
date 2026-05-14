/**
 * Refiner playground orchestrator.
 *
 * Runs the full pipeline against one (or more) sample files:
 *   1. refiner-locate.ts  (sibling: packages/ts-type-extractor)
 *   2. refiner-infer      (uv run --package ts-type-refiner)
 *   3. refiner-apply.ts   (sibling: packages/ts-type-extractor)
 *
 * Intermediate artifacts live under ./tmp/ inside this package so they
 * don't pollute /tmp and are easy to inspect after a run.
 *
 * Usage:
 *   pnpm --filter refiner-playground run    (writes *.refined.tsx siblings)
 *   pnpm --filter refiner-playground run:dry
 *   tsx src/run.ts [--target samples/Foo.tsx] [--dry-run] [--min-logprob -0.5]
 *                  [--out-suffix refined | --in-place]
 */

import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// packages/refiner-playground/src/run.ts → repo root is three levels up.
const PKG_DIR  = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(PKG_DIR, "..", "..");
const TS_EXTRACTOR_DIR = path.join(REPO_ROOT, "packages", "ts-type-extractor");
const REFINER_DIR      = path.join(REPO_ROOT, "packages", "ts-type-refiner");

const TMP_DIR        = path.join(PKG_DIR, "tmp");
const CANDIDATES     = path.join(TMP_DIR, "candidates.jsonl");
const EDITS          = path.join(TMP_DIR, "edits.jsonl");

const CHECKPOINT     = path.join(REFINER_DIR, "checkpoints", "refiner.pt");
const TOKENIZER      = path.join(REFINER_DIR, "tokenizer.json");

// ══════════════════════════════════════════════════════════════════════
// CLI
// ══════════════════════════════════════════════════════════════════════

interface Args {
  targets: string[];
  dryRun: boolean;
  minLogprob: string | null;
  rule: string | null;
  outSuffix: string;   // empty string = overwrite in place
}

function parseArgs(): Args {
  const argv = process.argv.slice(2);
  const targets: string[] = [];
  let dryRun = false;
  let minLogprob: string | null = null;
  let rule: string | null = null;
  // Default: write refined output to a sibling `<name>.refined.<ext>` so the
  // original input sample is never mutated. The pipeline is exploratory, the
  // model still hallucinates — keeping the input pristine is the safer default.
  let outSuffix = "refined";

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--dry-run") dryRun = true;
    else if (a === "--target" && argv[i + 1]) { targets.push(argv[++i]); }
    else if (a === "--min-logprob" && argv[i + 1]) { minLogprob = argv[++i]; }
    else if (a === "--rule" && argv[i + 1]) { rule = argv[++i]; }
    else if (a === "--in-place") { outSuffix = ""; }
    else if (a === "--out-suffix" && argv[i + 1]) { outSuffix = argv[++i].replace(/^\.+/, ""); }
    else if (!a.startsWith("--")) { targets.push(a); }
    else {
      console.error(`unknown arg: ${a}`);
      process.exit(1);
    }
  }

  if (targets.length === 0) {
    // Default: every sample file under ./samples (excluding any already-refined siblings).
    const samplesDir = path.join(PKG_DIR, "samples");
    targets.push(...fs.readdirSync(samplesDir)
      .filter((f) => /\.tsx?$/.test(f))
      .filter((f) => !/\.refined\.tsx?$/.test(f))
      .map((f) => path.join(samplesDir, f)));
  }

  return { targets, dryRun, minLogprob, rule, outSuffix };
}

// ══════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════

function banner(title: string): void {
  console.log(`\n${"═".repeat(60)}`);
  console.log(title);
  console.log(`${"═".repeat(60)}`);
}

function run(cmd: string, args: string[], cwd: string): void {
  console.log(`\n$ ${cmd} ${args.join(" ")}\n  (cwd: ${cwd})`);
  const r = spawnSync(cmd, args, { cwd, stdio: "inherit" });
  if (r.status !== 0) {
    console.error(`\n✗ command failed with exit code ${r.status}`);
    process.exit(r.status ?? 1);
  }
}

function ensureCheckpoint(): void {
  if (!fs.existsSync(CHECKPOINT)) {
    console.error(`✗ checkpoint not found: ${CHECKPOINT}`);
    console.error(`  train the model first: uv run --package ts-type-refiner refiner-train`);
    process.exit(1);
  }
  if (!fs.existsSync(TOKENIZER)) {
    console.error(`✗ tokenizer not found: ${TOKENIZER}`);
    process.exit(1);
  }
}

// ══════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════

function main(): void {
  const args = parseArgs();
  ensureCheckpoint();
  fs.mkdirSync(TMP_DIR, { recursive: true });

  banner(`REFINER PLAYGROUND  ${args.dryRun ? "(dry-run)" : ""}`);
  console.log(`  Targets:    ${args.targets.length}`);
  for (const t of args.targets) console.log(`    - ${path.relative(REPO_ROOT, t)}`);
  console.log(`  Candidates: ${path.relative(REPO_ROOT, CANDIDATES)}`);
  console.log(`  Edits:      ${path.relative(REPO_ROOT, EDITS)}`);
  console.log(`  Output:     ${args.outSuffix ? `sibling "*.${args.outSuffix}.<ext>" files` : "in-place (originals overwritten)"}`);

  // ── 1. Locate ────────────────────────────────────────────────────
  banner("[1/3] refiner-locate");
  const locateArgs = [
    "tsx", "src/refiner-locate.ts",
    ...args.targets,
    "--output", CANDIDATES,
  ];
  if (args.rule) locateArgs.push("--rule", args.rule);
  run("npx", locateArgs, TS_EXTRACTOR_DIR);

  if (!fs.existsSync(CANDIDATES) || fs.statSync(CANDIDATES).size === 0) {
    console.log("\n  No candidates found — pipeline stops here.");
    return;
  }

  // ── 2. Infer ─────────────────────────────────────────────────────
  banner("[2/3] refiner-infer");
  const inferArgs = [
    "run", "--package", "ts-type-refiner", "refiner-infer",
    "--input", CANDIDATES,
    "--output", EDITS,
    "--checkpoint", CHECKPOINT,
    "--tokenizer",  TOKENIZER,
  ];
  if (args.minLogprob !== null) inferArgs.push("--min-logprob", args.minLogprob);
  run("uv", inferArgs, REPO_ROOT);

  // ── 3. Apply ─────────────────────────────────────────────────────
  banner("[3/3] refiner-apply");
  const applyArgs = [
    "tsx", path.join(TS_EXTRACTOR_DIR, "src", "refiner-apply.ts"),
    "--input", EDITS,
  ];
  if (args.dryRun) applyArgs.push("--dry-run");  if (args.outSuffix) applyArgs.push("--out-suffix", args.outSuffix);  // Run from each target's directory so the relative `file` field resolves.
  // For multi-file runs we group by directory.
  const byDir = new Map<string, string[]>();
  for (const t of args.targets) {
    const d = path.dirname(path.resolve(t));
    const list = byDir.get(d);
    if (list) list.push(t); else byDir.set(d, [t]);
  }
  for (const dir of byDir.keys()) {
    run("npx", applyArgs, dir);
  }

  banner("DONE");
  if (args.dryRun) {
    console.log("  Dry-run only — no files were written.");
  } else if (args.outSuffix) {
    console.log(`  Refined output written as sibling *.${args.outSuffix}.<ext> files.`);
    console.log(`  Inspect with:  ls packages/refiner-playground/samples/*.${args.outSuffix}.*`);
    console.log(`  Diff against original:  diff -u samples/Foo.tsx samples/Foo.${args.outSuffix}.tsx`);
  } else {
    console.log(`  Files under ${args.targets.length} target(s) potentially modified in place.`);
    console.log(`  Inspect with:  git diff -- packages/refiner-playground/samples/`);
    console.log(`  Revert  with:  git checkout -- packages/refiner-playground/samples/`);
  }
}

main();
