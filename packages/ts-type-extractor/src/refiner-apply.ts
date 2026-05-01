/**
 * Refiner Applier
 *
 * Reads edits.jsonl (output of refiner-infer) and rewrites the
 * affected source files in place — replacing the byte range
 * [start, end) with `suggestion` for every edit where
 * `accepted == true`.
 *
 * Safety:
 *   - Edits are grouped by file and applied from the END of the file
 *     toward the start, so earlier offsets stay valid after each splice.
 *   - We re-read the file from disk and verify the slice [start, end)
 *     literally equals `degradedType`. If anything has changed since
 *     the candidate was located (file edited, different encoding,
 *     stale candidates JSONL), we SKIP that edit and report it.
 *   - --dry-run prints a unified-style diff without touching any file.
 *
 * Usage:
 *   npx tsx src/refiner-apply.ts --input edits.jsonl [--dry-run]
 */

import * as fs from "fs";
import * as path from "path";

// ══════════════════════════════════════════════════════════════════════
// Types
// ══════════════════════════════════════════════════════════════════════

interface Edit {
  id: string;
  file: string;
  start: number;
  end: number;
  degradedType: string;
  suggestion: string;
  accepted: boolean;
  reason: string;
  logprob: number | null;
  ruleValidatorPassed: boolean;
}

// ══════════════════════════════════════════════════════════════════════
// CLI
// ══════════════════════════════════════════════════════════════════════

function parseArgs() {
  const args = process.argv.slice(2);
  let input = "";
  let dryRun = false;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--input" && args[i + 1]) {
      input = args[i + 1];
      i++;
    } else if (args[i] === "--dry-run") {
      dryRun = true;
    }
  }
  if (!input) {
    console.error("usage: refiner-apply --input edits.jsonl [--dry-run]");
    process.exit(1);
  }
  return { input, dryRun };
}

// ══════════════════════════════════════════════════════════════════════
// Apply
// ══════════════════════════════════════════════════════════════════════

interface ApplyResult {
  file: string;
  applied: number;
  skipped: number;
  skippedReasons: string[];
}

function applyEditsToFile(
  filePath: string,
  edits: Edit[],
  dryRun: boolean,
): ApplyResult {
  const original = fs.readFileSync(filePath, "utf-8");

  // Sort descending by `start` so each splice keeps earlier offsets valid.
  const sorted = [...edits].sort((a, b) => b.start - a.start);

  let text = original;
  let applied = 0;
  let skipped = 0;
  const skippedReasons: string[] = [];

  for (const e of sorted) {
    const slice = text.slice(e.start, e.end);
    if (slice !== e.degradedType) {
      skipped++;
      skippedReasons.push(
        `[${e.start}:${e.end}] expected ${JSON.stringify(e.degradedType)} got ${JSON.stringify(slice)}`,
      );
      continue;
    }
    text = text.slice(0, e.start) + e.suggestion + text.slice(e.end);
    applied++;
  }

  // ── Diff preview ─────────────────────────────────────────────────
  if (applied > 0) {
    const rel = path.relative(process.cwd(), filePath);
    console.log(`\n  ── ${rel}  (+${applied}${skipped ? `, skipped ${skipped}` : ""})`);
    for (const e of [...sorted].reverse()) {
      const slice = original.slice(e.start, e.end);
      if (slice === e.degradedType) {
        // Find line number for nicer output
        const lineNo = original.slice(0, e.start).split("\n").length;
        console.log(`    L${lineNo}  ${JSON.stringify(e.degradedType)}  →  ${JSON.stringify(e.suggestion)}  (lp=${e.logprob?.toFixed(2) ?? "?"})`);
      }
    }
  }

  if (!dryRun && applied > 0) {
    fs.writeFileSync(filePath, text, "utf-8");
  }

  return { file: filePath, applied, skipped, skippedReasons };
}

// ══════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════

function main() {
  const { input, dryRun } = parseArgs();

  const inPath = path.resolve(input);
  if (!fs.existsSync(inPath)) {
    console.error(`Error: file not found: ${inPath}`);
    process.exit(1);
  }

  console.log(`\n${"═".repeat(60)}`);
  console.log(`REFINER APPLIER  ${dryRun ? "(dry-run)" : ""}`);
  console.log(`  Input: ${inPath}`);
  console.log(`${"═".repeat(60)}`);

  // ── Parse edits ──────────────────────────────────────────────────
  const lines = fs.readFileSync(inPath, "utf-8").split("\n").filter((l) => l.trim());
  const allEdits: Edit[] = lines.map((l) => JSON.parse(l));
  const accepted = allEdits.filter((e) => e.accepted);

  console.log(`  Total edits:    ${allEdits.length}`);
  console.log(`  Accepted:       ${accepted.length}`);
  console.log(`  Rejected:       ${allEdits.length - accepted.length}`);

  if (accepted.length === 0) {
    console.log("\n  Nothing to apply.\n");
    return;
  }

  // ── Group by file ────────────────────────────────────────────────
  const byFile = new Map<string, Edit[]>();
  for (const e of accepted) {
    const list = byFile.get(e.file);
    if (list) list.push(e);
    else byFile.set(e.file, [e]);
  }

  // ── Apply ────────────────────────────────────────────────────────
  let totalApplied = 0;
  let totalSkipped = 0;
  for (const [file, edits] of byFile) {
    if (!fs.existsSync(file)) {
      console.warn(`  SKIP file (not found): ${file}`);
      totalSkipped += edits.length;
      continue;
    }
    const r = applyEditsToFile(file, edits, dryRun);
    totalApplied += r.applied;
    totalSkipped += r.skipped;
    for (const reason of r.skippedReasons) {
      console.warn(`    skip: ${reason}`);
    }
  }

  console.log(`\n${"─".repeat(60)}`);
  console.log(`  Files modified: ${byFile.size}`);
  console.log(`  Edits applied:  ${totalApplied}${dryRun ? " (dry-run)" : ""}`);
  console.log(`  Edits skipped:  ${totalSkipped}`);
  console.log("");
}

main();
