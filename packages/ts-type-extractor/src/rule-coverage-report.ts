import * as fs from "fs";
import * as path from "path";

interface ExtractedRow {
  typeText: string;
}

interface PairRow {
  rule: string;
  degradedType: string;
  preciseType: string;
  target: string;
  siblings?: string;
  isNegative?: boolean;
  split?: "train" | "val";
}

interface Args {
  extracted: string;
  pairs: string;
  top: number;
  label: string;
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  let extracted = "";
  let pairs = "";
  let top = 20;
  let label = "coverage";

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--extracted" && args[i + 1]) {
      extracted = args[i + 1];
      i++;
    } else if (a === "--pairs" && args[i + 1]) {
      pairs = args[i + 1];
      i++;
    } else if (a === "--top" && args[i + 1]) {
      top = Math.max(1, parseInt(args[i + 1], 10) || 20);
      i++;
    } else if (a === "--label" && args[i + 1]) {
      label = args[i + 1];
      i++;
    }
  }

  if (!extracted || !pairs) {
    throw new Error("Usage: tsx src/rule-coverage-report.ts --extracted <file> --pairs <file> [--top 20] [--label x]");
  }

  return { extracted, pairs, top, label };
}

function readJsonl<T>(filePath: string): T[] {
  const p = path.resolve(filePath);
  if (!fs.existsSync(p)) {
    throw new Error(`File not found: ${p}`);
  }

  const txt = fs.readFileSync(p, "utf-8").trim();
  if (!txt) return [];
  return txt
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T);
}

function inc(map: Map<string, number>, key: string, by = 1): void {
  map.set(key, (map.get(key) || 0) + by);
}

function pct(n: number, d: number): string {
  if (d <= 0) return "0.0%";
  return `${((n / d) * 100).toFixed(1)}%`;
}

function topN(map: Map<string, number>, n: number): Array<[string, number]> {
  return [...map.entries()].sort((a, b) => b[1] - a[1]).slice(0, n);
}

// Identifiers we consider "trivial" — present in every TS file, recoverable
// from nothing. Recoverability counts non-trivial identifiers only.
const TRIVIAL_IDENTS = new Set<string>([
  "string", "number", "boolean", "unknown", "any", "never",
  "void", "null", "undefined", "object", "true", "false",
  "Array", "Promise", "Record", "Partial", "Required", "Readonly",
  "Pick", "Omit", "Exclude", "Extract", "NonNullable", "ReturnType",
  "Parameters", "InstanceType", "Awaited", "Map", "Set", "Date",
  "React", "JSX",
]);

const IDENT_RE = /[A-Za-z_$][A-Za-z0-9_$]*/g;

function extractIdentifiers(s: string): string[] {
  if (!s) return [];
  const out: string[] = [];
  for (const m of s.matchAll(IDENT_RE)) {
    const tok = m[0];
    if (TRIVIAL_IDENTS.has(tok)) continue;
    out.push(tok);
  }
  return out;
}

/**
 * Recoverability per pair: of the non-trivial identifiers appearing in the
 * target, what fraction also appear anywhere in the `siblings` string?
 *
 * Returns `null` when the target has no non-trivial identifiers (e.g.
 * `string`, `unknown`) — those pairs are trivially "recoverable" but skew
 * averages, so we exclude them from the score.
 */
function recoverabilityScore(target: string, siblings: string): number | null {
  const idents = extractIdentifiers(target);
  if (idents.length === 0) return null;
  const sibSet = new Set(extractIdentifiers(siblings));
  let hit = 0;
  for (const id of idents) if (sibSet.has(id)) hit++;
  return hit / idents.length;
}

function main(): void {
  const args = parseArgs();
  const extracted = readJsonl<ExtractedRow>(args.extracted);
  const pairs = readJsonl<PairRow>(args.pairs);

  const extractedByType = new Map<string, number>();
  const matchedByType = new Map<string, number>();
  const byRule = new Map<string, number>();
  const byDegraded = new Map<string, number>();
  // Step 1.7: positive/negative + recoverability per rule
  const posByRule = new Map<string, number>();
  const negByRule = new Map<string, number>();
  const trainByRule = new Map<string, number>();
  const valByRule = new Map<string, number>();
  // For recoverability: per rule sum and count of pairs that have a non-null score
  const recovSumByRule = new Map<string, number>();
  const recovCountByRule = new Map<string, number>();
  const recovFullByRule = new Map<string, number>();  // count where score == 1.0

  for (const row of extracted) {
    inc(extractedByType, row.typeText);
  }
  for (const row of pairs) {
    inc(byRule, row.rule);
    inc(byDegraded, row.degradedType);
    inc(matchedByType, row.preciseType);

    if (row.isNegative) inc(negByRule, row.rule);
    else inc(posByRule, row.rule);

    if (row.split === "train") inc(trainByRule, row.rule);
    else if (row.split === "val") inc(valByRule, row.rule);

    // Recoverability is meaningful only for positives — negatives have
    // target === degraded, the model is told the answer.
    if (!row.isNegative) {
      const score = recoverabilityScore(row.target ?? row.preciseType, row.siblings ?? "");
      if (score !== null) {
        recovSumByRule.set(row.rule, (recovSumByRule.get(row.rule) || 0) + score);
        inc(recovCountByRule, row.rule);
        if (score >= 0.999) inc(recovFullByRule, row.rule);
      }
    }
  }

  let unmatchedTotal = 0;
  const unmatchedByType = new Map<string, number>();
  for (const [typeText, total] of extractedByType.entries()) {
    const matched = matchedByType.get(typeText) || 0;
    if (total > matched) {
      const diff = total - matched;
      unmatchedTotal += diff;
      unmatchedByType.set(typeText, diff);
    }
  }

  const extractedTotal = extracted.length;
  const pairsTotal = pairs.length;

  console.log(`\n${"=".repeat(72)}`);
  console.log(`RULE COVERAGE REPORT: ${args.label}`);
  console.log(`${"=".repeat(72)}`);
  console.log(`extracted: ${path.resolve(args.extracted)}`);
  console.log(`pairs:     ${path.resolve(args.pairs)}`);
  console.log(`annotations: ${extractedTotal}`);
  console.log(`pairs:       ${pairsTotal}`);
  console.log(`coverage:    ${pct(pairsTotal, extractedTotal)}`);
  console.log(`unmatched:   ${unmatchedTotal} (${pct(unmatchedTotal, extractedTotal)})`);

  console.log(`\nPer-rule breakdown (${byRule.size} rules):`);
  console.log(
    "  " +
      "rule".padEnd(56) +
      "total".padStart(7) +
      "  pos".padStart(7) +
      "  neg".padStart(6) +
      " train".padStart(7) +
      "  val".padStart(6) +
      "  recov  full",
  );
  console.log("  " + "-".repeat(56 + 7 + 7 + 6 + 7 + 6 + 14));
  let macroRecov = 0;
  let macroRecovCount = 0;
  for (const [rule, count] of [...byRule.entries()].sort((a, b) => b[1] - a[1])) {
    const pos = posByRule.get(rule) || 0;
    const neg = negByRule.get(rule) || 0;
    const tr = trainByRule.get(rule) || 0;
    const va = valByRule.get(rule) || 0;
    const rs = recovSumByRule.get(rule) || 0;
    const rc = recovCountByRule.get(rule) || 0;
    const rfull = recovFullByRule.get(rule) || 0;
    const avg = rc > 0 ? rs / rc : NaN;
    const fullPct = rc > 0 ? rfull / rc : NaN;
    const recovStr = rc > 0 ? `${(avg * 100).toFixed(0)}%` : "n/a";
    const fullStr = rc > 0 ? `${(fullPct * 100).toFixed(0)}%` : "n/a";
    if (rc > 0) {
      macroRecov += avg;
      macroRecovCount++;
    }
    console.log(
      "  " +
        rule.padEnd(56) +
        String(count).padStart(7) +
        String(pos).padStart(7) +
        String(neg).padStart(6) +
        String(tr).padStart(7) +
        String(va).padStart(6) +
        recovStr.padStart(7) +
        fullStr.padStart(6),
    );
  }
  if (macroRecovCount > 0) {
    console.log(`\n  Macro recoverability (avg over ${macroRecovCount} non-trivial rules): ${(macroRecov / macroRecovCount * 100).toFixed(1)}%`);
    console.log(`  Target per plan: ≥80% for non-trivial rules.`);
  }

  console.log(`\nTop ${args.top} degraded literals:`);
  for (const [degraded, count] of topN(byDegraded, args.top)) {
    console.log(`  ${degraded.padEnd(40)} ${String(count).padStart(8)}  ${pct(count, pairsTotal)}`);
  }

  console.log(`\nTop ${args.top} unmatched precise types:`);
  for (const [typeText, count] of topN(unmatchedByType, args.top)) {
    console.log(`  ${typeText.padEnd(60)} ${String(count).padStart(8)}`);
  }
}

main();
