import * as fs from "fs";
import * as path from "path";

interface ExtractedRow {
  typeText: string;
}

interface PairRow {
  rule: string;
  degradedType: string;
  preciseType: string;
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

function main(): void {
  const args = parseArgs();
  const extracted = readJsonl<ExtractedRow>(args.extracted);
  const pairs = readJsonl<PairRow>(args.pairs);

  const extractedByType = new Map<string, number>();
  const matchedByType = new Map<string, number>();
  const byRule = new Map<string, number>();
  const byDegraded = new Map<string, number>();

  for (const row of extracted) {
    inc(extractedByType, row.typeText);
  }
  for (const row of pairs) {
    inc(byRule, row.rule);
    inc(byDegraded, row.degradedType);
    inc(matchedByType, row.preciseType);
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

  console.log(`\nRule histogram (${byRule.size} rules):`);
  for (const [rule, count] of [...byRule.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`  ${rule.padEnd(56)} ${String(count).padStart(8)}  ${pct(count, pairsTotal)}`);
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
