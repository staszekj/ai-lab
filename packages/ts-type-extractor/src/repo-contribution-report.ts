import * as fs from "fs";
import * as path from "path";

interface ExtractedRow {
  repo?: string;
}

interface PairRow {
  repo?: string;
  rule: string;
}

interface Args {
  extracted: string;
  pairs: string;
  top: number;
  minExtracted: number;
  label: string;
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  let extracted = "";
  let pairs = "";
  let top = 30;
  let minExtracted = 500;
  let label = "repo-contrib";

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--extracted" && args[i + 1]) {
      extracted = args[i + 1];
      i++;
    } else if (a === "--pairs" && args[i + 1]) {
      pairs = args[i + 1];
      i++;
    } else if (a === "--top" && args[i + 1]) {
      top = Math.max(1, parseInt(args[i + 1], 10) || 30);
      i++;
    } else if (a === "--min-extracted" && args[i + 1]) {
      minExtracted = Math.max(1, parseInt(args[i + 1], 10) || 500);
      i++;
    } else if (a === "--label" && args[i + 1]) {
      label = args[i + 1];
      i++;
    }
  }

  if (!extracted || !pairs) {
    throw new Error(
      "Usage: tsx src/repo-contribution-report.ts --extracted <file> --pairs <file> [--top 30] [--min-extracted 500] [--label x]",
    );
  }

  return { extracted, pairs, top, minExtracted, label };
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

function normalizeRepoName(raw?: string): string {
  return raw && raw.trim() ? raw.trim() : "<unknown>";
}

function main(): void {
  const args = parseArgs();
  const extracted = readJsonl<ExtractedRow>(args.extracted);
  const pairs = readJsonl<PairRow>(args.pairs);

  const extractedByRepo = new Map<string, number>();
  const pairsByRepo = new Map<string, number>();
  const rulesByRepo = new Map<string, Map<string, number>>();

  for (const row of extracted) {
    inc(extractedByRepo, normalizeRepoName(row.repo));
  }

  for (const row of pairs) {
    const repo = normalizeRepoName(row.repo);
    inc(pairsByRepo, repo);

    const byRule = rulesByRepo.get(repo) || new Map<string, number>();
    inc(byRule, row.rule);
    rulesByRepo.set(repo, byRule);
  }

  const totalExtracted = extracted.length;
  const totalPairs = pairs.length;

  const repos = [...new Set([...extractedByRepo.keys(), ...pairsByRepo.keys()])];
  const rows = repos.map((repo) => {
    const e = extractedByRepo.get(repo) || 0;
    const p = pairsByRepo.get(repo) || 0;
    return { repo, extracted: e, pairs: p, conv: e > 0 ? p / e : 0 };
  });

  rows.sort((a, b) => b.pairs - a.pairs);

  console.log(`\n${"=".repeat(78)}`);
  console.log(`REPO CONTRIBUTION REPORT: ${args.label}`);
  console.log(`${"=".repeat(78)}`);
  console.log(`extracted: ${path.resolve(args.extracted)}`);
  console.log(`pairs:     ${path.resolve(args.pairs)}`);
  console.log(`total annotations: ${totalExtracted}`);
  console.log(`total pairs:       ${totalPairs}`);
  console.log(`global coverage:   ${pct(totalPairs, totalExtracted)}`);
  console.log(`repos seen:        ${rows.length}`);

  console.log(`\nTop ${args.top} repos by pairs:`);
  for (const row of rows.slice(0, args.top)) {
    console.log(
      `  ${row.repo.padEnd(28)} pairs=${String(row.pairs).padStart(6)}  annotations=${String(row.extracted).padStart(7)}  conv=${pct(row.pairs, row.extracted)}`,
    );
  }

  console.log(`\nBottom ${args.top} repos by pairs (with annotations > 0):`);
  const nonEmpty = rows.filter((r) => r.extracted > 0).sort((a, b) => a.pairs - b.pairs);
  for (const row of nonEmpty.slice(0, args.top)) {
    console.log(
      `  ${row.repo.padEnd(28)} pairs=${String(row.pairs).padStart(6)}  annotations=${String(row.extracted).padStart(7)}  conv=${pct(row.pairs, row.extracted)}`,
    );
  }

  const lowYield = rows
    .filter((r) => r.extracted >= args.minExtracted && r.pairs === 0)
    .sort((a, b) => b.extracted - a.extracted);

  console.log(`\nLow-yield repos (annotations >= ${args.minExtracted}, pairs = 0): ${lowYield.length}`);
  for (const row of lowYield) {
    console.log(`  ${row.repo.padEnd(28)} annotations=${String(row.extracted).padStart(7)}`);
  }

  console.log(`\nTop rule per top ${Math.min(args.top, rows.length)} repo:`);
  for (const row of rows.slice(0, Math.min(args.top, rows.length))) {
    const byRule = rulesByRepo.get(row.repo);
    if (!byRule || byRule.size === 0) {
      console.log(`  ${row.repo.padEnd(28)} <none>`);
      continue;
    }
    const [rule, count] = [...byRule.entries()].sort((a, b) => b[1] - a[1])[0];
    console.log(`  ${row.repo.padEnd(28)} ${rule} (${count})`);
  }
}

main();
