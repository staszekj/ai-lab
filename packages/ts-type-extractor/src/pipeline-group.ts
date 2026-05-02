import * as fs from "fs";
import * as path from "path";
import { spawnSync } from "child_process";
import { REPO_GROUPS, type RepoGroupName } from "./repo-groups.js";

interface Args {
  group: RepoGroupName;
  context: number;
  extractOut: string;
  pairsOut: string;
  strict: boolean;
}

function isRepoGroupName(value: string): value is RepoGroupName {
  return value === "type-defs" || value === "usage";
}

function parseArgs(): Args {
  const args = process.argv.slice(2);

  let group: RepoGroupName = "usage";
  let context = 0;
  let extractOut = "";
  let pairsOut = "";
  let strict = false;

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--group" && args[i + 1]) {
      const candidate = args[i + 1];
      if (!isRepoGroupName(candidate)) {
        throw new Error(`Unsupported --group '${candidate}'. Use: type-defs | usage`);
      }
      group = candidate;
      i++;
    } else if (a === "--context" && args[i + 1]) {
      context = parseInt(args[i + 1], 10);
      i++;
    } else if (a === "--extract-out" && args[i + 1]) {
      extractOut = args[i + 1];
      i++;
    } else if (a === "--pairs-out" && args[i + 1]) {
      pairsOut = args[i + 1];
      i++;
    } else if (a === "--strict") {
      strict = true;
    }
  }

  if (!extractOut) extractOut = `data/extracted_${group}.jsonl`;
  if (!pairsOut) pairsOut = `data/training_pairs_${group}.jsonl`;

  return { group, context, extractOut, pairsOut, strict };
}

function runOrThrow(command: string, argv: string[]) {
  const res = spawnSync(command, argv, { stdio: "inherit" });
  if (res.status !== 0) {
    throw new Error(`Command failed: ${command} ${argv.join(" ")}`);
  }
}

function main() {
  const { group, context, extractOut, pairsOut, strict } = parseArgs();
  const configured = REPO_GROUPS[group];

  const existing: string[] = [];
  const missing: string[] = [];

  for (const rel of configured) {
    const abs = path.resolve(rel);
    if (fs.existsSync(abs)) existing.push(rel);
    else missing.push(rel);
  }

  console.log(`\n=== PIPELINE GROUP: ${group} ===`);
  console.log(`Configured: ${configured.length} repos`);
  console.log(`Existing:   ${existing.length} repos`);
  if (missing.length > 0) {
    console.log(`Missing:    ${missing.length} repos`);
    for (const m of missing) console.log(`  - ${m}`);
    if (strict) {
      throw new Error("Missing repositories in strict mode.");
    }
  }

  if (existing.length === 0) {
    throw new Error("No repositories found for selected group.");
  }

  const extractArgs = [
    "tsx",
    "src/extract.ts",
    ...existing,
    "--context",
    String(context),
    "--output",
    extractOut,
  ];

  if (group === "type-defs") {
    // Type-definition repos are mostly .d.ts files.
    extractArgs.push("--include-dts");
  }

  runOrThrow("npx", extractArgs);

  runOrThrow("npx", [
    "tsx",
    "src/degrade.ts",
    extractOut,
    "--output",
    pairsOut,
  ]);

  console.log("\nDone.");
  console.log(`  extracted: ${path.resolve(extractOut)}`);
  console.log(`  pairs:     ${path.resolve(pairsOut)}`);
}

main();
