/**
 * Materialize Encoder/Decoder Pairs
 *
 * Reads `training_pairs.jsonl` (rich metadata + degraded context + preciseType)
 * and produces `encoder_decoder_pairs.jsonl` with literal two-field rows:
 *
 *   { "input": "<encoder input>", "target": "<decoder target>", "rule": "..." }
 *
 * The `input` field is built using the exact same format as
 * `ts_type_refiner.prompt.build_refine_prompt` (Python). Both implementations
 * MUST stay in sync.
 *
 * The `rule` field is preserved for stratified train/val splits.
 */

import * as fs from "fs";
import * as path from "path";

interface TrainingPair {
  repo?: string;
  context: string;
  name: string;
  kind: string;
  degradedType: string;
  preciseType: string;
  rule: string;
  file: string;
  line: number;
  ast?: string;
  siblings?: string;
}

interface MaterializedPair {
  input: string;
  target: string;
  rule: string;
}

/**
 * Build encoder input prompt. Mirrors `build_refine_prompt` in
 * `ts_type_refiner/prompt.py` exactly.
 */
function buildRefinePrompt(p: TrainingPair): string {
  const meta = [
    `rule=${p.rule}`,
    `kind=${p.kind}`,
    `name=${p.name}`,
    `degraded=${p.degradedType}`,
  ];
  const ast = p.ast ?? p.siblings;
  if (ast) meta.push(`ast=${ast}`);

  return "[REFINE " + meta.join(" | ") + "]\n" + p.context;
}

function parseArgs() {
  const args = process.argv.slice(2);
  let input = "";
  let output = "";
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--output" && args[i + 1]) {
      output = args[i + 1];
      i++;
    } else if (!input) {
      input = args[i];
    }
  }
  if (!input) input = "data/training_pairs.jsonl";
  return { input, output };
}

function main() {
  const { input, output } = parseArgs();
  const inputPath = path.resolve(input);

  if (!fs.existsSync(inputPath)) {
    console.error(`Error: file not found: ${inputPath}`);
    process.exit(1);
  }

  console.log(`\n${"═".repeat(60)}`);
  console.log(`MATERIALIZE PAIRS — encoder input + decoder target`);
  console.log(`${"═".repeat(60)}`);
  console.log(`  Input: ${inputPath}\n`);

  const lines = fs.readFileSync(inputPath, "utf-8").trim().split("\n");
  const materialized: MaterializedPair[] = [];

  for (const line of lines) {
    if (!line) continue;
    const p: TrainingPair = JSON.parse(line);
    materialized.push({
      input: buildRefinePrompt(p),
      target: p.preciseType,
      rule: p.rule,
    });
  }

  const outputPath = output
    ? path.resolve(output)
    : path.join(path.dirname(inputPath), "encoder_decoder_pairs.jsonl");

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(
    outputPath,
    materialized.map((p) => JSON.stringify(p)).join("\n") + "\n",
    "utf-8",
  );

  console.log(`  Pairs materialized: ${materialized.length}`);
  console.log(`  Output: ${outputPath}`);
  console.log(`${"═".repeat(60)}\n`);
}

main();
