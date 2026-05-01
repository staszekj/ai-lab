/**
 * Refiner Locator
 *
 * Finds type annotations in TS/TSX source files that are CANDIDATES
 * for refinement by the ts-type-refiner model.
 *
 * A "candidate" is any type node whose text matches a known DEGRADED
 * shape (e.g. literally `string` for the string_literal_union rule).
 * For each candidate we emit precise character offsets so the applier
 * can rewrite the source losslessly.
 *
 * The output context format MUST match training data — that is, the
 * surrounding code already contains the degraded type (since we read
 * the file as-is). Training data was built the same way: degraded
 * type was substituted into the context string.
 *
 * Usage:
 *   npx tsx src/refiner-locate.ts <path...> [--context 7] [--output candidates.jsonl] [--rule <rule>]
 *
 * Currently only `string_literal_union→string` is supported because
 * Phase 1 was trained exclusively on that rule.
 */

import { Project, SyntaxKind, Node } from "ts-morph";
import * as path from "path";
import * as fs from "fs";

// ══════════════════════════════════════════════════════════════════════
// Types
// ══════════════════════════════════════════════════════════════════════

interface RefineCandidate {
  id: string;            // stable id: "<file>:<start>"
  file: string;          // absolute path on disk
  line: number;          // 1-based line of the annotated node
  start: number;         // 0-based char offset of typeNode in file (inclusive)
  end: number;           // 0-based char offset of typeNode in file (exclusive)
  kind: string;          // parameter | variable | return_type | property | type_assertion | generic_argument
  name: string;          // identifier name (or <as_expression>)
  context: string;       // ±radius lines around the annotation
  degradedType: string;  // verbatim text currently at [start, end)
  rule: string;          // which inverse-degradation rule generated this candidate
}

// ══════════════════════════════════════════════════════════════════════
// Inverse-degradation rules
// ──────────────────────────────────────────────────────────────────────
//
// A "candidate matcher" answers: does the literal text of this type
// node look like something the model was trained to refine?
// We MUST be conservative — every false positive becomes a wasted
// model call AND a potential bad edit.
//
// Each rule has a `name` matching the rule emitted by degrade.ts, so
// downstream stages can route candidates to rule-specific validators.
// ══════════════════════════════════════════════════════════════════════

interface CandidateRule {
  name: string;
  match: (typeText: string) => boolean;
}

const RULES: Record<string, CandidateRule> = {
  string_literal_union: {
    name: "string_literal_union→string",
    // Trained Phase 1: `string` → `'a' | 'b' | ...`
    match: (t) => t.trim() === "string",
  },
};

// ══════════════════════════════════════════════════════════════════════
// Helpers (mirrors extract.ts so contexts match training data)
// ══════════════════════════════════════════════════════════════════════

function getContext(sourceText: string, line: number, radius: number): string {
  const lines = sourceText.split("\n");
  const start = Math.max(0, line - 1 - radius);
  const end = Math.min(lines.length, line - 1 + radius + 1);
  return lines.slice(start, end).join("\n");
}

const SKIP_FILE_PATTERNS = [
  /_generated/,
  /\.generated\./,
  /generated\//,
  /\.gen\.ts/,
  /__generated__/,
  /\.d\.ts$/,
];

// ══════════════════════════════════════════════════════════════════════
// CLI
// ══════════════════════════════════════════════════════════════════════

function parseArgs() {
  const args = process.argv.slice(2);
  const paths: string[] = [];
  let contextRadius = 7;
  let output = "";
  let ruleName = "string_literal_union";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--context" && args[i + 1]) {
      contextRadius = parseInt(args[i + 1], 10);
      i++;
    } else if (args[i] === "--output" && args[i + 1]) {
      output = args[i + 1];
      i++;
    } else if (args[i] === "--rule" && args[i + 1]) {
      ruleName = args[i + 1];
      i++;
    } else {
      paths.push(args[i]);
    }
  }

  if (paths.length === 0) {
    console.error("usage: refiner-locate <path...> [--context N] [--output file] [--rule name]");
    process.exit(1);
  }
  return { paths, contextRadius, output, ruleName };
}

// ══════════════════════════════════════════════════════════════════════
// Locator
// ══════════════════════════════════════════════════════════════════════

function locateInProject(
  projectPath: string,
  contextRadius: number,
  rule: CandidateRule,
): RefineCandidate[] {
  const repoName = path.basename(projectPath);
  const t0 = Date.now();

  const project = new Project({ skipAddingFilesFromTsConfig: true });

  // Accept either a directory or a single file.
  const stat = fs.statSync(projectPath);
  if (stat.isFile()) {
    project.addSourceFileAtPath(projectPath);
  } else {
    project.addSourceFilesAtPaths(path.join(projectPath, "**/*.{ts,tsx}"));
  }

  const allFiles = project.getSourceFiles();
  const sourceFiles = allFiles.filter((sf) => {
    const rel = path.relative(projectPath, sf.getFilePath());
    return !SKIP_FILE_PATTERNS.some((pat) => pat.test(rel));
  });
  const total = sourceFiles.length;
  console.log(`\n  ┌─ ${repoName}: scanning ${total} files for rule "${rule.name}"`);

  const candidates: RefineCandidate[] = [];
  let filesProcessed = 0;
  let lastLog = Date.now();

  for (const sourceFile of sourceFiles) {
    const absFile = sourceFile.getFilePath();
    const sourceText = sourceFile.getFullText();

    // Visit every kind of node where the extractor pulled type annotations,
    // emit a candidate when its typeNode text matches the rule.
    const tryEmit = (
      typeNode: Node | undefined,
      kind: string,
      name: string,
      anchorLine: number,
    ) => {
      if (!typeNode) return;
      const text = typeNode.getText();
      if (!rule.match(text)) return;
      const start = typeNode.getStart(); // skips leading trivia
      const end = typeNode.getEnd();
      candidates.push({
        id: `${absFile}:${start}`,
        file: absFile,
        line: anchorLine,
        start,
        end,
        kind,
        name,
        context: getContext(sourceText, anchorLine, contextRadius),
        degradedType: text,
        rule: rule.name,
      });
    };

    sourceFile.forEachDescendant((node) => {
      if (node.getKind() === SyntaxKind.Parameter) {
        const param = node as import("ts-morph").ParameterDeclaration;
        tryEmit(param.getTypeNode(), "parameter", param.getName(), param.getStartLineNumber());
        return;
      }
      if (Node.isVariableDeclaration(node)) {
        tryEmit(node.getTypeNode(), "variable", node.getName(), node.getStartLineNumber());
        return;
      }
      if (
        Node.isFunctionDeclaration(node) ||
        Node.isArrowFunction(node) ||
        Node.isMethodDeclaration(node)
      ) {
        const ret = node.getReturnTypeNode();
        if (ret) {
          const name =
            Node.isFunctionDeclaration(node) || Node.isMethodDeclaration(node)
              ? (node.getName() ?? "<anonymous>")
              : (() => {
                  const parent = node.getParent();
                  return parent && Node.isVariableDeclaration(parent)
                    ? parent.getName()
                    : "<arrow>";
                })();
          tryEmit(ret, "return_type", name, node.getStartLineNumber());
        }
        return;
      }
      if (Node.isPropertySignature(node) || Node.isPropertyDeclaration(node)) {
        tryEmit(node.getTypeNode(), "property", node.getName(), node.getStartLineNumber());
        return;
      }
      if (Node.isAsExpression(node)) {
        tryEmit(node.getTypeNode(), "type_assertion", "<as_expression>", node.getStartLineNumber());
        return;
      }
      if (Node.isCallExpression(node)) {
        const typeArgs = node.getTypeArguments();
        if (typeArgs.length === 0) return;
        const funcName = node.getExpression().getText();
        for (const typeArg of typeArgs) {
          tryEmit(typeArg, "generic_argument", funcName, node.getStartLineNumber());
        }
      }
    });

    filesProcessed++;
    const now = Date.now();
    if (now - lastLog >= 2000 || filesProcessed === total) {
      const pct = ((filesProcessed / total) * 100).toFixed(0);
      const elapsed = ((now - t0) / 1000).toFixed(1);
      process.stdout.write(
        `\r  │  ${filesProcessed}/${total} files (${pct}%) — ${candidates.length} candidates — ${elapsed}s`,
      );
      lastLog = now;
    }
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(
    `\n  └─ ${repoName}: ${candidates.length} candidates in ${elapsed}s`,
  );
  return candidates;
}

// ══════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════

function main() {
  const { paths, contextRadius, output, ruleName } = parseArgs();
  const rule = RULES[ruleName];
  if (!rule) {
    console.error(`Unknown rule: ${ruleName}`);
    console.error(`Available: ${Object.keys(RULES).join(", ")}`);
    process.exit(1);
  }

  const outputPath = output ? path.resolve(output) : "";
  if (outputPath) fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  console.log(`\n${"═".repeat(60)}`);
  console.log(`REFINER LOCATOR — rule "${rule.name}", context ±${contextRadius} lines`);
  console.log(`  Inputs: ${paths.length} | Output: ${outputPath || "<stdout>"}`);
  console.log(`${"═".repeat(60)}`);

  const all: RefineCandidate[] = [];
  for (const p of paths) {
    const resolved = path.resolve(p);
    if (!fs.existsSync(resolved)) {
      console.error(`  SKIP: ${resolved} does not exist`);
      continue;
    }
    all.push(...locateInProject(resolved, contextRadius, rule));
  }

  // ── Summary ──────────────────────────────────────────────────────
  const byKind = new Map<string, number>();
  for (const c of all) byKind.set(c.kind, (byKind.get(c.kind) || 0) + 1);
  console.log(`\n${"─".repeat(60)}`);
  console.log(`  Total candidates: ${all.length}`);
  for (const [kind, count] of [...byKind.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`    ${kind.padEnd(20)} ${count}`);
  }

  // ── Output ───────────────────────────────────────────────────────
  const lines = all.map((c) => JSON.stringify(c));
  if (outputPath) {
    fs.writeFileSync(outputPath, lines.join("\n") + "\n", "utf-8");
    console.log(`\n  Output: ${outputPath}`);
  } else {
    for (const l of lines) process.stdout.write(l + "\n");
  }
}

main();
