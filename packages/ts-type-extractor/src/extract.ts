/**
 * TypeScript Type Extractor
 *
 * Parses TS/TSX files via AST and extracts every type annotation
 * with surrounding code context.
 *
 * Usage:
 *   npx tsx src/extract.ts <path...> [--context 0] [--output out.jsonl] [--include-dts]
 *   npx tsx src/extract.ts data/repos/radix-primitives data/repos/tanstack-query
 *
 * Default context radius is 0 (just the line containing the annotation).
 * Rationale: with multi-rule training, multiple candidates on nearby
 * lines would otherwise share identical multi-line context windows,
 * making the prompt ambiguous (the model can't tell which `string` /
 * `unknown` / etc. it's being asked to refine). A 1-line window forces
 * the model to lean on the IDENTIFIER NAME as the dominant signal,
 * which is usually present on the same line as the annotation.
 * MUST stay in sync with refiner-locate.ts default.
 */

import { Project, SyntaxKind, Node } from "ts-morph";
import * as path from "path";
import * as fs from "fs";

// ══════════════════════════════════════════════════════════════════════
// Types
// ══════════════════════════════════════════════════════════════════════

interface TypeAnnotation {
  repo: string;
  file: string;
  line: number;
  kind: string;
  name: string;
  typeText: string;
  context: string;
  parentName: string | null;
}

// ══════════════════════════════════════════════════════════════════════
// CLI
// ══════════════════════════════════════════════════════════════════════

function parseArgs() {
  const args = process.argv.slice(2);
  const paths: string[] = [];
  let contextRadius = 0;
  let output = "";
  let includeDts = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--context" && args[i + 1]) {
      contextRadius = parseInt(args[i + 1], 10);
      i++;
    } else if (args[i] === "--output" && args[i + 1]) {
      output = args[i + 1];
      i++;
    } else if (args[i] === "--include-dts") {
      includeDts = true;
    } else {
      paths.push(args[i]);
    }
  }

  if (paths.length === 0) paths.push("./samples");
  return { paths, contextRadius, output, includeDts };
}

// ══════════════════════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════════════════════

function getContext(
  sourceText: string,
  line: number,
  radius: number,
): string {
  const lines = sourceText.split("\n");
  const start = Math.max(0, line - 1 - radius);
  const end = Math.min(lines.length, line - 1 + radius + 1);
  return lines.slice(start, end).join("\n");
}

function getParentName(node: Node): string | null {
  let current = node.getParent();
  while (current) {
    if (
      Node.isFunctionDeclaration(current) ||
      Node.isArrowFunction(current) ||
      Node.isFunctionExpression(current) ||
      Node.isMethodDeclaration(current)
    ) {
      const parent = current.getParent();
      if (parent && Node.isVariableDeclaration(parent)) {
        return parent.getName();
      }
      if ("getName" in current && typeof current.getName === "function") {
        const name = current.getName();
        if (name) return name;
      }
    }
    if (Node.isClassDeclaration(current)) {
      return current.getName() ?? null;
    }
    current = current.getParent();
  }
  return null;
}

// ══════════════════════════════════════════════════════════════════════
// Extraction
// ══════════════════════════════════════════════════════════════════════

// Skip auto-generated files — they contain huge machine-made types
// that are useless for training and can hang the parser.
const SKIP_FILE_PATTERNS = [
  /_generated/,
  /\.generated\./,
  /generated\//,
  /\.gen\.ts/,
  /__generated__/
];

// Max type annotation length in characters.
// Anything longer is auto-gen noise, not a real hand-written type.
const MAX_TYPE_LENGTH = 200;

function extractFromProject(
  projectPath: string,
  contextRadius: number,
  includeDts: boolean,
): TypeAnnotation[] {
  const repoName = path.basename(projectPath);
  const t0 = Date.now();

  // Always use glob-based discovery to handle monorepos reliably.
  // tsconfig in monorepo roots often doesn't include sub-packages.
  const project = new Project({
    skipAddingFilesFromTsConfig: true,
  });
  project.addSourceFilesAtPaths(path.join(projectPath, "**/*.{ts,tsx}"));

  const allFiles = project.getSourceFiles();
  let skippedFiles = 0;
  const sourceFiles = allFiles.filter((sf) => {
    const rel = path.relative(projectPath, sf.getFilePath());
    if (!includeDts && rel.endsWith(".d.ts")) {
      skippedFiles++;
      return false;
    }
    if (SKIP_FILE_PATTERNS.some((pat) => pat.test(rel))) {
      skippedFiles++;
      return false;
    }
    return true;
  });
  const total = sourceFiles.length;
  console.log(`\n  ┌─ ${repoName}: ${total} files (${skippedFiles} generated/dts skipped)`);

  const annotations: TypeAnnotation[] = [];
  let filesProcessed = 0;
  let lastLog = Date.now();

  for (const sourceFile of sourceFiles) {
    const filePath = path.relative(projectPath, sourceFile.getFilePath());
    const sourceText = sourceFile.getFullText();
    const beforeCount = annotations.length;

    sourceFile.forEachDescendant((node) => {
      // Helper: push annotation only if type is short enough
      const pushIfShort = (ann: TypeAnnotation) => {
        if (ann.typeText.length <= MAX_TYPE_LENGTH) {
          annotations.push(ann);
        }
      };

      // ── 1. Parameter types ──────────────────────────────────────
      if (node.getKind() === SyntaxKind.Parameter) {
        const param = node as import("ts-morph").ParameterDeclaration;
        const typeNode = param.getTypeNode();
        if (typeNode) {
          pushIfShort({
            repo: repoName,
            file: filePath,
            line: param.getStartLineNumber(),
            kind: "parameter",
            name: param.getName(),
            typeText: typeNode.getText(),
            context: getContext(sourceText, param.getStartLineNumber(), contextRadius),
            parentName: getParentName(param),
          });
        }
      }

      // ── 2. Variable types ───────────────────────────────────────
      if (Node.isVariableDeclaration(node)) {
        const typeNode = node.getTypeNode();
        if (typeNode) {
          pushIfShort({
            repo: repoName,
            file: filePath,
            line: node.getStartLineNumber(),
            kind: "variable",
            name: node.getName(),
            typeText: typeNode.getText(),
            context: getContext(sourceText, node.getStartLineNumber(), contextRadius),
            parentName: getParentName(node),
          });
        }
      }

      // ── 3. Return types ─────────────────────────────────────────
      if (
        Node.isFunctionDeclaration(node) ||
        Node.isArrowFunction(node) ||
        Node.isMethodDeclaration(node)
      ) {
        const returnTypeNode = node.getReturnTypeNode();
        if (returnTypeNode) {
          const name =
            Node.isFunctionDeclaration(node) || Node.isMethodDeclaration(node)
              ? (node.getName() ?? "<anonymous>")
              : (() => {
                  const parent = node.getParent();
                  return parent && Node.isVariableDeclaration(parent)
                    ? parent.getName()
                    : "<arrow>";
                })();

          pushIfShort({
            repo: repoName,
            file: filePath,
            line: node.getStartLineNumber(),
            kind: "return_type",
            name,
            typeText: returnTypeNode.getText(),
            context: getContext(sourceText, node.getStartLineNumber(), contextRadius),
            parentName: getParentName(node),
          });
        }
      }

      // ── 4. Property types ───────────────────────────────────────
      if (Node.isPropertySignature(node) || Node.isPropertyDeclaration(node)) {
        const typeNode = node.getTypeNode();
        if (typeNode) {
          pushIfShort({
            repo: repoName,
            file: filePath,
            line: node.getStartLineNumber(),
            kind: "property",
            name: node.getName(),
            typeText: typeNode.getText(),
            context: getContext(sourceText, node.getStartLineNumber(), contextRadius),
            parentName: getParentName(node),
          });
        }
      }

      // ── 5. Type assertions (as casts) ──────────────────────────
      if (Node.isAsExpression(node)) {
        const typeNode = node.getTypeNode();
        if (typeNode) {
          pushIfShort({
            repo: repoName,
            file: filePath,
            line: node.getStartLineNumber(),
            kind: "type_assertion",
            name: "<as_expression>",
            typeText: typeNode.getText(),
            context: getContext(sourceText, node.getStartLineNumber(), contextRadius),
            parentName: getParentName(node),
          });
        }
      }

      // ── 6. Generic type arguments ──────────────────────────────
      if (Node.isCallExpression(node)) {
        const typeArgs = node.getTypeArguments();
        if (typeArgs.length > 0) {
          const funcName = node.getExpression().getText();
          for (const typeArg of typeArgs) {
            pushIfShort({
              repo: repoName,
              file: filePath,
              line: node.getStartLineNumber(),
              kind: "generic_argument",
              name: funcName,
              typeText: typeArg.getText(),
              context: getContext(sourceText, node.getStartLineNumber(), contextRadius),
              parentName: getParentName(node),
            });
          }
        }
      }
    });

    filesProcessed++;
    const now = Date.now();
    // Log progress every 2 seconds or on last file
    if (now - lastLog >= 2000 || filesProcessed === total) {
      const pct = ((filesProcessed / total) * 100).toFixed(0);
      const elapsed = ((now - t0) / 1000).toFixed(1);
      process.stdout.write(
        `\r  │  ${filesProcessed}/${total} files (${pct}%) — ${annotations.length} annotations — ${elapsed}s`,
      );
      lastLog = now;
    }
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(
    `\n  └─ ${repoName}: ${annotations.length} annotations in ${elapsed}s`,
  );

  return annotations;
}

// ══════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════

function main() {
  const { paths, contextRadius, output, includeDts } = parseArgs();
  const outputPath = output || "extracted_types.jsonl";
  const resolvedOutput = path.resolve(outputPath);
  fs.mkdirSync(path.dirname(resolvedOutput), { recursive: true });

  console.log(`\n${"═".repeat(60)}`);
  console.log(`TYPE EXTRACTOR — context ±${contextRadius} lines`);
  console.log(`  Include .d.ts: ${includeDts ? "yes" : "no"}`);
  console.log(`  Repos: ${paths.length} | Output: ${resolvedOutput}`);
  console.log(`${"═".repeat(60)}`);

  const allAnnotations: TypeAnnotation[] = [];
  const t0 = Date.now();

  for (let i = 0; i < paths.length; i++) {
    const p = paths[i];
    const resolved = path.resolve(p);
    if (!fs.existsSync(resolved)) {
      console.error(`  SKIP: ${resolved} does not exist`);
      continue;
    }
    const repoAnnotations = extractFromProject(resolved, contextRadius, includeDts);
    allAnnotations.push(...repoAnnotations);

    // Write incrementally after each repo so progress isn't lost
    const lines = allAnnotations.map((a) => JSON.stringify(a));
    fs.writeFileSync(resolvedOutput, lines.join("\n") + "\n", "utf-8");
    console.log(
      `  ✓ Saved: ${allAnnotations.length} total (${i + 1}/${paths.length} repos)`,
    );
  }

  const totalTime = ((Date.now() - t0) / 1000).toFixed(1);

  // ── Summary by kind ─────────────────────────────────────────────
  const byKind = new Map<string, number>();
  for (const a of allAnnotations) {
    byKind.set(a.kind, (byKind.get(a.kind) || 0) + 1);
  }
  console.log(`\n${"─".repeat(60)}`);
  console.log(`  Total: ${allAnnotations.length} annotations in ${totalTime}s`);
  for (const [kind, count] of [...byKind.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`    ${kind.padEnd(20)} ${count}`);
  }

  console.log(`\n  Output: ${resolvedOutput}`);
}

main();
