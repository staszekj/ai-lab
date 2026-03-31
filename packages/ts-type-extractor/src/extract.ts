/**
 * TypeScript Type Extractor for React Codebases
 * ===============================================
 *
 * Uses ts-morph (TypeScript AST parser) to extract type annotations
 * from React/TypeScript source files.
 *
 * Purpose:
 *   Build training data for an ML model that suggests precise React
 *   TypeScript types.  For example:
 *     Input:  "const handleClick = (e: Event) => ..."
 *     Target: "MouseEvent" (the more precise type)
 *
 * Pipeline:
 *   1. Parse .tsx/.ts files with ts-morph
 *   2. Walk the AST to find every type annotation
 *   3. Extract: { context, type, location }
 *   4. Output JSON lines — one training sample per line
 *
 * Usage:
 *   npx tsx src/extract.ts <path-to-react-project>
 *   npx tsx src/extract.ts ./samples
 */

import { Project, SyntaxKind, Node } from "ts-morph";
import * as path from "path";
import * as fs from "fs";

// ══════════════════════════════════════════════════════════════════════
// Types for extracted data
// ══════════════════════════════════════════════════════════════════════

interface TypeAnnotation {
  /** The file where this annotation was found */
  file: string;
  /** Line number (1-based) */
  line: number;
  /** What kind of annotation: parameter, variable, return type, etc. */
  kind: string;
  /** The name of the variable/parameter/property */
  name: string;
  /** The type annotation as written in source code */
  typeText: string;
  /** Surrounding code context (a few lines around the annotation) */
  context: string;
  /** Parent function/component name, if any */
  parentName: string | null;
}

// ══════════════════════════════════════════════════════════════════════
// Helper: get N lines of context around a given line
// ══════════════════════════════════════════════════════════════════════

function getContext(sourceText: string, line: number, radius: number = 3): string {
  const lines = sourceText.split("\n");
  const start = Math.max(0, line - 1 - radius);
  const end = Math.min(lines.length, line - 1 + radius + 1);
  return lines.slice(start, end).join("\n");
}

// ══════════════════════════════════════════════════════════════════════
// Helper: find the parent function/component name
// ══════════════════════════════════════════════════════════════════════

function getParentName(node: Node): string | null {
  let current = node.getParent();
  while (current) {
    if (
      Node.isFunctionDeclaration(current) ||
      Node.isArrowFunction(current) ||
      Node.isFunctionExpression(current) ||
      Node.isMethodDeclaration(current)
    ) {
      // For arrow functions assigned to a variable: const Foo = () => ...
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
// Main extraction logic
// ══════════════════════════════════════════════════════════════════════

function extractTypeAnnotations(projectPath: string): TypeAnnotation[] {
  console.log(`\n${"═".repeat(60)}`);
  console.log(`TYPE EXTRACTOR — scanning: ${projectPath}`);
  console.log(`${"═".repeat(60)}\n`);

  // Create a ts-morph project.
  // ts-morph wraps the TypeScript compiler API and gives us
  // a high-level interface to walk the AST.
  const tsconfigPath = path.join(projectPath, "tsconfig.json");
  const hasTsConfig = fs.existsSync(tsconfigPath);

  const project = new Project({
    // If the target project has a tsconfig, use it for type resolution.
    // Otherwise, just add files manually.
    ...(hasTsConfig ? { tsConfigFilePath: tsconfigPath } : {}),
    skipAddingFilesFromTsConfig: !hasTsConfig,
  });

  // If no tsconfig, add all .ts/.tsx files from the path
  if (!hasTsConfig) {
    const pattern = path.join(projectPath, "**/*.{ts,tsx}");
    project.addSourceFilesAtPaths(pattern);
  }

  const sourceFiles = project.getSourceFiles();
  console.log(`Found ${sourceFiles.length} TypeScript file(s)\n`);

  const annotations: TypeAnnotation[] = [];

  for (const sourceFile of sourceFiles) {
    const filePath = path.relative(projectPath, sourceFile.getFilePath());
    const sourceText = sourceFile.getFullText();
    console.log(`  Scanning: ${filePath}`);

    // ── 1. Function/method parameter types ────────────────────────
    // e.g. function handleClick(e: MouseEvent) { ... }
    //      const handler = (event: React.ChangeEvent<HTMLInputElement>) => ...
    sourceFile.forEachDescendant((node) => {
      // ts-morph v24 removed Node.isParameter — use SyntaxKind check
      if (node.getKind() === SyntaxKind.Parameter) {
        const param = node as import("ts-morph").ParameterDeclaration;
        const typeNode = param.getTypeNode();
        if (typeNode) {
          annotations.push({
            file: filePath,
            line: param.getStartLineNumber(),
            kind: "parameter",
            name: param.getName(),
            typeText: typeNode.getText(),
            context: getContext(sourceText, param.getStartLineNumber()),
            parentName: getParentName(param),
          });
        }
      }

      // ── 2. Variable declaration types ────────────────────────────
      // e.g. const count: number = 0
      //      const [state, setState]: [string, Dispatch<SetStateAction<string>>] = ...
      if (Node.isVariableDeclaration(node)) {
        const typeNode = node.getTypeNode();
        if (typeNode) {
          annotations.push({
            file: filePath,
            line: node.getStartLineNumber(),
            kind: "variable",
            name: node.getName(),
            typeText: typeNode.getText(),
            context: getContext(sourceText, node.getStartLineNumber()),
            parentName: getParentName(node),
          });
        }
      }

      // ── 3. Function return types ─────────────────────────────────
      // e.g. function App(): JSX.Element { ... }
      //      const handler = (): void => { ... }
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

          annotations.push({
            file: filePath,
            line: node.getStartLineNumber(),
            kind: "return_type",
            name,
            typeText: returnTypeNode.getText(),
            context: getContext(sourceText, node.getStartLineNumber()),
            parentName: getParentName(node),
          });
        }
      }

      // ── 4. Property types (interfaces, type aliases, class props) ──
      // e.g. interface Props { onClick: (e: MouseEvent) => void }
      if (Node.isPropertySignature(node) || Node.isPropertyDeclaration(node)) {
        const typeNode = node.getTypeNode();
        if (typeNode) {
          annotations.push({
            file: filePath,
            line: node.getStartLineNumber(),
            kind: "property",
            name: node.getName(),
            typeText: typeNode.getText(),
            context: getContext(sourceText, node.getStartLineNumber()),
            parentName: getParentName(node),
          });
        }
      }

      // ── 5. Type assertions / "as" casts ──────────────────────────
      // e.g. const el = document.getElementById("root") as HTMLDivElement
      if (Node.isAsExpression(node)) {
        const typeNode = node.getTypeNode();
        if (typeNode) {
          annotations.push({
            file: filePath,
            line: node.getStartLineNumber(),
            kind: "type_assertion",
            name: "<as_expression>",
            typeText: typeNode.getText(),
            context: getContext(sourceText, node.getStartLineNumber()),
            parentName: getParentName(node),
          });
        }
      }

      // ── 6. Generic type arguments ─────────────────────────────────
      // e.g. useState<string>(""), useRef<HTMLDivElement>(null)
      if (Node.isCallExpression(node)) {
        const typeArgs = node.getTypeArguments();
        if (typeArgs.length > 0) {
          const funcName = node.getExpression().getText();
          for (const typeArg of typeArgs) {
            annotations.push({
              file: filePath,
              line: node.getStartLineNumber(),
              kind: "generic_argument",
              name: funcName,
              typeText: typeArg.getText(),
              context: getContext(sourceText, node.getStartLineNumber()),
              parentName: getParentName(node),
            });
          }
        }
      }
    });
  }

  return annotations;
}

// ══════════════════════════════════════════════════════════════════════
// CLI entry point
// ══════════════════════════════════════════════════════════════════════

function main() {
  const targetPath = process.argv[2] || "./samples";
  const resolvedPath = path.resolve(targetPath);

  if (!fs.existsSync(resolvedPath)) {
    console.error(`Error: path does not exist: ${resolvedPath}`);
    process.exit(1);
  }

  const annotations = extractTypeAnnotations(resolvedPath);

  // ── Print summary ───────────────────────────────────────────────
  console.log(`\n${"─".repeat(60)}`);
  console.log(`EXTRACTION SUMMARY`);
  console.log(`${"─".repeat(60)}`);
  console.log(`  Total annotations: ${annotations.length}`);

  // Count by kind
  const byKind = new Map<string, number>();
  for (const a of annotations) {
    byKind.set(a.kind, (byKind.get(a.kind) || 0) + 1);
  }
  for (const [kind, count] of [...byKind.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`    ${kind.padEnd(20)} ${count}`);
  }

  // Count by file
  const byFile = new Map<string, number>();
  for (const a of annotations) {
    byFile.set(a.file, (byFile.get(a.file) || 0) + 1);
  }
  console.log(`\n  By file:`);
  for (const [file, count] of [...byFile.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`    ${file.padEnd(40)} ${count}`);
  }

  // ── Output as JSON lines (one per annotation) ──────────────────
  const outputPath = path.join(resolvedPath, "..", "extracted_types.jsonl");
  const lines = annotations.map((a) => JSON.stringify(a));
  fs.writeFileSync(outputPath, lines.join("\n") + "\n", "utf-8");
  console.log(`\n  Output: ${outputPath}`);
  console.log(`  Format: JSON Lines (one annotation per line)`);

  // ── Show a few examples ─────────────────────────────────────────
  console.log(`\n${"─".repeat(60)}`);
  console.log(`SAMPLE ANNOTATIONS (first 5)`);
  console.log(`${"─".repeat(60)}`);
  for (const a of annotations.slice(0, 5)) {
    console.log(`\n  [${a.kind}] ${a.name}: ${a.typeText}`);
    console.log(`  File: ${a.file}:${a.line}`);
    if (a.parentName) console.log(`  Parent: ${a.parentName}`);
    console.log(`  Context:`);
    for (const line of a.context.split("\n")) {
      console.log(`    ${line}`);
    }
  }

  console.log(`\n${"═".repeat(60)}`);
  console.log(`Done. ${annotations.length} type annotations extracted.`);
  console.log(`${"═".repeat(60)}\n`);
}

main();
