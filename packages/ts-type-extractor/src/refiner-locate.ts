/**
 * Refiner Locator
 *
 * Finds type annotations in TS/TSX source files that are CANDIDATES
 * for refinement by the ts-type-refiner model.
 *
 * A "candidate" is any type node whose text matches a known DEGRADED
 * shape (e.g. literally `string` for `string_literal_union`, `unknown`
 * for `utility_type`, `T[]` for `tuple→array`, etc.). For each
 * candidate we emit precise character offsets so the applier can
 * rewrite the source losslessly.
 *
 * The output context format MUST match training data — that is, the
 * surrounding code already contains the degraded type (since we read
 * the file as-is). Training data was built the same way: the degraded
 * type was substituted into the context string.
 *
 * Rule coverage: this file mirrors the 24 rules that produced training
 * pairs in `degrade.ts`. Each `RULES[key]` is a matcher on the
 * DEGRADED literal text — its job is to spot type nodes whose shape
 * suggests the model was trained to refine them. Validators (in
 * Python `ts_type_refiner/validators.py`) provide the matching
 * inverse on the model's output.
 *
 * Collision note: `returntype→unknown` and `utility_type→unknown`
 * both degrade to literal `unknown`. We cannot disambiguate at
 * locator time (the source text is identical), so we emit a single
 * candidate per `unknown` site under `utility_type→unknown`
 * (the more common rule in training data) and rely on the validator
 * to accept either precise form (ReturnType<...> or Extract|Exclude|
 * Omit|Pick<...>).
 *
 * Usage:
 *   npx tsx src/refiner-locate.ts <path...> [--context 0] \
 *       [--output candidates.jsonl] [--rule <name|all>]
 *
 * Default context radius is 0 (single line). MUST match extract.ts —
 * the model was trained on whatever window extract.ts produced, so
 * mismatched radii at inference time = out-of-distribution prompts.
 */

import { Project, SyntaxKind, Node } from "ts-morph";
import * as path from "path";
import * as fs from "fs";

// ══════════════════════════════════════════════════════════════════════
// Types
// ══════════════════════════════════════════════════════════════════════

interface RefineCandidate {
  id: string;
  file: string;
  line: number;
  start: number;
  end: number;
  kind: string;
  name: string;
  context: string;
  degradedType: string;
  rule: string;
}

// ══════════════════════════════════════════════════════════════════════
// Inverse-degradation rules
// ──────────────────────────────────────────────────────────────────────
//
// A "candidate matcher" answers: does the literal text of this type
// node look like something the model was trained to refine?
// We MUST be conservative — every false positive becomes a wasted
// model call AND a potential bad edit. The Python-side validator is
// the second gate; the log-prob threshold is the third.
//
// Each rule has a `name` matching the rule emitted by degrade.ts, so
// downstream stages can route candidates to rule-specific validators.
// ══════════════════════════════════════════════════════════════════════

interface CandidateRule {
  name: string;
  match: (typeText: string) => boolean;
}

// Helper: collapse whitespace so "HTMLElement | null" matches even if
// the source uses `HTMLElement\n  | null`.
const norm = (s: string): string => s.replace(/\s+/g, " ").trim();

const RULES: Record<string, CandidateRule> = {
  // ── React events / handlers ────────────────────────────────────
  "react_event": {
    name: "react_event→synthetic",
    match: (t) => t === "React.SyntheticEvent",
  },
  "react_handler": {
    name: "react_handler→generic_handler",
    match: (t) => t === "React.EventHandler<React.SyntheticEvent>",
  },

  // ── DOM elements ───────────────────────────────────────────────
  "html_element": {
    name: "html_element→generic",
    match: (t) => t === "HTMLElement",
  },
  "html_element_nullable": {
    name: "html_element_nullable→generic",
    match: (t) => norm(t) === "HTMLElement | null",
  },
  "svg_element": {
    name: "svg_element→generic",
    match: (t) => t === "SVGElement",
  },
  "dom_event": {
    name: "dom_event→generic",
    match: (t) => t === "Event",
  },

  // ── React refs ─────────────────────────────────────────────────
  "ref_element": {
    name: "ref_element→generic",
    match: (t) => t === "React.RefObject<HTMLElement>",
  },
  "mutable_ref_element": {
    name: "mutable_ref_element→generic",
    match: (t) => t === "React.MutableRefObject<HTMLElement>",
  },
  "ref_specific": {
    name: "ref_specific→unknown",
    match: (t) => t === "React.RefObject<unknown>",
  },
  "mutable_ref_specific": {
    name: "mutable_ref_specific→unknown",
    match: (t) => t === "React.MutableRefObject<unknown>",
  },

  // ── Primitive unions ───────────────────────────────────────────
  "string_literal_union": {
    name: "string_literal_union→string",
    // degrade rule 11: `'a' | 'b' | ...` → `string`
    match: (t) => t.trim() === "string",
  },
  "numeric_literal_union": {
    name: "numeric_literal_union→number",
    // degrade rule 12: `1 | 2 | 3` → `number`
    match: (t) => t.trim() === "number",
  },
  "boolean_literal": {
    name: "boolean_literal→boolean",
    // degrade rule 13: `true` or `false` → `boolean`
    match: (t) => t.trim() === "boolean",
  },
  "mixed_literal_union": {
    name: "mixed_literal_union→string_boolean",
    // degrade rule 14: `boolean | 'indeterminate'` → `string | boolean`
    match: (t) => norm(t) === "string | boolean",
  },

  // ── Tuples / callbacks ─────────────────────────────────────────
  // ⚠ HIGH FALSE-POSITIVE RATE: most `T[]` in real code are genuine
  // arrays, not degraded tuples. The validator + log-prob threshold
  // are the safety net here — the locator is intentionally permissive.
  "tuple": {
    name: "tuple→array",
    match: (t) => /^\s*\w+\[\]\s*$/.test(t),
  },
  "callback": {
    name: "callback→generic_callback",
    // degrade rule 16: `(specific args) => RET` → `(...args: any[]) => RET`
    // RET ∈ {void, boolean, string, number, unknown, any, never, undefined, null}
    match: (t) =>
      /^\(\s*\.\.\.args:\s*any\[\]\s*\)\s*=>\s*(void|boolean|string|number|unknown|any|never|undefined|null)\s*$/.test(t),
  },
  "callback_return": {
    name: "callback_return→unknown",
    // degrade rule 17: `() => SpecificReturn` → `() => unknown`
    match: (t) => /^\(\s*\)\s*=>\s*unknown\s*$/.test(t),
  },

  // ── React component props / element refs ──────────────────────
  "component_props_ref": {
    name: "component_props_ref→generic",
    match: (t) => t === "React.ComponentPropsWithRef<any>",
  },
  "component_props": {
    name: "component_props→generic",
    match: (t) => t === "React.ComponentPropsWithoutRef<any>",
  },
  "element_ref": {
    name: "element_ref→generic",
    match: (t) => t === "React.ElementRef<any>",
  },

  // ── DOM objects → object ──────────────────────────────────────
  "dom_object": {
    name: "dom_object→object",
    // ⚠ Bare `object` is rare in TS but ambiguous (could degrade
    // from any of DOMRect, DataTransfer, Selection, …). Validator
    // narrows to that specific allow-list.
    match: (t) => t.trim() === "object",
  },

  // ── unknown → ReturnType<…> | Extract|Exclude|Omit|Pick<…> ────
  // BOTH `returntype→unknown` (rule 22) and `utility_type→unknown`
  // (rule 23) degrade to literal `unknown`. We cannot tell them
  // apart from source text, so we emit a single candidate per
  // `unknown` site under the more-trained rule (`utility_type`,
  // 609 vs 246 pairs) and let the Python validator accept either
  // precise form.
  "utility_type": {
    name: "utility_type→unknown",
    match: (t) => t.trim() === "unknown",
  },

  // ── Generics over a specific type → generic over unknown ──────
  "record": {
    name: "record→generic",
    match: (t) => norm(t) === "Record<string, unknown>",
  },
  "promise": {
    name: "promise→generic",
    match: (t) => t === "Promise<unknown>",
  },
  "map": {
    name: "map→generic",
    match: (t) => norm(t) === "Map<unknown, unknown>",
  },
  "set": {
    name: "set→generic",
    match: (t) => t === "Set<unknown>",
  },
};

// Stable order — matches degrade.ts numbering for readability.
// "First match wins" semantics depend on this ordering.
const RULE_ORDER: readonly string[] = [
  "react_event", "react_handler",
  "html_element", "html_element_nullable", "svg_element", "dom_event",
  "ref_element", "mutable_ref_element", "ref_specific", "mutable_ref_specific",
  "string_literal_union", "numeric_literal_union", "boolean_literal", "mixed_literal_union",
  "tuple", "callback", "callback_return",
  "component_props_ref", "component_props", "element_ref",
  "dom_object", "utility_type",
  "record", "promise", "map", "set",
];

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

// First matching rule wins — same priority semantics as degrade.ts's
// sequential rule list. With the matcher set above most rules are
// mutually exclusive, but ordering still matters for the few overlaps
// (e.g. `(...args: any[]) => unknown` matches `callback` not
// `callback_return`).
function classify(typeText: string, ruleKeys: readonly string[]): CandidateRule | null {
  for (const key of ruleKeys) {
    const rule = RULES[key];
    if (rule.match(typeText)) return rule;
  }
  return null;
}

// ══════════════════════════════════════════════════════════════════════
// CLI
// ══════════════════════════════════════════════════════════════════════

function parseArgs() {
  const args = process.argv.slice(2);
  const paths: string[] = [];
  let contextRadius = 0;
  let output = "";
  let ruleName = "all"; // default: scan for all 24 trained rules

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
    console.error("usage: refiner-locate <path...> [--context N] [--output file] [--rule name|all]");
    console.error(`available rules: all, ${RULE_ORDER.join(", ")}`);
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
  ruleKeys: readonly string[],
): RefineCandidate[] {
  const repoName = path.basename(projectPath);
  const t0 = Date.now();

  const project = new Project({ skipAddingFilesFromTsConfig: true });

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
  const ruleLabel =
    ruleKeys.length === RULE_ORDER.length ? "all 24 rules" : ruleKeys.join(", ");
  console.log(`\n  ┌─ ${repoName}: scanning ${total} files for ${ruleLabel}`);

  const candidates: RefineCandidate[] = [];
  let filesProcessed = 0;
  let lastLog = Date.now();

  for (const sourceFile of sourceFiles) {
    const absFile = sourceFile.getFilePath();
    const sourceText = sourceFile.getFullText();

    const tryEmit = (
      typeNode: Node | undefined,
      kind: string,
      name: string,
      anchorLine: number,
    ) => {
      if (!typeNode) return;
      const text = typeNode.getText();
      const rule = classify(text, ruleKeys);
      if (!rule) return;
      const start = typeNode.getStart();
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

  // Resolve which rule keys to run with: "all" → every configured
  // rule; otherwise either a key from RULES or one of the trained
  // rule names (e.g. "string_literal_union→string") for convenience.
  let ruleKeys: string[];
  if (ruleName === "all") {
    ruleKeys = [...RULE_ORDER];
  } else if (RULES[ruleName]) {
    ruleKeys = [ruleName];
  } else {
    const found = RULE_ORDER.find((k) => RULES[k].name === ruleName);
    if (!found) {
      console.error(`Unknown rule: ${ruleName}`);
      console.error(`Available: all, ${RULE_ORDER.join(", ")}`);
      process.exit(1);
    }
    ruleKeys = [found];
  }

  const outputPath = output ? path.resolve(output) : "";
  if (outputPath) fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  const ruleLabel =
    ruleKeys.length === RULE_ORDER.length
      ? "all 24 rules"
      : ruleKeys.map((k) => RULES[k].name).join(", ");
  console.log(`\n${"═".repeat(60)}`);
  console.log(`REFINER LOCATOR — ${ruleLabel}, context ±${contextRadius} lines`);
  console.log(`  Inputs: ${paths.length} | Output: ${outputPath || "<stdout>"}`);
  console.log(`${"═".repeat(60)}`);

  const all: RefineCandidate[] = [];
  for (const p of paths) {
    const resolved = path.resolve(p);
    if (!fs.existsSync(resolved)) {
      console.error(`  SKIP: ${resolved} does not exist`);
      continue;
    }
    all.push(...locateInProject(resolved, contextRadius, ruleKeys));
  }

  // ── Summary ──────────────────────────────────────────────────────
  const byKind = new Map<string, number>();
  const byRule = new Map<string, number>();
  for (const c of all) {
    byKind.set(c.kind, (byKind.get(c.kind) || 0) + 1);
    byRule.set(c.rule, (byRule.get(c.rule) || 0) + 1);
  }
  console.log(`\n${"─".repeat(60)}`);
  console.log(`  Total candidates: ${all.length}`);
  console.log(`  By kind:`);
  for (const [kind, count] of [...byKind.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`    ${kind.padEnd(20)} ${count}`);
  }
  if (byRule.size > 1) {
    console.log(`  By rule:`);
    for (const [rule, count] of [...byRule.entries()].sort((a, b) => b[1] - a[1])) {
      console.log(`    ${rule.padEnd(40)} ${count}`);
    }
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
