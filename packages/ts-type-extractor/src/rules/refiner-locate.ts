/**
 * Refiner Locator
 *
 * Finds degraded type annotations in TS/TSX code and emits candidates for
 * ts-type-refiner.
 *
 * IMPORTANT: Some degraded shapes are inherently ambiguous (notably `unknown`)
 * because multiple degradation rules collapse into the same literal. To handle
 * this, locator emits MULTIPLE rule hypotheses for the same candidate span.
 * Inference then runs per-hypothesis validators and keeps the best accepted one.
 */

import { fileURLToPath } from "url";

import { Project, SyntaxKind, Node } from "ts-morph";
import * as path from "path";
import * as fs from "fs";
import { buildSiblings, getContainingDeclName, applyContainingBoost } from "../ts-data/siblings.js";

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
  siblings: string;
}

export interface CandidateRule {
  key: string;
  name: string;
  match: (typeText: string) => boolean;
}

const norm = (s: string): string => s.replace(/\s+/g, " ").trim();

// Keep low-support rules in code, but mute them in locator for now.
const MUTED_RULE_KEYS = new Set<string>([
  "dom_add_event_listener_options",
  "react_component_props_without_ref",
  "dom_intersection_observer_init",
  "dom_mutation_observer_init",
  "jsx_intrinsic_keyof",
  "astro_api_route",
  "react_element_ref",
  "tanstack_infinite_data",
  "astro_infer_get_static_props_type",
  "dom_element_internals_intersection",
  "astro_get_static_paths",
]);

// Rule numbering mirrors degrade.ts DEGRADATION_RULES and validators.py (ts-type-refiner) — keep in sync.
export const RULES: CandidateRule[] = [
  // 1
  { key: "react_event_handler", name: "react_event_handler→generic", match: (t) => t === "React.EventHandler<React.SyntheticEvent>" },
  // 1b
  { key: "react_specific_event_handler_alias", name: "react_specific_event_handler_alias→generic", match: (t) => t === "React.EventHandler<React.SyntheticEvent>" },
  // 2
  { key: "react_event", name: "react_event→synthetic", match: (t) => t === "React.SyntheticEvent" },
  // 3
  { key: "react_component_props_with_ref", name: "react_component_props_with_ref→any", match: (t) => t === "React.ComponentPropsWithRef<any>" },
  // 4
  { key: "react_component_props_without_ref", name: "react_component_props_without_ref→any", match: (t) => t === "React.ComponentPropsWithoutRef<any>" },
  // 5
  { key: "react_element_ref", name: "react_element_ref→any", match: (t) => t === "React.ElementRef<any>" },
  // 6
  { key: "react_refobject", name: "react_refobject→unknown", match: (t) => t === "React.RefObject<unknown>" },
  // 7
  { key: "react_mutable_refobject", name: "react_mutable_refobject→unknown", match: (t) => t === "React.MutableRefObject<unknown>" },
  // 8
  {
    key: "react_dispatch_setstateaction",
    name: "react_dispatch_setstateaction→unknown",
    match: (t) => t === "React.Dispatch<React.SetStateAction<unknown>>",
  },
  // 9
  { key: "jsx_intrinsic_keyof", name: "jsx_intrinsic_keyof→string", match: (t) => t === "string" },
  // 10
  { key: "string_literal_union", name: "string_literal_union→string", match: (t) => t === "string" },
  // 11
  { key: "template_literal_type", name: "template_literal_type→string", match: (t) => t === "string" },
  // 11b
  { key: "html_specific_element", name: "html_specific_element→html_element", match: (t) => t === "HTMLElement" },
  // 11c
  {
    key: "html_specific_element_nullable",
    name: "html_specific_element_nullable→html_element_nullable",
    match: (t) => norm(t) === "HTMLElement | null",
  },
  // 11d
  { key: "custom_event", name: "custom_event→unknown", match: (t) => t === "CustomEvent<unknown>" },
  // 11e
  { key: "record_string_value", name: "record_string_value→unknown", match: (t) => norm(t) === "Record<string, unknown>" },
  // 11f
  { key: "map", name: "map→unknown", match: (t) => norm(t) === "Map<unknown, unknown>" },
  // 11g
  { key: "set", name: "set→unknown", match: (t) => t === "Set<unknown>" },
  // 11h
  {
    key: "dom_add_event_listener_options",
    name: "dom_add_event_listener_options→event_listener_options",
    match: (t) => t === "EventListenerOptions",
  },

  // Ambiguous UNKNOWN rules: emit all hypotheses for `unknown`.
  // 12
  { key: "conditional_type", name: "conditional_type→unknown", match: (t) => t === "unknown" },
  // 13
  { key: "indexed_access_type", name: "indexed_access_type→unknown", match: (t) => t === "unknown" },
  // 14
  { key: "utility_type", name: "utility_type→unknown", match: (t) => t === "unknown" },
  // 14b
  { key: "dom_mutation_observer_init", name: "dom_mutation_observer_init→unknown", match: (t) => t === "unknown" },
  // 14c
  { key: "dom_intersection_observer_init", name: "dom_intersection_observer_init→unknown", match: (t) => t === "unknown" },
  // 14d
  { key: "dom_shadow_root_init", name: "dom_shadow_root_init→unknown", match: (t) => t === "unknown" },
  // 14e
  { key: "dom_css_style_declaration", name: "dom_css_style_declaration→unknown", match: (t) => t === "unknown" },
  // 14f
  {
    key: "dom_element_internals_intersection",
    name: "dom_element_internals_intersection→unknown",
    match: (t) => t === "unknown",
  },

  // 15
  { key: "promise", name: "promise→unknown", match: (t) => t === "Promise<unknown>" },
  // 16
  { key: "readonly_array", name: "readonly_array→unknown", match: (t) => t === "ReadonlyArray<unknown>" },
  // 17
  { key: "tanstack_use_query_result", name: "tanstack_use_query_result→unknown", match: (t) => norm(t) === "UseQueryResult<unknown, unknown>" },
  // 18
  {
    key: "tanstack_use_infinite_query_result",
    name: "tanstack_use_infinite_query_result→unknown",
    match: (t) => norm(t) === "UseInfiniteQueryResult<unknown, unknown>",
  },
  // 18b
  {
    key: "tanstack_query_observer_result",
    name: "tanstack_query_observer_result→unknown",
    match: (t) => norm(t) === "QueryObserverResult<unknown, unknown>",
  },
  // 18c
  {
    key: "tanstack_infinite_data",
    name: "tanstack_infinite_data→unknown",
    match: (t) => norm(t) === "InfiniteData<unknown, unknown>",
  },
  // 18cc
  {
    key: "tanstack_infinite_query_observer_result",
    name: "tanstack_infinite_query_observer_result→unknown",
    match: (t) => norm(t) === "InfiniteQueryObserverResult<unknown, unknown>",
  },
  // 18d
  {
    key: "tanstack_query_function_context",
    name: "tanstack_query_function_context→unknown",
    match: (t) => norm(t) === "QueryFunctionContext<unknown>",
  },
  // 18e
  {
    key: "astro_infer_get_static_props_type",
    name: "astro_infer_get_static_props_type→unknown",
    match: (t) => norm(t) === "InferGetStaticPropsType<unknown>",
  },
  // 18f
  {
    key: "astro_infer_get_static_paths_type",
    name: "astro_infer_get_static_paths_type→unknown",
    match: (t) => norm(t) === "InferGetStaticPathsType<unknown>",
  },
  // 18g
  { key: "astro_api_route", name: "astro_api_route→unknown", match: (t) => t === "unknown" },
  // 18h
  { key: "astro_get_static_paths", name: "astro_get_static_paths→unknown", match: (t) => t === "unknown" },
  // 19
  { key: "astro_collection_entry", name: "astro_collection_entry→any", match: (t) => t === "CollectionEntry<any>" },
];

const SKIP_FILE_PATTERNS = [
  /_generated/,
  /\.generated\./,
  /generated\//,
  /\.gen\.ts/,
  /__generated__/,
  /\.d\.ts$/,
];

function getContext(sourceText: string, line: number, radius: number): string {
  const lines = sourceText.split("\n");
  const start = Math.max(0, line - 1 - radius);
  const end = Math.min(lines.length, line - 1 + radius + 1);
  return lines.slice(start, end).join("\n");
}


function parseArgs() {
  const args = process.argv.slice(2);
  const paths: string[] = [];
  let contextRadius = 0;
  let output = "";
  let ruleName = "all";

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
    console.error(`available rules: all, ${RULES.map((r) => r.key).join(", ")}`);
    process.exit(1);
  }

  return { paths, contextRadius, output, ruleName };
}

function classifyAll(typeText: string, activeRules: readonly CandidateRule[]): CandidateRule[] {
  const out: CandidateRule[] = [];
  for (const rule of activeRules) {
    if (rule.match(typeText)) out.push(rule);
  }
  return out;
}

function locateInProject(
  projectPath: string,
  contextRadius: number,
  activeRules: readonly CandidateRule[],
): RefineCandidate[] {
  const repoName = path.basename(projectPath);
  const t0 = Date.now();

  const project = new Project({ skipAddingFilesFromTsConfig: true });
  const stat = fs.statSync(projectPath);
  // When invoked on a single file, use its parent dir as the relative-path
  // base so emitted `file` fields are non-empty (apply needs a real path).
  const baseDir = stat.isFile() ? path.dirname(projectPath) : projectPath;
  if (stat.isFile()) {
    project.addSourceFileAtPath(projectPath);
  } else {
    project.addSourceFilesAtPaths(path.join(projectPath, "**/*.{ts,tsx}"));
  }

  const allFiles = project.getSourceFiles();
  const sourceFiles = allFiles.filter((sf) => {
    const rel = path.relative(baseDir, sf.getFilePath());
    return !SKIP_FILE_PATTERNS.some((pat) => pat.test(rel));
  });

  console.log(`\n  ┌─ ${repoName}: scanning ${sourceFiles.length} files`);

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
      const text = norm(typeNode.getText());
      const matches = classifyAll(text, activeRules);
      if (matches.length === 0) return;

      const start = typeNode.getStart();
      const end = typeNode.getEnd();
      const id = `${absFile}:${start}`;
      const file = path.relative(baseDir, absFile);
      const context = getContext(sourceText, anchorLine, contextRadius);
      // typeNode owns the type; its parent is the declaration that holds it.
      const decl = typeNode.getParent() ?? typeNode;
      const baseSiblings = buildSiblings(decl, kind);
      const containingDecl = getContainingDeclName(decl);

      for (const rule of matches) {
        const siblings = applyContainingBoost(baseSiblings, containingDecl, rule.name);
        candidates.push({
          id,
          file,
          line: anchorLine,
          start,
          end,
          kind,
          name,
          context,
          degradedType: text,
          rule: rule.name,
          siblings,
        });
      }
    };

    sourceFile.forEachDescendant((node) => {
      if (Node.isParameterDeclaration(node)) {
        tryEmit(node.getTypeNode(), "parameter", node.getName(), node.getStartLineNumber());
      }

      if (Node.isVariableDeclaration(node)) {
        tryEmit(node.getTypeNode(), "variable", node.getName(), node.getStartLineNumber());
      }

      if (Node.isPropertySignature(node) || Node.isPropertyDeclaration(node)) {
        tryEmit(node.getTypeNode(), "property", node.getName(), node.getStartLineNumber());
      }

      if (Node.isFunctionDeclaration(node) || Node.isMethodDeclaration(node) || Node.isArrowFunction(node)) {
        const name =
          Node.isFunctionDeclaration(node) || Node.isMethodDeclaration(node)
            ? (node.getName() ?? "<anonymous>")
            : "<arrow>";
        tryEmit(node.getReturnTypeNode(), "return_type", name, node.getStartLineNumber());
      }

      if (Node.isAsExpression(node)) {
        tryEmit(node.getTypeNode(), "type_assertion", "<as_expression>", node.getStartLineNumber());
      }

      if (Node.isCallExpression(node)) {
        const fn = node.getExpression().getText();
        for (const typeArg of node.getTypeArguments()) {
          tryEmit(typeArg, "generic_argument", fn, node.getStartLineNumber());
        }
      }

      if (node.getKind() === SyntaxKind.TypeAliasDeclaration) {
        const alias = node.asKind(SyntaxKind.TypeAliasDeclaration);
        if (alias) {
          tryEmit(alias.getTypeNode(), "type_alias", alias.getName(), alias.getStartLineNumber());
        }
      }
    });

    filesProcessed++;
    const now = Date.now();
    if (now - lastLog >= 2000 || filesProcessed === sourceFiles.length) {
      const pct = ((filesProcessed / sourceFiles.length) * 100).toFixed(0);
      const elapsed = ((now - t0) / 1000).toFixed(1);
      process.stdout.write(
        `\r  │  ${filesProcessed}/${sourceFiles.length} files (${pct}%) — ${candidates.length} candidates — ${elapsed}s`,
      );
      lastLog = now;
    }
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`\n  └─ ${repoName}: ${candidates.length} candidates in ${elapsed}s`);
  return candidates;
}

function main() {
  const { paths, contextRadius, output, ruleName } = parseArgs();
  const enabledRules = RULES.filter((r) => !MUTED_RULE_KEYS.has(r.key));

  let activeRules: CandidateRule[];
  if (ruleName === "all") {
    activeRules = enabledRules;
  } else {
    activeRules = enabledRules.filter((r) => r.key === ruleName || r.name === ruleName);
    if (activeRules.length === 0) {
      if (MUTED_RULE_KEYS.has(ruleName)) {
        console.error(`rule '${ruleName}' is currently muted`);
        process.exit(1);
      }
      console.error(`unknown rule '${ruleName}'`);
      process.exit(1);
    }
  }

  const outPath = path.resolve(output || "candidates.jsonl");
  fs.mkdirSync(path.dirname(outPath), { recursive: true });

  console.log(`\n${"═".repeat(60)}`);
  console.log(`REFINER LOCATOR — context ±${contextRadius} lines`);
  console.log(`  Repos: ${paths.length} | Rules: ${ruleName} | Output: ${outPath}`);
  console.log(`${"═".repeat(60)}`);

  const all: RefineCandidate[] = [];
  const seen = new Set<string>();

  for (const p of paths) {
    const resolved = path.resolve(p);
    if (!fs.existsSync(resolved)) {
      console.error(`  SKIP: ${resolved} does not exist`);
      continue;
    }

    const repoCandidates = locateInProject(resolved, contextRadius, activeRules);

    for (const c of repoCandidates) {
      const key = `${c.id}|${c.rule}`;
      if (seen.has(key)) continue;
      seen.add(key);
      all.push(c);
    }

    const lines = all.map((c) => JSON.stringify(c));
    fs.writeFileSync(outPath, lines.join("\n") + "\n", "utf-8");
    console.log(`  ✓ Saved: ${all.length} total candidates`);
  }

  const byRule = new Map<string, number>();
  for (const c of all) {
    byRule.set(c.rule, (byRule.get(c.rule) || 0) + 1);
  }

  console.log(`\n${"─".repeat(60)}`);
  console.log(`  Total: ${all.length} candidates`);
  for (const [rule, count] of [...byRule.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`    ${rule.padEnd(55)} ${count}`);
  }
  console.log(`\n  Output: ${outPath}`);
}

// Only run when invoked directly (not when imported, e.g. by `negatives.ts`).
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main();
}
