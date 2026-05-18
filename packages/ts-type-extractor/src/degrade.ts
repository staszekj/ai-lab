/**
 * Type Degradation — Generate Training Pairs
 *
 * New ruleset derived from type-definition repositories:
 * - TypeScript lib.dom / native browser
 * - DefinitelyTyped React
 * - Astro
 * - TanStack Query/Router
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";
import { applyContainingBoost } from "./siblings.js";
import { collectNegatives } from "./negatives.js";

interface TypeAnnotation {
  repo?: string;
  file: string;
  line: number;
  kind: string;
  name: string;
  typeText: string;
  context: string;
  /** Offsets of `typeText` within `context` (Step 1.3, optional for back-compat). */
  typeStart?: number;
  typeEnd?: number;
  siblings?: string;
  containingDecl?: string | null;
}

/**
 * Enriched encoder/decoder training pair.
 *
 * - `input` / `target` / `rule` — consumed by the model (see `dataset.py`).
 * - `repo` / `file` / `line` / `kind` / `name` / `degradedType` / `preciseType` /
 *   `siblings` — diagnostics consumed by `rule-coverage-report.ts`
 *   and `repo-contribution-report.ts`. Ignored by the model.
 *
 * Future fields planned by the data-quality plan:
 */
interface TrainingPair {
  input: string;
  target: string;
  rule: string;

  repo?: string;
  file: string;
  line: number;
  kind: string;
  name: string;
  degradedType: string;
  preciseType: string;
  siblings?: string;
  /** Step 1.4: when true, this pair teaches the model to preserve an
   * already-precise type (degraded === target === typeText). */
  isNegative?: boolean;
  /** Step 1.5: deterministic content-hash split ("train" | "val"). */
  split?: "train" | "val";
}

/**
 * Build encoder input prompt. Mirrors `build_refine_prompt` in
 * `ts_type_refiner/prompt.py` exactly — both implementations MUST stay in sync.
 *
 * Wire format v2 (see `prompt.py::PROMPT_VERSION`):
 *   `[REFINE rule=... | kind=... | name=... | degraded=... | siblings=...]\n---\n<code>`
 */
export const PROMPT_VERSION = 2;

function buildRefinePrompt(args: {
  context: string;
  name: string;
  kind: string;
  rule: string;
  degradedType: string;
  siblings?: string;
}): string {
  const meta = [
    `rule=${args.rule}`,
    `kind=${args.kind}`,
    `name=${args.name}`,
    `degraded=${args.degradedType}`,
  ];
  if (args.siblings) meta.push(`siblings=${args.siblings}`);

  return "[REFINE " + meta.join(" | ") + "]\n---\n" + args.context;
}

type DegradationResult = { degraded: string; rule: string } | null;
type DegradationRule = (typeText: string) => DegradationResult;

// Keep low-support rules in code, but mute them for current training pair generation.
const MUTED_RULES = new Set<string>([
  "dom_add_event_listener_options→event_listener_options",
  "react_component_props_without_ref→any",
  "dom_intersection_observer_init→unknown",
  "dom_mutation_observer_init→unknown",
  "jsx_intrinsic_keyof→string",
  "astro_api_route→unknown",
  "react_element_ref→any",
  "tanstack_infinite_data→unknown",
  "astro_infer_get_static_props_type→unknown",
  "dom_element_internals_intersection→unknown",
  "astro_get_static_paths→unknown",
]);

const SIMPLE_TYPES = new Set([
  "void", "boolean", "string", "number",
  "unknown", "any", "never", "undefined", "null",
]);

const norm = (s: string): string => s.replace(/\s+/g, " ").trim();

function splitUnion(t: string): string[] {
  return t.split(/\s*\|\s*/).map((p) => p.trim()).filter(Boolean);
}

function isStringLiteral(tok: string): boolean {
  return /^'[^']*'$/.test(tok) || /^"[^"]*"$/.test(tok);
}

const DEGRADATION_RULES: DegradationRule[] = [
  // React ecosystem
  (t) => {
    if (/^React\.\w+EventHandler(?:<.+>)?$/.test(t) && t !== "React.EventHandler<React.SyntheticEvent>") {
      return { degraded: "React.EventHandler<React.SyntheticEvent>", rule: "react_event_handler→generic" };
    }
    return null;
  },
  (t) => {
    if (/^(Mouse|Keyboard|Pointer|Touch|Drag|Focus|Change|Clipboard|Composition|Animation|Transition|Form|Wheel)EventHandler(?:<.+>)?$/.test(t)) {
      return { degraded: "React.EventHandler<React.SyntheticEvent>", rule: "react_specific_event_handler_alias→generic" };
    }
    return null;
  },
  (t) => {
    if (/^React\.\w+Event(?:<.+>)?$/.test(t) && t !== "React.SyntheticEvent") {
      return { degraded: "React.SyntheticEvent", rule: "react_event→synthetic" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^React\.ComponentPropsWithRef<(.+)>$/);
    if (m && m[1].trim() !== "any") {
      return { degraded: "React.ComponentPropsWithRef<any>", rule: "react_component_props_with_ref→any" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^React\.ComponentPropsWithoutRef<(.+)>$/);
    if (m && m[1].trim() !== "any") {
      return { degraded: "React.ComponentPropsWithoutRef<any>", rule: "react_component_props_without_ref→any" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^React\.ElementRef<(.+)>$/);
    if (m && m[1].trim() !== "any") {
      return { degraded: "React.ElementRef<any>", rule: "react_element_ref→any" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^React\.RefObject<(.+)>$/);
    if (m && !["unknown", "any"].includes(m[1].trim())) {
      return { degraded: "React.RefObject<unknown>", rule: "react_refobject→unknown" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^React\.MutableRefObject<(.+)>$/);
    if (m && !["unknown", "any"].includes(m[1].trim())) {
      return { degraded: "React.MutableRefObject<unknown>", rule: "react_mutable_refobject→unknown" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^React\.Dispatch<\s*React\.SetStateAction<(.+)>\s*>$/);
    if (m && !["unknown", "any"].includes(m[1].trim())) {
      return {
        degraded: "React.Dispatch<React.SetStateAction<unknown>>",
        rule: "react_dispatch_setstateaction→unknown",
      };
    }
    return null;
  },
  (t) => {
    if (norm(t) === "keyof JSX.IntrinsicElements") {
      return { degraded: "string", rule: "jsx_intrinsic_keyof→string" };
    }
    return null;
  },

  // Literal / template families
  (t) => {
    const parts = splitUnion(t);
    if (parts.length >= 2 && parts.every(isStringLiteral)) {
      return { degraded: "string", rule: "string_literal_union→string" };
    }
    return null;
  },
  (t) => {
    if (t.includes("`") && t !== "string") {
      return { degraded: "string", rule: "template_literal_type→string" };
    }
    return null;
  },

  // Native browser + custom elements
  (t) => {
    if (/^HTML\w+Element$/.test(t) && t !== "HTMLElement") {
      return { degraded: "HTMLElement", rule: "html_specific_element→html_element" };
    }
    return null;
  },
  (t) => {
    if (/^HTML\w+Element\s*\|\s*null$/.test(t) && t !== "HTMLElement | null") {
      return { degraded: "HTMLElement | null", rule: "html_specific_element_nullable→html_element_nullable" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^CustomEvent<(.+)>$/);
    if (m && !["unknown", "any"].includes(m[1].trim())) {
      return { degraded: "CustomEvent<unknown>", rule: "custom_event→unknown" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^Record<\s*string\s*,\s*(.+)\s*>$/);
    if (m && !["unknown", "any"].includes(m[1].trim())) {
      return { degraded: "Record<string, unknown>", rule: "record_string_value→unknown" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^Map<\s*(.+?)\s*,\s*(.+?)\s*>$/);
    if (m && !(m[1].trim() === "unknown" && m[2].trim() === "unknown")) {
      return { degraded: "Map<unknown, unknown>", rule: "map→unknown" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^Set<\s*(.+?)\s*>$/);
    if (m && !["unknown", "any"].includes(m[1].trim())) {
      return { degraded: "Set<unknown>", rule: "set→unknown" };
    }
    return null;
  },
  (t) => {
    if (t === "AddEventListenerOptions") {
      return { degraded: "EventListenerOptions", rule: "dom_add_event_listener_options→event_listener_options" };
    }
    return null;
  },

  // Ambiguous UNKNOWN families (locator will emit multi-hypothesis)
  (t) => {
    if (/\bextends\b/.test(t) && /\?/.test(t) && /:/.test(t)) {
      return { degraded: "unknown", rule: "conditional_type→unknown" };
    }
    return null;
  },
  (t) => {
    if (/^[A-Za-z0-9_$.<>,\s]+\[[^\]]+\]$/.test(t) && !t.endsWith("[]")) {
      return { degraded: "unknown", rule: "indexed_access_type→unknown" };
    }
    return null;
  },
  (t) => {
    if (/^(Extract|Exclude|Pick|Omit|Partial|Required|Readonly|NonNullable|Parameters|ReturnType|InstanceType|Awaited)</.test(t)) {
      return { degraded: "unknown", rule: "utility_type→unknown" };
    }
    return null;
  },
  (t) => {
    if (t === "MutationObserverInit") {
      return { degraded: "unknown", rule: "dom_mutation_observer_init→unknown" };
    }
    return null;
  },
  (t) => {
    if (t === "IntersectionObserverInit") {
      return { degraded: "unknown", rule: "dom_intersection_observer_init→unknown" };
    }
    return null;
  },
  (t) => {
    if (t === "ShadowRootInit") {
      return { degraded: "unknown", rule: "dom_shadow_root_init→unknown" };
    }
    return null;
  },
  (t) => {
    if (t === "CSSStyleDeclaration") {
      return { degraded: "unknown", rule: "dom_css_style_declaration→unknown" };
    }
    return null;
  },
  (t) => {
    if (/^ElementInternals\s*&\s*.+$/.test(t)) {
      return { degraded: "unknown", rule: "dom_element_internals_intersection→unknown" };
    }
    return null;
  },

  // Generic wrappers from type-defs
  (t) => {
    const m = t.match(/^Promise<(.+)>$/);
    if (m && !SIMPLE_TYPES.has(m[1].trim())) {
      return { degraded: "Promise<unknown>", rule: "promise→unknown" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^ReadonlyArray<(.+)>$/);
    if (m && !["unknown", "any"].includes(m[1].trim())) {
      return { degraded: "ReadonlyArray<unknown>", rule: "readonly_array→unknown" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^UseQueryResult<\s*(.+?)\s*,\s*(.+?)\s*>$/);
    if (m && !(m[1].trim() === "unknown" && m[2].trim() === "unknown")) {
      return { degraded: "UseQueryResult<unknown, unknown>", rule: "tanstack_use_query_result→unknown" };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^UseInfiniteQueryResult<\s*(.+?)\s*,\s*(.+?)\s*>$/);
    if (m && !(m[1].trim() === "unknown" && m[2].trim() === "unknown")) {
      return {
        degraded: "UseInfiniteQueryResult<unknown, unknown>",
        rule: "tanstack_use_infinite_query_result→unknown",
      };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^QueryObserverResult<\s*(.+?)\s*,\s*(.+?)\s*>$/);
    if (m && !(m[1].trim() === "unknown" && m[2].trim() === "unknown")) {
      return {
        degraded: "QueryObserverResult<unknown, unknown>",
        rule: "tanstack_query_observer_result→unknown",
      };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^InfiniteData<\s*(.+?)(?:\s*,\s*(.+?)\s*)?>$/);
    if (!m) return null;
    const a = m[1].trim();
    const b = m[2]?.trim();
    if (a === "unknown" && (!b || b === "unknown")) return null;
    return {
      degraded: "InfiniteData<unknown, unknown>",
      rule: "tanstack_infinite_data→unknown",
    };
  },
  (t) => {
    const m = t.match(/^InfiniteQueryObserverResult<\s*(.+?)\s*,\s*(.+?)\s*>$/);
    if (m && !(m[1].trim() === "unknown" && m[2].trim() === "unknown")) {
      return {
        degraded: "InfiniteQueryObserverResult<unknown, unknown>",
        rule: "tanstack_infinite_query_observer_result→unknown",
      };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^QueryFunctionContext<\s*(.+)\s*>$/);
    if (m && m[1].trim() !== "unknown") {
      return {
        degraded: "QueryFunctionContext<unknown>",
        rule: "tanstack_query_function_context→unknown",
      };
    }
    return null;
  },

  // Astro
  (t) => {
    const m = t.match(/^InferGetStaticPropsType<(.+)>$/);
    if (m && m[1].trim() !== "unknown") {
      return {
        degraded: "InferGetStaticPropsType<unknown>",
        rule: "astro_infer_get_static_props_type→unknown",
      };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^InferGetStaticPathsType<(.+)>$/);
    if (m && m[1].trim() !== "unknown") {
      return {
        degraded: "InferGetStaticPathsType<unknown>",
        rule: "astro_infer_get_static_paths_type→unknown",
      };
    }
    return null;
  },
  (t) => {
    if (t === "APIRoute") {
      return {
        degraded: "unknown",
        rule: "astro_api_route→unknown",
      };
    }
    return null;
  },
  (t) => {
    if (t === "GetStaticPaths") {
      return {
        degraded: "unknown",
        rule: "astro_get_static_paths→unknown",
      };
    }
    return null;
  },
  (t) => {
    const m = t.match(/^CollectionEntry<(.+)>$/);
    if (m && m[1].trim() !== "any") {
      return { degraded: "CollectionEntry<any>", rule: "astro_collection_entry→any" };
    }
    return null;
  },
];

function degradeType(typeText: string): DegradationResult {
  const t = norm(typeText);
  for (const rule of DEGRADATION_RULES) {
    const result = rule(t);
    if (!result) continue;
    if (MUTED_RULES.has(result.rule)) continue;
    return result;
  }
  return null;
}

function parseArgs() {
  const args = process.argv.slice(2);
  let input = "";
  let output = "";
  let negativeRatio = 0.25;
  let valPct = 15;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--output" && args[i + 1]) {
      output = args[i + 1];
      i++;
    } else if (args[i] === "--negative-ratio" && args[i + 1]) {
      const n = parseFloat(args[i + 1]);
      if (!Number.isFinite(n) || n < 0) {
        console.error(`Error: invalid --negative-ratio: ${args[i + 1]}`);
        process.exit(1);
      }
      negativeRatio = n;
      i++;
    } else if (args[i] === "--val-pct" && args[i + 1]) {
      const n = parseInt(args[i + 1], 10);
      if (!Number.isFinite(n) || n < 0 || n > 100) {
        console.error(`Error: invalid --val-pct: ${args[i + 1]}`);
        process.exit(1);
      }
      valPct = n;
      i++;
    } else if (!input) {
      input = args[i];
    }
  }

  return { input, output, negativeRatio, valPct };
}

function main() {
  const { input, output, negativeRatio, valPct } = parseArgs();
  const inputPath = path.resolve(input || "extracted_types.jsonl");

  if (!fs.existsSync(inputPath)) {
    console.error(`Error: file not found: ${inputPath}`);
    process.exit(1);
  }

  console.log(`\n${"═".repeat(60)}`);
  console.log(`TYPE DEGRADATION — generating training pairs`);
  console.log(`${"═".repeat(60)}`);
  console.log(`  Input: ${inputPath}\n`);

  const lines = fs.readFileSync(inputPath, "utf-8").trim().split("\n");
  const annotations: TypeAnnotation[] = lines.map((l: string) => JSON.parse(l));

  console.log(`  Total annotations: ${annotations.length}`);

  const trainingPairs: TrainingPair[] = [];
  const skippedTypes = new Map<string, number>();
  const appliedRules = new Map<string, number>();

  const t0 = Date.now();
  let lastLog = t0;
  const total = annotations.length;

  for (let i = 0; i < total; i++) {
    const ann = annotations[i];
    const result = degradeType(ann.typeText);
    if (result) {
      // AST-targeted single-span replace (Step 1.3): use the offsets emitted
      // by `extract.ts` so we never accidentally clobber a peer occurrence of
      // the same type text on the same line (e.g. `Record<string, string>`).
      // Falls back to split/join only if offsets are missing or invalid
      // (legacy `extracted_*.jsonl` files).
      let modifiedContext: string;
      const hasOffsets =
        typeof ann.typeStart === "number" &&
        typeof ann.typeEnd === "number" &&
        ann.context.slice(ann.typeStart, ann.typeEnd) === ann.typeText;
      if (hasOffsets) {
        modifiedContext =
          ann.context.slice(0, ann.typeStart!) +
          result.degraded +
          ann.context.slice(ann.typeEnd!);
      } else {
        modifiedContext = ann.context.split(ann.typeText).join(result.degraded);
      }
      const siblings = applyContainingBoost(
        ann.siblings ?? "",
        ann.containingDecl ?? null,
        result.rule,
      );
      const input = buildRefinePrompt({
        context: modifiedContext,
        name: ann.name,
        kind: ann.kind,
        rule: result.rule,
        degradedType: result.degraded,
        siblings,
      });

      trainingPairs.push({
        input,
        target: ann.typeText,
        rule: result.rule,
        repo: ann.repo,
        file: ann.file,
        line: ann.line,
        kind: ann.kind,
        name: ann.name,
        degradedType: result.degraded,
        preciseType: ann.typeText,
        siblings,
      });
      appliedRules.set(result.rule, (appliedRules.get(result.rule) || 0) + 1);
    } else {
      skippedTypes.set(ann.typeText, (skippedTypes.get(ann.typeText) || 0) + 1);
    }

    const done = i + 1;
    const now = Date.now();
    if (now - lastLog >= 2000 || done % 10_000 === 0 || done === total) {
      const pct = ((done / total) * 100).toFixed(1);
      const elapsed = (now - t0) / 1000;
      const rate = done / Math.max(elapsed, 1e-6);
      const etaSec = (total - done) / Math.max(rate, 1e-6);
      process.stdout.write(
        `\r  ${done}/${total} (${pct}%)  pairs=${trainingPairs.length}  ${rate.toFixed(0)}/s  ETA ${etaSec.toFixed(1)}s   `,
      );
      lastLog = now;
    }
  }
  process.stdout.write("\n");

  const positiveCount = trainingPairs.length;

  // ── Step 1.4: per-rule hard negatives ─────────────────────────────────
  // For every active rule (positive count > 0), grab annotations whose
  // `typeText` already matches the rule's degraded shape and emit no-op
  // pairs `(degraded=typeText, target=typeText, isNegative=true)`. Teaches
  // the model to preserve types that are already correct.
  const negatives = negativeRatio > 0
    ? collectNegatives(annotations, appliedRules, negativeRatio)
    : [];
  const negativeRules = new Map<string, number>();
  for (const { ann, rule } of negatives) {
    const siblings = applyContainingBoost(
      ann.siblings ?? "",
      ann.containingDecl ?? null,
      rule,
    );
    const promptInput = buildRefinePrompt({
      context: ann.context,
      name: ann.name,
      kind: ann.kind,
      rule,
      degradedType: ann.typeText,
      siblings,
    });
    trainingPairs.push({
      input: promptInput,
      target: ann.typeText,
      rule,
      repo: ann.repo,
      file: ann.file,
      line: ann.line,
      kind: ann.kind,
      name: ann.name,
      degradedType: ann.typeText,
      preciseType: ann.typeText,
      siblings,
      isNegative: true,
    });
    negativeRules.set(rule, (negativeRules.get(rule) || 0) + 1);
  }

  console.log(`\n${"─".repeat(60)}`);
  console.log(`DEGRADATION SUMMARY`);
  console.log(`${"─".repeat(60)}`);
  console.log(`  Positive pairs:           ${positiveCount}`);
  console.log(`  Negative pairs:           ${negatives.length}  (ratio=${negativeRatio})`);
  console.log(`  Total training pairs:     ${trainingPairs.length}`);
  console.log(`  Annotations skipped:      ${annotations.length - positiveCount}`);
  console.log(`  Conversion rate:          ${((positiveCount / annotations.length) * 100).toFixed(1)}%`);

  console.log(`\n  Rules applied (positives + negatives):`);
  const allRules = new Set<string>([...appliedRules.keys(), ...negativeRules.keys()]);
  const ruleRows = [...allRules].map((r) => ({
    rule: r,
    pos: appliedRules.get(r) || 0,
    neg: negativeRules.get(r) || 0,
  })).sort((a, b) => (b.pos + b.neg) - (a.pos + a.neg));
  for (const row of ruleRows) {
    console.log(`    ${row.rule.padEnd(55)} pos=${String(row.pos).padStart(5)}  neg=${String(row.neg).padStart(5)}`);
  }

  console.log(`\n  Top 30 skipped types:`);
  const sortedSkipped = [...skippedTypes.entries()].sort((a, b) => b[1] - a[1]);
  for (const [type, count] of sortedSkipped.slice(0, 30)) {
    console.log(`    ${count.toString().padStart(4)}  ${type}`);
  }

  // ── Step 1.5: dedup + content-hash split ─────────────────────────────
  // Dedup by SHA1(input + "\t" + target) so identical encoder/decoder pairs
  // (very common in usage repos with copy-pasted props) don't dominate.
  // Then assign `split` per-input via SHA1(input) % 100, guaranteeing that
  // duplicate prompts can never straddle the train/val boundary.
  const sha1 = (s: string) => crypto.createHash("sha1").update(s).digest("hex");

  const seen = new Set<string>();
  const deduped: TrainingPair[] = [];
  let dupCount = 0;
  for (const p of trainingPairs) {
    const key = sha1(p.input + "\t" + p.target);
    if (seen.has(key)) {
      dupCount++;
      continue;
    }
    seen.add(key);
    deduped.push(p);
  }

  const splitCounts = { train: 0, val: 0 };
  const splitByRule = new Map<string, { train: number; val: number }>();
  for (const p of deduped) {
    // First 8 hex chars → 32-bit unsigned, mod 100.
    const bucket = parseInt(sha1(p.input).slice(0, 8), 16) % 100;
    const split: "train" | "val" = bucket < valPct ? "val" : "train";
    p.split = split;
    splitCounts[split]++;
    const row = splitByRule.get(p.rule) ?? { train: 0, val: 0 };
    row[split]++;
    splitByRule.set(p.rule, row);
  }

  console.log(`\n${"─".repeat(60)}`);
  console.log(`DEDUP + SPLIT`);
  console.log(`${"─".repeat(60)}`);
  console.log(`  Before dedup:  ${trainingPairs.length}`);
  console.log(`  Duplicates:    ${dupCount}`);
  console.log(`  After dedup:   ${deduped.length}`);
  console.log(`  Split (val-pct=${valPct}):  train=${splitCounts.train}  val=${splitCounts.val}  (${(splitCounts.val / deduped.length * 100).toFixed(1)}% val)`);
  console.log(`\n  Per-rule split:`);
  const splitRows = [...splitByRule.entries()].sort((a, b) => (b[1].train + b[1].val) - (a[1].train + a[1].val));
  for (const [rule, c] of splitRows) {
    console.log(`    ${rule.padEnd(55)} train=${String(c.train).padStart(5)}  val=${String(c.val).padStart(4)}`);
  }

  const outputPath = output
    ? path.resolve(output)
    : path.join(path.dirname(inputPath), "encoder_decoder_pairs.jsonl");
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const outputLines = deduped.map((p) => JSON.stringify(p));
  fs.writeFileSync(outputPath, outputLines.join("\n") + "\n", "utf-8");

  console.log(`\n  Output: ${outputPath}`);
  console.log(`  ${deduped.length} training pairs ready.`);
  console.log(`${"═".repeat(60)}\n`);
}

main();
