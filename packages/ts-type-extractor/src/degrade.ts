/**
 * Type Degradation — Generate Training Pairs
 *
 * Takes extracted annotations and creates training pairs by degrading
 * precise types to generic ones.  The ML model learns to reverse this.
 *
 * CRITICAL: The context is modified so the precise type is REPLACED
 * with the degraded type.  This prevents the model from "cheating"
 * by copying the answer from the surrounding code.
 *
 * Usage:
 *   npx tsx src/degrade.ts <extracted.jsonl> [--output pairs.jsonl]
 */

import * as fs from "fs";
import * as path from "path";

// ══════════════════════════════════════════════════════════════════════
// Types
// ══════════════════════════════════════════════════════════════════════

interface TypeAnnotation {
  file: string;
  line: number;
  kind: string;
  name: string;
  typeText: string;
  context: string;
  parentName: string | null;
}

interface TrainingPair {
  context: string;
  name: string;
  kind: string;
  degradedType: string;
  preciseType: string;
  rule: string;
  file: string;
  line: number;
}

// ══════════════════════════════════════════════════════════════════════
// Degradation rules
// ══════════════════════════════════════════════════════════════════════

type DegradationResult = { degraded: string; rule: string } | null;
type DegradationRule = (typeText: string) => DegradationResult;

/**
 * Match a callback/function type using balanced parentheses.
 * Returns params string and return type, or null.
 *
 *   "(event: React.MouseEvent<HTMLButtonElement>) => void"
 *    → { params: "event: React.MouseEvent<HTMLButtonElement>", returnType: "void" }
 */
function parseCallbackType(t: string): { params: string; returnType: string } | null {
  if (!t.startsWith("(")) return null;
  let depth = 0;
  let i: number;
  for (i = 0; i < t.length; i++) {
    if (t[i] === "(") depth++;
    if (t[i] === ")") depth--;
    if (depth === 0) break;
  }
  if (depth !== 0) return null;
  const params = t.slice(1, i);
  const rest = t.slice(i + 1).trim();
  if (!rest.startsWith("=>")) return null;
  const returnType = rest.slice(2).trim();
  if (!returnType) return null;
  return { params, returnType };
}

const SIMPLE_TYPES = new Set([
  "void", "boolean", "string", "number",
  "unknown", "any", "never", "undefined", "null",
]);

const DEGRADATION_RULES: DegradationRule[] = [
  // ── 1. React event types → React.SyntheticEvent ─────────────────
  (t) => {
    if (/^React\.\w+Event(<[^>]+>)?$/.test(t) && !t.startsWith("React.SyntheticEvent")) {
      return { degraded: "React.SyntheticEvent", rule: "react_event→synthetic" };
    }
    return null;
  },

  // ── 2. React event handler types ────────────────────────────────
  (t) => {
    if (/^React\.\w+EventHandler(<[^>]+>)?$/.test(t) && !t.includes("EventHandler<React.SyntheticEvent>")) {
      return { degraded: "React.EventHandler<React.SyntheticEvent>", rule: "react_handler→generic_handler" };
    }
    return null;
  },

  // ── 3. HTML*Element → HTMLElement ───────────────────────────────
  (t) => {
    const m = t.match(/^HTML(\w+)Element$/);
    if (m && m[1] !== "") {
      return { degraded: "HTMLElement", rule: "html_element→generic" };
    }
    return null;
  },

  // ── 4. HTML*Element | null → HTMLElement | null ─────────────────
  (t) => {
    const m = t.match(/^HTML(\w+)Element\s*\|\s*null$/);
    if (m && m[1] !== "") {
      return { degraded: "HTMLElement | null", rule: "html_element_nullable→generic" };
    }
    return null;
  },

  // ── 5. SVG*Element → SVGElement ─────────────────────────────────
  (t) => {
    const m = t.match(/^SVG(\w+)Element$/);
    if (m && m[1] !== "") {
      return { degraded: "SVGElement", rule: "svg_element→generic" };
    }
    return null;
  },

  // ── 6. DOM native events → Event ───────────────────────────────
  (t) => {
    const domEvents = new Set([
      "PointerEvent", "KeyboardEvent", "MouseEvent", "FocusEvent",
      "TouchEvent", "WheelEvent", "DragEvent", "InputEvent",
      "CompositionEvent", "ClipboardEvent", "AnimationEvent",
      "TransitionEvent", "UIEvent", "CustomEvent",
    ]);
    if (domEvents.has(t)) {
      return { degraded: "Event", rule: "dom_event→generic" };
    }
    return null;
  },

  // ── 7. React.RefObject<SpecificElement> → React.RefObject<HTMLElement>
  (t) => {
    const m = t.match(/^React\.RefObject<HTML(\w+)Element>$/);
    if (m && m[1] !== "") {
      return { degraded: "React.RefObject<HTMLElement>", rule: "ref_element→generic" };
    }
    return null;
  },

  // ── 8. React.MutableRefObject<HTML*Element> → React.MutableRefObject<HTMLElement>
  (t) => {
    const m = t.match(/^React\.MutableRefObject<HTML(\w+)Element>$/);
    if (m && m[1] !== "") {
      return { degraded: "React.MutableRefObject<HTMLElement>", rule: "mutable_ref_element→generic" };
    }
    return null;
  },

  // ── 9. React.RefObject<non-element specific> → React.RefObject<unknown>
  (t) => {
    const m = t.match(/^React\.RefObject<(.+)>$/);
    if (m && !["HTMLElement", "unknown", "any"].includes(m[1]) && !m[1].startsWith("HTML")) {
      return { degraded: "React.RefObject<unknown>", rule: "ref_specific→unknown" };
    }
    return null;
  },

  // ── 10. React.MutableRefObject<non-element> → React.MutableRefObject<unknown>
  (t) => {
    const m = t.match(/^React\.MutableRefObject<(.+)>$/);
    if (m && !["HTMLElement", "unknown", "any"].includes(m[1]) && !m[1].startsWith("HTML")) {
      return { degraded: "React.MutableRefObject<unknown>", rule: "mutable_ref_specific→unknown" };
    }
    return null;
  },

  // ── 11. String literal unions → string ──────────────────────────
  (t) => {
    const parts = t.split(/\s*\|\s*/);
    if (parts.length >= 2 && parts.every((p) => /^["']/.test(p.trim()))) {
      return { degraded: "string", rule: "string_literal_union→string" };
    }
    return null;
  },

  // ── 12. Numeric literal unions → number ─────────────────────────
  (t) => {
    const parts = t.split(/\s*\|\s*/);
    if (parts.length >= 2 && parts.every((p) => /^-?\d+(\.\d+)?$/.test(p.trim()))) {
      return { degraded: "number", rule: "numeric_literal_union→number" };
    }
    return null;
  },

  // ── 13. Boolean literal types → boolean ─────────────────────────
  (t) => {
    if (t === "true" || t === "false") {
      return { degraded: "boolean", rule: "boolean_literal→boolean" };
    }
    return null;
  },

  // ── 14. Mixed literal/boolean unions → string | boolean ─────────
  //   e.g. boolean | 'indeterminate' → string | boolean
  (t) => {
    const parts = t.split(/\s*\|\s*/);
    if (parts.length >= 2) {
      const hasBoolean = parts.some((p) => p === "boolean" || p === "true" || p === "false");
      const hasStringLiteral = parts.some((p) => /^["']/.test(p.trim()));
      if (hasBoolean && hasStringLiteral) {
        return { degraded: "string | boolean", rule: "mixed_literal_union→string_boolean" };
      }
    }
    return null;
  },

  // ── 15. Tuple of same type → type[] ─────────────────────────────
  (t) => {
    const m = t.match(/^\[(\w+)(?:,\s*\1)+\]$/);
    if (m) {
      return { degraded: `${m[1]}[]`, rule: "tuple→array" };
    }
    return null;
  },

  // ── 16. Callback with params → (...args: any[]) => returnType ──
  //   Uses balanced parentheses for reliable matching.
  (t) => {
    const cb = parseCallbackType(t);
    if (!cb) return null;
    // Skip already-generic callbacks
    if (cb.params.trim() === "...args: any[]") return null;
    if (cb.params.trim() === "") return null; // handled by rule 17
    const degradedReturn = SIMPLE_TYPES.has(cb.returnType) ? cb.returnType : "unknown";
    return {
      degraded: `(...args: any[]) => ${degradedReturn}`,
      rule: "callback→generic_callback",
    };
  },

  // ── 17. Parameterless callback with specific return → () => unknown
  (t) => {
    const cb = parseCallbackType(t);
    if (!cb) return null;
    if (cb.params.trim() !== "") return null;
    if (SIMPLE_TYPES.has(cb.returnType)) return null; // already generic
    return {
      degraded: `() => unknown`,
      rule: "callback_return→unknown",
    };
  },

  // ── 18. React.ComponentPropsWithRef<"tag"> → React.ComponentPropsWithRef<any>
  (t) => {
    const m = t.match(/^React\.ComponentPropsWithRef<(.+)>$/);
    if (m && m[1] !== "any") {
      return { degraded: "React.ComponentPropsWithRef<any>", rule: "component_props_ref→generic" };
    }
    return null;
  },

  // ── 19. React.ComponentPropsWithoutRef<"tag"> → React.ComponentPropsWithoutRef<any>
  (t) => {
    const m = t.match(/^React\.ComponentPropsWithoutRef<(.+)>$/);
    if (m && m[1] !== "any") {
      return { degraded: "React.ComponentPropsWithoutRef<any>", rule: "component_props→generic" };
    }
    return null;
  },

  // ── 20. React.ElementRef<"tag"> → React.ElementRef<any> ────────
  (t) => {
    const m = t.match(/^React\.ElementRef<(.+)>$/);
    if (m && m[1] !== "any") {
      return { degraded: "React.ElementRef<any>", rule: "element_ref→generic" };
    }
    return null;
  },

  // ── 21. DOMRect / DataTransfer / Selection → object ─────────────
  (t) => {
    const domObjects = new Set(["DOMRect", "DataTransfer", "Selection", "DOMRectReadOnly", "CSSStyleDeclaration"]);
    if (domObjects.has(t)) {
      return { degraded: "object", rule: "dom_object→object" };
    }
    return null;
  },

  // ── 22. ReturnType<typeof X> → unknown ──────────────────────────
  (t) => {
    if (t.startsWith("ReturnType<")) {
      return { degraded: "unknown", rule: "returntype→unknown" };
    }
    return null;
  },

  // ── 23. Extract<...> / Exclude<...> / Omit<...> / Pick<...> → unknown
  (t) => {
    if (/^(Extract|Exclude|Omit|Pick)</.test(t)) {
      return { degraded: "unknown", rule: "utility_type→unknown" };
    }
    return null;
  },

  // ── 24. Record<string, SpecificType> → Record<string, unknown> ─
  (t) => {
    const m = t.match(/^Record<string,\s*(.+)>$/);
    if (m && m[1] !== "unknown" && m[1] !== "any") {
      return { degraded: "Record<string, unknown>", rule: "record→generic" };
    }
    return null;
  },

  // ── 25. Promise<SpecificType> → Promise<unknown> ───────────────
  (t) => {
    const m = t.match(/^Promise<(.+)>$/);
    if (m && !SIMPLE_TYPES.has(m[1]) && m[1] !== "Response") {
      return { degraded: "Promise<unknown>", rule: "promise→generic" };
    }
    return null;
  },

  // ── 26. Map<K, V> → Map<unknown, unknown> ──────────────────────
  (t) => {
    const m = t.match(/^Map<(.+),\s*(.+)>$/);
    if (m) {
      return { degraded: "Map<unknown, unknown>", rule: "map→generic" };
    }
    return null;
  },

  // ── 27. Set<T> → Set<unknown> ──────────────────────────────────
  (t) => {
    const m = t.match(/^Set<(.+)>$/);
    if (m && !SIMPLE_TYPES.has(m[1])) {
      return { degraded: "Set<unknown>", rule: "set→generic" };
    }
    return null;
  },
];

// ══════════════════════════════════════════════════════════════════════
// Apply degradation
// ══════════════════════════════════════════════════════════════════════

function degradeType(typeText: string): DegradationResult {
  for (const rule of DEGRADATION_RULES) {
    const result = rule(typeText);
    if (result) return result;
  }
  return null;
}

// ══════════════════════════════════════════════════════════════════════
// CLI
// ══════════════════════════════════════════════════════════════════════

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

  return { input, output };
}

// ══════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════

function main() {
  const { input, output } = parseArgs();
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
  const annotations: TypeAnnotation[] = lines.map((l) => JSON.parse(l));

  console.log(`  Total annotations: ${annotations.length}`);

  const trainingPairs: TrainingPair[] = [];
  const skippedTypes = new Map<string, number>();
  const appliedRules = new Map<string, number>();

  for (const ann of annotations) {
    const result = degradeType(ann.typeText);
    if (result) {
      // CRITICAL: Replace precise type with degraded type in context
      // so the model can't copy the answer from surrounding code.
      const modifiedContext = ann.context.split(ann.typeText).join(result.degraded);

      trainingPairs.push({
        context: modifiedContext,
        name: ann.name,
        kind: ann.kind,
        degradedType: result.degraded,
        preciseType: ann.typeText,
        rule: result.rule,
        file: ann.file,
        line: ann.line,
      });
      appliedRules.set(result.rule, (appliedRules.get(result.rule) || 0) + 1);
    } else {
      skippedTypes.set(ann.typeText, (skippedTypes.get(ann.typeText) || 0) + 1);
    }
  }

  // ── Summary ─────────────────────────────────────────────────────
  console.log(`\n${"─".repeat(60)}`);
  console.log(`DEGRADATION SUMMARY`);
  console.log(`${"─".repeat(60)}`);
  console.log(`  Training pairs generated: ${trainingPairs.length}`);
  console.log(`  Annotations skipped:      ${annotations.length - trainingPairs.length}`);
  console.log(
    `  Conversion rate:          ${((trainingPairs.length / annotations.length) * 100).toFixed(1)}%`,
  );

  console.log(`\n  Rules applied:`);
  for (const [rule, count] of [...appliedRules.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`    ${rule.padEnd(45)} ${count}`);
  }

  console.log(`\n  Top 30 skipped types:`);
  const sortedSkipped = [...skippedTypes.entries()].sort((a, b) => b[1] - a[1]);
  for (const [type, count] of sortedSkipped.slice(0, 30)) {
    console.log(`    ${count.toString().padStart(4)}  ${type}`);
  }

  // ── Output ──────────────────────────────────────────────────────
  const outputPath = output
    ? path.resolve(output)
    : path.join(path.dirname(inputPath), "training_pairs.jsonl");
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  const outputLines = trainingPairs.map((p) => JSON.stringify(p));
  fs.writeFileSync(outputPath, outputLines.join("\n") + "\n", "utf-8");

  console.log(`\n  Output: ${outputPath}`);
  console.log(`  ${trainingPairs.length} training pairs ready.`);
  console.log(`${"═".repeat(60)}\n`);
}

main();
