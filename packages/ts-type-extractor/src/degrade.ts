/**
 * Type Degradation — Generate Training Pairs
 * ============================================
 *
 * Takes extracted type annotations (from extract.ts) and creates
 * training pairs by "degrading" precise types to generic ones.
 *
 * The model's job will be to REVERSE the degradation:
 *   Input:  context + generic type  (e.g. "handleClick(e: Event)")
 *   Target: precise type            (e.g. "React.MouseEvent<HTMLButtonElement>")
 *
 * Degradation hierarchy (precise → generic):
 *
 *   HTML Elements:
 *     HTMLButtonElement  → HTMLElement
 *     HTMLInputElement   → HTMLElement
 *     HTMLDivElement     → HTMLElement
 *     HTMLSpanElement    → HTMLElement
 *     ...all HTML*Element → HTMLElement
 *
 *   React Events:
 *     React.MouseEvent<HTMLButtonElement>    → React.SyntheticEvent
 *     React.KeyboardEvent<HTMLInputElement>  → React.SyntheticEvent
 *     React.PointerEvent                    → React.SyntheticEvent
 *     React.ChangeEvent<HTMLInputElement>    → React.SyntheticEvent
 *     React.FocusEvent<HTMLInputElement>     → React.SyntheticEvent
 *     React.DragEvent<HTMLDivElement>        → React.SyntheticEvent
 *     React.FormEvent<HTMLFormElement>       → React.SyntheticEvent
 *
 *   DOM Events:
 *     PointerEvent     → Event
 *     KeyboardEvent    → Event
 *     MouseEvent       → Event
 *     FocusEvent       → Event
 *     TouchEvent       → Event
 *
 *   React Refs:
 *     React.RefObject<HTMLButtonElement>     → React.RefObject<HTMLElement>
 *     React.MutableRefObject<HTMLDivElement> → React.MutableRefObject<HTMLElement>
 *
 *   Union types with specific literals:
 *     "primary" | "secondary" | "danger"     → string
 *
 *   Specific number types:
 *     [number, number]  → number[]
 *
 * Usage:
 *   npx tsx src/degrade.ts <path-to-extracted-types.jsonl>
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
  /** The code context around the annotation */
  context: string;
  /** The variable/parameter name */
  name: string;
  /** What kind: parameter, variable, property, etc. */
  kind: string;
  /** The degraded (generic) type — this is the MODEL INPUT */
  degradedType: string;
  /** The original precise type — this is the MODEL TARGET */
  preciseType: string;
  /** What degradation rule was applied */
  rule: string;
  /** Source file */
  file: string;
  /** Source line */
  line: number;
}

// ══════════════════════════════════════════════════════════════════════
// Degradation rules
// ══════════════════════════════════════════════════════════════════════

/**
 * Each rule takes a precise type string and returns:
 *   - { degraded, rule } if the rule applies
 *   - null if it doesn't apply
 */
type DegradationRule = (typeText: string) => { degraded: string; rule: string } | null;

const DEGRADATION_RULES: DegradationRule[] = [
  // ── Rule 1: React event types with generic params ─────────────
  // React.MouseEvent<HTMLButtonElement> → React.SyntheticEvent
  // React.ChangeEvent<HTMLInputElement> → React.SyntheticEvent
  (t) => {
    const match = t.match(
      /^React\.(Mouse|Keyboard|Pointer|Change|Focus|Drag|Form|Touch|Wheel|Clipboard|Composition|Animation|Transition|UI)Event(<[^>]+>)?$/
    );
    if (match) return { degraded: "React.SyntheticEvent", rule: "react_event→synthetic" };
    return null;
  },

  // ── Rule 2: React event handler types ─────────────────────────
  // React.PointerEventHandler<E> → React.EventHandler<React.SyntheticEvent>
  (t) => {
    const match = t.match(
      /^React\.(Mouse|Keyboard|Pointer|Change|Focus|Drag|Form|Touch)EventHandler(<[^>]+>)?$/
    );
    if (match) return { degraded: "React.EventHandler<React.SyntheticEvent>", rule: "react_handler→generic_handler" };
    return null;
  },

  // ── Rule 3: Specific HTML element types → HTMLElement ─────────
  // HTMLButtonElement → HTMLElement
  // HTMLInputElement  → HTMLElement
  (t) => {
    const match = t.match(/^HTML(\w+)Element$/);
    if (match && match[1] !== "") {
      // Don't degrade HTMLElement itself
      return { degraded: "HTMLElement", rule: `html_${match[1].toLowerCase()}→html_element` };
    }
    return null;
  },

  // ── Rule 4: HTML element | null → HTMLElement | null ──────────
  (t) => {
    const match = t.match(/^HTML(\w+)Element\s*\|\s*null$/);
    if (match && match[1] !== "") {
      return { degraded: "HTMLElement | null", rule: `html_${match[1].toLowerCase()}_nullable→html_element_nullable` };
    }
    return null;
  },

  // ── Rule 5: DOM native events → Event ─────────────────────────
  // PointerEvent → Event, KeyboardEvent → Event, MouseEvent → Event
  (t) => {
    const domEvents = [
      "PointerEvent", "KeyboardEvent", "MouseEvent", "FocusEvent",
      "TouchEvent", "WheelEvent", "DragEvent", "InputEvent",
      "CompositionEvent", "ClipboardEvent", "AnimationEvent",
      "TransitionEvent", "UIEvent",
    ];
    if (domEvents.includes(t)) {
      return { degraded: "Event", rule: `dom_${t.toLowerCase()}→event` };
    }
    return null;
  },

  // ── Rule 6: React.RefObject<SpecificElement> → React.RefObject<HTMLElement>
  (t) => {
    const match = t.match(/^React\.RefObject<HTML(\w+)Element>$/);
    if (match && match[1] !== "") {
      return { degraded: "React.RefObject<HTMLElement>", rule: "ref_specific→ref_generic" };
    }
    return null;
  },

  // ── Rule 7: React.MutableRefObject<SpecificElement>
  (t) => {
    const match = t.match(/^React\.MutableRefObject<HTML(\w+)Element>$/);
    if (match && match[1] !== "") {
      return { degraded: "React.MutableRefObject<HTMLElement>", rule: "mutable_ref_specific→mutable_ref_generic" };
    }
    return null;
  },

  // ── Rule 8: String literal unions → string ────────────────────
  // "primary" | "secondary" | "danger" → string
  (t) => {
    // Match: "foo" | "bar" | "baz" (all parts are quoted strings)
    const parts = t.split(/\s*\|\s*/);
    if (parts.length >= 2 && parts.every((p) => /^["']/.test(p.trim()))) {
      return { degraded: "string", rule: "string_literal_union→string" };
    }
    return null;
  },

  // ── Rule 9: Specific tuple → array ────────────────────────────
  // [number, number] → number[]
  (t) => {
    const match = t.match(/^\[(\w+)(?:,\s*\1)+\]$/);
    if (match) {
      return { degraded: `${match[1]}[]`, rule: "tuple→array" };
    }
    return null;
  },

  // ── Rule 10: Specific callback signature → generic function ───
  // (value: string) => void → (...args: any[]) => void
  (t) => {
    const match = t.match(/^\([^)]+\)\s*=>\s*(void|boolean|string|number)$/);
    if (match) {
      return { degraded: `(...args: any[]) => ${match[1]}`, rule: "specific_callback→generic_callback" };
    }
    return null;
  },

  // ── Rule 11: DOMRect → object ─────────────────────────────────
  (t) => {
    if (t === "DOMRect") {
      return { degraded: "object", rule: "domrect→object" };
    }
    return null;
  },

  // ── Rule 12: DataTransfer → object ────────────────────────────
  (t) => {
    if (t === "DataTransfer") {
      return { degraded: "object", rule: "datatransfer→object" };
    }
    return null;
  },

  // ── Rule 13: ReturnType<typeof X> → unknown ───────────────────
  (t) => {
    if (t.startsWith("ReturnType<")) {
      return { degraded: "unknown", rule: "returntype→unknown" };
    }
    return null;
  },
];

// ══════════════════════════════════════════════════════════════════════
// Apply degradation
// ══════════════════════════════════════════════════════════════════════

function degradeType(typeText: string): { degraded: string; rule: string } | null {
  for (const rule of DEGRADATION_RULES) {
    const result = rule(typeText);
    if (result) return result;
  }
  return null;
}

// ══════════════════════════════════════════════════════════════════════
// Main
// ══════════════════════════════════════════════════════════════════════

function main() {
  const inputPath = process.argv[2] || "/home/stan/workspace/ai-lab/data/extracted_types.jsonl";
  const resolvedInput = path.resolve(inputPath);

  if (!fs.existsSync(resolvedInput)) {
    console.error(`Error: file not found: ${resolvedInput}`);
    process.exit(1);
  }

  console.log(`\n${"═".repeat(60)}`);
  console.log(`TYPE DEGRADATION — generating training pairs`);
  console.log(`${"═".repeat(60)}`);
  console.log(`  Input: ${resolvedInput}\n`);

  const lines = fs.readFileSync(resolvedInput, "utf-8").trim().split("\n");
  const annotations: TypeAnnotation[] = lines.map((l) => JSON.parse(l));

  console.log(`  Total annotations: ${annotations.length}`);

  const trainingPairs: TrainingPair[] = [];
  const skippedTypes = new Map<string, number>();
  const appliedRules = new Map<string, number>();

  for (const ann of annotations) {
    const result = degradeType(ann.typeText);
    if (result) {
      trainingPairs.push({
        context: ann.context,
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
  console.log(`  Conversion rate:          ${((trainingPairs.length / annotations.length) * 100).toFixed(1)}%`);

  console.log(`\n  Rules applied:`);
  for (const [rule, count] of [...appliedRules.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`    ${rule.padEnd(45)} ${count}`);
  }

  console.log(`\n  Top skipped types (no degradation rule):`);
  const sortedSkipped = [...skippedTypes.entries()].sort((a, b) => b[1] - a[1]);
  for (const [type, count] of sortedSkipped.slice(0, 20)) {
    console.log(`    ${count.toString().padStart(4)}  ${type}`);
  }

  // ── Output training pairs ───────────────────────────────────────
  const outputDir = path.dirname(resolvedInput);
  const outputPath = path.join(outputDir, "training_pairs.jsonl");
  const outputLines = trainingPairs.map((p) => JSON.stringify(p));
  fs.writeFileSync(outputPath, outputLines.join("\n") + "\n", "utf-8");

  console.log(`\n  Output: ${outputPath}`);
  console.log(`  Format: JSON Lines (one training pair per line)`);

  // ── Show examples ───────────────────────────────────────────────
  console.log(`\n${"─".repeat(60)}`);
  console.log(`SAMPLE TRAINING PAIRS (first 10)`);
  console.log(`${"─".repeat(60)}`);

  for (const pair of trainingPairs.slice(0, 10)) {
    console.log(`\n  [${pair.kind}] ${pair.name}`);
    console.log(`  Degraded (input):  ${pair.degradedType}`);
    console.log(`  Precise (target):  ${pair.preciseType}`);
    console.log(`  Rule:              ${pair.rule}`);
    console.log(`  File:              ${pair.file}:${pair.line}`);
  }

  // ── Stats by degradation direction ──────────────────────────────
  console.log(`\n${"─".repeat(60)}`);
  console.log(`UNIQUE DEGRADATION MAPPINGS`);
  console.log(`${"─".repeat(60)}`);

  const mappings = new Map<string, number>();
  for (const pair of trainingPairs) {
    const key = `${pair.degradedType}  ←  ${pair.preciseType}`;
    mappings.set(key, (mappings.get(key) || 0) + 1);
  }
  for (const [mapping, count] of [...mappings.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`    ${count.toString().padStart(4)}  ${mapping}`);
  }

  console.log(`\n${"═".repeat(60)}`);
  console.log(`Done. ${trainingPairs.length} training pairs ready.`);
  console.log(`${"═".repeat(60)}\n`);
}

main();
