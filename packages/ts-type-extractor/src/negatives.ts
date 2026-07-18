/**
 * Per-rule hard negatives (data-quality plan §1.4).
 *
 * For every active rule `R` we want to teach the model to **preserve** types
 * that already have the rule's degraded shape (e.g. for `string_literal_union→string`,
 * inputs where `typeText === "string"`). These are emitted as pairs
 * `(degraded = typeText, target = typeText, rule = R, isNegative = true)`.
 *
 * Shape predicates live in `refiner-locate.ts::RULES` (single source of truth
 * shared between the locator and this module).
 */

import { RULES } from "./rules/refiner-locate.js";

const norm = (s: string): string => s.replace(/\s+/g, " ").trim();

/** Minimal subset of `TypeAnnotation` needed for negative collection. */
export interface NegativeCandidate {
  typeText: string;
  repo?: string;
  file: string;
  line: number;
  name: string;
}

export interface SelectedNegative<A extends NegativeCandidate> {
  ann: A;
  rule: string;
}

/** Build `name → predicate` map from the locator's RULES table. */
function buildPredicateMap(): Map<string, (t: string) => boolean> {
  const m = new Map<string, (t: string) => boolean>();
  for (const r of RULES) m.set(r.name, r.match);
  return m;
}

/**
 * Select hard-negative annotations per rule.
 *
 * For every rule with `positiveCounts[rule] > 0`, scan `annotations` for ones
 * whose `typeText` already matches the rule's degraded shape and pick the first
 * `round(positiveCount * ratio)` of them in deterministic order.
 *
 * Negatives cannot collide with positives because the degradation rules in
 * `degrade.ts` always require `typeText` to differ from the degraded form, so
 * the matching pools are disjoint by construction.
 */
export function collectNegatives<A extends NegativeCandidate>(
  annotations: A[],
  positiveCounts: Map<string, number>,
  ratio: number,
): SelectedNegative<A>[] {
  if (ratio <= 0) return [];
  const predicates = buildPredicateMap();

  // Deterministic ordering: by (repo, file, line, name).
  const sorted = [...annotations].sort((a, b) => {
    const ar = a.repo ?? "";
    const br = b.repo ?? "";
    if (ar !== br) return ar < br ? -1 : 1;
    if (a.file !== b.file) return a.file < b.file ? -1 : 1;
    if (a.line !== b.line) return a.line - b.line;
    if (a.name !== b.name) return a.name < b.name ? -1 : 1;
    return 0;
  });

  const out: SelectedNegative<A>[] = [];
  for (const [ruleName, positiveCount] of positiveCounts.entries()) {
    if (positiveCount <= 0) continue;
    const predicate = predicates.get(ruleName);
    if (!predicate) continue; // rule has no shape predicate registered
    const limit = Math.round(positiveCount * ratio);
    if (limit <= 0) continue;

    let taken = 0;
    for (const ann of sorted) {
      if (!predicate(norm(ann.typeText))) continue;
      out.push({ ann, rule: ruleName });
      taken++;
      if (taken >= limit) break;
    }
  }
  return out;
}
