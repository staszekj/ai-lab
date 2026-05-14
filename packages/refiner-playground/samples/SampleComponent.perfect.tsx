/**
 * Sample React component — **gold-standard refined output**.
 *
 * Hand-written reference for what `SampleComponent.tsx` (the degraded
 * input) would look like if the ts-type-refiner did a perfect job.
 * Use this as the target when measuring model quality:
 *
 *   diff -u samples/SampleComponent.refined.tsx samples/SampleComponent.perfect.tsx
 *
 * Every change vs. the input is tightening — narrower types, but every
 * value site that compiled before still compiles. `label` stays
 * `string` (it really is a free-form string; refining it would be a
 * false positive).
 */

import * as React from "react";
import { useState } from "react";

// ── Button with two degraded props + one degraded event handler ──
export interface ButtonProps {
  // Genuine free-form string — left untouched intentionally.
  // A perfect refiner abstains here.
  label: string;
  // string_literal_union: variant is used as a discriminator in
  // `data-variant` and a CSS class suffix; "primary" | "secondary"
  // | "danger" is the canonical narrowing.
  variant: "primary" | "secondary" | "danger";
  // react_event: the handler is wired to a <button onClick={…}>.
  // The precise React event for a button click is MouseEvent<HTMLButtonElement>.
  onClick: (e: React.MouseEvent<HTMLButtonElement>) => void;
}

export function Button({ label, variant, onClick }: ButtonProps) {
  return (
    <button
      onClick={onClick}
      data-variant={variant}
      className={`btn btn-${variant}`}
    >
      {label}
    </button>
  );
}

// ── Async function with widened return type ──────────────────────
// promise_unknown: the endpoint returns a user object. Narrow to the
// minimal record the function actually exposes. In a real codebase
// this would alias a shared `User` type — we inline it here so the
// sample stays self-contained.
export async function fetchUser(
  id: number,
): Promise<{ id: number; name: string }> {
  const res = await fetch(`/api/users/${id}`);
  return res.json();
}

// ── Lookup tables widened to `unknown` values ────────────────────
// record_string_value: values are plain display strings.
export const LABELS: Record<string, string> = {
  submit: "Submit",
  cancel: "Cancel",
};

// map: keys are tag names (string), values are counts (number).
export const tagToCount: Map<string, number> = new Map();

// set: route paths are strings.
export const visitedRoutes: Set<string> = new Set();

// ── useState with widened state type ─────────────────────────────
// string_literal_union: status is a finite state machine label.
export function StatusBadge() {
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle",
  );
  void setStatus;
  return <span data-status={status}>{status}</span>;
}
