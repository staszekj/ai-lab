/**
 * Sample React component used as **input** for the ts-type-refiner.
 *
 * Every type annotation below is intentionally a *degraded* (widened /
 * loose) form that one of the trained refiner rules should be able to
 * tighten. The code still type-checks: each degraded type is a valid
 * supertype of the precise one, so values keep flowing.
 *
 * Refiner targets in this file (rule → precise form the model should suggest):
 *
 *   - string                       → "primary" | "secondary" | "danger"   (string_literal_union)
 *   - React.SyntheticEvent         → React.MouseEvent<HTMLButtonElement>  (react_event)
 *   - Promise<unknown>             → Promise<{ id: number; name: string }> (promise_unknown)
 *   - Record<string, unknown>      → Record<string, string>               (record_string_value)
 *   - Map<unknown, unknown>        → Map<string, number>                  (map)
 *   - Set<unknown>                 → Set<string>                          (set)
 *
 * The `label` field is a *genuine* `string` — the refiner is expected
 * to either leave it alone or be rejected by the log-prob threshold.
 * Useful for measuring false-positive rate on bare-string targets.
 */

import * as React from "react";
import { useState } from "react";

// ── Button with two degraded props + one degraded event handler ──
export interface ButtonProps {
  label: string;                                // genuine string (FP target)
  variant: string;                              // ← string_literal_union
  onClick: (e: React.SyntheticEvent) => void;   // ← react_event
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
export async function fetchUser(id: number): Promise<unknown> {
  const res = await fetch(`/api/users/${id}`);
  return res.json();
}

// ── Lookup tables widened to `unknown` values ────────────────────
export const LABELS: Record<string, unknown> = {
  submit: "Submit",
  cancel: "Cancel",
};

export const tagToCount: Map<unknown, unknown> = new Map();
export const visitedRoutes: Set<unknown> = new Set();

// ── useState with widened state type ─────────────────────────────
export function StatusBadge() {
  const [status, setStatus] = useState<string>("idle");
  void setStatus;
  return <span data-status={status}>{status}</span>;
}
