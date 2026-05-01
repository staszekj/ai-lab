"""
Output validators for ts-type-refiner predictions.

Each validator is a pure function `(suggestion: str) -> (ok, reason)`
that mechanically inverts ONE rule from `degrade.ts`. They are the
SAFETY NET on top of the model: even if the network outputs garbage,
we never propose an edit unless the suggestion shape-matches the
expected precise form.

This is "shape-level" verification (level 1 of 3). Stronger options
(round-trip degrade, full TS compile of the patched file) are out of
scope for Phase 2.

Coverage: this file mirrors all 24 rules that produced training
pairs. Keys in `VALIDATORS` MUST match the `rule` field emitted by
`refiner-locate.ts` (and the `rule` recorded in training_pairs.jsonl).

Collision note: `returntype→unknown` and `utility_type→unknown` both
degrade to literal `unknown`, so the locator emits a single candidate
per `unknown` site under `utility_type→unknown`. Both keys map to
the same combined validator that accepts EITHER precise form.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Tuple


# ══════════════════════════════════════════════════════════════════════
# Building blocks
# ══════════════════════════════════════════════════════════════════════

# Single string literal: 'a' or "a" — both quote styles accepted to
# mirror degrade.ts rule 11 which uses `/^["']/`.
_STR_LIT = r"(?:'[^']*'|\"[^\"]*\")"
# Single numeric literal (positive or negative, integer or decimal).
_NUM_LIT = r"-?\d+(?:\.\d+)?"
# Boolean literal token.
_BOOL_LIT = r"(?:true|false)"

# Identifier-ish token used as a "simple" type name (e.g. `string`,
# `Foo`, `MyType`). Deliberately permissive — the validator's job is
# shape, not type resolution.
_IDENT = r"[A-Za-z_$][A-Za-z0-9_$]*"

# Set of return types `degrade.ts` recognises as "simple" (rule 16).
SIMPLE_RETURN_TYPES = {
    "void", "boolean", "string", "number",
    "unknown", "any", "never", "undefined", "null",
}

# DOM event names collapsed to bare `Event` by degrade rule 6.
DOM_EVENTS = {
    "PointerEvent", "KeyboardEvent", "MouseEvent", "FocusEvent",
    "TouchEvent", "WheelEvent", "DragEvent", "InputEvent",
    "CompositionEvent", "ClipboardEvent", "AnimationEvent",
    "TransitionEvent", "UIEvent", "CustomEvent",
}

# DOM "object-ish" types collapsed to bare `object` by degrade rule 21.
DOM_OBJECTS = {
    "DOMRect", "DataTransfer", "Selection",
    "DOMRectReadOnly", "CSSStyleDeclaration",
}

Validator = Callable[[str], Tuple[bool, str]]


def _ok() -> Tuple[bool, str]:
    return True, "ok"


def _fail(reason: str) -> Tuple[bool, str]:
    return False, reason


# ══════════════════════════════════════════════════════════════════════
# Per-rule validators (24 total — one per trained rule)
# ══════════════════════════════════════════════════════════════════════

# ── Rule 1: react_event→synthetic ─────────────────────────────────────
# Precise form must be `React.<Specific>Event[<…>]` and NOT
# `React.SyntheticEvent` (the degraded form).
_RE_REACT_EVENT = re.compile(rf"^React\.{_IDENT}Event(?:<.+>)?$")

def validate_react_event(s: str) -> Tuple[bool, str]:
    s = s.strip()
    if not _RE_REACT_EVENT.match(s):
        return _fail("expected React.<Specific>Event[<…>]")
    if s.startswith("React.SyntheticEvent"):
        return _fail("still degraded (React.SyntheticEvent)")
    return _ok()


# ── Rule 2: react_handler→generic_handler ─────────────────────────────
_RE_REACT_HANDLER = re.compile(rf"^React\.{_IDENT}EventHandler(?:<.+>)?$")

def validate_react_handler(s: str) -> Tuple[bool, str]:
    s = s.strip()
    if not _RE_REACT_HANDLER.match(s):
        return _fail("expected React.<Specific>EventHandler[<…>]")
    if "EventHandler<React.SyntheticEvent>" in s:
        return _fail("still degraded (generic EventHandler)")
    return _ok()


# ── Rule 3: html_element→generic ──────────────────────────────────────
_RE_HTML_ELEMENT = re.compile(rf"^HTML{_IDENT}Element$")

def validate_html_element(s: str) -> Tuple[bool, str]:
    s = s.strip()
    if not _RE_HTML_ELEMENT.match(s):
        return _fail("expected HTML<Specific>Element")
    if s == "HTMLElement":
        return _fail("still degraded (HTMLElement)")
    return _ok()


# ── Rule 4: html_element_nullable→generic ─────────────────────────────
_RE_HTML_ELEMENT_NULL = re.compile(rf"^HTML{_IDENT}Element\s*\|\s*null$")

def validate_html_element_nullable(s: str) -> Tuple[bool, str]:
    s = s.strip()
    if not _RE_HTML_ELEMENT_NULL.match(s):
        return _fail("expected HTML<Specific>Element | null")
    if s.startswith("HTMLElement"):
        return _fail("still degraded (HTMLElement | null)")
    return _ok()


# ── Rule 5: svg_element→generic ───────────────────────────────────────
_RE_SVG_ELEMENT = re.compile(rf"^SVG{_IDENT}Element$")

def validate_svg_element(s: str) -> Tuple[bool, str]:
    s = s.strip()
    if not _RE_SVG_ELEMENT.match(s):
        return _fail("expected SVG<Specific>Element")
    if s == "SVGElement":
        return _fail("still degraded (SVGElement)")
    return _ok()


# ── Rule 6: dom_event→generic ─────────────────────────────────────────
def validate_dom_event(s: str) -> Tuple[bool, str]:
    s = s.strip()
    if s not in DOM_EVENTS:
        return _fail(f"expected one of {sorted(DOM_EVENTS)}")
    return _ok()


# ── Rule 7: ref_element→generic ───────────────────────────────────────
_RE_REF_HTML = re.compile(rf"^React\.RefObject<HTML{_IDENT}Element>$")

def validate_ref_element(s: str) -> Tuple[bool, str]:
    s = s.strip()
    if not _RE_REF_HTML.match(s):
        return _fail("expected React.RefObject<HTML<Specific>Element>")
    if s == "React.RefObject<HTMLElement>":
        return _fail("still degraded")
    return _ok()


# ── Rule 8: mutable_ref_element→generic ───────────────────────────────
_RE_MUT_REF_HTML = re.compile(rf"^React\.MutableRefObject<HTML{_IDENT}Element>$")

def validate_mutable_ref_element(s: str) -> Tuple[bool, str]:
    s = s.strip()
    if not _RE_MUT_REF_HTML.match(s):
        return _fail("expected React.MutableRefObject<HTML<Specific>Element>")
    if s == "React.MutableRefObject<HTMLElement>":
        return _fail("still degraded")
    return _ok()


# ── Rule 9: ref_specific→unknown ──────────────────────────────────────
_RE_REF_GENERIC = re.compile(r"^React\.RefObject<(.+)>$")

def validate_ref_specific(s: str) -> Tuple[bool, str]:
    s = s.strip()
    m = _RE_REF_GENERIC.match(s)
    if not m:
        return _fail("expected React.RefObject<…>")
    inner = m.group(1).strip()
    if inner in {"unknown", "any"} or inner.startswith("HTML"):
        return _fail("still degraded or HTML element ref")
    return _ok()


# ── Rule 10: mutable_ref_specific→unknown ─────────────────────────────
_RE_MUT_REF_GENERIC = re.compile(r"^React\.MutableRefObject<(.+)>$")

def validate_mutable_ref_specific(s: str) -> Tuple[bool, str]:
    s = s.strip()
    m = _RE_MUT_REF_GENERIC.match(s)
    if not m:
        return _fail("expected React.MutableRefObject<…>")
    inner = m.group(1).strip()
    if inner in {"unknown", "any"} or inner.startswith("HTML"):
        return _fail("still degraded or HTML element ref")
    return _ok()


# ── Rule 11: string_literal_union→string ──────────────────────────────
_RE_STR_UNION = re.compile(rf"^\s*{_STR_LIT}(\s*\|\s*{_STR_LIT})+\s*$")

def validate_string_literal_union(s: str) -> Tuple[bool, str]:
    if not _RE_STR_UNION.match(s):
        return _fail("expected 'a' | 'b' | …")
    return _ok()


# ── Rule 12: numeric_literal_union→number ─────────────────────────────
_RE_NUM_UNION = re.compile(rf"^\s*{_NUM_LIT}(\s*\|\s*{_NUM_LIT})+\s*$")

def validate_numeric_literal_union(s: str) -> Tuple[bool, str]:
    if not _RE_NUM_UNION.match(s):
        return _fail("expected 1 | 2 | …")
    return _ok()


# ── Rule 13: boolean_literal→boolean ──────────────────────────────────
_RE_BOOL_LIT = re.compile(rf"^\s*{_BOOL_LIT}\s*$")

def validate_boolean_literal(s: str) -> Tuple[bool, str]:
    if not _RE_BOOL_LIT.match(s):
        return _fail("expected literal `true` or `false`")
    return _ok()


# ── Rule 14: mixed_literal_union→string_boolean ───────────────────────
# Precise form is a union with at least one boolean-ish part AND at
# least one quoted string literal. Mirrors degrade.ts logic exactly.
def validate_mixed_literal_union(s: str) -> Tuple[bool, str]:
    parts = [p.strip() for p in re.split(r"\s*\|\s*", s.strip())]
    if len(parts) < 2:
        return _fail("expected union of ≥2 parts")
    has_bool = any(p in {"boolean", "true", "false"} for p in parts)
    has_str  = any(re.match(r"^['\"]", p) for p in parts)
    if not (has_bool and has_str):
        return _fail("expected union with both boolean and string literal parts")
    return _ok()


# ── Rule 15: tuple→array ──────────────────────────────────────────────
# Precise form is `[T, T, ...]` — same simple-type element repeated.
# Matches degrade.ts pattern `^\[(\w+)(?:,\s*\1)+\]$`.
_RE_TUPLE_REPEATED = re.compile(rf"^\s*\[\s*({_IDENT})(?:\s*,\s*\1)+\s*\]\s*$")

def validate_tuple(s: str) -> Tuple[bool, str]:
    if not _RE_TUPLE_REPEATED.match(s):
        return _fail("expected [T, T, …] tuple of the same simple type")
    return _ok()


# ── Rule 16: callback→generic_callback ────────────────────────────────
# Precise form: `(specific args) => RET` — non-empty params AND not
# the degraded shape `(...args: any[]) => RET`. RET ∈ SIMPLE_RETURN_TYPES.
_RE_CALLBACK = re.compile(r"^\((.*)\)\s*=>\s*(.+)$", re.DOTALL)

def validate_callback(s: str) -> Tuple[bool, str]:
    s = s.strip()
    m = _RE_CALLBACK.match(s)
    if not m:
        return _fail("expected (params) => returnType")
    params = m.group(1).strip()
    if params == "":
        return _fail("empty params (would match callback_return rule)")
    if params == "...args: any[]":
        return _fail("still degraded ((...args: any[]) => …)")
    return _ok()


# ── Rule 17: callback_return→unknown ──────────────────────────────────
# Precise form: `() => SpecificReturn` where SpecificReturn is NOT a
# member of SIMPLE_RETURN_TYPES (otherwise degrade.ts would not have
# fired this rule).
_RE_NULLARY_CB = re.compile(r"^\(\s*\)\s*=>\s*(.+)$", re.DOTALL)

def validate_callback_return(s: str) -> Tuple[bool, str]:
    s = s.strip()
    m = _RE_NULLARY_CB.match(s)
    if not m:
        return _fail("expected () => returnType")
    ret = m.group(1).strip()
    if ret in SIMPLE_RETURN_TYPES:
        return _fail(f"return type still simple ({ret})")
    return _ok()


# ── Rule 18: component_props_ref→generic ──────────────────────────────
_RE_CP_WITH_REF = re.compile(r"^React\.ComponentPropsWithRef<(.+)>$")

def validate_component_props_ref(s: str) -> Tuple[bool, str]:
    s = s.strip()
    m = _RE_CP_WITH_REF.match(s)
    if not m:
        return _fail("expected React.ComponentPropsWithRef<…>")
    if m.group(1).strip() == "any":
        return _fail("still degraded (<any>)")
    return _ok()


# ── Rule 19: component_props→generic ──────────────────────────────────
_RE_CP_WITHOUT_REF = re.compile(r"^React\.ComponentPropsWithoutRef<(.+)>$")

def validate_component_props(s: str) -> Tuple[bool, str]:
    s = s.strip()
    m = _RE_CP_WITHOUT_REF.match(s)
    if not m:
        return _fail("expected React.ComponentPropsWithoutRef<…>")
    if m.group(1).strip() == "any":
        return _fail("still degraded (<any>)")
    return _ok()


# ── Rule 20: element_ref→generic ──────────────────────────────────────
_RE_ELEMENT_REF = re.compile(r"^React\.ElementRef<(.+)>$")

def validate_element_ref(s: str) -> Tuple[bool, str]:
    s = s.strip()
    m = _RE_ELEMENT_REF.match(s)
    if not m:
        return _fail("expected React.ElementRef<…>")
    if m.group(1).strip() == "any":
        return _fail("still degraded (<any>)")
    return _ok()


# ── Rule 21: dom_object→object ────────────────────────────────────────
def validate_dom_object(s: str) -> Tuple[bool, str]:
    s = s.strip()
    if s not in DOM_OBJECTS:
        return _fail(f"expected one of {sorted(DOM_OBJECTS)}")
    return _ok()


# ── Rules 22 & 23 (combined): unknown → ReturnType<…> | utility<…> ────
# The locator emits a single candidate per `unknown` site under
# `utility_type→unknown`. We accept EITHER precise form because, from
# the source `unknown`, there is no way to know which the user wanted.
_RE_RETURN_TYPE = re.compile(r"^ReturnType<.+>$")
_RE_UTILITY     = re.compile(r"^(?:Extract|Exclude|Omit|Pick)<.+>$")

def validate_unknown_specific(s: str) -> Tuple[bool, str]:
    s = s.strip()
    if _RE_RETURN_TYPE.match(s) or _RE_UTILITY.match(s):
        return _ok()
    return _fail("expected ReturnType<…> or Extract|Exclude|Omit|Pick<…>")


# ── Rule 24: record→generic ───────────────────────────────────────────
_RE_RECORD = re.compile(r"^Record<\s*string\s*,\s*(.+)>$")

def validate_record(s: str) -> Tuple[bool, str]:
    s = s.strip()
    m = _RE_RECORD.match(s)
    if not m:
        return _fail("expected Record<string, …>")
    inner = m.group(1).strip()
    if inner in {"unknown", "any"}:
        return _fail("still degraded (Record<string, unknown>)")
    return _ok()


# ── Rule 25: promise→generic ──────────────────────────────────────────
_RE_PROMISE = re.compile(r"^Promise<(.+)>$")

def validate_promise(s: str) -> Tuple[bool, str]:
    s = s.strip()
    m = _RE_PROMISE.match(s)
    if not m:
        return _fail("expected Promise<…>")
    inner = m.group(1).strip()
    if inner in SIMPLE_RETURN_TYPES:
        return _fail(f"inner type still simple ({inner})")
    return _ok()


# ── Rule 26: map→generic ──────────────────────────────────────────────
_RE_MAP = re.compile(r"^Map<\s*(.+?)\s*,\s*(.+)>$")

def validate_map(s: str) -> Tuple[bool, str]:
    s = s.strip()
    m = _RE_MAP.match(s)
    if not m:
        return _fail("expected Map<K, V>")
    k, v = m.group(1).strip(), m.group(2).strip()
    if k == "unknown" and v == "unknown":
        return _fail("still degraded (Map<unknown, unknown>)")
    return _ok()


# ── Rule 27: set→generic ──────────────────────────────────────────────
_RE_SET = re.compile(r"^Set<(.+)>$")

def validate_set(s: str) -> Tuple[bool, str]:
    s = s.strip()
    m = _RE_SET.match(s)
    if not m:
        return _fail("expected Set<…>")
    inner = m.group(1).strip()
    if inner in SIMPLE_RETURN_TYPES:
        return _fail(f"inner type still simple ({inner})")
    return _ok()


# ══════════════════════════════════════════════════════════════════════
# Routing table — keys MUST match the `rule` field emitted by
# refiner-locate.ts and recorded in training_pairs.jsonl.
#
# Both `returntype→unknown` and `utility_type→unknown` map to the
# same combined validator (see rules 22 & 23 above).
# ══════════════════════════════════════════════════════════════════════

VALIDATORS: Dict[str, Validator] = {
    # 1 — 10
    "react_event→synthetic":            validate_react_event,
    "react_handler→generic_handler":    validate_react_handler,
    "html_element→generic":             validate_html_element,
    "html_element_nullable→generic":    validate_html_element_nullable,
    "svg_element→generic":              validate_svg_element,
    "dom_event→generic":                validate_dom_event,
    "ref_element→generic":              validate_ref_element,
    "mutable_ref_element→generic":      validate_mutable_ref_element,
    "ref_specific→unknown":             validate_ref_specific,
    "mutable_ref_specific→unknown":     validate_mutable_ref_specific,
    # 11 — 17
    "string_literal_union→string":      validate_string_literal_union,
    "numeric_literal_union→number":     validate_numeric_literal_union,
    "boolean_literal→boolean":          validate_boolean_literal,
    "mixed_literal_union→string_boolean": validate_mixed_literal_union,
    "tuple→array":                      validate_tuple,
    "callback→generic_callback":        validate_callback,
    "callback_return→unknown":          validate_callback_return,
    # 18 — 21
    "component_props_ref→generic":      validate_component_props_ref,
    "component_props→generic":          validate_component_props,
    "element_ref→generic":              validate_element_ref,
    "dom_object→object":                validate_dom_object,
    # 22 & 23 — collide on `unknown`, share validator
    "returntype→unknown":               validate_unknown_specific,
    "utility_type→unknown":             validate_unknown_specific,
    # 24 — 27
    "record→generic":                   validate_record,
    "promise→generic":                  validate_promise,
    "map→generic":                      validate_map,
    "set→generic":                      validate_set,
}
