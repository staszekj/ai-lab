"""
Output validators for ts-type-refiner predictions.

Rules mirror the new type-def-driven degradation set.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Tuple

Validator = Callable[[str], Tuple[bool, str]]


def _ok() -> Tuple[bool, str]:
    return True, "ok"


def _fail(reason: str) -> Tuple[bool, str]:
    return False, reason


_SIMPLE_TYPES = {
    "void", "boolean", "string", "number",
    "unknown", "any", "never", "undefined", "null",
}


def _strip(s: str) -> str:
    return s.strip()


# 1
_RE_REACT_EVENT_HANDLER = re.compile(r"^React\.\w+EventHandler(?:<.+>)?$")


def validate_react_event_handler(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    if not _RE_REACT_EVENT_HANDLER.match(s):
        return _fail("expected React.<Specific>EventHandler[<…>]")
    if s == "React.EventHandler<React.SyntheticEvent>":
        return _fail("still degraded")
    return _ok()


# 1b
_RE_REACT_SPECIFIC_HANDLER_ALIAS = re.compile(
    r"^(Mouse|Keyboard|Pointer|Touch|Drag|Focus|Change|Clipboard|Composition|Animation|Transition|Form|Wheel)EventHandler(?:<.+>)?$"
)


def validate_react_specific_event_handler_alias(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    if not _RE_REACT_SPECIFIC_HANDLER_ALIAS.match(s):
        return _fail("expected specific React event handler alias")
    return _ok()


# 2
_RE_REACT_EVENT = re.compile(r"^React\.\w+Event(?:<.+>)?$")


def validate_react_event(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    if not _RE_REACT_EVENT.match(s):
        return _fail("expected React.<Specific>Event[<…>]")
    if s == "React.SyntheticEvent":
        return _fail("still degraded")
    return _ok()


# 3
_RE_CPWR = re.compile(r"^React\.ComponentPropsWithRef<(.+)>$")


def validate_react_component_props_with_ref(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    m = _RE_CPWR.match(s)
    if not m:
        return _fail("expected React.ComponentPropsWithRef<…>")
    if m.group(1).strip() == "any":
        return _fail("still degraded")
    return _ok()


# 4
_RE_CPWOR = re.compile(r"^React\.ComponentPropsWithoutRef<(.+)>$")


def validate_react_component_props_without_ref(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    m = _RE_CPWOR.match(s)
    if not m:
        return _fail("expected React.ComponentPropsWithoutRef<…>")
    if m.group(1).strip() == "any":
        return _fail("still degraded")
    return _ok()


# 5
_RE_ELEMENT_REF = re.compile(r"^React\.ElementRef<(.+)>$")


def validate_react_element_ref(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    m = _RE_ELEMENT_REF.match(s)
    if not m:
        return _fail("expected React.ElementRef<…>")
    if m.group(1).strip() == "any":
        return _fail("still degraded")
    return _ok()


# 6
_RE_REFOBJECT = re.compile(r"^React\.RefObject<(.+)>$")


def validate_react_refobject(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    m = _RE_REFOBJECT.match(s)
    if not m:
        return _fail("expected React.RefObject<…>")
    inner = m.group(1).strip()
    if inner in {"unknown", "any"}:
        return _fail("still degraded")
    return _ok()


# 7
_RE_MUT_REFOBJECT = re.compile(r"^React\.MutableRefObject<(.+)>$")


def validate_react_mutable_refobject(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    m = _RE_MUT_REFOBJECT.match(s)
    if not m:
        return _fail("expected React.MutableRefObject<…>")
    inner = m.group(1).strip()
    if inner in {"unknown", "any"}:
        return _fail("still degraded")
    return _ok()


# 8
_RE_DISPATCH_SSA = re.compile(r"^React\.Dispatch<\s*React\.SetStateAction<(.+)>\s*>$")


def validate_react_dispatch_setstateaction(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    m = _RE_DISPATCH_SSA.match(s)
    if not m:
        return _fail("expected React.Dispatch<React.SetStateAction<…>>")
    inner = m.group(1).strip()
    if inner in {"unknown", "any"}:
        return _fail("still degraded")
    return _ok()


# 9

def validate_jsx_intrinsic_keyof(s: str) -> Tuple[bool, str]:
    if _strip(s) == "keyof JSX.IntrinsicElements":
        return _ok()
    return _fail("expected keyof JSX.IntrinsicElements")


# 10
_RE_STR_LIT = r"(?:'[^']*'|\"[^\"]*\")"
_RE_STR_UNION = re.compile(rf"^\s*{_RE_STR_LIT}(\s*\|\s*{_RE_STR_LIT})+\s*$")


def validate_string_literal_union(s: str) -> Tuple[bool, str]:
    if _RE_STR_UNION.match(s):
        return _ok()
    return _fail("expected string literal union")


# 11

def validate_template_literal_type(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    if "`" in s:
        return _ok()
    return _fail("expected template literal type")


# 11b
_RE_HTML_SPECIFIC = re.compile(r"^HTML\w+Element$")


def validate_html_specific_element(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    if not _RE_HTML_SPECIFIC.match(s):
        return _fail("expected HTML<Specific>Element")
    if s == "HTMLElement":
        return _fail("still degraded")
    return _ok()


# 11c
_RE_HTML_SPECIFIC_NULLABLE = re.compile(r"^HTML\w+Element\s*\|\s*null$")


def validate_html_specific_element_nullable(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    if not _RE_HTML_SPECIFIC_NULLABLE.match(s):
        return _fail("expected HTML<Specific>Element | null")
    if s == "HTMLElement | null":
        return _fail("still degraded")
    return _ok()


# 11d
_RE_CUSTOM_EVENT = re.compile(r"^CustomEvent<(.+)>$")


def validate_custom_event(s: str) -> Tuple[bool, str]:
    m = _RE_CUSTOM_EVENT.match(_strip(s))
    if not m:
        return _fail("expected CustomEvent<…>")
    if m.group(1).strip() in {"unknown", "any"}:
        return _fail("still degraded")
    return _ok()


# 11e
_RE_RECORD_STRING_VALUE = re.compile(r"^Record<\s*string\s*,\s*(.+)\s*>$")


def validate_record_string_value(s: str) -> Tuple[bool, str]:
    m = _RE_RECORD_STRING_VALUE.match(_strip(s))
    if not m:
        return _fail("expected Record<string, …>")
    if m.group(1).strip() in {"unknown", "any"}:
        return _fail("still degraded")
    return _ok()


# 11f
_RE_MAP = re.compile(r"^Map<\s*(.+?)\s*,\s*(.+?)\s*>$")


def validate_map(s: str) -> Tuple[bool, str]:
    m = _RE_MAP.match(_strip(s))
    if not m:
        return _fail("expected Map<K, V>")
    a, b = m.group(1).strip(), m.group(2).strip()
    if a == "unknown" and b == "unknown":
        return _fail("still degraded")
    return _ok()


# 11g
_RE_SET = re.compile(r"^Set<\s*(.+?)\s*>$")


def validate_set(s: str) -> Tuple[bool, str]:
    m = _RE_SET.match(_strip(s))
    if not m:
        return _fail("expected Set<T>")
    if m.group(1).strip() in {"unknown", "any"}:
        return _fail("still degraded")
    return _ok()


# 11h
def validate_dom_add_event_listener_options(s: str) -> Tuple[bool, str]:
    if _strip(s) == "AddEventListenerOptions":
        return _ok()
    return _fail("expected AddEventListenerOptions")


# 12

def validate_conditional_type(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    if "extends" in s and "?" in s and ":" in s:
        return _ok()
    return _fail("expected conditional type")


# 13
_RE_INDEXED_ACCESS = re.compile(r"^[A-Za-z0-9_$.<>,\s]+\[[^\]]+\]$")


def validate_indexed_access_type(s: str) -> Tuple[bool, str]:
    s = _strip(s)
    if s.endswith("[]"):
        return _fail("array syntax, not indexed access")
    if _RE_INDEXED_ACCESS.match(s):
        return _ok()
    return _fail("expected indexed access type")


# 14
_RE_UTILITY = re.compile(r"^(Extract|Exclude|Pick|Omit|Partial|Required|Readonly|NonNullable|Parameters|ReturnType|InstanceType|Awaited)<.+>$")


def validate_utility_type(s: str) -> Tuple[bool, str]:
    if _RE_UTILITY.match(_strip(s)):
        return _ok()
    return _fail("expected utility type")


# 14b
def validate_dom_mutation_observer_init(s: str) -> Tuple[bool, str]:
    if _strip(s) == "MutationObserverInit":
        return _ok()
    return _fail("expected MutationObserverInit")


# 14c
def validate_dom_intersection_observer_init(s: str) -> Tuple[bool, str]:
    if _strip(s) == "IntersectionObserverInit":
        return _ok()
    return _fail("expected IntersectionObserverInit")


# 14d
def validate_dom_shadow_root_init(s: str) -> Tuple[bool, str]:
    if _strip(s) == "ShadowRootInit":
        return _ok()
    return _fail("expected ShadowRootInit")


# 14e
def validate_dom_css_style_declaration(s: str) -> Tuple[bool, str]:
    if _strip(s) == "CSSStyleDeclaration":
        return _ok()
    return _fail("expected CSSStyleDeclaration")


# 14f
_RE_ELEMENT_INTERNALS_INTERSECTION = re.compile(r"^ElementInternals\s*&\s*.+$")


def validate_dom_element_internals_intersection(s: str) -> Tuple[bool, str]:
    if _RE_ELEMENT_INTERNALS_INTERSECTION.match(_strip(s)):
        return _ok()
    return _fail("expected ElementInternals intersection type")


# 15
_RE_PROMISE = re.compile(r"^Promise<(.+)>$")


def validate_promise(s: str) -> Tuple[bool, str]:
    m = _RE_PROMISE.match(_strip(s))
    if not m:
        return _fail("expected Promise<…>")
    inner = m.group(1).strip()
    if inner in _SIMPLE_TYPES:
        return _fail("still too generic/simple")
    return _ok()


# 16
_RE_READONLY_ARRAY = re.compile(r"^ReadonlyArray<(.+)>$")


def validate_readonly_array(s: str) -> Tuple[bool, str]:
    m = _RE_READONLY_ARRAY.match(_strip(s))
    if not m:
        return _fail("expected ReadonlyArray<…>")
    inner = m.group(1).strip()
    if inner in {"unknown", "any"}:
        return _fail("still degraded")
    return _ok()


# 17
_RE_USE_QUERY_RESULT = re.compile(r"^UseQueryResult<\s*(.+?)\s*,\s*(.+?)\s*>$")


def validate_tanstack_use_query_result(s: str) -> Tuple[bool, str]:
    m = _RE_USE_QUERY_RESULT.match(_strip(s))
    if not m:
        return _fail("expected UseQueryResult<TData, TError>")
    a, b = m.group(1).strip(), m.group(2).strip()
    if a == "unknown" and b == "unknown":
        return _fail("still degraded")
    return _ok()


# 18
_RE_USE_INF_QUERY_RESULT = re.compile(r"^UseInfiniteQueryResult<\s*(.+?)\s*,\s*(.+?)\s*>$")


def validate_tanstack_use_infinite_query_result(s: str) -> Tuple[bool, str]:
    m = _RE_USE_INF_QUERY_RESULT.match(_strip(s))
    if not m:
        return _fail("expected UseInfiniteQueryResult<TData, TError>")
    a, b = m.group(1).strip(), m.group(2).strip()
    if a == "unknown" and b == "unknown":
        return _fail("still degraded")
    return _ok()


# 18b
_RE_QUERY_OBSERVER_RESULT = re.compile(r"^QueryObserverResult<\s*(.+?)\s*,\s*(.+?)\s*>$")


def validate_tanstack_query_observer_result(s: str) -> Tuple[bool, str]:
    m = _RE_QUERY_OBSERVER_RESULT.match(_strip(s))
    if not m:
        return _fail("expected QueryObserverResult<TData, TError>")
    a, b = m.group(1).strip(), m.group(2).strip()
    if a == "unknown" and b == "unknown":
        return _fail("still degraded")
    return _ok()


# 18c
_RE_INFINITE_DATA = re.compile(r"^InfiniteData<\s*(.+?)(?:\s*,\s*(.+?)\s*)?>$")


def validate_tanstack_infinite_data(s: str) -> Tuple[bool, str]:
    m = _RE_INFINITE_DATA.match(_strip(s))
    if not m:
        return _fail("expected InfiniteData<TData[, TPageParam]>")
    a = m.group(1).strip()
    b = m.group(2).strip() if m.group(2) else None
    if a == "unknown" and (b is None or b == "unknown"):
        return _fail("still degraded")
    return _ok()


# 18cc
_RE_INFINITE_QUERY_OBSERVER_RESULT = re.compile(r"^InfiniteQueryObserverResult<\s*(.+?)\s*,\s*(.+?)\s*>$")


def validate_tanstack_infinite_query_observer_result(s: str) -> Tuple[bool, str]:
    m = _RE_INFINITE_QUERY_OBSERVER_RESULT.match(_strip(s))
    if not m:
        return _fail("expected InfiniteQueryObserverResult<TData, TError>")
    a, b = m.group(1).strip(), m.group(2).strip()
    if a == "unknown" and b == "unknown":
        return _fail("still degraded")
    return _ok()


# 18d
_RE_QUERY_FUNCTION_CONTEXT = re.compile(r"^QueryFunctionContext<\s*(.+)\s*>$")


def validate_tanstack_query_function_context(s: str) -> Tuple[bool, str]:
    m = _RE_QUERY_FUNCTION_CONTEXT.match(_strip(s))
    if not m:
        return _fail("expected QueryFunctionContext<TQueryKey>")
    if m.group(1).strip() == "unknown":
        return _fail("still degraded")
    return _ok()


# 18e
_RE_ASTRO_INFER_GET_STATIC_PROPS = re.compile(r"^InferGetStaticPropsType<(.+)>$")


def validate_astro_infer_get_static_props_type(s: str) -> Tuple[bool, str]:
    m = _RE_ASTRO_INFER_GET_STATIC_PROPS.match(_strip(s))
    if not m:
        return _fail("expected InferGetStaticPropsType<…>")
    if m.group(1).strip() == "unknown":
        return _fail("still degraded")
    return _ok()


# 18f
_RE_ASTRO_INFER_GET_STATIC_PATHS = re.compile(r"^InferGetStaticPathsType<(.+)>$")


def validate_astro_infer_get_static_paths_type(s: str) -> Tuple[bool, str]:
    m = _RE_ASTRO_INFER_GET_STATIC_PATHS.match(_strip(s))
    if not m:
        return _fail("expected InferGetStaticPathsType<…>")
    if m.group(1).strip() == "unknown":
        return _fail("still degraded")
    return _ok()


# 18g
def validate_astro_api_route(s: str) -> Tuple[bool, str]:
    if _strip(s) == "APIRoute":
        return _ok()
    return _fail("expected APIRoute")


# 18h
def validate_astro_get_static_paths(s: str) -> Tuple[bool, str]:
    if _strip(s) == "GetStaticPaths":
        return _ok()
    return _fail("expected GetStaticPaths")


# 19
_RE_ASTRO_COLLECTION_ENTRY = re.compile(r"^CollectionEntry<(.+)>$")


def validate_astro_collection_entry(s: str) -> Tuple[bool, str]:
    m = _RE_ASTRO_COLLECTION_ENTRY.match(_strip(s))
    if not m:
        return _fail("expected CollectionEntry<…>")
    if m.group(1).strip() == "any":
        return _fail("still degraded")
    return _ok()


MUTED_RULES = {
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
}

ALL_VALIDATORS: Dict[str, Validator] = {
    "react_event_handler→generic": validate_react_event_handler,
    "react_specific_event_handler_alias→generic": validate_react_specific_event_handler_alias,
    "react_event→synthetic": validate_react_event,
    "react_component_props_with_ref→any": validate_react_component_props_with_ref,
    "react_component_props_without_ref→any": validate_react_component_props_without_ref,
    "react_element_ref→any": validate_react_element_ref,
    "react_refobject→unknown": validate_react_refobject,
    "react_mutable_refobject→unknown": validate_react_mutable_refobject,
    "react_dispatch_setstateaction→unknown": validate_react_dispatch_setstateaction,
    "jsx_intrinsic_keyof→string": validate_jsx_intrinsic_keyof,
    "string_literal_union→string": validate_string_literal_union,
    "template_literal_type→string": validate_template_literal_type,
    "html_specific_element→html_element": validate_html_specific_element,
    "html_specific_element_nullable→html_element_nullable": validate_html_specific_element_nullable,
    "custom_event→unknown": validate_custom_event,
    "record_string_value→unknown": validate_record_string_value,
    "map→unknown": validate_map,
    "set→unknown": validate_set,
    "dom_add_event_listener_options→event_listener_options": validate_dom_add_event_listener_options,
    "conditional_type→unknown": validate_conditional_type,
    "indexed_access_type→unknown": validate_indexed_access_type,
    "utility_type→unknown": validate_utility_type,
    "dom_mutation_observer_init→unknown": validate_dom_mutation_observer_init,
    "dom_intersection_observer_init→unknown": validate_dom_intersection_observer_init,
    "dom_shadow_root_init→unknown": validate_dom_shadow_root_init,
    "dom_css_style_declaration→unknown": validate_dom_css_style_declaration,
    "dom_element_internals_intersection→unknown": validate_dom_element_internals_intersection,
    "promise→unknown": validate_promise,
    "readonly_array→unknown": validate_readonly_array,
    "tanstack_use_query_result→unknown": validate_tanstack_use_query_result,
    "tanstack_use_infinite_query_result→unknown": validate_tanstack_use_infinite_query_result,
    "tanstack_query_observer_result→unknown": validate_tanstack_query_observer_result,
    "tanstack_infinite_data→unknown": validate_tanstack_infinite_data,
    "tanstack_infinite_query_observer_result→unknown": validate_tanstack_infinite_query_observer_result,
    "tanstack_query_function_context→unknown": validate_tanstack_query_function_context,
    "astro_infer_get_static_props_type→unknown": validate_astro_infer_get_static_props_type,
    "astro_infer_get_static_paths_type→unknown": validate_astro_infer_get_static_paths_type,
    "astro_api_route→unknown": validate_astro_api_route,
    "astro_get_static_paths→unknown": validate_astro_get_static_paths,
    "astro_collection_entry→any": validate_astro_collection_entry,
}

VALIDATORS: Dict[str, Validator] = {
    rule: validator
    for rule, validator in ALL_VALIDATORS.items()
    if rule not in MUTED_RULES
}
