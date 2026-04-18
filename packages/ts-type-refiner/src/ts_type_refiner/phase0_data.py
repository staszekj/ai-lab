"""
Phase 0 — Proof-of-concept training pairs.

Hardcoded (source_context, target_type) pairs for overfitting test.
Source = TypeScript code with a degraded (generic) type.
Target = the precise type the model should generate.
"""

TRAINING_PAIRS: list[tuple[str, str]] = [
    # ── string literal unions ────────────────────────────────────────
    (
        "let enabled : string = 'on'",
        "'on' | 'off'",
    ),
    (
        "const direction : string = 'ltr'",
        "'ltr' | 'rtl'",
    ),
    (
        "const size : string = 'md'",
        "'sm' | 'md' | 'lg'",
    ),
    (
        "const color : string = 'red'",
        "'red' | 'green' | 'blue'",
    ),
    (
        "function f ( action : string )",
        "'click' | 'hover'",
    ),
    # ── slightly different patterns ──────────────────────────────────
    (
        "let x : string = 'a'",
        "'a' | 'b' | 'c'",
    ),
    (
        "const action : string = 'realClick'",
        "'realClick' | 'realTouch'",
    ),
    (
        "var name : string = 'on'",
        "'on' | 'off'",
    ),
]
