"""
Prompt formatting for ts-type-refiner.

We prepend compact, deterministic metadata to each source context so the model
can disambiguate candidates that would otherwise have identical code windows
(e.g. multiple `unknown`/`string` annotations on one line).
"""

from __future__ import annotations


def build_refine_prompt(
    *,
    context: str,
    name: str,
    kind: str,
    rule: str,
    degraded_type: str,
    file: str | None = None,
    line: int | None = None,
) -> str:
    meta = [
        f"rule={rule}",
        f"kind={kind}",
        f"name={name}",
        f"degraded={degraded_type}",
    ]
    if file:
        meta.append(f"file={file}")
    if line is not None:
        meta.append(f"line={line}")

    # Fixed delimiter token makes parsing easy for both human debugging and model.
    return "[REFINE " + " | ".join(meta) + "]\n" + context
