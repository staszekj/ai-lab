"""
Prompt formatting for ts-type-refiner.

We prepend compact, deterministic metadata to each source context so the model
can disambiguate candidates that would otherwise have identical code windows
(e.g. multiple `unknown`/`string` annotations on one line).

Versioning: bump :data:`PROMPT_VERSION` whenever the wire format changes.
The trainer stamps the value into the checkpoint so inference can refuse to
run against a mismatched model. Mirror in `degrade.ts::PROMPT_VERSION`.
"""

from __future__ import annotations

# v2: replaced legacy `ast=` meta key with `siblings=`, added explicit `\n---\n`
#     separator between metadata and code so the parser can't be confused
#     when the code window starts with `[`.
PROMPT_VERSION = 2


def build_refine_prompt(
    *,
    context: str,
    name: str,
    kind: str,
    rule: str,
    degraded_type: str,
    siblings: str | None = None,
) -> str:
    meta = [
        f"rule={rule}",
        f"kind={kind}",
        f"name={name}",
        f"degraded={degraded_type}",
    ]
    if siblings:
        meta.append(f"siblings={siblings}")

    return "[REFINE " + " | ".join(meta) + "]\n---\n" + context
