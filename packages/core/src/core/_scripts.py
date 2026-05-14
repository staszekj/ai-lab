"""
Entry points for uv scripts.

Usage from monorepo root:
    uv run --package core presentation-encoder-decoder
"""

import runpy


def presentation_encoder_decoder() -> None:
    runpy.run_module("core.presentation_encoder_decoder", run_name="__main__", alter_sys=True)
