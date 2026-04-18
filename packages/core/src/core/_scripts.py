"""
Entry points for uv scripts.

Usage from monorepo root:
    uv run --package core presentation-mini-gpt
    uv run --package core presentation-transformer-block
    uv run --package core presentation-decoder-block
    uv run --package core presentation-encoder-decoder
    uv run --package core presentation-loss-backward

    uv run --package core manual-mini-gpt
    uv run --package core manual-transformer-block
    uv run --package core manual-decoder-block
    uv run --package core manual-encoder-decoder
    uv run --package core manual-loss-backward
"""

import runpy


# ── presentation ─────────────────────────────────────────────────────

def presentation_mini_gpt() -> None:
    runpy.run_module("core.presentation_mini_gpt", run_name="__main__", alter_sys=True)


def presentation_transformer_block() -> None:
    runpy.run_module("core.presentation_transformer_block", run_name="__main__", alter_sys=True)


def presentation_decoder_block() -> None:
    runpy.run_module("core.presentation_decoder_block", run_name="__main__", alter_sys=True)


def presentation_encoder_decoder() -> None:
    runpy.run_module("core.presentation_encoder_decoder", run_name="__main__", alter_sys=True)


def presentation_loss_backward() -> None:
    runpy.run_module("core.presentation_loss_backward", run_name="__main__", alter_sys=True)


# ── manual ────────────────────────────────────────────────────────────

def manual_mini_gpt() -> None:
    runpy.run_module("core.manual_mini_gpt", run_name="__main__", alter_sys=True)


def manual_transformer_block() -> None:
    runpy.run_module("core.manual_transformer_block", run_name="__main__", alter_sys=True)


def manual_decoder_block() -> None:
    runpy.run_module("core.manual_decoder_block", run_name="__main__", alter_sys=True)


def manual_encoder_decoder() -> None:
    runpy.run_module("core.manual_encoder_decoder", run_name="__main__", alter_sys=True)


def manual_loss_backward() -> None:
    runpy.run_module("core.manual_loss_backward", run_name="__main__", alter_sys=True)
