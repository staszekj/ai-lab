#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
uv run python3 -m core.presentation_transformer_block
