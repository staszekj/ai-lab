"""
Checkpoint helpers — the ONLY module that touches `torch.save / load`.

A "checkpoint" in this codebase is a single `.pt` file containing:

    {
        "model_state_dict": <Tensor weights>,
        "model_config":     <dict — the EncoderDecoderConfig fields>,
        ... arbitrary user-supplied extras (epoch, val_acc, optimizer
            state, anything else the caller wants persisted) ...
    }

The `model_config` dict is mandatory because state_dict shapes are
not enough to rebuild the module: e.g. `(vocab, d_model)` is recorded
in the embedding tensor's shape but `num_heads`, `num_layers`,
`max_seq_len` are not — they must be saved explicitly.

Centralising save / load here means there is exactly ONE place in the
codebase that needs to migrate when the checkpoint format evolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Union

import torch

from .encoder_decoder_model import EncoderDecoderConfig, EncoderDecoderModel


PathLike = Union[str, Path]


# ══════════════════════════════════════════════════════════════════════
# Loaded checkpoint container
# ══════════════════════════════════════════════════════════════════════

@dataclass
class LoadedCheckpoint:
    """
    A parsed checkpoint, with the three structural fields we always
    care about pulled out explicitly. Everything else (`epoch`,
    `val_accuracy`, `optimizer_state_dict`, …) lives in `extras`.
    """

    model_config: Dict[str, Any]
    state_dict:   Dict[str, Any]
    extras:       Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════

def save(
    model: EncoderDecoderModel,
    path: PathLike,
    *,
    model_config: Dict[str, Any],
    **extras: Any,
) -> None:
    """
    Persist `model` to `path` along with `model_config` (required) and
    any number of arbitrary `extras` (validation accuracy, epoch number,
    optimizer state, …).

    `model_config` is taken as an explicit argument rather than read
    from `model.cfg` so callers stay in full control of WHAT they
    serialize — e.g. they may want to record the *requested* hyper-
    parameters, which include things derived at training time
    (`max_seq_len = max(observed)+4`).
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config":     dict(model_config),
        **extras,
    }
    torch.save(payload, path)


# ══════════════════════════════════════════════════════════════════════
# Load
# ══════════════════════════════════════════════════════════════════════

def load(path: PathLike, device: torch.device) -> LoadedCheckpoint:
    """
    Read a checkpoint produced by `save` (or by an older training
    script that wrote the same three keys directly). The returned
    `LoadedCheckpoint` is a structural view — pass `state_dict` to
    `model.load_state_dict`, pass `model_config` to `build_model`.
    """

    # `weights_only=False` is needed because we serialise plain Python
    # dicts (model_config, extras), not just tensors. Trusted-source
    # only — these checkpoints are produced by our own training code.
    raw = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" not in raw or "model_config" not in raw:
        raise ValueError(
            f"Checkpoint at {path} is missing required keys "
            f"'model_state_dict' and/or 'model_config'. "
            f"Found keys: {sorted(raw)}"
        )

    state_dict   = raw["model_state_dict"]
    model_config = dict(raw["model_config"])
    extras       = {k: v for k, v in raw.items()
                    if k not in ("model_state_dict", "model_config")}

    return LoadedCheckpoint(
        model_config = model_config,
        state_dict   = state_dict,
        extras       = extras,
    )


# ══════════════════════════════════════════════════════════════════════
# Build
# ══════════════════════════════════════════════════════════════════════

def build_model(
    model_config: Dict[str, Any],
    device: torch.device | None = None,
) -> EncoderDecoderModel:
    """
    Construct an `EncoderDecoderModel` from a `model_config` dict.

    `model_config` must contain exactly the fields of
    `EncoderDecoderConfig` (vocab_size, max_seq_len, d_model,
    num_heads, d_ff, num_layers). Extra keys are rejected so we never
    silently drop typo'd hyper-parameters.
    """

    cfg = EncoderDecoderConfig(**model_config)
    model = EncoderDecoderModel(cfg)
    if device is not None:
        model = model.to(device)
    return model
