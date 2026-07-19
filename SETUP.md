# Setup Guide

## PyTorch Configuration

This project supports three configurations:

- **CPU** (default) — works everywhere, slower
- **CUDA 12.1** — Linux/Windows with NVIDIA GPU, fast
- **CPU-only (macOS)** — ARM64 optimized, single-threaded

### Setup Instructions

#### Option 1: CPU (Default)
```bash
# Uses pyproject.toml (CPU-only config)
uv lock
```

#### Option 2: CUDA (Linux/Windows with GPU)
```bash
cp packages/ts-type-refiner/pyproject-cuda.toml packages/ts-type-refiner/pyproject.toml
uv lock
```

Then restore to CPU default:
```bash
cp packages/ts-type-refiner/pyproject-cpu.toml packages/ts-type-refiner/pyproject.toml
uv lock
```

### Switching to CUDA

Only on Linux/Windows with NVIDIA GPU:

```bash
cp packages/ts-type-refiner/pyproject-cuda.toml packages/ts-type-refiner/pyproject.toml
uv lock
# Now training/inference will use CUDA
```

### Inference Performance

CPU device is ~25x faster than MPS on macOS for small models:
- **CPU**: 2.5 cand/s
- **MPS**: 0.1 cand/s

See `packages/ts-type-refiner/src/ts_type_refiner/inference/infer.py` for device selection logic.

## Downloading the Checkpoint

The project requires `refiner.pt` — a pre-trained TypeScript type refiner model.

### Automatic Download (Recommended)

```bash
pnpm fetch-pt
```

This script:
- Downloads from Google Drive automatically
- Backs up existing checkpoint with `from-google-drive` suffix
- Installs `gdown` if needed
- Places file at: `packages/ts-type-refiner/checkpoints/refiner.pt`

### Manual Download

If automatic download fails:

1. Open: https://drive.google.com/file/d/1mY2bXLAsJ8aq_sADcDDzeG9RqfPwtl6H/view
2. Click "Download anyway" (Google will warn about file size)
3. Move downloaded `refiner.pt` to:
   ```
   packages/ts-type-refiner/checkpoints/refiner.pt
   ```

**Size:** ~33 MB

### Quick Start

```bash
# 1. Download checkpoint
pnpm fetch-pt

# 2. Run the refiner
pnpm refine

# 3. Check types compile
pnpm typecheck
```
