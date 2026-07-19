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

### Lock Files

- **`uv-cuda.lock`** — committed, from GPU machine. Use with CUDA config.
- **`uv.lock`** — **local only**, generated per-platform by `uv lock`
- **`uv-cpu.lock`** — **local only**, alternative CPU naming

Each platform generates its own lock file to ensure wheel compatibility:
- macOS ARM64: `pytorch==2.5.1` (CPU)
- Linux x86_64: `pytorch==2.5.1` (CPU)
- Windows x64: `pytorch==2.5.1` (CPU)

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
