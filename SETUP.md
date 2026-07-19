## Downloading the Checkpoint

The project requires `refiner.pt` — a pre-trained TypeScript type refiner model.

### Manual Download (Recommended)

If automatic download fails or you prefer manual control:

1. Open: https://drive.google.com/file/d/1mY2bXLAsJ8aq_sADcDDzeG9RqfPwtl6H/view
2. Click "Download anyway" (Google will warn about file size)
3. Move downloaded `refiner.pt` to:
   ```
   packages/ts-type-refiner/checkpoints/refiner.pt
   ```

**Size:** ~33 MB

### Automatic Download (Optional)

Alternative method using `gdown`:

```bash
pnpm fetch-pt
```

This script:
- Downloads from Google Drive automatically
- Backs up existing checkpoint with `from-google-drive` suffix
- Installs `gdown` if needed
- Places file at: `packages/ts-type-refiner/checkpoints/refiner.pt`

### Quick Start

```bash
# 1. Download checkpoint
pnpm fetch-pt

# 2. Run the refiner
pnpm refine

# 3. Check types compile
pnpm typecheck
```
