## Downloading Model Artifacts

The project requires two paired artifacts:

- `packages/ts-type-refiner/checkpoints/refiner.pt`
- `packages/ts-type-refiner/checkpoints/tokenizer.json`

`refiner.pt` and `tokenizer.json` are an inseparable pair. Never update only one.

### Manual Download (Recommended)

If automatic download fails or you prefer manual control:

1. Open folder:
   https://drive.google.com/drive/folders/1198fm78bonAy-A4FfH4jh6-kPHpburDR
2. Download these two files from that folder:
   - `refiner.pt`
   - `tokenizer.json`
3. Put `refiner.pt` here:
   ```
   packages/ts-type-refiner/checkpoints/refiner.pt
   ```
4. Put `tokenizer.json` here:
   ```
   packages/ts-type-refiner/checkpoints/tokenizer.json
   ```

Both files must come from the same release pair.

**Size:** ~33 MB

### Automatic Download (Optional)

Alternative method using `gdown`:

```bash
pnpm fetch-pt
```

This script:
- Downloads `refiner.pt` and `tokenizer.json` from the shared Google Drive `checkpoints` folder
- Downloads both files to temporary paths and only then swaps them into place
- If local `checkpoints/` exists, archives it as `checkpoints-YYYYMMDD-HHMMSS`
- Installs `gdown` if needed
- Places files at:
   - `packages/ts-type-refiner/checkpoints/refiner.pt`
   - `packages/ts-type-refiner/checkpoints/tokenizer.json`

Tokenizer path used by `refiner-infer` and `pnpm refine`:
`packages/ts-type-refiner/checkpoints/tokenizer.json`

### Quick Start

```bash
# 1. Download artifact pair
pnpm fetch-pt

# 2. Run the refiner
pnpm refine

# 3. Check types compile
pnpm typecheck
```
