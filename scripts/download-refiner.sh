#!/bin/bash
# Download refiner artifacts (checkpoint + tokenizer) from Google Drive using gdown.
# If local checkpoints already exist, archive the whole directory with a
# timestamp suffix before replacing it with freshly downloaded artifacts.

set -euo pipefail

CHECKPOINTS_FOLDER_URL="https://drive.google.com/drive/folders/1198fm78bonAy-A4FfH4jh6-kPHpburDR?usp=drive_link"

CHECKPOINTS_DIR="packages/ts-type-refiner/checkpoints"
CHECKPOINT_PATH="${CHECKPOINTS_DIR}/refiner.pt"
TOKENIZER_PATH="${CHECKPOINTS_DIR}/tokenizer.json"
TMP_DOWNLOAD_DIR="packages/ts-type-refiner/.checkpoints_download_tmp"

mkdir -p "$(dirname "$CHECKPOINTS_DIR")"

if ! command -v gdown &> /dev/null; then
    echo "❌ gdown not found. Installing with uv..."
    uv pip install gdown
fi

cleanup_tmp() {
    rm -rf "$TMP_DOWNLOAD_DIR"
}

trap cleanup_tmp EXIT

mkdir -p "$TMP_DOWNLOAD_DIR"

echo "📥 Downloading checkpoints folder from Google Drive..."
uv run gdown --folder "$CHECKPOINTS_FOLDER_URL" -O "$TMP_DOWNLOAD_DIR" --quiet

DOWNLOADED_CHECKPOINT="${TMP_DOWNLOAD_DIR}/refiner.pt"
DOWNLOADED_TOKENIZER="${TMP_DOWNLOAD_DIR}/tokenizer.json"

if [ ! -f "$DOWNLOADED_CHECKPOINT" ]; then
    echo "✗ Missing refiner.pt in downloaded folder"
    exit 1
fi

if [ ! -f "$DOWNLOADED_TOKENIZER" ]; then
    echo "✗ Missing tokenizer.json in downloaded folder"
    exit 1
fi

if file "$DOWNLOADED_CHECKPOINT" | grep -q "HTML"; then
    echo "✗ Download failed for refiner.pt: received HTML error page"
    exit 1
fi

if file "$DOWNLOADED_TOKENIZER" | grep -q "HTML"; then
    echo "✗ Download failed for tokenizer.json: received HTML error page"
    exit 1
fi

# Validate tokenizer JSON before touching active files.
python3 -m json.tool "$DOWNLOADED_TOKENIZER" >/dev/null

# Archive current checkpoints directory with a timestamp if it exists.
if [ -d "$CHECKPOINTS_DIR" ]; then
    TS=$(date +"%Y%m%d-%H%M%S")
    ARCHIVE_DIR="${CHECKPOINTS_DIR}-${TS}"
    echo "✓ Archiving existing checkpoints directory → ${ARCHIVE_DIR}"
    mv "$CHECKPOINTS_DIR" "$ARCHIVE_DIR"
fi

mkdir -p "$CHECKPOINTS_DIR"
cp "$DOWNLOADED_CHECKPOINT" "$CHECKPOINT_PATH"
cp "$DOWNLOADED_TOKENIZER" "$TOKENIZER_PATH"

CHECKPOINT_SIZE=$(du -h "$CHECKPOINT_PATH" | cut -f1)
TOKENIZER_SIZE=$(du -h "$TOKENIZER_PATH" | cut -f1)

echo "✓ Download complete: $CHECKPOINT_PATH ($CHECKPOINT_SIZE)"
echo "✓ Download complete: $TOKENIZER_PATH ($TOKENIZER_SIZE)"
echo "✓ Pair ready: checkpoint and tokenizer are synchronized"
