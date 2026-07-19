#!/bin/bash
# Download refiner.pt checkpoint from Google Drive using gdown

FILE_ID="1mY2bXLAsJ8aq_sADcDDzeG9RqfPwtl6H"
OUTPUT_PATH="packages/ts-type-refiner/checkpoints/refiner.pt"
BACKUP_PATH="packages/ts-type-refiner/checkpoints/refiner.from-google-drive.pt"

# Create directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Check if gdown is installed, if not install it with uv
if ! command -v gdown &> /dev/null; then
    echo "❌ gdown not found. Installing with uv..."
    uv pip install gdown
fi

# Check if file already exists and is valid (not HTML error page)
if [ -f "$OUTPUT_PATH" ]; then
    # Check if it's a valid PyTorch file (not HTML error)
    if file "$OUTPUT_PATH" | grep -q "HTML"; then
        echo "⚠️  Invalid checkpoint (HTML error page), re-downloading..."
        rm "$OUTPUT_PATH"
    else
        echo "✓ Backing up existing refiner.pt → refiner.from-google-drive.pt"
        mv "$OUTPUT_PATH" "$BACKUP_PATH"
    fi
fi

echo "📥 Downloading refiner.pt from Google Drive using gdown..."
uv run gdown "$FILE_ID" -O "$OUTPUT_PATH" --quiet

# Verify download
if [ ! -f "$OUTPUT_PATH" ]; then
    echo "✗ Download failed!"
    exit 1
fi

if file "$OUTPUT_PATH" | grep -q "HTML"; then
    echo "✗ Download failed: received HTML error page"
    exit 1
fi

SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
echo "✓ Download complete: $OUTPUT_PATH ($SIZE)"
