#!/bin/bash
# Download refiner.pt checkpoint from Google Drive

FILE_ID="1mY2bXLAsJ8aq_sADcDDzeG9RqfPwtl6H"
OUTPUT_PATH="packages/ts-type-refiner/checkpoints/refiner.pt"
BACKUP_PATH="packages/ts-type-refiner/checkpoints/refiner.from-google-drive.pt"

# Create directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Check if file already exists
if [ -f "$OUTPUT_PATH" ]; then
    echo "✓ Backing up existing refiner.pt → refiner.from-google-drive.pt"
    mv "$OUTPUT_PATH" "$BACKUP_PATH"
fi

echo "📥 Downloading refiner.pt from Google Drive..."
curl -L "https://drive.google.com/uc?id=$FILE_ID&export=download" -o "$OUTPUT_PATH"

if [ -f "$OUTPUT_PATH" ]; then
    echo "✓ Download complete: $OUTPUT_PATH"
else
    echo "✗ Download failed!"
    exit 1
fi
