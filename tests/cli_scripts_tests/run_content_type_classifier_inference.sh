#!/bin/bash
set -uo pipefail

LOG_FILE="$1"

# Create a JSONL file with a single text
INPUT_FILE="/tmp/content_type_dataset/data.jsonl"
mkdir -p "$(dirname "$INPUT_FILE")"
> "$INPUT_FILE"
texts=(
    "Hi, great video! I am now a subscriber."
)
for text in "${texts[@]}"; do
    echo "{\"text\": \"$text\"}" >> "$INPUT_FILE"
done
echo "✅ JSONL file '$INPUT_FILE' successfully created."

INPUT_DIR="/tmp/content_type_dataset"
OUTPUT_DIR="/tmp/content_type_classifier_inference_output"
mkdir -p "$OUTPUT_DIR"

content_type_classifier_inference \
  --input-data-dir "$INPUT_DIR" \
  --input-file-type "jsonl" \
  --input-text-field "text" \
  --output-data-dir "$OUTPUT_DIR" \
  > "$LOG_FILE" 2>&1
EXIT_CODE=$?

exit $EXIT_CODE
