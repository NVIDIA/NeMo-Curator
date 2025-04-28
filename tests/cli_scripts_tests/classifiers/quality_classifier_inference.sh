#!/bin/bash
set -uo pipefail

LOG_FILE="$1"

# Create a JSONL file with a single text
INPUT_FILE="/tmp/quality_dataset/data.jsonl"
mkdir -p "$(dirname "$INPUT_FILE")"
> "$INPUT_FILE"
texts=(
    "Traveling to Europe during the off-season can be a more budget-friendly option."
)
for text in "${texts[@]}"; do
    echo "{\"text\": \"$text\"}" >> "$INPUT_FILE"
done
echo "✅ JSONL file '$INPUT_FILE' successfully created."

INPUT_DIR="/tmp/quality_dataset"
OUTPUT_DIR="/tmp/quality_classifier_inference_output"
mkdir -p "$OUTPUT_DIR"

quality_classifier_inference \
  --input-data-dir "$INPUT_DIR" \
  --input-file-type "jsonl" \
  --input-text-field "text" \
  --output-data-dir "$OUTPUT_DIR" \
  > "$LOG_FILE" 2>&1
EXIT_CODE=$?

exit $EXIT_CODE
