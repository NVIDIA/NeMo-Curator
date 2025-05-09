#!/bin/bash
set -uo pipefail

LOG_FILE="$1"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "❌ HF_TOKEN is not set. Skipping test." >> "$LOG_FILE"
  exit 0
else
  echo "✅ HF_TOKEN is set"
fi

# Create a JSONL file with a single text
INPUT_FILE="/tmp/instruction_data_guard_dataset/data.jsonl"
mkdir -p "$(dirname "$INPUT_FILE")"
> "$INPUT_FILE"
texts=(
    "Instruction: Find a route between San Diego and Phoenix which passes through Nevada. Input: . Response: Drive to Las Vegas with highway 15 and from there drive to Phoenix with highway 93."
)
for text in "${texts[@]}"; do
    echo "{\"text\": \"$text\"}" >> "$INPUT_FILE"
done
echo "✅ JSONL file '$INPUT_FILE' successfully created."

INPUT_DIR="/tmp/instruction_data_guard_dataset"
OUTPUT_DIR="/tmp/instruction_data_guard_classifier_inference_output"
mkdir -p "$OUTPUT_DIR"

instruction_data_guard_classifier_inference \
  --input-data-dir "$INPUT_DIR" \
  --input-file-type "jsonl" \
  --input-text-field "text" \
  --output-data-dir "$OUTPUT_DIR" \
  --token "$HF_TOKEN" \
  > "$LOG_FILE" 2>&1
EXIT_CODE=$?

exit $EXIT_CODE
