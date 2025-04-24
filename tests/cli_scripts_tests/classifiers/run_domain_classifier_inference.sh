#!/bin/bash
set -uo pipefail

LOG_FILE="$1"

INPUT_DIR="/tmp/domain_dataset"
OUTPUT_DIR="/tmp/domain_classifier_inference_output"
mkdir -p "$OUTPUT_DIR"

domain_classifier_inference \
  --input-data-dir "$INPUT_DIR" \
  --input-file-type "jsonl" \
  --input-text-field "text" \
  --output-data-dir "$OUTPUT_DIR" \
  > "$LOG_FILE" 2>&1
EXIT_CODE=$?

exit $EXIT_CODE
