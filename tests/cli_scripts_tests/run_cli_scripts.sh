#!/bin/bash
set -euo pipefail

LOG_FILE="/tmp/run_cli_scripts.log"
FAILURES=()

# Generate test data
echo "🛠️ Generating test data..."
python3 /tests/scripts_tests/generate_input_data.py
DOMAIN_DATA_DIR=/tmp/domain_dataset

# Domain classifier inference
echo "▶️ Running domain_classifier_inference..."
OUTPUT_DIR=/tmp/domain_classifier_inference_output
mkdir -p $OUTPUT_DIR
LOG_FILE="/tmp/domain_classifier_inference.log"
domain_classifier_inference \
  --input-data-dir "$DOMAIN_DATA_DIR" \
  --input-file-type "jsonl" \
  --input-text-field "text" \
  --output-data-dir "$OUTPUT_DIR" \
  > "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ domain_classifier_inference failed! Here's the log output:"
    cat "$LOG_FILE"
    FAILURES+=("domain_classifier_inference")
else
    echo "✅ domain_classifier_inference passed"
fi

# Final summary
if [ ${#FAILURES[@]} -ne 0 ]; then
    echo ""
    echo "🚨 Some CLI scripts failed:"
    for f in "${FAILURES[@]}"; do
        echo "- $f"
    done
    exit 1
else
    echo ""
    echo "🎉 All CLI scripts passed."
    exit 0
fi
