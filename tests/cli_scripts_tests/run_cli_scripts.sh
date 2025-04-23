#!/bin/bash
set -euo pipefail

TEST_DIR="$(dirname "$0")"
FAILURES=()

declare -A SCRIPTS
SCRIPTS["domain_classifier_inference"]="$TEST_DIR/run_domain_classifier_inference.sh"
SCRIPTS["quality_classifier_inference"]="$TEST_DIR/run_quality_classifier_inference.sh"

# Generate test data
echo "🛠️ Generating test data..."
bash generate_input_data.sh

# Loop through each script and run it, logging the output and duration
for NAME in "${!SCRIPTS[@]}"; do
    SCRIPT_PATH="${SCRIPTS[$NAME]}"
    LOG_FILE="/tmp/${NAME}.log"

    echo "▶️ Running $NAME..."
    START_TIME=$(date +%s)

    bash "$SCRIPT_PATH" "$LOG_FILE"
    EXIT_CODE=$?

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo "📄 Log output for $NAME:"
    echo "------------------------"
    cat "$LOG_FILE"
    echo "------------------------"

    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ $NAME failed in ${DURATION}s."
        FAILURES+=("$NAME")
    else
        echo "✅ $NAME passed in ${DURATION}s."
    fi
    echo ""
done

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
