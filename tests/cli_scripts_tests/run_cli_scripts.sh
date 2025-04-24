#!/bin/bash
set -uo pipefail

TEST_DIR="$(dirname "$0")"
FAILURES=()

declare -A SCRIPTS
SCRIPTS["aegis_classifier_inference"]="$TEST_DIR/run_aegis_classifier_inference.sh"
SCRIPTS["content_type_classifier_inference"]="$TEST_DIR/run_content_type_classifier_inference.sh"
SCRIPTS["domain_classifier_inference"]="$TEST_DIR/run_domain_classifier_inference.sh"
SCRIPTS["fineweb_edu_classifier_inference"]="$TEST_DIR/run_fineweb_edu_classifier_inference.sh"
SCRIPTS["fineweb_mixtral_edu_classifier_inference"]="$TEST_DIR/run_fineweb_mixtral_edu_classifier_inference.sh"
SCRIPTS["fineweb_nemotron_edu_classifier_inference"]="$TEST_DIR/run_fineweb_nemotron_edu_classifier_inference.sh"
SCRIPTS["instruction_data_guard_classifier_inference"]="$TEST_DIR/run_instruction_data_guard_classifier_inference.sh"
SCRIPTS["multilingual_domain_classifier_inference"]="$TEST_DIR/run_multilingual_domain_classifier_inference.sh"
SCRIPTS["prompt_task_complexity_classifier_inference"]="$TEST_DIR/run_prompt_task_complexity_classifier_inference.sh"
SCRIPTS["quality_classifier_inference"]="$TEST_DIR/run_quality_classifier_inference.sh"

# Generate test data
echo "🛠️ Generating test data..."
bash generate_input_data.sh
echo ""

# Loop through each script and run it, logging the output and duration
echo "🔍 Running CLI scripts..."
echo ""
for NAME in "${!SCRIPTS[@]}"; do
    SCRIPT_PATH="${SCRIPTS[$NAME]}"
    LOG_FILE="/tmp/${NAME}.log"

    echo "▶️ Running $NAME..."
    START_TIME=$(date +%s)

    bash "$SCRIPT_PATH" "$LOG_FILE"
    EXIT_CODE=$?

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

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

if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "⚠️ WARNING: HF_TOKEN is not set. The following CLI scripts were skipped:"
    echo "➖ run_aegis_classifier_inference.sh"
    echo "➖ run_instruction_data_guard_classifier_inference.sh"
    echo ""
fi

# Final summary
if [ ${#FAILURES[@]} -ne 0 ]; then
    echo ""
    echo "🚨 Some CLI scripts failed:"
    for f in "${FAILURES[@]}"; do
        echo "❌ $f"
    done
    exit 1
else
    echo ""
    echo "🎉 All CLI scripts passed."
    exit 0
fi
