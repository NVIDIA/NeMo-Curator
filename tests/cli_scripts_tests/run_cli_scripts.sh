#!/bin/bash
set -uo pipefail

TEST_DIR="$(dirname "$0")"
FAILURES=()

declare -A SCRIPTS
relative_paths=(
    "classifiers/aegis_classifier_inference"
    "classifiers/content_type_classifier_inference"
    "classifiers/domain_classifier_inference"
    "classifiers/fineweb_edu_classifier_inference"
    "classifiers/fineweb_mixtral_edu_classifier_inference"
    "classifiers/fineweb_nemotron_edu_classifier_inference"
    "classifiers/instruction_data_guard_classifier_inference"
    "classifiers/multilingual_domain_classifier_inference"
    "classifiers/prompt_task_complexity_classifier_inference"
    "classifiers/quality_classifier_inference"
)
for relative_path in "${relative_paths[@]}"; do
    SCRIPTS["$relative_path"]="$TEST_DIR/$relative_path.sh"
done

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
    echo "➖ classifiers/aegis_classifier_inference"
    echo "➖ classifiers/instruction_data_guard_classifier_inference"
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
