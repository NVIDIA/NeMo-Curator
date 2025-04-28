#!/bin/bash
set -uo pipefail

LOG_FILE="$1"

# Create first batch of JSONL files with domain data
INPUT_FILE="/tmp/multilingual_domain_dataset/batch_1.jsonl"
mkdir -p "$(dirname "$INPUT_FILE")"
> "$INPUT_FILE"
texts=(
    # Chinese
    "量子计算将彻底改变密码学领域。"
    # Spanish
    "Invertir en fondos indexados es una estrategia popular para el crecimiento financiero a largo plazo."
    # English
    "Recent advancements in gene therapy offer new hope for treating genetic disorders."
)
for text in "${texts[@]}"; do
    echo "{\"text\": \"$text\"}" >> "$INPUT_FILE"
done
echo "✅ JSONL file '$INPUT_FILE' successfully created."

# Create second batch of JSONL files with domain data
INPUT_FILE="/tmp/multilingual_domain_dataset/batch_2.jsonl"
mkdir -p "$(dirname "$INPUT_FILE")"
> "$INPUT_FILE"
texts=(
    # Hindi
    "ऑनलाइन शिक्षण प्लेटफार्मों ने छात्रों के शैक्षिक संसाधनों तक पहुंचने के तरीके को बदल दिया है।"
    # Bengali
    "অফ-সিজনে ইউরোপ ভ্রমণ করা আরও বাজেট-বান্ধব বিকল্প হতে পারে।"
)
for text in "${texts[@]}"; do
    echo "{\"text\": \"$text\"}" >> "$INPUT_FILE"
done
echo "✅ JSONL file '$INPUT_FILE' successfully created."

INPUT_DIR="/tmp/multilingual_domain_dataset"
OUTPUT_DIR="/tmp/multilingual_domain_classifier_inference_output"
mkdir -p "$OUTPUT_DIR"

multilingual_domain_classifier_inference \
  --input-data-dir "$INPUT_DIR" \
  --input-file-type "jsonl" \
  --input-text-field "text" \
  --output-data-dir "$OUTPUT_DIR" \
  > "$LOG_FILE" 2>&1
EXIT_CODE=$?

exit $EXIT_CODE
