#!/bin/bash

# Create first batch of JSONL files with domain data
OUTPUT_FILE="/tmp/domain_dataset/batch_1.jsonl"
mkdir -p "$(dirname "$OUTPUT_FILE")"
> "$OUTPUT_FILE"
texts=(
    "Quantum computing is set to revolutionize the field of cryptography."
    "Investing in index funds is a popular strategy for long-term financial growth."
    "Recent advancements in gene therapy offer new hope for treating genetic disorders."
)
for text in "${texts[@]}"; do
    echo "{\"text\": \"$text\"}" >> "$OUTPUT_FILE"
done
echo "✅ JSONL file '$OUTPUT_FILE' successfully created."

# Create second batch of JSONL files with domain data
OUTPUT_FILE="/tmp/domain_dataset/batch_2.jsonl"
mkdir -p "$(dirname "$OUTPUT_FILE")"
> "$OUTPUT_FILE"
texts=(
    "Online learning platforms have transformed the way students access educational resources."
    "Traveling to Europe during the off-season can be a more budget-friendly option."
)
for text in "${texts[@]}"; do
    echo "{\"text\": \"$text\"}" >> "$OUTPUT_FILE"
done
echo "✅ JSONL file '$OUTPUT_FILE' successfully created."
