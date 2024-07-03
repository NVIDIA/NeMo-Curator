#!/bin/bash
# SemDeDup Pipeline

echo "Running SemDeDup pipeline..."
echo "---------------------------------"
# Config file path
CONFIG_FILE="config.yaml"
INPUT_DATA_DIR="/datasets/semdedup/c4/realnewslike/modified"
INPUT_FILE_TYPE="json"

echo "CONFIG_FILE: $CONFIG_FILE"
# Load config.yaml variables
CACHE_DIR=$(grep -P '^cache_dir:\s*' "$CONFIG_FILE" | awk '{print $2}' | tr -d "'")
CLUSTERING_LOC=$(awk '/^clustering:/,/save_loc:/{ if ($1 == "save_loc:") print $2 }' "$CONFIG_FILE" | tr -d "'\"")

echo "Cache Directory: $CACHE_DIR"
echo "Save Location for sem-dedup: $CACHE_DIR/$CLUSTERING_LOC"

# Optionally delete the CACHE_DIR path
# echo "Deleting existing CACHE_DIR directory: $CACHE_DIR"
# rm -rf "$CACHE_DIR"
# rm -rf "$CLUSTERING_LOC"
mkdir -p "$CACHE_DIR"

# Step 1: Compute embeddings
echo "Running compute_embeddings.py..."
python compute_embeddings.py --input-data-dir "$INPUT_DATA_DIR" --input-file-type "json"

# Step 2: Clustering
echo "Running clustering.py..."
python clustering.py

# Step 3: Extract Dedup Data
echo "Running extract_dedup_data.py..."
python extract_dedup_data.py

## Optional: End-to-End Example
# python3 end_to_end_example.py --input-data-dir "/datasets/semdedup/c4/realnewslike/modified" --input-file-type "jsonl"

echo "--------------------------------------------"
echo "Pipeline execution completed successfully."
echo "--------------------------------------------"
