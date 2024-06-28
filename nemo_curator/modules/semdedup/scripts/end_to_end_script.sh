#!/bin/bash
# SemDeDup Pipeline

echo "Running SemDeDup pipeline..."
echo "---------------------------------"
# Config file path
CONFIG_FILE="configs/config.yaml"
echo "CONFIG_FILE: $CONFIG_FILE"
# Load config.yaml variables
ROOT=$(grep -P '^root:\s*' "$CONFIG_FILE" | awk '{print $2}' | tr -d "'")
SAVE_LOC=$(awk '/^clustering:/,/save_loc:/{ if ($1 == "save_loc:") print $2 }' "$CONFIG_FILE" | tr -d "'\"")

echo "ROOT Directory: $ROOT"
echo "Save Location for sem-dedup: $ROOT/$SAVE_LOC"

# Optionally delete the root path
# echo "Deleting existing root directory: $ROOT"
#rm -rf "$ROOT"
#mkdir -p "$ROOT"
rm -rf "$ROOT/$SAVE_LOC"

# Step 1: Compute embeddings
echo "Running compute_embeddings.py..."
python compute_embeddings.py

# Step 2: Clustering
echo "Running clustering.py..."
python clustering.py

# Step 3: Sort the clusters
echo "Running sort_clusters.py..."
python sort_clusters.py

# Step 4: Run SemDeDup
# Helps find duplicates with in the same cluster
echo "Running semdedup.py..."
python semdedup.py

# Step 5: Extract deduplicated data
echo "Running extract_dedup_data.py..."
python extract_dedup_data.py
echo "---------------------------------"
echo "Pipeline execution completed successfully."
