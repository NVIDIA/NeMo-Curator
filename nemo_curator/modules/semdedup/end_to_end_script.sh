#!/bin/bash
# SemDeDup Pipeline


# Config file path
CONFIG_FILE="config.yaml"

# Load config.yaml variables
ROOT=$(grep -P '^root:\s*' "$CONFIG_FILE" | awk '{print $2}' | tr -d "'")

# Delete the root path
if [ -d "$ROOT" ]; then
    echo "Deleting existing root directory: $ROOT"
    rm -rf "$ROOT"
    mkdir -p "$ROOT"
fi

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
echo "Running semdedup.py..."
python semdedup.py

# Step 5: Extract deduplicated data
echo "Running extract_dedup_data.py..."
python extract_dedup_data.py

echo "Pipeline execution completed successfully."
