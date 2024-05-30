#!/usr/bin/env python3

import os
import sys
from collections import Counter
from difflib import SequenceMatcher

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from manifest_utils import *
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform


def generate_ngrams(path, n):
    parts = path.split(os.path.sep)
    ngrams = [os.path.sep.join(parts[i : i + n]) for i in range(len(parts) - n + 1)]
    return ngrams


def analyze_paths_ngrams(data, n=2, min_occurrences=5):
    ngram_counter = Counter()
    paths = data["Path"].unique()

    for path in paths:
        ngrams = generate_ngrams(path, n)
        ngram_counter.update(ngrams)

    common_ngrams = {k: v for k, v in ngram_counter.items() if v >= min_occurrences}

    print(f"\nCommon {n}-grams in paths (occurring at least {min_occurrences} times):")
    for ngram, count in sorted(common_ngrams.items(), key=lambda x: x[1], reverse=True):
        print(f"{ngram}: {count} occurrences")


def path_similarity(path1, path2):
    matcher = SequenceMatcher(None, path1, path2)
    return matcher.ratio()


def analyze_paths(data, min_cluster_size=5):
    paths = data["Path"].unique()
    path_count = len(paths)

    similarity_matrix = np.zeros((path_count, path_count))

    for i in range(path_count):
        for j in range(i + 1, path_count):
            similarity = path_similarity(paths[i], paths[j])
            similarity_matrix[i, j] = similarity_matrix[j, i] = 1 - similarity

    clustering = linkage(squareform(similarity_matrix), method="average")
    clusters = fcluster(clustering, t=0.5, criterion="distance")

    unique_clusters = np.unique(clusters)
    print(
        f"\nCommon patterns in paths (clusters with at least {min_cluster_size} paths):"
    )

    for cluster_id in unique_clusters:
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) >= min_cluster_size:
            print(f"\nCluster {cluster_id}:")
            for index in cluster_indices:
                print(f"  {paths[index]}")


# # change collection_dir to where you want to store the data
# collection_dir = f"/home/scratch.{os.environ['USER']}_research_1/llm_data"
# if not os.path.exists(collection_dir):
#     sys.exit("Collection directory does not exist: " + collection_dir)

# Read the CSV file
# input file should come from command line argument argv[1]
# check number of arguments
if len(sys.argv) != 2:
    sys.exit("Usage: report_gpu_tree_manifest.py <input_file>")
input_file = sys.argv[1]

if not os.path.exists(input_file):
    sys.exit("Input file does not exist: " + input_file)

# input_file = os.path.join(collection_dir, "gpu_tree", "gpu_tree_manifest.csv")
# input_file = os.path.join(collection_dir, "gpu_tree", "gpu_tree_cpp_manifest.csv") # TODO: Rerun with this one.
data = pd.read_csv(input_file)

# preprocess
data = manifest_on_open(data)

# Report Percentile of Sizes
print(f"{len(data)} Files")
print(f"Top 10 percentiles (Size_MB):")
print(data["Size_MB"].quantile(q=np.arange(0.9, 1.0, 0.01)))

print(f"Top 10 files (size):")
for index, row in data.sort_values("Size_MB")[-10:].iterrows():
    print(f"{row['Path']} : {row['Lines']} Lines {row['Size_MB']}MB")


file_size_summary = (
    data.groupby("Extension")["Size_MB"]
    .sum()
    .reset_index()
    .sort_values("Size_MB", ascending=False)
)

# Round file sizes to the nearest MB
file_size_summary["Size_MB"] = file_size_summary["Size_MB"].apply(round)
# only include if Size_MB > 5
file_size_summary = file_size_summary[file_size_summary["Size_MB"] > 5]


# Configure pandas to display all rows
pd.set_option("display.max_rows", None)

# Print the total file size summary
print("File size summary (>5 MB):")
print(file_size_summary.to_string(index=False))

# Create a text-based histogram for different file types
print("\nFile size text-based histogram:")

max_width = 50
max_value = file_size_summary["Size_MB"].max()

for index, row in file_size_summary.iterrows():
    num_bars = int((row["Size_MB"] / max_value) * max_width)
    print(f"{row['Extension']:<15} | {'#' * num_bars}")

# Create separate text-based histograms for ".V" and ".CPP" file extensions

collection_metrics = ["Size_KB", "Lines"]
file_ext_sets = [
    [".V", ".VH"],
    [".VX", ".VXH"],
    [".CPP", ".HPP", ".C", ".H"],
    [".SPEC"],
    ["Makefile", "Makeppfile"],
]

for collection_metric in collection_metrics:
    for file_ext_set in file_ext_sets:
        print_histogram(data, file_ext_set, collection_metric, do_log=True)

# try to find commonalities in paths
analyze_paths_ngrams(data, n=2, min_occurrences=100)
analyze_paths_ngrams(data, n=3, min_occurrences=100)
analyze_paths_ngrams(data, n=4, min_occurrences=100)

analyze_paths(data, min_cluster_size=1000)

print("WARNING: TEMPORARILY RUNNING ON A DIFFERENT MANIFEST")

# Create a histogram for different file types
fig, ax = plt.subplots(figsize=(10, 6))
data.groupby("Extension")["Size"].sum().plot(kind="bar", ax=ax)
ax.set_xlabel("File Extension")
ax.set_ylabel("File Size (bytes)")
ax.set_title("File Size Histogram by File Extension")

# Save the histogram to a file
plt.savefig(
    os.path.dirname(input_file) + "/file_size_histogram.png", bbox_inches="tight"
)
# print(os.path.dirname(input_file)+'/file_size_histogram.png')
# Display the histogram
plt.show()
