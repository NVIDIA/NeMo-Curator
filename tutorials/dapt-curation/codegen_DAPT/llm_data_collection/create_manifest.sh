#!/bin/bash
#
# run in GPU tree or other tree in a scratch space

# collection_dir="/home/scratch.${USER}_research_1/llm_data"
# # ensure collection_dir exists
# if [ ! -d "$collection_dir" ]; then
#     echo "Collection directory does not exist: $collection_dir"
#     exit 1
# fi
# mkdir -p $collection_dir/gpu_tree

# output_file="$collection_dir/gpu_tree/gpu_tree_manifest.csv"

# Check if the output file was provided
# need to provide two arguments: directory to search and output_file.csv
# optional argument to collect from a gzip based dir


if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 directory_to_search output_file.csv [append]"
  exit 1
fi

root_collection_dir="$1"
output_file="$2"
append_mode="$3"

# normalize root_collection_dir to abspath
root_collection_dir=$(realpath "$root_collection_dir")

# ensure root_collection_dir exists
if [ ! -d "$root_collection_dir" ]; then
    echo "Root collection directory does not exist: $root_collection_dir"
    exit 1
fi

# ensure gzip_root is either a 1 or a 0, or empty
if [ -n "$append_mode" ] && [ "$append_mode" != "1" ] && [ "$append_mode" != "0" ]; then
    echo "append_mode must be either 1 or 0"
    exit 1
fi

# ensure output_file doesn't exist, otherwise throw an error
if [ -z "$append_mode" ] || [ "$append_mode" == "0" ]; then # if append_mode is 0 or empty
  if [ -f "$output_file" ]; then
      echo "Output file already exists: $output_file"
      exit 1
  fi
fi

echo "Collecting from $root_collection_dir..."

# Create the output file and write the header
if [ -z "$append_mode" ] || [ "$append_mode" == "0" ]; then # if append_mode is 0 or empty
  echo "Path,Size,Lines" > "$output_file"
fi

# Find all files and process them
find "$root_collection_dir" -type f -print0 | while IFS= read -r -d '' file; do
  # Check if the file is a text file
  file_description=$(file --brief --mime-type "$file")
  if [[ "$file_description" == text/* ]]; then
    # Get the number of lines
    lines=$(wc -l < "$file")

    # Get the file size in bytes
    size=$(stat -c%s "$file")   # for Linux
    #size=$(stat -f%z "$file")   # for macOS

    # Save the data to the CSV file
    printf '"%s",%s,%s\n' "$file" "$size" "$lines" >> "$output_file"
  fi
done

echo "Done! Results saved to $output_file"
