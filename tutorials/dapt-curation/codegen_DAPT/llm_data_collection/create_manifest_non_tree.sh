#!/bin/bash
#
# Collect from a non-tree source, such as collect confluence, nvwiki, etc...
# Looks for .gz files instead of text files.

function uncompressed_size() {
    gzip -l "$1" | awk 'NR==2 {print $2}'
}

# Check if the output file was provided
# need to provide two arguments: directory to search and output_file.csv
# optional argument to collect from a gzip based dir
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 directory_to_search output_file.csv [gzip_root]"
  exit 1
fi

root_collection_dir="$1"
output_file="$2"
gzip_root="$3"

# normalize root_collection_dir to abspath
root_collection_dir=$(realpath "$root_collection_dir")

# ensure root_collection_dir exists
if [ ! -d "$root_collection_dir" ]; then
    echo "Root collection directory does not exist: $root_collection_dir"
    exit 1
fi
# ensure output_file doesn't exist, otherwise throw an error
if [ -f "$output_file" ]; then
    echo "Output file already exists: $output_file"
    exit 1
fi
# ensure gzip_root is either a 1 or a 0, or empty
if [ -n "$gzip_root" ] && [ "$gzip_root" != "1" ] && [ "$gzip_root" != "0" ]; then
    echo "gzip_root must be either 1 or 0"
    exit 1
fi

echo "Collecting from $root_collection_dir..."

# Create the output file and write the header
echo "Path,Size,Lines" > "$output_file"

# Find all files and process them
find "$root_collection_dir" -name '*.gz' -print0 | while IFS= read -r -d '' file; do
    # Get the number of lines
    lines=$(zcat "$file" | wc -l)

    # Get the file size in bytes
    # size=$(stat -c%s "$file")
    size=$(uncompressed_size "$file")

    # Save the data to the CSV file
    # printf '"%s",%s,%s\n' "${file/.gz/}" "$size" "$lines" >> "$output_file"
    printf '"%s",%s,%s\n' "$file" "$size" "$lines" >> "$output_file"
done

echo "Done! Results saved to $output_file"
