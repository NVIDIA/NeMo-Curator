#!/bin/bash

# get directory this script lives in
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 dir_list output_file.csv"
  exit 1
fi

dir_list_file="$1"
output_file="$2"

if [ -f "$output_file" ]; then
    echo "Output file already exists: $output_file"
    exit 1
fi

# for each line in dir_list_file call create_manifest.
# set append_mode to 0 for first call, 1 for all others
append_mode=0
while IFS= read -r dir_to_process; do
  echo "Processing $dir_to_process"
  # ensure dir exists
    if [ ! -d "$dir_to_process" ]; then
        echo "Directory does not exist: $dir_to_process"
        exit 1
    fi
  $script_dir/create_manifest.sh "$dir_to_process" "$output_file" "$append_mode"
  append_mode=1
done < "$dir_list_file"

echo "Done."
