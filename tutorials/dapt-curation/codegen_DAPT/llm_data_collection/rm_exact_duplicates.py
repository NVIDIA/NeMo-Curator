#!/usr/bin/env python3

# reports and then offers to remove duplicates across different data source directories

import os
import sys
import re
from tqdm import tqdm

def is_md5(filename):
    """Check if the filename matches the MD5 hash format followed by .gz."""
    base_filename = os.path.basename(filename)
    name, ext = os.path.splitext(base_filename)
    name, ext2 = os.path.splitext(name)
    return len(name) == 32 and all(c in '0123456789abcdefABCDEF' for c in name) and ext == '.gz'

def find_duplicates(paths):
    """Find duplicate files in the given paths."""
    found_files = {}
    duplicates = []

    for path in paths:
        for root, dirs, files in os.walk(path):
            for filename in files:
                if is_md5(filename):
                    full_path = os.path.abspath(os.path.join(root, filename))
                    if filename in found_files:
                        duplicates.append(full_path)
                    else:
                        found_files[filename] = full_path

    return duplicates

if __name__ == "__main__":
    # Get paths from command line arguments
    paths = sys.argv[1:]

    duplicates = find_duplicates(paths)

    # Write out a list of all duplicate file paths
    print("Saving lists of duplicates to duplicates.txt...")
    with open('duplicates.txt', 'w') as f:
        for duplicate in duplicates:
            f.write(duplicate + '\n')

    print(f"Found {len(duplicates)} duplicates.")
    print("First 10 lines look like:")
    for i in duplicates[:10]:
        print(f"\t{i}")
    print("Last 10 lines look like:")
    for i in duplicates[-10:]:
        print(f"\t{i}")
    print(f"Remove?")
    response = input("y/n: ")
    if response == 'y':
        for duplicate in tqdm(duplicates):
            os.remove(duplicate)
        print("Removed.")
    
