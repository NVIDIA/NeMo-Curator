#!/usr/bin/env python3

# reports duplicates within a specific category (using info.txt)

import csv
import os
import re
import sys


def is_md5(filename):
    """Check if the filename matches the MD5 hash format followed by .gz."""
    base_filename = os.path.basename(filename)
    name, ext = os.path.splitext(base_filename)
    while "." in name:
        name, ext2 = os.path.splitext(name)
    return len(name) == 32 and all(c in "0123456789abcdefABCDEF" for c in name)


def find_duplicates(paths):
    """Find duplicate files in the given paths."""
    found_files = {}
    duplicates = {}

    for path in paths:
        for root, dirs, files in os.walk(path):
            if "info.txt" in files:
                with open(os.path.join(root, "info.txt"), "r") as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        collected_file = row["CollectedFile"]
                        if is_md5(collected_file):
                            path = row["Path"]
                            if collected_file in found_files:
                                if found_files[collected_file] not in duplicates:
                                    duplicates[found_files[collected_file]] = [path]
                                else:
                                    duplicates[found_files[collected_file]].append(path)
                            else:
                                found_files[collected_file] = path
    return duplicates


if __name__ == "__main__":
    # Get paths from command line arguments
    paths = sys.argv[1:]

    duplicates = find_duplicates(paths)

    # Print out a list of duplicates for each first file that is duplicated
    for first_file, duplicate_files in duplicates.items():
        print(f"First file: {first_file}")
        print("\tDuplicates:")
        for duplicate_file in duplicate_files:
            print(f"\t{duplicate_file}")
        print()
