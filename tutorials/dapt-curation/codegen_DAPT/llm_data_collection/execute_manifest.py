#!/usr/bin/env python3

import fileinput
import gzip
import hashlib
import io
import os
import sys
import tempfile

# import matplotlib as mpl
# import matplotlib.pyplot as plt
from collections import Counter

import numpy as np
import pandas as pd
from manifest_utils import *


def get_local_dir_path(orig_name):
    # get basename of orig_name (no extensions)
    basename = os.path.splitext(os.path.basename(orig_name))[0]

    # get directory name of orig_name
    dirname = os.path.dirname(orig_name)

    # local directory path is dirname/basename_collected
    local_dir = os.path.join(dirname, basename)

    return local_dir


def compress_and_hash_file(input_file, local_dir, extension=""):
    """This function compresses a file object and renames it with its MD5 hash."""
    # Create a hash object
    h = hashlib.md5()

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=local_dir)
    temp_output_filename = temp_file.name

    lines = io.BytesIO()
    for line in input_file:
        lines.write(line)
    lines.seek(0)

    with gzip.open(temp_output_filename, "wb") as out_file:
        while True:
            # Read only 1024 bytes at a time
            chunk = lines.read(1024)
            if not chunk:
                break
            h.update(chunk)
            out_file.write(chunk)

    output_filename = f"{h.hexdigest()}{extension}.gz"

    output_full_path = os.path.join(local_dir, output_filename)
    if os.path.exists(output_full_path):
        # if file already exists, then delete the temporary file
        os.remove(temp_output_filename)
    else:
        # Rename the temporary file to the final output filename
        # change permissions to read/write for user and read for group
        os.chmod(temp_output_filename, 0o664)
        os.rename(temp_output_filename, output_full_path)

    # return output_filename
    return output_filename


def copy_multifile_to_local_dir(paths, local_dir):
    global collection_log

    path = paths[0]

    path = path.encode("latin1")

    # determine extenion for new file (dropping .gz if necessary), detect file open hook based on .gz suffix
    extension = os.path.splitext(path)[1].decode("utf-8")
    base_filename = os.path.splitext(path)[0]
    if extension.upper() == ".GZ":
        f_open = gzip.open
        # Create the final output extension
        while (
            os.path.splitext(base_filename)[1] != ""
            and os.path.splitext(base_filename)[0].upper() == ".GZ"
        ):
            extension = os.path.splitext(base_filename)[1].decode("utf-8")
            base_filename = os.path.splitext(base_filename)[0]
        if extension.upper() == ".GZ":
            extension = ""
    else:
        f_open = open

    # use fileinput to concatenate file paths into single file object
    with fileinput.input(
        files=[os.path.abspath(path.encode("latin1")) for path in paths],
        mode="rb",
        openhook=f_open,
    ) as input_files:
        filename = compress_and_hash_file(input_files, local_dir, extension)

    # add to collection_log
    collection_log = pd.concat(
        [collection_log, pd.DataFrame([{"CollectedFile": filename, "Path": paths}])],
        ignore_index=True,
    )


def copy_file_to_local_dir(path, local_dir):
    global collection_log

    path = path.encode("latin1")

    # conform path to absolute path
    path = os.path.abspath(path)

    # determine extenion for new file (dropping .gz if necessary), detect file open hook based on .gz suffix
    extension = os.path.splitext(path)[1].decode("utf-8")
    base_filename = os.path.splitext(path)[0]
    if extension.upper() == ".GZ":
        f_open = gzip.open
        # Create the final output extension
        while (
            os.path.splitext(base_filename)[1] != ""
            and os.path.splitext(base_filename)[0].upper() == ".GZ"
        ):
            extension = os.path.splitext(base_filename)[1].decode("utf-8")
            base_filename = os.path.splitext(base_filename)[0]
        if extension.upper() == ".GZ":
            extension = ""
    else:
        f_open = open

    with f_open(path, "rb") as input_file:
        filename = compress_and_hash_file(input_file, local_dir, extension)

    # add to collection_log
    collection_log = pd.concat(
        [collection_log, pd.DataFrame([{"CollectedFile": filename, "Path": path}])],
        ignore_index=True,
    )


# Read the CSV file
# input file should come from command line argument argv[1]
# check number of arguments
if len(sys.argv) < 2:
    sys.exit(f"Usage: {sys.argv[0]} <input_file1> [<input_file2> ...]")

input_files = sys.argv[1:]

for input_file in input_files:
    if not os.path.exists(input_file):
        sys.exit(f"Input file does not exist: {input_file}")
    print(f"Collecting from {input_file}...")

    if "bugs" in input_file:
        print("Unified BUGS detected. Grouping files by bug id.")
        unified_bugs_p4_mode = True
    else:
        unified_bugs_p4_mode = False

    # read input file
    data = pd.read_csv(input_file, encoding="latin1")

    # get the local_dir and create it if it doesn't already exist
    local_dir = get_local_dir_path(input_file)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # create a dataframe that will be Path and md5sum
    collection_log = pd.DataFrame(columns=["CollectedFile", "Path"])

    if unified_bugs_p4_mode:

        # if bugs/p4 unified mode, then group by file name prefix (which should be the bug_id)
        data["BugID"] = data["Path"].apply(lambda x: os.path.basename(x).split("_")[0])

        for bug_id, bug_group in data.groupby("BugID"):
            copy_multifile_to_local_dir(list(bug_group["Path"]), local_dir)

    else:

        # for each file, get the path
        data["Path"].apply(lambda path: copy_file_to_local_dir(path, local_dir))

    # write collection log to collected.txt
    collection_log.to_csv(os.path.join(local_dir, "info.txt"), index=False)

    print("Done.")

# end of file
