#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import sys
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from collections import Counter

from manifest_utils import *

def what_is_gen_type(file_path, encoding='ISO-8859-1'):
    with open(file_path, 'r', encoding=encoding) as file:
        for i, line in enumerate(file):
            if i > 15:
                break
            if "GENERATED" in line.upper() and ("DO NOT EDIT" in line.upper() or "AUTO" in line.upper()):
                if "VIVA" in line.upper():
                    return "GeneratedViva"
                elif "NESS" in line.upper():
                    return "GeneratedNess"
                elif "HESS" in line.upper():
                    return "GeneratedHess"
                elif "MANUALS" in line.upper():
                    return "GeneratedManuals"
                else:
                    return "GeneratedOther"
            if "FILE CREATED BY" in line.upper():
                return "GeneratedOther"
    return ""

def filter_executable_shebang(data):
    return data[data['ExecutableShebang'] == True]

def filter_extensions(data, extensions):
    # convert extensions to upper case
    extensions = [ext.upper() for ext in extensions]
    data = data[data['Extension'].isin(extensions)]
    return data

def filter_min_lines(data, min_size):
    if min_size != None:
        data = data[data['Lines'] >= min_size]
    return data

def filter_max_lines(data, max_size):
    if max_size != None:
        data = data[data['Lines'] <= max_size]
    return data

def filter_lines(data, min_size, max_size):
    return filter_min_lines(filter_max_lines(data, max_size), min_size)

def filter_size(data, min_size, max_size):
    if min_size != None:
        data = data[data['Size_MB'] >= min_size]
    if max_size != None:
        data = data[data['Size_MB'] <= max_size]
    return data

# Looks for "do not edit" in first five lines, cases insensitive
# indicative of Viva, Ness, and other generators
def test_gen_type(data):
    data = data.copy()
    data['GeneratedType'] = data['Path'].apply(lambda x: what_is_gen_type(x))
    return data

def is_executable_shebang(file_path):
    # return False if file is not executable
    if not os.access(file_path, os.X_OK):
        return False
    # return True if the first line contains a she-bang
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        for i, line in enumerate(file):
            if i > 0:
                break
            if line.startswith("#!"):
                return True
    return False

def test_executable_shebang(data):
    data = data.copy()
    data['ExecutableShebang'] = data['Path'].apply(lambda x: is_executable_shebang(x))
    return data

def get_filter_out_path(orig_name, subsetname):
    # error if string subsetname is not alphanumeric
    if not subsetname.isalnum():
        sys.exit("subsetname must be alphanumeric")

    output_file = os.path.splitext(input_file)[0] + f"_filtered_{subsetname.lower()}" + os.path.splitext(input_file)[1]
    return output_file

def total_size_mb(data):
    return data['Size_MB'].sum()

def save_filtered(data, orig_name, subsetname, extensions):
    # error if string subsetname is not alphanumeric
    if not subsetname.isalnum():
        sys.exit("subsetname must be alphanumeric")

    print(f"\n# {subsetname} Summary ({extensions})")
    print(f"\tTotal number of files: {len(data)}")
    print(f"\tTotal Size: {int(total_size_mb(data))} MB (uncompressed)")

    data = manifest_on_close(data)

    # write data csv
    # only write csv if there is data
    if len(data) > 0:
        data.to_csv(get_filter_out_path(orig_name, subsetname), index=False, encoding='latin1')
    else:
        print(f"\tWarning: No data for {subsetname}. Skipping...")

# Read the CSV file
# input file should come from command line argument argv[1]
# check number of arguments
if len(sys.argv) != 2:
    sys.exit("Usage: report_gpu_tree_manifest.py <input_file>")
input_file = sys.argv[1]
if not os.path.exists(input_file):
    sys.exit("Input file does not exist: " + input_file)

data = pd.read_csv(input_file, encoding='latin1')

data = manifest_on_open(data)

common_min_lines = 10
# common_max_lines = 5000
common_max_lines = 20000
common_min_size = None # MB
common_max_size = None # MB

min_lines = common_min_lines
max_lines = common_max_lines
min_size = common_min_size
max_size = common_max_size

if "nvwiki" in input_file or "confluence" in input_file or "doc" in input_file or "pdf" in input_file or "email" in input_file:
    subsets = [('Text', ['.TXT'], False)]
    if "pdf" in input_file and "doc" not in input_file:
        min_lines = 10
        max_lines = None
    if "vivid" in input_file:
        # manual override to do less filtering of vivid specific docs
        min_lines = 1
        max_lines = None 
elif "nvbugs" in input_file or "perforce" in input_file:
    max_lines = 5000
    max_size = 0.1 # MB
    subsets = [('None', ['NO_EXTENSION'], False)]
else:
    subsets = [
        ("Viva", ['.VX', '.VXH'], True),
        ("VerilogVHDL", ['.V', '.VH', '.VHDL'], True),
        ("CPP", ['.C', '.CPP', '.H', '.HPP'], True),
        ("Python", ['.PY'], False),
        ("SV", ['.SV'], True),
        ("GV", ['.GV'], True),
        ("Config", ['.CONFIG'], False),
        ("Makefile", ['Makefile', 'Makeppfile', '.mk'], False),
        ("Perl", ['.PM', '.PL'], True),
        ("Tcl", ['.TCL'], True),
        ("Spec", ['.spec'], False),
        ("Yaml", ['.yaml', '.yml'], False)
    ]

    subsets.extend([("Spice", ['.sp', '.cir', '.cmd', '.spf'], False),
                    ("VerilogAnalog", ['.va'], False),
                    ("Skill", ['.il'], False)
    ])

    subsets.extend([("PT", [".ptsh", ".proc"], False)]) # for PrimeTime
    subsets.extend([("EP3", [".ep3", ".ep3.pp"], False)]) # for FV

    subsets.extend([('Script', ['NO_EXTENSION'], False, True)]) # she-bang scripts

total_size_accum = 0
for cur_subset in subsets:
    # if not os.path.exists(get_filter_out_path(input_file, "Viva")):
    cur_data = filter_size(filter_lines(filter_extensions(data, cur_subset[1]), min_lines, max_lines), min_size, max_size)
    total_size_accum += total_size_mb(cur_data) # always add to total size
    if len(cur_subset) > 3 and cur_subset[3]:
        # Check for she-bang
        cur_data = test_executable_shebang(cur_data)
        cur_data = filter_executable_shebang(cur_data)
    if cur_subset[2]:
        cur_data = test_gen_type(cur_data)
        # for each unique generated type, save a separate file
        for gen_type in cur_data['GeneratedType'].unique():
            cur_data_gen = cur_data[cur_data['GeneratedType'] == gen_type]
            save_filtered(cur_data_gen, input_file, cur_subset[0] + gen_type, cur_subset[1])
    else:
        save_filtered(cur_data, input_file, cur_subset[0], cur_subset[1])
    # print_histogram(cur_data, cur_subset[1], 'Lines', do_log=True)
print(f"\n\nTotal Size (uncompressed): {int(total_size_accum)} MB")
