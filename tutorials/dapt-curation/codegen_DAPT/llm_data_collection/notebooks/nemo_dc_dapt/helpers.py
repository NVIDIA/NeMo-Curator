# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip
import json
import os

from .docbuilder import DAPTtxtExtractor, DAPTtxtIterator


def write_jsonl(
    input_filename: str, output_dir: str, original_path: str, dump_every_n: int = 10000
):
    """
    Convert a file to JSONL format and write it to the specified output directory.

    Args:
        input_filename (str): The path to the input file.
        output_dir (str): The path to the output directory.
        original_path (str): The original path to the input file
        dump_every_n (int, optional): The number of records to dump to file at once.
    """
    if os.path.isdir(output_dir):
        if len(os.listdir(output_dir)) > 0:
            print(
                f"Output directory '{output_dir}' is not empty. Adding conversion to JSONL."
            )
    else:
        os.makedirs(output_dir)

    print(f"Converting '{input_filename}' to JSONL format (output: '{output_dir}')...")

    original_path = original_path[1:].replace("'", "")
    basename = os.path.basename(input_filename)
    iterator = DAPTtxtIterator()
    extractor = DAPTtxtExtractor()
    to_dump = []
    dump_ctr = 0

    if "pdf" not in original_path:
        if "wiki" in original_path:
            with gzip.open(original_path, "rt") as fp:
                lines = len(fp.readlines())
        else:
            with open(original_path, "r") as fp:
                lines = len(fp.readlines())
    else:
        with gzip.open(original_path, "rt") as fp:
            lines = len(fp.readlines())

    def dump_to_file(to_dump, dump_ctr):
        """Helper function to facilitate dumping to file."""
        output_filename = f"{basename}-{dump_ctr}.jsonl"
        with open(os.path.join(output_dir, output_filename), "w") as output_file:
            output_file.writelines(to_dump)
        # Empty out the list and increment the counter.
        return [], dump_ctr + 1

    for item in iterator.iterate(input_filename):
        record_meta, content = item
        extracted = extractor.extract(content)

        if extracted is None:
            continue

        text_meta, text = extracted

        if text is None:
            continue
        ext = os.path.splitext(os.path.basename(original_path))[1]
        if ext == ".gz":
            ext = ".txt"

        if ext in (".VX", ".VXH"):
            category = "Viva"
        elif ext in (".V", ".VH", ".VHDL", ".v"):
            category = "VerilogVHDL"
        elif ext in (".C", ".CPP", ".H", ".HPP", ".c", ".h", ".cpp"):
            category = "CPP"
        elif ext in (".PY", ".py"):
            category = "Python"
        elif ext in (".CONFIG", ".config"):
            category = "Config"
        elif ext in ("Makefile", "Makeppfile", ".mk"):
            category = "Makefile"
        elif ext in (".PM", ".PL", ".pm", ".pl"):
            category = "Perl"
        elif ext in (".TCL", ".tcl"):
            category = "Tcl"
        elif ext in (".spec"):
            category = "Spec"
        elif ext in (".yaml", ".yml"):
            category = "Yaml"
        elif ext in (".sp", ".cir", ".cmd", ".spf", ".spice"):
            category = "Spice"
        elif ext in (".va"):
            category = "VerilogAnalog"
        elif ext in (".txt", ".json", ".xml", ".html", ".pdf"):
            category = "text"
        else:
            category = "other"

        if category == "text":
            file_type = "text"
        else:
            file_type = "code"

        line = {
            "text": text,
            "file_extension": ext,
            "category": category,
            "file_type": file_type,
            "lines": lines,
            "size_MB": os.path.getsize(original_path) / (1 << 20),
            "compressed_size_MB": os.path.getsize(input_filename) / (1 << 20),
            "orig_path": original_path,
            **text_meta,
            **record_meta,
        }
        json_out = json.dumps(line, ensure_ascii=False)
        to_dump.append(json_out + "\n")
        #         print("dump length",len(to_dump))
        #         print("text length:",len(content))

        # Should we dump what we have so far?
        if len(to_dump) >= dump_every_n:
            to_dump, dump_ctr = dump_to_file(to_dump, dump_ctr)
    #             print("dumped file: ",input_filename)

    # Dump the remaining records.
    if to_dump:
        dump_to_file(to_dump, dump_ctr)


def count_lines(input_filename: str):
    with gzip.open(input_filename, "rt") as fp:
        lines = len(fp.readlines())
    return lines
