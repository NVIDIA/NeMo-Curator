import os
import subprocess

import ftfy

from .util import sort_struct


def _extract_raw(path):
    """Calls the pdftotext binary to extract HTML-formatted text from PDF"""
    # Make sure the file exists
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found: {}".format(path))

    # Use the -bbox-layout option to preserve the layout of the PDF
    # Set the out-file to "-" to print to stdout
    cmd = ["pdftotext", "-bbox-layout", path, "-"]
    # Yield the output line-by-line
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            try:
                yield line.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue


def _extract_pages(raw_lines):
    """Turns a generator of lines into a generator of pages"""
    parse_flag = False
    page = []
    for line in raw_lines:
        if line.startswith("</doc"):
            parse_flag = False
        if parse_flag:
            page.append(line)
            if line.startswith("</page"):
                yield page
                page = []
        if line.startswith("<doc"):
            parse_flag = True


def _parse_line(raw_line):
    """Parse a raw line into genre, pos, and text"""
    meta, text = raw_line.split(">", 1)
    genre, *pos = meta.split(" ")
    genre = genre[1:]
    pos = tuple(float(x.split("=")[1].strip('"')) for x in pos)
    text = text.split("<")[0]
    # Normalize strange characters like ligatures
    text = ftfy.fixes.fix_latin_ligatures(text)
    # Fix HTML entities
    text = ftfy.fixes.unescape_html(text)
    return genre, pos, text


def _parse_pages(raw_pages, multi_col=True):
    """Parses a list of raw page HTML into words"""
    # The organization is
    # page > flow > block > line > word
    # Each word has a position and a text
    # The data structure is a list of lists of lists etc

    for page in raw_pages:
        genres = {"page": [], "flow": [], "block": [], "line": [], "word": []}
        order = {"/page": "flow", "/flow": "block", "/block": "line", "/line": "word"}
        for line in page:
            genre, pos, text = _parse_line(line)
            if genre in genres:
                data = ([], pos, text)
                data = (text, pos) if genre == "word" else data[:-1]
                genres[genre].append(data)
            if genre in order:
                this_section = genre[1:]
                sub_section = order[genre]
                # By default, flows don't have position data
                # We can get it by applying min / max on the block positions
                if this_section == "flow":
                    # This is a quick hack to hide a bug in the underlying tool
                    # Sometimes non-overlapping blocks get grouped into a flow
                    leftmost = min(genres["block"], key=lambda x: x[1][0])
                    new_flow = []
                    for block in genres["block"]:
                        # If the block doesn't overlap the leftmost
                        if block[1][0] > leftmost[1][2]:
                            # Move it to a new flow
                            new_flow.append(block)
                    for block in new_flow:
                        genres["block"].remove(block)
                    xmin = min(x[1][0] for x in genres["block"])
                    ymin = min(x[1][1] for x in genres["block"])
                    xmax = max(x[1][2] for x in genres["block"])
                    ymax = max(x[1][3] for x in genres["block"])
                    pos = (xmin, ymin, xmax, ymax)
                    empty = genres[this_section][-1][0]
                    genres[this_section][-1] = (empty, pos)
                    # Append new_flow as well, if it exists
                    if new_flow:
                        xmin = min(x[1][0] for x in new_flow)
                        ymin = min(x[1][1] for x in new_flow)
                        xmax = max(x[1][2] for x in new_flow)
                        ymax = max(x[1][3] for x in new_flow)
                        pos = (xmin, ymin, xmax, ymax)
                        genres[this_section].insert(-1, (new_flow, pos))
                # Prepare for the next section
                genres[this_section][-1][0].extend(genres[sub_section])
                genres[sub_section] = []
        yield sort_struct(genres["page"][0], multi_col=multi_col)


def parse(path, multi_col=True):
    """Parses a given PDF into a collection of words and associated data"""
    raw_lines = _extract_raw(path)
    raw_pages = _extract_pages(raw_lines)
    parsed_pages = _parse_pages(raw_pages, multi_col=multi_col)
    return parsed_pages


if __name__ == "__main__":
    raise RuntimeError("This file is importable, but not executable")
