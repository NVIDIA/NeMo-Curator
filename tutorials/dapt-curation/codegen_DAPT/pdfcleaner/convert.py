import argparse
import gzip
import multiprocessing
import os
import sys
from pathlib import Path

from pdfcleaner.filters import academic_filter, manuals_filter
from pdfcleaner.parser import parse
from pdfcleaner.util import dehyphenate_page, dump_text, strip_pos

parser = argparse.ArgumentParser()
parser.add_argument("--path")
parser.add_argument("--outpath")
parser.add_argument(
    "--filter",
    default="academic",
    choices=[
        "academic",
        "manuals",
    ],
)
parser.add_argument("--clobber", action="store_true", help="Override existing files")
args = parser.parse_args()

# ensure args.path exists
if not os.path.exists(args.path):
    sys.exit(f"Error: {args.path} does not exist")

request = []
for root, dirs, files in os.walk(args.path):
    for filename in files:
        if filename.upper().endswith(".PDF"):
            request.append(os.path.join(root, filename))


def handle(filename):
    rel_path = os.path.relpath(filename, args.path)

    output_file_path = os.path.join(args.outpath, rel_path)

    # Create necessary subdirectories in output directory
    Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)

    outfile_full_path = output_file_path + ".txt.gz"

    if (not args.clobber) and os.path.isfile(outfile_full_path):
        # skip if file already exists
        outfile_full_path_safe = outfile_full_path.encode("utf-8", "replace").decode(
            "utf-8"
        )
        print(f"Output file {outfile_full_path_safe} already exists. Skipping...")
        return

    try:
        if args.filter == "academic":
            parsed_pages = parse(filename)
            parsed_pages = academic_filter(parsed_pages)
        elif args.filter == "manuals":
            parsed_pages = parse(filename, multi_col=False)
            parsed_pages = manuals_filter(parsed_pages)
        with gzip.open(outfile_full_path, "wt") as f:
            for page in parsed_pages:
                page = dump_text(dehyphenate_page(strip_pos(page))) + "\n"
                f.write(page)
    except Exception as e:
        print(
            f"Error {type(e)}: {filename.encode('utf-8', 'replace').decode('utf-8')}. Skipping..."
        )
        return


pool = multiprocessing.Pool()
for filename in request:
    pool.apply_async(handle, [filename])

pool.close()
pool.join()
