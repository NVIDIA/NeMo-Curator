import argparse 
import os
import shutil
from typing import Any

from docbuilder import TedTalksDownloader
from helpers import write_jsonl

from nemo_curator import ParallelScoreFilter, JointScoreFilter, Sequential
from nemo_curator.filters import LengthRatioFilter, HistogramFilter
from nemo_curator.datasets import ParallelDataset
from nemo_curator.utils.script_utils import ArgumentHelper
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import get_all_files_paths_under




TED_DE_URL = "https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/t/en-de/raw/ted_dev_en-de.raw.de"
TED_EN_URL = "https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/t/en-de/raw/ted_dev_en-de.raw.en"
SRC_LANG="en"
TGT_LANG="de"

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "data")
JSONL_ROOT_DIR = os.path.join(DATA_DIR, "jsonl")


def download_and_convert_to_jsonl() -> str:
    downloader = TedTalksDownloader(DATA_DIR)

    # ted files is a dict of filenames, 'src', 'tgt' keys and their corresponding file paths as values
    ted_files = downloader.download(TED_EN_URL, TED_DE_URL, force=False)
    src_jsonl_dir = os.path.join(JSONL_ROOT_DIR, "src_val")
    tgt_jsonl_dir = os.path.join(JSONL_ROOT_DIR, "tgt_val")
    write_jsonl(ted_files['src'], src_jsonl_dir, force=True)
    write_jsonl(ted_files['tgt'], tgt_jsonl_dir, force=True)
    
    return src_jsonl_dir, tgt_jsonl_dir
    
def filter_dataset(dataset: ParallelDataset) -> ParallelDataset:
    filters = Sequential(
        [
            JointScoreFilter(
                LengthRatioFilter(max_ratio=2, src_lang=SRC_LANG,
                tgt_lang=TGT_LANG),
                score_field='length_ratio',
                score_type=float,
                
            ),
            ParallelScoreFilter(
                HistogramFilter(lang=SRC_LANG),
                HistogramFilter(lang=TGT_LANG),
                src_score='src_hist',
                tgt_score='tgt_hist',
                score_type=int,
            )
        ]
    )
    filtered_dataset = filters(dataset)
    return filtered_dataset

def run_curation_pipeline(args: Any, src_jsonl_dir: str, tgt_jsonl_dir: str) -> None:
    # Initialize the Dask cluster.
    client = get_client(**ArgumentHelper.parse_client_args(args))
    print(f"Running curation pipeline on '{src_jsonl_dir} and {tgt_jsonl_dir}'...")
    src_files = [
        fp
        for fp in get_all_files_paths_under(src_jsonl_dir, recurse_subdirectories=False)
        if fp.endswith(".jsonl")
    ]
    tgt_files = [
        fp
        for fp in get_all_files_paths_under(tgt_jsonl_dir, recurse_subdirectories=False)
        if fp.endswith(".jsonl")
    ]
    print("Reading the data...")
    bitext_dataset = ParallelDataset.read_simple_bitext(src_files, tgt_files,
                                                        SRC_LANG, TGT_LANG
                                                        )
    curation_steps = Sequential(
        [
            filter_dataset,
        ]
    )

    dataset = curation_steps(bitext_dataset)
    print("Executing the pipeline...")
    dataset = dataset.persist()

    print(f"Original dataset length: {len(bitext_dataset.df)}")
    print(f"After dataprep: {len(dataset.df)}")
    print("Writing the results to disk...")

    raise NotImplementedError("writing not finished yet, filters not checked")
    # # Overwrite existing files in the curated directory.
    # out_path = os.path.join(DATA_DIR, "curated")

    # if os.path.isdir(out_path):
    #     shutil.rmtree(out_path)

    # os.makedirs(out_path)
    # dataset.to_json(out_path, write_to_filename=True)
    client.close()


def main():
    parser = argparse.ArgumentParser()
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 8)

    src_jsonl_val_dir, tgt_jsonl_val_dir = download_and_convert_to_jsonl()
    run_curation_pipeline(args, src_jsonl_val_dir, tgt_jsonl_val_dir)

if __name__ == "__main__":
    main()