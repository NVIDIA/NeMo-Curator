import argparse 
import os
import shutil

from functools import partial
from typing import Any

from docbuilder import TedTalksDownloader

from nemo_curator import ParallelScoreFilter, JointScoreFilter, Sequential
from nemo_curator.filters import LengthRatioFilter, HistogramFilter, COMETQualityEstimationFilter
from nemo_curator.datasets import ParallelDataset
from nemo_curator.utils.script_utils import ArgumentHelper
from nemo_curator.utils.distributed_utils import get_client


TED_DE_URL = "https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/t/en-de/raw/ted_train_en-de.raw.de"
TED_EN_URL = "https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/t/en-de/raw/ted_train_en-de.raw.en"
SRC_LANG="en"
TGT_LANG="de"

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "data")


def download_files() -> str:
    downloader = TedTalksDownloader(DATA_DIR)

    # ted files is a dict of filenames, 'src', 'tgt' keys and their corresponding file paths as values
    ted_files = downloader.download(TED_EN_URL, TED_DE_URL, force=False)
    return ted_files['src'], ted_files['tgt']
    
def filter_dataset(dataset: ParallelDataset, gpu: bool = False) -> ParallelDataset:
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
            ),
            JointScoreFilter(
                COMETQualityEstimationFilter(gpu=gpu),
            )
        ]
    )
    filtered_dataset = filters(dataset) 
    return filtered_dataset

def run_curation_pipeline(args: Any, src_file: str, tgt_file: str) -> None:
    # Initialize the Dask cluster.
    client = get_client(**ArgumentHelper.parse_client_args(args))
    print(f"Running curation pipeline on '{src_file} and {tgt_file}'...")

    print("Reading the data...")
    
    bitext_dataset = ParallelDataset.read_simple_bitext(
        src_file,
        tgt_file,
        SRC_LANG,
        TGT_LANG,
        add_filename=True
    )
    curation_steps = Sequential(
        [
            partial(filter_dataset, gpu=(args.device == "gpu")),
        ]
    )

    dataset = curation_steps(bitext_dataset)
    print("Executing the pipeline...")
    dataset = dataset.persist()

    print(f"Original dataset length: {len(bitext_dataset.df)}")
    print(f"After dataprep: {len(dataset.df)}")
    print("Writing the results to disk...")

    #raise NotImplementedError("writing not finished yet, filters not checked")

    # Overwrite existing files in the curated directory.
    out_path = os.path.join(DATA_DIR, "curated")

    if os.path.isdir(out_path):
        shutil.rmtree(out_path)

    os.makedirs(out_path)
    dataset.to_bitext(out_path, write_to_filename=True)
    client.close()


def main():
    parser = argparse.ArgumentParser()
    args = ArgumentHelper(parser).add_distributed_args().parse_args()
    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 8)

    src_file, tgt_file = download_files()
    run_curation_pipeline(args, src_file, tgt_file)

if __name__ == "__main__":
    main()
