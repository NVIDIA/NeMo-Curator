import argparse
import json
import os
import random
from typing import Any, List
import dask
from openai import AsyncOpenAI
from omegaconf import OmegaConf

from nemo_curator import AsyncOpenAIClient, ScoreFilter, Sequential
from nemo_curator.datasets import DocumentDataset
from retriever_evalset_generator import RetrieverEvalSetGenerator
from filters import EasinessFilter, AnswerabilityFilter
from nemo_curator.modules.filter import ScoreFilter, Score
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir
from nemo_curator.utils.script_utils import ArgumentHelper



def run_sdg_pipeline(args: Any,
                     ) -> DocumentDataset: 
    
    if args.input_format == "rawdoc":
        input_dataset = DocumentDataset.read_json(args.input_file)
    else:
        raise ValueError("Error: Only rawdoc format supported")

    cfg = OmegaConf.load(args.pipeline_config)
    sdg_pipeline = Sequential(
        [
            RetrieverEvalSetGenerator(cfg),
            ScoreFilter(EasinessFilter(cfg),
                        text_field = ["text", "question"],
                        score_field = "easiness_scores"),
            ScoreFilter(AnswerabilityFilter(cfg),
                        text_field = ["text", "question"],
                        score_field = "answerability_scores"),
         ]
    )
    generated_dataset = sdg_pipeline(input_dataset)
    generated_dataset.persist()
    return generated_dataset
    
def write_to_beir(args: Any, dataset: DocumentDataset):
    dataset.to_json(args.output_dir) 

def main():
    parser = argparse.ArgumentParser()
    # parser = ArgumentHelper(parser).add_distributed_args()
    parser.add_argument(
        "--input_file",
        type=str,
        default="",
        help="File path of input file containing document chunks for synthetic data generation",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="rawdoc",  
        help="The synthetic data generation framework supports two input formats rawdoc or squad.",
    )
    parser.add_argument(
        "--pipeline_config",
        type=str,
        default="",  
        help="Pipeline configuartion yaml file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output dir for generated data",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="The API key to use for the synthetic data generation LLM client.",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=120,
        help="The timeout value for API calls in seconds.",
    )
    args = parser.parse_args()

    os.environ['NVIDIA_API_KEY'] = args.api_key

    generated_dataset = run_sdg_pipeline(args)
    write_to_beir(args, generated_dataset)

if __name__ == "__main__":
    main()