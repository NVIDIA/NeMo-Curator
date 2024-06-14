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
import argparse
import os


class ArgumentHelper:
    """
    A helper class to add common arguments to an argparse.ArgumentParser instance.
    """

    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser

    @staticmethod
    def __attach_bool_arg(
        parser: argparse.ArgumentParser,
        flag_name: str,
        default: bool = False,
        help: str = None,
    ):
        attr_name = flag_name.replace("-", "_")
        help = flag_name.replace("-", " ") if help is None else help
        parser.add_argument(
            "--{}".format(flag_name),
            dest=attr_name,
            action="store_true",
            help=help,
        )
        parser.add_argument(
            "--no-{}".format(flag_name),
            dest=attr_name,
            action="store_false",
            help=help,
        )

        parser.set_defaults(**{attr_name: default})

    def add_arg_batch_size(
        self,
        default: int = 64,
        help: str = "Number of files to read into memory at a time.",
    ):
        self.parser.add_argument(
            "--batch-size",
            type=int,
            default=default,
            help=help,
        )

    def add_arg_device(self):
        self.parser.add_argument(
            "--device",
            type=str,
            default="gpu",
            help="Device to run the script on. Either 'cpu' or 'gpu'.",
        )

    def add_arg_enable_spilling(self):
        self.parser.add_argument("--enable-spilling", action="store_true")

    def add_arg_language(self, help: str):
        self.parser.add_argument(
            "--language",
            type=str,
            default="en",
            help=help,
        )

    def add_arg_log_dir(self, default: str):
        self.parser.add_argument(
            "--log-dir",
            type=str,
            default=default,
            help="The output log directory where node and local"
            " ranks will write their respective log files",
        )

    def add_arg_input_data_dir(
        self,
        help: str = "Input directory consisting of .jsonl files that are accessible "
        "to all nodes. Use this for a distributed file system",
    ):
        self.parser.add_argument(
            "--input-data-dir",
            type=str,
            default=None,
            help=help,
        )

    def add_arg_input_file_type(
        self,
        choices=None,
        help="File type of the dataset to be read in. Supported file formats "
        "include 'jsonl' (default), 'pickle', or 'parquet'.",
    ):
        self.parser.add_argument(
            "--input-file-type",
            type=str,
            default="jsonl",
            choices=choices,
            help=help,
        )

    def add_arg_input_local_data_dir(self):
        self.parser.add_argument(
            "--input-local-data-dir",
            type=str,
            default=None,
            help="Input directory consisting of dataset files. "
            "Use this argument when a distributed file system is not available.",
        )

    def add_arg_input_meta(self):
        self.parser.add_argument(
            "--input-meta",
            type=str,
            default=None,
            help="A string formatted as a dictionary, which outlines the field names and "
            "their respective data types within the JSONL input files.",
        )

    def add_arg_input_text_field(self):
        self.parser.add_argument(
            "--input-text-field",
            type=str,
            default="text",
            help="The name of the field within each datapoint object of the input "
            "file that contains the text.",
        )

    def add_arg_minhash_length(self):
        self.parser.add_argument(
            "--minhash-length",
            type=int,
            default=260,
            help="The minhash signature length of each input document",
        )

    def add_arg_nvlink_only(self):
        self.parser.add_argument(
            "--nvlink-only",
            action="store_true",
            help="Start a local cluster with only NVLink enabled."
            "Only applicable when protocol=ucx and no scheduler file/address is specified",
        )

    def add_arg_output_data_dir(self, help: str):
        self.parser.add_argument(
            "--output-data-dir",
            type=str,
            required=True,
            help=help,
        )

    def add_arg_output_dir(
        self, required=False, help: str = "The output directory to write results in"
    ):
        self.parser.add_argument(
            "--output-dir",
            type=str,
            required=required,
            help=help,
        )

    def add_arg_output_file_type(
        self,
        choices=None,
        help="File type the dataset will be written to. Supported file formats "
        "include 'jsonl' (default), 'pickle', or 'parquet'.",
    ):
        self.parser.add_argument(
            "--output-file-type",
            type=str,
            default="jsonl",
            choices=choices,
            help=help,
        )

    def add_arg_output_train_file(self, help: str, default: str = None):
        self.parser.add_argument(
            "--output-train-file",
            type=str,
            default=default,
            help=help,
        )

    def add_arg_protocol(self):
        self.parser.add_argument(
            "--protocol",
            type=str,
            default="ucx",
            help="Protcol to use for dask cluster. "
            "Note: This only applies to the localCUDACluster. If providing an user created "
            "cluster refer to "
            "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-protocol",  # noqa: E501
        )

    def add_arg_rmm_pool_size(self):
        self.parser.add_argument(
            "--rmm-pool-size",
            type=str,
            default="14GB",
            help="Initial pool size to use for the RMM Pool Memory allocator. "
            "Note: This only applies to the localCUDACluster. If providing an user created "
            "cluster refer to "
            "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-rmm-pool-size",  # noqa: E501
        )

    def add_arg_scheduler_address(self):
        self.parser.add_argument(
            "--scheduler-address",
            type=str,
            default=None,
            help="Address to the scheduler of a created dask cluster. If not provided"
            "a single node LocalCUDACluster will be started.",
        )

    def add_arg_scheduler_file(self):
        self.parser.add_argument(
            "--scheduler-file",
            type=str,
            default=None,
            help="Path to the scheduler file of a created dask cluster. If not provided"
            " a single node LocalCUDACluster will be started.",
        )

    def add_arg_seed(
        self,
        default=42,
        help: str = "If specified, the random seed used for shuffling.",
    ):
        self.parser.add_argument(
            "--seed",
            type=int,
            default=default,
            help=help,
        )

    def add_arg_set_torch_to_use_rmm(self):
        self.parser.add_argument("--set-torch-to-use-rmm", action="store_true")

    def add_arg_shuffle(self, help: str):
        ArgumentHelper.__attach_bool_arg(
            self.parser,
            "shuffle",
            help=help,
        )

    def add_arg_text_ddf_blocksize(self):
        self.parser.add_argument(
            "--text-ddf-blocksize",
            type=int,
            default=256,
            help="The block size for chunking jsonl files for text ddf in mb",
        )

    def add_args_add_id(self):
        self.add_distributed_args()
        self.parser.add_argument(
            "--id-field-name",
            type=str,
            default="adlr_id",
            help="The name of the field that will contain the id value. "
            "Default is 'adlr_id'",
        )
        self.parser.add_argument(
            "--id-prefix",
            type=str,
            default="doc_id",
            help="The prefix to the id number that will be assigned to the "
            "document. When performing deduplication jointly with different"
            "datasets, it is helpful to provide a prefix that denotes that a "
            "document belongs to a particular dataset (e.g., wiki for documents"
            "that come from the wikipedia dataset)",
        )
        self.parser.add_argument(
            "--starting-index",
            type=int,
            default=None,
            help="If supplied, determines the starting index from which to start "
            "indexing the documents. By default, it is unspecified, and uses an id"
            " scheme that is fast to calculate and is not guaranteed to be ordered.",
        )

    def add_args_blend_datasets(self):
        self.add_distributed_args()
        self.parser.add_argument(
            "--input-data-dirs",
            type=str,
            default=None,
            help="Comma-separated list of directories consisting of dataset "
            "files that are accessible to all nodes.",
        )
        self.parser.add_argument(
            "--target-samples",
            type=int,
            default=10000,
            help="The number of samples to be included in the output dataset."
            " There may be more samples in order to accurately reflect the "
            "weight balance, but there will never be less",
        )
        self.parser.add_argument(
            "--weights",
            type=str,
            default=None,
            help="Comma-separated list of floating-point weights corresponding "
            "to each dataset passed in --input-data-dirs",
        )

    def add_args_common_crawl(self):
        self.parser.add_argument(
            "--cc-data-domain-prefix",
            type=str,
            default="https://data.commoncrawl.org",
            help="The prefix that will be prepended to each WARC "
            "file to create the URL. By default this value is "
            " 'https://data.commoncrawl.org'",
        )
        self.parser.add_argument(
            "--cc-index-prefix",
            type=str,
            default="https://index.commoncrawl.org",
            help="The prefix of the URL to the Common Crawl index. "
            "By default this value is 'https://index.commoncrawl.org'",
        )
        self.parser.add_argument(
            "--output-warc-url-file",
            type=str,
            default=None,
            required=True,
            help="The output file to which the WARC urls will be written",
        )
        self.parser.add_argument(
            "--starting-snapshot",
            type=str,
            default="2020-50",
            help="The starting snapshot to download. All WARC urls will be written "
            "between the dates specified by --starting-snapshot "
            "and --ending-snapshot. Snapshots must be specified by YYYY-WeekNumber "
            "(e.g., '2020-50' or '2021-04'). For the CC-NEWS dataset, "
            "(specified with the '--cc-news' flag) this changes to "
            "Year-Month (YYYY-MM)",
        )
        self.parser.add_argument(
            "--ending-snapshot",
            type=str,
            default="2020-50",
            help="The last snapshot for which WARC urls will be retrieved. "
            "Snapshots must be specified by YYYY-WeekNumber "
            "(e.g., '2020-50' or '2021-04')",
        )

        self.__(
            self.parser,
            "cc-news",
            help="Specify --cc-news in order to download WARC URLs for "
            "the CC-NEWS dataset instead of the CC-MAIN datasets. If this "
            "is specified, then it is assumed that the format for the start "
            "and end snapshots is 'YYYY-MM' (Year-Month). All WARC URLs between "
            "the specified years and months will be download",
        )

    def add_args_compute_minhashes(self):
        self.parser.add_argument(
            "--char-ngram",
            type=int,
            default=5,
            help="The number of consecutive characters to include in a sliding "
            "window when creating the document shingles for computing "
            "minhash signatures.",
        )
        self.parser.add_argument(
            "--hash-bytes",
            type=int,
            default=4,
            help="Number of bytes per computed minhash "
            "(default is an unsigned 32-bit integer)",
        )
        self.parser.add_argument(
            "--output-minhash-dir",
            type=str,
            required=True,
            help="Output directory where minhashes will be written. "
            "Each file is a parquet file that contains two series, the document ids, "
            "and a series of lists, each list denoting the minhash signature for that document id.",
        )

    def add_args_connected_components(self):
        self.parser.add_argument(
            "--cache-dir",
            type=str,
            help="The cache directory to write intermediate results to",
        )
        self.parser.add_argument(
            "--jaccard-pairs-path",
            type=str,
            help="The directory containing the jaccard results",
        )
        self.parser.add_argument(
            "--jaccard-threshold",
            type=int,
            default=0.8,
            help="Jaccard threshold below which we don't consider documents"
            " to be duplicate",
        )

    def add_args_create_k8s_dask_cluster(self):
        self.parser.add_argument(
            "-n",
            "--name",
            type=str,
            default="rapids-dask",
            help="The name of the DaskCluster which you would be able to inspect via `kubectl describe daskcluster <name>`.",
        )
        self.parser.add_argument(
            "-w", "--n_workers", type=int, default=2, help="Number of workers"
        )
        self.parser.add_argument(
            "-g",
            "--n_gpus_per_worker",
            type=int,
            default=None,
            help="Number of GPUs per worker. If not specified, the Dask Cluster defaults to a CPU cluster.",
        )
        self.parser.add_argument(
            "-c",
            "--n_cpus_per_worker",
            type=int,
            default=None,
            help="Number of CPUs per worker. Provide this flag if you want to limit your CPU resources and K8s will throttle the workers to make sure this limit is satisfied.",
        )
        self.parser.add_argument(
            "-i",
            "--image",
            type=str,
            default="nvcr.io/nvidia/nemo:24.03.framework",
            help="The image used for the Dask Cluster scheduler and workers.",
        )
        self.parser.add_argument(
            "-s",
            "--image_pull_secret",
            type=str,
            default=None,
            help="If --image is from a private registry, specify the appropriate pull secret you created to allow these to be pulled.",
        )
        self.parser.add_argument(
            "-p",
            "--pvcs",
            type=parse_pvcs,
            default="",
            help="Comma sep PVC specificiation of $pvc_name_1:$mount_path_1,$pvc_name_2:$mount_path_2. Example: foo:/foo,bar:/bar mounts pvcs named foo and bar to /foo and /bar respectively.",
        )

    def add_args_download_and_extract(self):
        self.add_distributed_args()
        self.parser.add_argument(
            "--builder-config-file",
            type=str,
            required=True,
            help="YAML file that contains paths to implementations of a downloader, "
            "iterator and extractor that will be used in this program "
            "to build the documents that make up the output dataset",
        )
        ArgumentHelper.__attach_bool_arg(
            self.parser,
            "download-only",
            help="Specify this flag if you desire to only download the data"
            "files and not extract text from the downloaded files",
        )
        ArgumentHelper.__attach_bool_arg(
            self.parser,
            "keep-downloaded-files",
            help="If this flag is set to true, the downloaded data files "
            "will be kept on disk and not removed after extraction",
        )
        ArgumentHelper.__attach_bool_arg(
            self.parser,
            "overwrite-existing-json",
            help="If this flag is specified, then the json data will be "
            "overwritten if downloading from the the same file.",
        )
        self.parser.add_argument(
            "--output-json-dir",
            type=str,
            default=None,
            help="Output directory to store the extracted text in jsonl files",
        )
        self.parser.add_argument(
            "--output-download-dir",
            type=str,
            default=None,
            help="The directory to where data files will be written "
            "in 'download-only' mode. Specify this argument only when "
            "the '--download-only flag is specified'.",
        )
        self.parser.add_argument(
            "--input-url-file",
            type=str,
            default=None,
            help="Input directory consisting of .jsonl files that are accessible "
            "to all nodes. Use this for a distributed file system",
        )

    def add_args_find_exact_duplicates(self):
        self.parser.add_argument(
            "--hash-method",
            type=str,
            default="md5",
            help="Hash Method to use for exact dedup",
        )

    def add_args_find_matching_ngrams(self):
        self.add_distributed_args()
        self.parser.add_argument(
            "--input-task-ngrams",
            type=str,
            default=None,
            help="",
        )
        self.parser.add_argument(
            "--max-ngram-size",
            type=int,
            default=13,
            help="The maximum n-gram size to consider within the dataset",
        )
        self.parser.add_argument(
            "--min-ngram-size",
            type=int,
            default=8,
            help="The minimum n-gram size to consider within the datset",
        )
        self.parser.add_argument(
            "--output-matched-ngram-data",
            type=str,
            default=None,
            help="Output dictionary that contains the output matched n-grams "
            "and the frequency of their matches, min-ngram size, max-ngram "
            "size and the frequencies of n-gram sizes. All of these data will be "
            "used by remove_matching_grams for which this program is a prequisite",
        )

    def add_args_find_pii_and_deidentify(self):
        self.parser.add_argument(
            "--anonymize-action",
            type=str,
            default="replace",
            help="Anonymization action. Choose from among: redact, hash, mask and replace",
        )
        self.parser.add_argument(
            "--chars-to-mask",
            type=int,
            default=100,
            help="The number of characters to mask. Only applicable if anonymize action is mask",
        )
        self.parser.add_argument(
            "--hash-type",
            type=str,
            default=None,
            help="The hash type. Choose from among: sha256, sha512 or md5",
        )
        self.parser.add_argument(
            "--masking-char",
            type=str,
            default="*",
            help="The masking character. Only applicable if anonymize action is mask",
        )
        self.parser.add_argument(
            "--new-value",
            type=str,
            default=None,
            help="The new value to replace with. Only applicable if anonymize action is replace",
        )
        self.parser.add_argument(
            "--supported-entities",
            type=str,
            default=None,
            help="Comma separated list of PII entity types. None implies all supported types",
        )
        self.parser.add_argument(
            "--text-field",
            type=str,
            default="text",
            help="The input field within each JSONL or CSV object on which the PII redactor will "
            "operate. By default, the redactor will operate on the 'text' "
            "field but other fields can be specified such as 'url' or 'id'.",
        )

    def add_args_filter_documents(self):
        self.add_distributed_args()
        self.parser.add_argument(
            "--filter-config-file",
            type=str,
            required=True,
            help="The input filter configuration file that contains the "
            "path to the filter module as well as the filter parameters",
        )
        ArgumentHelper.__attach_bool_arg(
            self.parser,
            "filter-only",
            default=False,
            help="Specifying this flag will indicate to the code that only the "
            "filtering operation should be performed and that scores should not be "
            "computed. This flag should be specified if scores have been "
            "pre-computed on the documents (e.g., the code was run without the "
            "'--output-retained-document-dir' argument) and users desire to apply "
            "the filter using the pre-computed scores",
        )
        self.parser.add_argument(
            "--id-field",
            type=str,
            default="adlr_id",
            help="The name of the field within each object of the dataset "
            "file that assigns a unqiue ID to each document. "
            "If this is specified and found within the object, a list of all "
            "ids will be written to the output score directory such that each line"
            "is consistent with the lines of the written score files ",
        )
        ArgumentHelper.__attach_bool_arg(
            self.parser,
            "keep-node-scores-tmp-dir",
            default=False,
            help="If multiple nodes are used when computing scores, "
            "each node will write out its scores to a temporary directory "
            "shared across all nodes. Then, the rank 0 node will "
            "concatenate all of the scores creating the output file. "
            "By default, this directory is removed after concatenation, "
            "however users can keep this temporary directory by specifying "
            "the flag --keep-node-scores-tmp-dir ",
        )
        self.parser.add_argument(
            "--log-frequency",
            type=int,
            default=10000,
            help="The frequency with which to write log messages when "
            "computing scores. By default a log message will "
            "be written every 10000 documents in a file",
        )
        ArgumentHelper.__attach_bool_arg(
            self.__parser,
            "log-scores",
            default=False,
            help="Specifying this flag will cause the computed scores to be "
            "logged as additional keys for each document. This only applies to "
            "filters with 'log_score: True' in the config. This can aid in "
            "performing an interactive quality check of the documents.",
        )
        self.parser.add_argument(
            "--output-document-score-dir",
            type=str,
            default=None,
            help="The output directory to where the computed document scores will "
            "be written. For each filter, its score will be written to a separate "
            "file where each line of the file corresponds to the score computed "
            "for each document in the corpus within this directory. This only applies to "
            "filters with 'log_score: True' in the config. If this directory is not "
            "specified, then filter scores will not be written",
        )
        self.parser.add_argument(
            "--output-removed-document-dir",
            type=str,
            default=None,
            help="The output directory to where documents that are removed during "
            "filtering will be written. This argument is mainly for quality control "
            "in order examine documents that are not preserved during filtering. "
            "If it is not specified and the retained-document-dir is specified, "
            "then only the retained documents will be written to disk",
        )
        self.parser.add_argument(
            "--output-retained-document-dir",
            type=str,
            default=None,
            help="The output directory to where documents that are "
            "retained during filtering will be written. If this argument "
            "is not specified, then the document scores from the "
            "filter(s) will be written to the document meta data in place",
        )

    def add_args_jaccard_compute(self):
        self.parser.add_argument(
            "--ngram-size",
            type=int,
            default=5,
            help="Size of ngram to use during jaccard similarity",
        )
        self.parser.add_argument(
            "--shuffled-docs-path",
            type=str,
            help="The directory containing the shuffled documents",
        )

    def add_args_jaccard_shuffle(self):
        self.parser.add_argument(
            "--bucket-mapping-ddf-blocksize",
            type=int,
            default=256,
            help="The block size for for anchor_docs_with_bk ddf in mb",
        )
        self.parser.add_argument(
            "--bucket-parts-per-worker",
            default=8,
            type=int,
            help="The number of bucket parts to process per worker per batch",
        )
        self.parser.add_argument(
            "--input-bucket-mapping-dir",
            type=str,
            help="The directory containing anchor docs with bk files",
        )
        self.parser.add_argument(
            "--parts-per-worker",
            default=1,
            type=int,
            help="The number of parts to process per worker per batch",
        )

    def add_args_make_data_shards(self):
        self.add_distributed_args()
        self.parser.add_argument(
            "--output-file-size",
            type=str,
            default="100M",
            help="Approximate size of output files. Must specify with a string and "
            "with the unit K, M or G for kilo, mega or gigabytes",
        )
        self.parser.add_argument(
            "--output-resharded-dir",
            type=str,
            default=None,
            required=True,
            help="Output directory to where the sharded "
            ".jsonl files will be written",
        )
        self.parser.add_argument(
            "--prefix",
            type=str,
            default="",
            help="Prefix to use to prepend to output file number",
        )
        self.parser.add_argument(
            "--start-index",
            type=int,
            default=0,
            help="Starting index for naming the output files",
        )

    def add_args_map_buckets(self):
        self.parser.add_argument(
            "--input-bucket-dir",
            type=str,
            help="The directory containing bucket information files",
        )
        self.parser.add_argument(
            "--input-bucket-field",
            type=str,
            default="_bucket_id",
            help="Name of the column containing minhashes",
        )
        self.parser.add_argument(
            "--shuffle-type",
            type=str,
            default="tasks",
            help="Type of shuffle to use before writing to parquet",
        )

    def add_args_minhash_lsh(self):
        self.parser.add_argument(
            "--buckets-per-shuffle",
            type=int,
            required=True,
            help="Number of buckets to shuffle per batch",
        )
        self.parser.add_argument(
            "--input-minhash-field",
            type=str,
            default="_minhash_signature",
            help="Name of the column containing minhashes",
        )
        self.parser.add_argument(
            "--num-bands",
            type=int,
            default=20,
            help="The number of minhashes to compute for each document.",
        )
        self.parser.add_argument(
            "--output-bucket-dir",
            type=str,
            required=True,
            help="Output directory where minhashes will be written. "
            "Each file parquet file consiting of document and bucket IDs",
        )

    def add_args_prepare_task_data(self):
        self.add_distributed_args()
        self.parser.add_argument(
            "--task-config-file",
            type=str,
            default=None,
            required=True,
            help="YAML configuration file that contains task information. "
            "YAML files for already implemented tasks can be found in the config "
            "directory that is located in the root directory of this repository.",
        )
        self.parser.add_argument(
            "--output-task-ngrams",
            type=str,
            default="./task_ngrams.pkl",
            help="N-grams computed from input task data. N-grams are stored "
            "as keys to a dictionary and the values of the dictionary "
            "are the frequencies of which the n-grams occurr within a "
            "training dataset (they are initialized to zero within this program)",
        )

    def add_args_prepare_fasttext_training_data(self):
        self.add_distributed_args()
        self.parser.add_argument(
            "--input-json-field",
            type=str,
            default="text",
            help="The input field within each JSON object on which the filter will "
            "operate. By default, the filter will operate on the 'text' "
            "field but other fields can be specified such as 'url' or 'id'.",
        )
        self.parser.add_argument(
            "--label",
            type=str,
            default=None,
            required=True,
            help="The label to be used at the beginning of each sample "
            "in the output file. For example '__label__hq' could be "
            "used for the high-quality (positive) samples",
        )

    def add_args_quality_classifier_inference(self):
        self.parser.add_argument("--num-labels", type=int, default=3)

    def add_args_remove_matching_ngrams(self):
        self.add_distributed_args()
        self.parser.add_argument(
            "--input-matched-ngrams",
            type=str,
            default=None,
            required=True,
            help="Input dictionary (.pkl file), that contains matched "
            "n-gram data from the find_matching_ngrams code",
        )
        self.parser.add_argument(
            "--match-threshold",
            type=int,
            default=10,
            help="A threshold that determines if a matched n-gram will be "
            "considered for removal in remove_matching_ngrams. N-grams that "
            "exceed this number of matches in the training dataset will not be "
            "considered during the removal stage",
        )
        self.parser.add_argument(
            "--max-document-splits",
            type=int,
            default=10,
            help="A threshold used to determine if a document should be removed "
            "from the corpus if it is split more than "
            "--max-document-splits number of times",
        )
        self.parser.add_argument(
            "--output-removed-doc-dir",
            type=str,
            default=None,
            help="Output directory to where removed documents will be written. "
            "Documents will be removed from the corpus if they are split more "
            "than --max-document-splits number of times, or if the user specifies "
            "that they be removed via the flag, --remove-split-docs",
        )
        self.parser.add_argument(
            "--output-task-deduped-dir",
            type=str,
            default=None,
            required=True,
            help="Output directory to where task-deduplicated (split) "
            "documents will be written",
        )

    def add_args_sample_dataframe(self):
        self.parser.add_argument(
            "--num_samples",
            type=int,
            help="The number of rows to sample",
            required=True,
        )

    def add_args_separate_by_metadata(self):
        self.add_distributed_args()
        self.parser.add_argument(
            "--input-metadata-field",
            type=str,
            default="language",
            help="The name of the field within each datapoint object of the input "
            "file that the dataset should be separated by.",
        )
        self.parser.add_argument(
            "--output-metadata-distribution",
            type=str,
            help="Output json file containing the frequency of documents "
            "that occur for a particular metadata.",
        )
        ArgumentHelper.__attach_bool_arg(
            self.parser,
            "remove-input-dir",
            default=False,
            help="Specify '--remove-input-dir' to remove the original "
            "input directory. This is false by default.",
        )
        ArgumentHelper.__attach_bool_arg(
            self.parser,
            "remove-metadata-field",
            default=False,
            help="Option of whether to remove the metadata field "
            "after filtering. Useful only in the case in which one metadata "
            "is desired to be separated from the others",
        )

    def add_args_text_cleaning(self):
        self.add_distributed_args()
        self.parser.add_argument(
            "--output-clean-dir",
            type=str,
            required=True,
            help="The output directory to where the cleaned "
            "jsonl files will be written",
        )

    def add_args_train_fasttext(self):
        self.add_arg_seed(default=1992)
        self.parser.add_argument(
            "--fasttext-files-dir",
            type=str,
            default=None,
            required=True,
            help="The input directory containing the file(s) "
            "containing the prepared FastText samples",
        )
        self.parser.add_argument(
            "--high-quality-label",
            type=str,
            default="__label__hq",
            help="The label assigned to the high quality samples "
            "when preparing the data",
        )
        self.add_arg_output_train_file(
            help="The concatenated, shuffled samples used "
            "to train the skip-gram classifier",
            default="./fasttext_samples.train",
        )
        self.parser.add_argument(
            "--output-validation-file",
            type=str,
            default="./fasttext_samples.valid",
            help="The concatenated, shuffled samples used to "
            "for computing validation metrics",
        )
        self.parser.add_argument(
            "--validation-split",
            type=float,
            default=0.9,
            help="The training validation split",
        )
        self.parser.add_argument(
            "--output-model",
            type=str,
            default=None,
            required=True,
            help="The output trained skip-gram classifier written "
            "as a FastText model",
        )
        self.parser.add_argument(
            "--wordNgrams",
            type=int,
            default=2,
            help="The size of the word n-gram used to train the classifier "
            "(default is bigram)",
        )
        self.parser.add_argument(
            "--learning-rate",
            type=float,
            default=0.1,
            help="The learning rate used to train the classifier",
        )
        self.parser.add_argument(
            "--num-epochs",
            type=int,
            default=5,
            help="Number of epochs used to train the classifier",
        )
        self.parser.add_argument(
            "--word-vector-dim",
            type=int,
            default=100,
            help="Size of word vectors to be computed by the model",
        )
        self.parser.add_argument(
            "--output-predictions",
            type=str,
            default=None,
            help="The output predictions on the validation data. If a file "
            "is not specified, the predictions are not written to file",
        )

    def add_args_verify_classification(self):
        self.parser.add_argument(
            "--results_file_path",
            type=str,
            required=True,
            help="The path of the input files",
        )
        self.parser.add_argument(
            "--expected_results_file_path",
            type=str,
            required=True,
            help="The path of the expected_result file",
        )
        self.parser.add_argument(
            "--results_pred_column",
            type=str,
            default="pred",
            help="The prediction column name for the input files",
        )
        self.parser.add_argument(
            "--expected_pred_column",
            type=str,
            default="pred",
            help="The prediction column name for the expected_result file",
        )

    def add_args_wikipedia(self):
        self.add_arg_language(help="Desired language of the Wikipedia dump")
        self.parser.add_argument(
            "--wikidumps-index-baseurl",
            type=str,
            default="https://dumps.wikimedia.org",
            help="The base url for all Wikipedia dumps",
        )
        self.parser.add_argument(
            "--output-url-file",
            type=str,
            default="wikipedia_urls_latest.txt",
            help="The output file to which the urls containing "
            "the latest dump data will be written",
        )

    def add_distributed_args(self) -> argparse.ArgumentParser:
        """
        Adds default set of arguments that are needed for Dask cluster setup
        """
        self.parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="Device to run the script on. Either 'cpu' or 'gpu'.",
        )
        self.parser.add_argument(
            "--files-per-partition",
            type=int,
            default=2,
            help="Number of jsonl files to combine into single partition",
        )
        self.parser.add_argument(
            "--n-workers",
            type=int,
            default=os.cpu_count(),
            help="The number of workers to run in total on the Dask CPU cluster",
        )
        self.parser.add_argument(
            "--num-files",
            type=int,
            default=None,
            help="Upper limit on the number of json files to process",
        )
        self.parser.add_argument(
            "--nvlink-only",
            action="store_true",
            help="Start a local cluster with only NVLink enabled."
            "Only applicable when protocol=ucx and no scheduler file/address is specified",
        )
        self.parser.add_argument(
            "--protocol",
            type=str,
            default="tcp",
            help="Protcol to use for dask cluster"
            "Note: This only applies to the localCUDACluster. If providing an user created "
            "cluster refer to"
            "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-protocol",  # noqa: E501
        )
        self.parser.add_argument(
            "--rmm-pool-size",
            type=str,
            default=None,
            help="Initial pool size to use for the RMM Pool Memory allocator"
            "Note: This only applies to the LocalCUDACluster. If providing an user created "
            "cluster refer to"
            "https://docs.rapids.ai/api/dask-cuda/stable/api.html#cmdoption-dask-cuda-rmm-pool-size",  # noqa: E501
        )
        self.parser.add_argument(
            "--scheduler-address",
            type=str,
            default=None,
            help="Address to the scheduler of a created dask cluster. If not provided"
            "a single node Cluster will be started.",
        )
        self.parser.add_argument(
            "--scheduler-file",
            type=str,
            default=None,
            help="Path to the scheduler file of a created dask cluster. If not provided"
            " a single node Cluster will be started.",
        )
        self.parser.add_argument(
            "--threads-per-worker",
            type=int,
            default=1,
            help="The number of threads ot launch per worker on the Dask CPU cluster. Usually best set at 1 due to the GIL.",
        )

        return self.parser

    @staticmethod
    def parse_client_args(args: argparse.Namespace):
        """
        Extracts relevant arguments from an argparse namespace to pass to get_client
        """
        relevant_args = [
            "scheduler_address",
            "scheduler_file",
            "n_workers",
            "threads_per_worker",
            "nvlink_only",
            "protocol",
            "rmm_pool_size",
            "enable_spilling",
            "set_torch_to_use_rmm",
        ]
        dict_args = vars(args)

        parsed_args = {arg: dict_args[arg] for arg in relevant_args if arg in dict_args}
        if "device" in dict_args:
            parsed_args["cluster_type"] = dict_args["device"]

        return parsed_args

    @staticmethod
    def parse_distributed_classifier_args(
        description="Default distributed classifier argument parser",
    ) -> argparse.ArgumentParser:
        """
        Adds default set of arguments that are common to multiple stages
        of the pipeline
        """
        parser = argparse.ArgumentParser(
            description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser = ArgumentHelper(parser).add_distributed_args()
        # Set low default RMM pool size for classifier
        # to allow pytorch to grow its memory usage
        # by default
        parser.set_defaults(rmm_pool_size="512MB")
        parser.add_argument(
            "--input-data-dir",
            type=str,
            help="The path of the input files",
            required=True,
        )
        parser.add_argument(
            "--output-data-dir",
            type=str,
            help="The path of the output files",
            required=True,
        )
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path to the model file",
            required=True,
        )
        parser.add_argument(
            "--input-file-type",
            type=str,
            help="The type of the input files",
            required=True,
        )
        parser.add_argument(
            "--output-file-type",
            type=str,
            default="jsonl",
            help="The type of the output files",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=128,
            help="The batch size to be used for inference",
        )
        ArgumentHelper.__attach_bool_arg(
            parser, "autocast", default=True, help="Whether to use autocast or not"
        )
        ArgumentHelper.__attach_bool_arg(
            parser,
            "enable-spilling",
            default=True,
            help="Whether to enable spilling or not",
        )

        # Setting to False makes it more stable for long running jobs
        # possibly because of memory fragmentation
        ArgumentHelper.__attach_bool_arg(
            parser,
            "set-torch-to-use-rmm",
            default=False,
            help="Whether to set torch to use RMM or not",
        )

        return parser

    @staticmethod
    def parse_gpu_dedup_args(description: str) -> argparse.ArgumentParser:
        """
        Adds default set of arguments that are common to multiple stages
        of the pipeline
        """

        parser = argparse.ArgumentParser(
            description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        argumentHelper = ArgumentHelper(parser)

        argumentHelper.add_distributed_args()

        # Set default device to GPU for dedup
        argumentHelper.parser.set_defaults(device="gpu")
        argumentHelper.parser.add_argument(
            "--input-data-dirs",
            type=str,
            nargs="+",
            default=None,
            help="Input directories consisting of .jsonl files that are accessible "
            "to all nodes. This path must be accessible by all machines in the cluster",
        )
        argumentHelper.parser.add_argument(
            "--input-json-text-field",
            type=str,
            default="text",
            help="The name of the field within each json object of the jsonl "
            "file that contains the text from which minhashes will be computed. ",
        )
        argumentHelper.parser.add_argument(
            "--input-json-id-field",
            type=str,
            default="adlr_id",
            help="The name of the field within each json object of the jsonl "
            "file that assigns a unqiue ID to each document. "
            "Can be created by running the script "
            "'./prospector/add_id.py' which adds the field 'adlr_id' "
            "to the documents in a distributed fashion",
        )
        argumentHelper.parser.add_argument(
            "--log-dir",
            type=str,
            default="./logs/",
            help="The output log directory where node and local",
        )
        argumentHelper.parser.add_argument(
            "--profile-path",
            type=str,
            default=None,
            help="Path to save dask profile",
        )

        return argumentHelper.parser
