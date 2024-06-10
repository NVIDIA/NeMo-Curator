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

from nemo_curator.utils.download_utils import get_common_crawl_urls
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):
    urls = get_common_crawl_urls(
        args.starting_snapshot,
        args.ending_snapshot,
        data_domain_prefix=args.cc_data_domain_prefix,
        index_prefix=args.cc_index_prefix,
        news=args.cc_news,
    )

    with open(args.output_warc_url_file, "w") as fp:
        for url in urls:
            fp.write(url)
            fp.write("\n")


def attach_args(
    parser=argparse.ArgumentParser(
        """
Pulls URLs of WARC files stored within the common crawl data repository
and writes them to file so that they can be used to subsequently
download the WARC files.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    ArgumentHelper(parser).add_common_crawl_args()

    return parser


def console_script():
    main(attach_args().parse_args())
