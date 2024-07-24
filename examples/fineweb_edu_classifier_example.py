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
import time

import torch
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from transformers import AutoConfig, AutoModelForSequenceClassification

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper

EDU_CLASSIFIER_PATH = "HuggingFaceTB/fineweb-edu-classifier"


class FinewebEduClassifier(HFModel):
    def __init__(self, path_or_name, max_mem_gb="16", autocast=False):
        self.path_or_name = path_or_name
        self.autocast = autocast
        super().__init__(path_or_name=path_or_name, max_mem_gb=max_mem_gb)

    def load_model(self, device="cuda"):
        model = AutoModelForSequenceClassification.from_pretrained(self.path_or_name)
        model = model.to(device)
        model = self.configure_forward(model, self.autocast)
        return model

    @staticmethod
    def configure_forward(model, autocast=True):
        original_forward = model.forward

        def custom_forward(*args, **kwargs):
            if autocast:
                with torch.cuda.amp.autocast():
                    output = original_forward(*args, **kwargs)
            return output.logits.squeeze(-1).float()

        model.forward = custom_forward
        return model

    def load_config(self):
        return AutoConfig.from_pretrained(self.path_or_name)


def run_pipeline(
    dataset: DocumentDataset, model: HFModel, batch_size: int = 256, input_column="text"
) -> DocumentDataset:

    ddf = dataset.df
    pipe = op.Sequential(
        op.Tokenizer(
            model,
            cols=[input_column],
            tokenizer_type="sentencepiece",
            max_length=model.max_seq_length(),
        ),
        op.Predictor(
            model,
            sorted_data_loader=True,
            batch_size=batch_size,
            pred_output_col="fineweb-edu-score",
        ),
        keep_cols=ddf.columns.tolist(),
    )
    ddf = pipe(ddf)
    # Go from list to scalar
    ddf["fineweb-edu-score"] = ddf["fineweb-edu-score"].list.get(0)
    return DocumentDataset(ddf)


def attach_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fineweb EDU Classifier",
    )
    argumentHelper = ArgumentHelper(parser)
    argumentHelper.add_distributed_args()
    argumentHelper.add_arg_input_data_dir(required=True)
    argumentHelper.add_arg_output_data_dir(help="Output directory to write the results")
    argumentHelper.add_arg_input_text_field()
    argumentHelper.add_arg_batch_size(help="The batch size to be used for inference")
    argumentHelper.add_arg_autocaset()
    argumentHelper.add_arg_max_mem_gb_classifier()

    parser.set_defaults(device="gpu")
    parser.set_defaults(rmm_pool_size="512MB")
    parser.set_defaults(set_torch_to_use_rmm=False)

    return parser


def main(args):
    global_st = time.time()

    client = get_client(**ArgumentHelper.parse_client_args(args))

    input_dataset = DocumentDataset.read_json(
        [args.input_data_dir], backend="cudf", add_filename=True
    )

    fineweb_edu_classifier = FinewebEduClassifier(
        EDU_CLASSIFIER_PATH,
        max_mem_gb=args.max_mem_gb_classifier,
        autocast=args.autocast,
    )

    result_dataset = run_pipeline(
        input_dataset,
        fineweb_edu_classifier,
        batch_size=args.batch_size,
        input_column=args.input_text_field,
    )

    result_dataset.to_json(output_file_dir=args.output_data_dir, write_to_filename=True)

    print(
        f"Total time taken for fine-web classifier inference: {time.time() - global_st} s",
        flush=True,
    )
    client.close()


if __name__ == "__main__":
    main(attach_args().parse_args())
