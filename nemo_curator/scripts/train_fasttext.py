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
import json
import os
import random

import fasttext
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args):

    # Set the random seed for shuffling
    random.seed(args.seed)

    # Read in all samples into a list and shuffle the samples
    documents = []
    for ifile in get_all_files_paths_under(args.fasttext_files_dir):
        if os.path.splitext(ifile)[-1] == ".txt":
            with open(ifile, "r") as fp:
                documents += fp.readlines()
    random.shuffle(documents)

    # Get total number of samples
    nlines = len(documents)

    num_train = round(nlines * args.validation_split)

    # Split into training and validation samples
    train_samples = documents[:num_train]
    valid_samples = documents[num_train:]

    # Write out the training and validation samples
    with open(args.output_train_file, "w") as fp:
        for document in train_samples:
            fp.write(document)

    with open(args.output_validation_file, "w") as fp:
        for document in valid_samples:
            fp.write(document)

    # Train the model
    model = fasttext.train_supervised(
        input=args.output_train_file,
        lr=args.learning_rate,
        dim=args.word_vector_dim,
        epoch=args.num_epochs,
        wordNgrams=args.wordNgrams,
    )

    # Save the classifier as a FastText model
    model.save_model(args.output_model)

    if args.output_predictions is not None:
        fout = open(args.output_predictions, "wb")

    # Read in the model and compute accuracy and other metrics on the data
    hq_label = args.high_quality_label
    prds, lbls = [], []
    with open(args.output_validation_file, "r") as f:
        for line in tqdm(f.readlines()):
            # Split the text and the label
            label_t, doc = line.split(" ", 1)
            doc = doc.rstrip()
            labels_p, scores = model.predict(doc, k=2)
            # Write the predictions to file
            if args.output_predictions is not None:
                line = {
                    "text": doc,
                    "label": label_t,
                    f"{labels_p[0]}": scores[0],
                    f"{labels_p[1]}": scores[1],
                }
                myjson = json.dumps(line, ensure_ascii=False)
                fout.write(myjson.encode("utf-8"))
                fout.write("\n".encode("utf-8"))
            # Save predictions and labels
            prds.append(1) if labels_p[0] == hq_label else prds.append(0)
            lbls.append(1) if label_t == hq_label else lbls.append(0)

    # Print out the metrics computed on the validation data
    tn, fp, fn, tp = confusion_matrix(prds, lbls).ravel()
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)

    print(f"Acc={accuracy} Prec={precision} Rec={recall} f1={f1}")


def attach_args(
    parser=argparse.ArgumentParser(
        """
Train a skip-gram quality classifier with FastText

Takes as input files with prepared samples for training
a skip-gram classifier with FastText, trains a skip-gram
classifier on the input samples and writes the trained classifier
out to disk as a FastText model.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
):
    ArgumentHelper(parser).add_train_fasttext_args()

    return parser


def console_script():
    main(attach_args().parse_args())
