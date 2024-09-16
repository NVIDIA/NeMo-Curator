import os

os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "False"
import argparse
import re
import time
from dataclasses import dataclass
from functools import lru_cache

import cudf
import ctranslate2
import numpy as np
import torch
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
from dask.distributed import get_worker
from nltk.tokenize import sent_tokenize
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from nemo_curator.classifiers.base import DistributedDataClassifier
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, load_object_on_worker
from nemo_curator.utils.script_utils import ArgumentHelper

try:
    from IndicTransToolkit import IndicProcessor
except ImportError:
    raise ImportError(
        "IndicTransToolkit not found. Please install it using the following command: \n"
        + "pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
    )

TERMINAL_PUNCTUATIONS = (
    ".",
    "!",
    "?",
    ":",
    ",",
    ";",
    ")",
    "}",
    "]",
    '"',
    "'",
    "”",
    "’",
)
START_PUNCTUATIONS = ("(", "{", "[", "'", '"', "“", "‘")


@dataclass
class TranslationConfig:
    pretrained_model_name_or_path: str
    ct2_model_path: str
    string_tok_inf: bool = True
    max_words_per_sen: int = 200
    target_lang_code: str = "hin_Deva"

class CT2CustomModel():
    def __init__(self, config: TranslationConfig, device="cuda"):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            trust_remote_code = True,
        )
        self.model = ctranslate2.Translator(model_path=config.ct2_model_path, device=device)
        self.string_tok_inf = config.string_tok_inf

    def clean_extra_tokens(self, token_2d):
        results=[]
        for token_1d in token_2d:
            result = []
            for t in token_1d:
                if t==self.tokenizer.pad_token or t==self.tokenizer.bos_token or t==self.tokenizer.eos_token or t==self.tokenizer.unk_token:
                    pass
                else:
                    result.append(t)
            results.append(result)
        return results

    def __call__(self, batch):
        token_ids_2d=batch['input_ids']
        token_ids_1d = token_ids_2d.view(-1).tolist()
        tokens_1d = self.tokenizer.convert_ids_to_tokens(token_ids_1d)
        tokens_2d = [tokens_1d[i:i + token_ids_2d.size(1)] for i in range(0, len(tokens_1d), token_ids_2d.size(1))]
        tokenss = self.clean_extra_tokens(tokens_2d)

        tr_res = self.model.translate_batch(
            tokenss,
            min_decoding_length=0,
            max_decoding_length=256,
            beam_size=5,
            num_hypotheses=1,
        )
        translations = ["".join(x.hypotheses[0]) for x in tr_res]
        return translations

class ModelForSeq2SeqModel(HFModel):
    def __init__(self, config):
        self.trans_config = config
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        self.string_tok_inf = config.string_tok_inf
        super().__init__(self.trans_config.pretrained_model_name_or_path)

    def load_model(self, device="cuda"):
        model = CT2CustomModel(
            self.trans_config
        )
        return model

    def load_config(self):
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    @lru_cache(maxsize=1)
    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

    def max_seq_length(self) -> int:
        return self.config.max_source_positions

    @lru_cache(maxsize=1)
    def load_cfg(self):
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        return config


class IndicTranslation(DistributedDataClassifier):
    def __init__(
        self,
        ct2_model_path: str,
        pretrained_model_name_or_path: str = "ai4bharat/indictrans2-en-indic-1B",
        input_column: str = "indic_proc_text",
        batch_size: int = 128,
        autocast: bool = False,
        string_tok_inf: bool = True,
        target_lang_code: str = "hin_Deva"
    ):
        self.ct2_model_path = ct2_model_path
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.input_column = input_column
        self.batch_size = batch_size
        self.autocast = autocast
        self.string_tok_inf = string_tok_inf

        self.translation_config = TranslationConfig(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            ct2_model_path=self.ct2_model_path,
            string_tok_inf=self.string_tok_inf,
            target_lang_code=target_lang_code
        )
        self.model = ModelForSeq2SeqModel(self.translation_config)
        super().__init__(
            model=self.model,
            batch_size=self.batch_size,
            device_type="cuda",
            autocast=self.autocast,
            labels=None,
            filter_by=None,
            out_dim=None,
            pred_column=None,
            max_chars=None,
        )

    def preprocess_df(self, df: cudf.DataFrame) -> cudf.DataFrame:
        ip = load_object_on_worker(
            "IndicProcessor", IndicProcessor, {"inference": True}
        )
        indices = df["text"].index.to_arrow().to_pylist()
        sentences = df["text"].to_arrow().to_pylist()
        sentences = ip.preprocess_batch(
            sentences, src_lang="eng_Latn", tgt_lang=self.translation_config.target_lang_code#"hin_Deva"
        )
        df["indic_proc_text"] = cudf.Series(sentences, index=indices)
        return df

    def translate_tokens(self, df: cudf.DataFrame) -> cudf.DataFrame:
        worker = get_worker()
        if hasattr(worker, "IndicProcessor"):
            ip = getattr(worker, "IndicProcessor")
        else:
            ip = load_object_on_worker(
                "IndicProcessor", IndicProcessor, {"inference": True}
            )
        tokenizer = self.model.load_tokenizer()
        indices = df["translation"].index.to_arrow().to_pylist()
        generated_tokens = df["translation"].to_arrow().to_pylist()
        converted_tokens = []
        for g in generated_tokens:
            converted_tokens.append(tokenizer.convert_tokens_to_string(g))
        converted_tokens = ip.postprocess_batch(converted_tokens, lang=self.translation_config.target_lang_code)#"hin_Deva")
        df["translation"] = cudf.Series(data=converted_tokens,index=indices)
        return df

    def has_alphabet_characters(self, text: str) -> bool:
        return any(c.isalpha() for c in text)

    def custom_tokenize(self, text: str):
        split_text = re.split(
            r"(\#{2,}|\_{2,}|\…{2,}|\+{2,}|\.{2,}|\-{3,}|\*{2,}|\~{2,}|\={2,}|\!{2,}|\n|\t|\‣|\⁃|\⁌|\⁍|\●|\○|\•|\·|\◘|\◦|\⦾|\⦿|\|)",
            text,
        )
        split_text = [s for s in split_text if len(s) > 0]
        tokenized_sentences = []
        len_flag = False
        for line in split_text:
            # Tokenize sentences using NLTK's sent_tokenize function
            if self.has_alphabet_characters(line) == True:
                sentences = sent_tokenize(line)
                i = 0
                j = 0
                curr_tokenized_snt = []
                non_translation_str = ""
                # Comparing the list of tokenized sentences (using NLTK) and actual sentence and preserving the spaces,
                # newline and other special characters
                while i < len(line):
                    if j < len(sentences):
                        stripped_sent = sentences[j].strip()
                        if len(stripped_sent) == 0:
                            j += 1
                            continue
                        # If tokenized sentence matches then moving to next sentence
                        if line[i] == stripped_sent[0]:
                            if non_translation_str != "":
                                curr_tokenized_snt.append(non_translation_str)
                            curr_tokenized_snt.append(stripped_sent)
                            i += len(stripped_sent)
                            j += 1
                            non_translation_str = ""
                        else:
                            non_translation_str += line[i]
                            i += 1
                    else:
                        non_translation_str += line[i]
                        i += 1
                if non_translation_str != "":
                    curr_tokenized_snt.append(non_translation_str)
                # Add the tokenized sentences to the list
                tokenized_sentences.extend(curr_tokenized_snt)
            else:
                tokenized_sentences.append(line)

        tokenized_sentence_len = []
        for sentence in tokenized_sentences:
            sent = sentence.split()
            # removing the sentences with word length greater than threshold as the model may not be able translate it due to constraint on output token size
            if len(sent) <= self.translation_config.max_words_per_sen:
                tokenized_sentence_len.append(sentence)

        return tokenized_sentence_len

    def process_input_text(self, df: cudf.DataFrame) -> cudf.DataFrame:
        df = df.to_pandas()
        df["text"] = df["text"].apply(self.custom_tokenize)
        df["doc_id"] = np.arange(1, len(df) + 1)
        df = df.explode("text", ignore_index=True)
        df = df.reset_index(drop=False)
        df = cudf.DataFrame.from_pandas(df)
        return df

    def combine_text(self, df: cudf.DataFrame) -> cudf.DataFrame:
        engligh_stop_flag = df["text"].str.endswith(".")
        hindi_stop_flag = df["translation"].str.endswith("|")
        df["translation"][~engligh_stop_flag & hindi_stop_flag] = df[
            "translation"
        ].str.rstrip("|")
        df["translation"] = df["translation"].str.strip()
        return df

    def grouping(self, df: cudf.DataFrame) -> cudf.DataFrame:
        df = df.to_pandas()
        agg_funcs = {
            "translation": lambda s: "".join(s),
            "text": lambda s: "".join(s),
        }
        other_columns = {
            col: "first"
            for col in df.columns
            if col not in agg_funcs and col != "doc_id"
        }

        agg_funcs.update(other_columns)
        df = df.groupby("doc_id").agg(agg_funcs).reset_index()
        df = cudf.DataFrame.from_pandas(df)
        return df

    def atleast_letter(self, df: cudf.DataFrame, column_name: str) -> cudf.DataFrame:
        df = df.to_pandas()
        df["isalpha"] = df[column_name].apply(self.has_alphabet_characters)
        df = cudf.DataFrame(df)
        return df

    def _run_classifier(self, dataset: DocumentDataset) -> DocumentDataset:
        ddf = dataset.df
        # Applying process_input_text for following :
        # 1. nltk tokenization to break doc into sentences
        # 2. craeting a row w.r.t each sentence.
        # 3. Process sentences strip symbols from start and end
        ddf = ddf.map_partitions(self.process_input_text, enforce_metadata=False)
        ddf["text"] = ddf["text"].astype("str")

        ddf["word_count"] = ddf["text"].str.split().list.len()
        ddf["word_count"] = ddf["word_count"].astype("int64")
        ddf_true = ddf[(ddf["word_count"] <= self.translation_config.max_words_per_sen)]
        # To filter for atleast one unicode letter in text
        has_letter = ddf_true.map_partitions(self.atleast_letter, column_name="text")
        ddf_trans = ddf_true[has_letter["isalpha"]]
        ddf = ddf_trans.drop(columns="word_count")
        ## ddf false operations
        ddf_false = ddf_true[~has_letter["isalpha"]]
        ddf_false = ddf_false.drop(columns="word_count")
        ddf_false["translation"] = ddf_false["text"]
        # Applying preprocess_df for Indic preprocessing
        ddf["text"] = ddf["text"].astype("str")
        ddf_meta = ddf._meta.copy()
        ddf_meta["indic_proc_text"] = ""
        ddf = ddf.map_partitions(self.preprocess_df, meta=ddf_meta)
        
        columns = ddf.columns.tolist()
        pipe = op.Sequential(
            op.Tokenizer(
                self.model, cols=[self.input_column], tokenizer_type="default", max_length=255
            ),
            op.Predictor(
                self.model,
                sorted_data_loader=True,
                batch_size=self.batch_size,
                pred_output_col="translation",
            ),
            keep_cols=columns,
        )
        ddf = pipe(ddf)
        translated_meta = ddf._meta.copy()
        translated_meta["translation"] = "DUMMY_STRING"
        ddf = ddf.map_partitions(self.translate_tokens, meta=translated_meta)
        # print(f"{ddf['translation'].head(20)}")
        ddf = ddf.map_partitions(self.combine_text, meta=translated_meta)

        # Merging translated and non-translated samples
        ddf_true["false_translation"] = ddf_false["translation"]
        ddf_true["false_translation"] = ddf_true["false_translation"].fillna("")
        ddf_true["translation"] = ddf["translation"]
        ddf_true["translation"] = ddf_true["translation"].fillna("")
        ddf_true["translation"] = (
            ddf_true["translation"] + ddf_true["false_translation"]
        )

        ddf = ddf_true.map_partitions(self.grouping)
        return DocumentDataset(ddf)


def attach_args():
    parser = ArgumentHelper.parse_distributed_classifier_args()
    parser.set_defaults(
        pretrained_model_name_or_path="ai4bharat/indictrans2-en-indic-1B"
    )
    parser.set_defaults(input_text_field="text")
    parser.set_defaults(device="gpu")
    return parser


def main(args):
    print(f"Arguments parsed = {args}")
    st = time.time()
    client = get_client(**ArgumentHelper.parse_client_args(args))
    print(client.dashboard_link)
    translator_model = IndicTranslation(
        ct2_model_path=args.ct2_model_path,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        input_column=args.input_text_field,
        batch_size=args.batch_size,
        autocast=args.autocast,
        string_tok_inf=True,
        target_lang_code=args.tgt_lang
    )
    input_files = [
        os.path.join(args.input_data_dir, x) for x in os.listdir(args.input_data_dir)
    ]
    input_dataset = DocumentDataset.read_json(
        input_files, backend="cudf", add_filename=True
    )
    result_dataset = translator_model(dataset=input_dataset)

    result_dataset.to_json(output_file_dir=args.output_data_dir, write_to_filename=True)
    print(f"Total time taken for translation: {time.time()-st} seconds", flush=True)
    client.close()


if __name__ == "__main__":
    parser = attach_args()
    parser.add_argument('--ct2-model-path',type=str,required=True,help="CT2 Model directory")
    parser.add_argument('--tgt-lang',default="hin_Deva",type=str,help="Language code for which translation will run")
    main(parser.parse_args())
