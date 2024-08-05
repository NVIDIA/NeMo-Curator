import os

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
os.environ["CUDF_SPILL"] = "on"
import time
from dataclasses import dataclass
from functools import lru_cache
import pandas as pd
import cudf
import torch
import torch.nn as nn
from crossfit import op
from crossfit.backend.torch.hf.model import HFModel
import multiprocessing
import pickle
import dask_cudf
import dask.dataframe 
from numba import cuda
from nltk.tokenize import sent_tokenize
import re

# TODO: Import for IndicTransTokenizer
# Check if we want to add this as an dependency
try:
    from IndicTransTokenizer import IndicProcessor
except ImportError:
    raise ImportError(
        "IndicTransTokenizer not found. Please install it using the following command: \n"
        + "pip install git+https://github.com/VarunGumma/IndicTransTokenizer.git"
    )
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from nemo_curator.utils.distributed_utils import (
    get_client,
    load_object_on_worker,
    read_data,
    write_to_disk,
)
from nemo_curator.utils.script_utils import (
    parse_client_args,
    parse_distributed_classifier_args,
)
from dask.distributed import get_worker
MAX_OUT_LEN= 256
MAX_TOKEN_LEN=200

@dataclass
class TranslationConfig:
    pretrained_model_name_or_path: str
    max_length: int = 50
    num_beams: int = 5
    autocast: bool = False


class CustomModel(nn.Module):
    def __init__(self, config: TranslationConfig, pretrained_model_name_or_path: str, autocast: bool = False):
        super().__init__()
        self.config = config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        
    @torch.no_grad()
    def _forward(self, batch):
        return self.model.generate(
            **batch,
            use_cache=True,
            min_length=0,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            num_return_sequences=1,
            repetition_penalty=1.2,
        )

    def forward(self, batch):
        outputs = self._forward(batch)
        return outputs


class ModelForSeq2SeqModel(HFModel):
    def __init__(self, config):
        self.trans_config = config
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.trans_config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )
        super().__init__(self.trans_config.pretrained_model_name_or_path)

    def load_model(self, device="cuda"):
        model = CustomModel(
            self.trans_config, self.trans_config.pretrained_model_name_or_path, self.trans_config.autocast
        )
        model = model.to(device)
        model.half()
        model.eval()
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

def preprocess_df(df):
    ip = load_object_on_worker("IndicProcessor", IndicProcessor, {"inference": True})
    indices = df['indic_proc_text'].index.to_arrow().to_pylist()
    sentences = df["indic_proc_text"].to_arrow().to_pylist()
    sentences = ip.preprocess_batch(
        sentences, src_lang="eng_Latn", tgt_lang="hin_Deva"
    )
    df["indic_proc_text"] = cudf.Series(sentences,index=indices)
    return df

def translate_tokens(df, model):
    worker = get_worker()
    if hasattr(worker, "IndicProcessor"):
        ip = getattr(worker, "IndicProcessor")
    else:
        ip = load_object_on_worker("IndicProcessor", IndicProcessor, {"inference": True})
    tokenizer = model.load_tokenizer()
    indices = df["translation"].index.to_arrow().to_pylist()
    generated_tokens = df["translation"].to_arrow().to_pylist()
    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            src=False,
        )
    generated_tokens = ip.postprocess_batch(generated_tokens, lang="hin_Deva")
    df["translation"] = cudf.Series(data=generated_tokens,index=indices)
    return df


def parse_arguments():
    parser = parse_distributed_classifier_args()
    parser.add_argument(
        "--input-column",
        type=str,
        required=False,
        default="text",
        help="The column name in the input data that contains the text to be translated",
    )
    return parser.parse_args()


TERMINAL_PUNCTUATIONS = (".", "!", "?", ":", ",", ";", ")", "}", "]", '"', "'", "”", "’")
START_PUNCTUATIONS = ("(", "{", "[", "'", '"', "“", "‘")
def truncate_start_end_symbols(input_string):
    start = ""
    mid = ""
    end = ""
    flag = True
    for char in input_string:
        if char.isalnum() == False:
            if flag:
                start += char
            else:
                end += char
        else:
            flag = False
            end = ""
    mid = input_string[len(start) : len(input_string) - len(end)]
    while len(start):
        if start[-1] in START_PUNCTUATIONS:
            mid = start[-1] + mid
            start = start[: len(start) - 1]
        else:
            break
    while len(end):
        if end[0] in TERMINAL_PUNCTUATIONS:
            mid += end[0]
            end = end[1:]
        else:
            break
    return pd.Series([start, mid, end], index=['start_sym', 'indic_proc_text', 'end_sym'])

def has_alphabet_characters(text):
    return any(c.isalpha() for c in text)

def sent_tokenizer(line):
    return sent_tokenize(line)

def custom_tokenize(text):
    split_text = re.split(r"(\#{2,}|\_{2,}|\…{2,}|\+{2,}|\.{2,}|\-{3,}|\*{2,}|\~{2,}|\={2,}|\!{2,}|\n|\t|\‣|\⁃|\⁌|\⁍|\●|\○|\•|\·|\◘|\◦|\⦾|\⦿|\|)", text)
    split_text = [s for s in split_text if len(s) > 0]
    tokenized_sentences = []
    len_flag = False
    for line in split_text:
        # Tokenize sentences using NLTK's sent_tokenize function
        if has_alphabet_characters(line) == True:
            sentences = sent_tokenizer(line)
            i = 0
            j = 0
            tokenized_snt_lst = []
            str = ""
            while i < len(line):
                ch = line[i]
                if j < len(sentences):
                    stripped_sent = sentences[j].strip()
                    if len(stripped_sent) == 0:
                        j += 1
                        continue
                    if ch == stripped_sent[0]:
                        if str != "":
                            tokenized_snt_lst.append(str)
                        tokenized_snt_lst.append(stripped_sent)
                        i += len(stripped_sent)
                        j += 1
                        str = ""
                    else:
                        str += ch
                        i += 1
                else:
                    str += ch
                    i += 1
            if str != "":
                tokenized_snt_lst.append(str)
            # Add the tokenized sentences to the list
            tokenized_sentences.extend(tokenized_snt_lst)
        else:
            tokenized_sentences.append(line)

    tokenized_sentence_len = []
    for sentence in tokenized_sentences:
        sent = sentence.split()
        # removing the sentences with word length greater than threshold as the model may not be able translate it due to constraint on output token size
        if len(sent) <= MAX_TOKEN_LEN:
            tokenized_sentence_len.append(sentence)
        else:
            pass
    return tokenized_sentence_len


def process_input_text(ddf):
    ddf = ddf.to_pandas()
    ddf['indic_proc_text'] = ddf['text'].apply(custom_tokenize)
    ddf['doc_id'] = range(1, len(ddf) + 1)
    ddf = ddf.explode('indic_proc_text')
    ddf = ddf.reset_index(drop=True)
    ddf=ddf.drop('text', axis=1)
    ddf[['start_sym', 'indic_proc_text', 'end_sym']] = ddf['indic_proc_text'].apply(truncate_start_end_symbols)#, result_type='expand')
    ddf = cudf.DataFrame.from_pandas(ddf)
    return ddf

def concat_strings(series):
    return ''.join(series)

def combine_text(ddf):
    engligh_stop_flag = ddf["indic_proc_text"].str.endswith(".")
    hindi_stop_flag = ddf["translation"].str.endswith("|")
    ddf["translation"][~engligh_stop_flag & hindi_stop_flag] = ddf["translation"].str.rstrip("|")
    ddf["translation"] = ddf["start_sym"] + ddf["translation"].str.strip() + ddf["end_sym"]
    ddf['indic_proc_text'] = ddf["start_sym"] + ddf["indic_proc_text"] + ddf["end_sym"]
    return ddf

def grouping(ddf):
    ddf = ddf.to_pandas()
    agg_funcs = {'translation': concat_strings, 'indic_proc_text': concat_strings}
    other_columns = {col: 'first' for col in ddf.columns if col not in agg_funcs and col != 'doc_id'}

    agg_funcs.update(other_columns)
    ddf = ddf.groupby('doc_id').agg(agg_funcs).reset_index()
    ddf = cudf.DataFrame.from_pandas(ddf)
    return ddf

def combine_text_false(ddf):
    ddf["translation"]=ddf["start_sym"] + ddf["indic_proc_text"] + ddf["end_sym"]
    ddf["indic_proc_text"] = ddf["translation"]    
    return ddf

def post_output(output):
    import torch.nn.functional as F
    padded_tensor = output
    if output.size(1) <  MAX_OUT_LEN:
        pad_size = MAX_OUT_LEN - output.size(1)
        padded_tensor  = F.pad(output, (0, pad_size), "constant", 1)
    return padded_tensor

def main_func(args):
    st = time.time()
    translation_config = TranslationConfig(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        max_length=256,
        num_beams=5,
        autocast=args.autocast,
    )
    global MAX_OUT_LEN
    MAX_OUT_LEN = translation_config.max_length
    rdst = time.time()
    input_files = [
        os.path.join(args.input_data_dir, x) for x in os.listdir(args.input_data_dir)
    ]
    
    ddf = read_data(
        input_files,
        file_type=args.input_file_type,
        backend="cudf",
        files_per_partition=1,
        add_filename=True,
    )
    
    rdet = time.time()
    ddf_meta = ddf._meta.copy()

    ddf_meta['indic_proc_text'] = ""
    ddf_meta['doc_id'] = ""
    ddf_meta['start_sym'] = ""
    ddf_meta['end_sym'] = ""
    ddf_meta = ddf_meta.drop('text',axis=1)
    ddf = ddf.map_partitions(process_input_text, meta=ddf_meta)
    ddf['word_count'] = ddf['indic_proc_text'].str.split().list.len()
    ddf['word_count'] = ddf['word_count'].astype('int64')

    ddf_true = ddf[(ddf['word_count'] <= MAX_TOKEN_LEN)]
    ddf_trans = ddf_true[ddf_true['indic_proc_text'].str.contains('[a-zA-Z]',regex=True)]
    ddf_false=ddf_true[~(ddf_true['indic_proc_text'].str.contains('[a-zA-Z]',regex=True))]

    ddf=ddf_trans.drop(columns='word_count')

    ## ddf false operations
    ddf_false=ddf_false.drop(columns='word_count')
    ddf_false_meta = ddf_false._meta.copy()
    ddf_false_meta["translation"]=""
    ddf_false = ddf_false.map_partitions(combine_text_false, meta=ddf_false_meta)

    ddf['indic_proc_text'] = ddf['indic_proc_text'].astype('str')
    ddf = ddf.map_partitions(preprocess_df, meta=ddf_meta)

    columns = ddf.columns.tolist()
    model = ModelForSeq2SeqModel(translation_config)

    pipe = op.Sequential(
        op.Tokenizer(model, cols=[args.input_column], tokenizer_type="sentencepiece"),
        op.Predictor(
            model,
            sorted_data_loader=True,
            batch_size=args.batch_size,
            pred_output_col="translation",
            post=post_output,
        ),
        keep_cols=columns,
    )
    ddf = pipe(ddf)
    translated_meta = ddf._meta.copy()
    translated_meta["translation"] = "DUMMY_STRING"
    ddf = ddf.map_partitions(translate_tokens, model=model, meta=translated_meta)
    ddf = ddf.map_partitions(combine_text, meta=translated_meta)

    ddf_true['false_translation'] = ddf_false['translation']
    ddf_true['false_translation'] = ddf_true['false_translation'].fillna('')
    ddf_true['translation'] = ddf['translation']
    ddf_true['translation'] = ddf_true['translation'].fillna('')
    ddf_true['translation'] = ddf_true['translation'] + ddf_true['false_translation']

    ddf = ddf_true.map_partitions(grouping)
    write_to_disk(
        ddf,
        output_file_dir=args.output_data_dir,
        write_to_filename=True,
        output_type=args.output_file_type,
    )

    print(f"Total time taken for translation: {time.time()-st} seconds", flush=True)

def main():
    args = parse_arguments()
    print(f"Arguments parsed = {args}")
    client = get_client(**parse_client_args(args))
    print(client.dashboard_link)
    main_func(args)
    client.close()


if __name__ == "__main__":
    main()
