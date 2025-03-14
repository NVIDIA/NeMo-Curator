from nemo_curator import (
    DocumentSplitter,
    DocumentJoiner,
    get_client,
    Sequential,
    Score,
    ScoreFilter,
    Modify,
    Filter,
)
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import TokenCountFilter, SubstringFilter
from nemo_curator.modifiers import (
    QuotationRemover,
    LineRemover,
    MarkdownRemover,
    Slicer,
)
from nemo_curator.services import OpenAIClient
from nemo_curator.synthetic import (
    NemotronCCGenerator,
    NemotronCCDiverseQAPostprocessor,
    NemotronCCKnowledgeListPostprocessor,
)
from transformers import AutoTokenizer
from openai import OpenAI
import pandas as pd
from nemo_curator.synthetic.prompts import (
    NEMOTRON_CC_SYSTEM_PROMPT,
    WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
    DIVERSE_QA_PROMPT_TEMPLATE,
    DISTILL_PROMPT_TEMPLATE,
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
)
from tqdm import tqdm


def get_prefix_token_count(
    tokenizer: AutoTokenizer, system_prompt: str, user_prompt_template: str
):
    user_prompt = user_prompt_template.format(document="placeholder")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prefix_tokens = tokenizer.apply_chat_template(messages)

    return len(prefix_tokens)


def build_preprocessing_pipeline(
    tokenizer: AutoTokenizer,
    text_field: str,
    system_prompt: str,
    user_prompt_template: str,
    min_document_tokens: int,
    min_segment_tokens: int,
    max_input_tokens: int,
):
    # Construct filters for document filtering
    document_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, min_tokens=min_document_tokens
    )

    # Construct filters for segment filtering
    prefix_token_count = get_prefix_token_count(
        tokenizer, system_prompt, user_prompt_template
    )
    max_segment_tokens = max_input_tokens - prefix_token_count - 2
    long_segment_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, max_tokens=max_segment_tokens
    )
    short_segment_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, min_tokens=min_segment_tokens
    )

    preprocessing_pipeline = Sequential(
        [
            # Filter out documents that are too short
            ScoreFilter(
                document_token_count_filter,
                text_field=text_field,
                score_field="document_token_count",
                score_type=int,
            ),
            # Split documents into segments
            DocumentSplitter(
                separator="\n", text_field=text_field, segment_id_field="segment_id"
            ),
            # Filter out segments that are too long
            ScoreFilter(
                long_segment_token_count_filter,
                text_field=text_field,
                score_field="segment_token_count",
                score_type=int,
            ),
            # Join adjacent short segments
            DocumentJoiner(
                separator="\n",
                text_field=text_field,
                segment_id_field="segment_id",
                document_id_field="id",
                max_length=max_segment_tokens,
                length_field="segment_token_count",
                drop_segment_id_field=False,
            ),
            # Filter out segments that are too short even after joining
            Filter(
                short_segment_token_count_filter.keep_document,
                filter_field="segment_token_count",
            ),
        ]
    )

    return preprocessing_pipeline


def build_wikipedia_postprocessing_pipeline(
    tokenizer: AutoTokenizer, rephrased_field: str
):
    MAX_REPHRASED_TOKENS = 510
    MIN_DOCUMENT_TOKENS = 50

    long_segment_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, max_tokens=MAX_REPHRASED_TOKENS
    )
    document_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, min_tokens=MIN_DOCUMENT_TOKENS
    )
    postprocessing_pipeline = Sequential(
        [
            # Filter by token count
            ScoreFilter(
                long_segment_token_count_filter,
                text_field=rephrased_field,
                score_field="rephrased_segment_token_count",
                score_type=int,
            ),
            # Remove markdown formatting
            Modify(MarkdownRemover(), text_field=rephrased_field),
            # Remove documents not starting with the specified prefix
            ScoreFilter(
                SubstringFilter(
                    substring="Here is a paraphrased version:", position="prefix"
                ),
                text_field=rephrased_field,
                score_field="substring",
                score_type=int,
            ),
            # Remove the paraphrase prefix
            Modify(
                Slicer(
                    left="Here is a paraphrased version:",
                    include_left=False,
                    strip=True,
                ),
                text_field=rephrased_field,
            ),
            # Remove quotation marks
            Modify(QuotationRemover(), text_field=rephrased_field),
            # Concat paragraphs belonging to the same document
            DocumentJoiner(
                separator="\n",
                text_field=rephrased_field,
                segment_id_field="segment_id",
                document_id_field="id",
                drop_segment_id_field=False,
            ),
            # Filter out documents that are too short
            ScoreFilter(
                document_token_count_filter,
                text_field=rephrased_field,
                score_field="rephrased_document_token_count",
                score_type=int,
            ),
        ]
    )

    return postprocessing_pipeline


def wikipedia_rephraser(
    dataset: DocumentDataset,
    text_field: str,
    openai_client: OpenAI,
    tokenizer: AutoTokenizer,
    api_model_name: str,
    n_entries:int=5
):
    client = OpenAIClient(openai_client)
    nemotron_cc = NemotronCCGenerator(client)
    rephrased_field = "rephrased"
    config = {
        "MIN_DOCUMENT_TOKENS": 30,
        "MIN_SEGMENT_TOKENS": 10,
        "MAX_INPUT_TOKENS": 512,
        "MAX_OUTPUT_TOKENS": 512,
        "TOP_K": 0,
        "TOP_P": 0.9,
        "END_STRINGS": "['</s>']",
        "TEMPERATURE": 0.5,
    }

    preprocessing_pipeline = build_preprocessing_pipeline(
        tokenizer,
        text_field,
        NEMOTRON_CC_SYSTEM_PROMPT,
        WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
        config["MIN_DOCUMENT_TOKENS"],
        config["MIN_SEGMENT_TOKENS"],
        config["MAX_INPUT_TOKENS"],
    )

    print("Running Wikipedia rephraser preprocessing pipeline")

    dataset = preprocessing_pipeline(dataset)

    first_entries = dataset.df.head(n_entries)
    print("Using only a small portion of the input dataset to save time: ")
    print(first_entries)

    rewritten_texts = []
    for text in tqdm(first_entries[text_field], desc="Rephrasing texts"):
        rewritten_text = nemotron_cc.rewrite_to_wikipedia_style(
            text,
            api_model_name,
            model_kwargs={
                "top_k": config["TOP_K"],
                "top_p": config["TOP_P"],
                "stop": config["END_STRINGS"],
                "max_tokens": config["MAX_OUTPUT_TOKENS"],
                "temperature": config["TEMPERATURE"],
            },
        )
        rewritten_texts.append(rewritten_text[0])

    first_entries[rephrased_field] = rewritten_texts

    rephrased_dataset = DocumentDataset.from_pandas(first_entries)

    print("Running Wikipedia rephraser postprocessing pipeline")
    postprocessing_pipeline = build_wikipedia_postprocessing_pipeline(
        tokenizer, rephrased_field
    )

    rephrased_dataset = postprocessing_pipeline(rephrased_dataset)
    print("Wikipedia rephraser postprocessing complete.")
    return rephrased_dataset


def build_diverse_qa_postprocessing_pipeline(
    tokenizer: AutoTokenizer, text_field: str, llm_response_field: str
):
    MAX_REPHRASED_TOKENS = 598
    MIN_DOCUMENT_TOKENS = 100

    long_segment_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, max_tokens=MAX_REPHRASED_TOKENS
    )
    document_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, min_tokens=MIN_DOCUMENT_TOKENS
    )
    postprocessing_pipeline = Sequential(
        [
            # Filter by token count
            ScoreFilter(
                long_segment_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_segment_token_count",
                score_type=int,
            ),
            # Remove markdown formatting
            Modify(MarkdownRemover(), text_field=llm_response_field),
            # Reformat QA pairs
            NemotronCCDiverseQAPostprocessor(
                tokenizer, text_field=text_field, response_field=llm_response_field
            ),
            # Filter out documents that are too short
            ScoreFilter(
                document_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_document_token_count",
                score_type=int,
            ),
        ]
    )

    return postprocessing_pipeline


def diverse_qa(
    input_data_path: str,
    output_path: str,
    final_output_path: str,
    text_field: str,
    llm_response_field: str,
    postprocess_only: bool,
    openai_client: OpenAI = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key="<insert-NV-api-key>"),
):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    client = OpenAIClient(openai_client)
    nemotron_cc = NemotronCCGenerator(client, tokenizer)
    api_model_name = "nvdev/nv-mistralai/mistral-nemo-12b-instruct"

    if not postprocess_only:
        dataset = DocumentDataset.read_json(input_data_path)

        MIN_DOCUMENT_TOKENS = 30
        MIN_SEGMENT_TOKENS = 30
        MAX_INPUT_TOKENS = 1000
        preprocessing_pipeline = build_preprocessing_pipeline(
            tokenizer,
            text_field,
            NEMOTRON_CC_SYSTEM_PROMPT,
            DIVERSE_QA_PROMPT_TEMPLATE,
            MIN_DOCUMENT_TOKENS,
            MIN_SEGMENT_TOKENS,
            MAX_INPUT_TOKENS,
        )

        dataset = preprocessing_pipeline(dataset)

        first_entries = dataset.df.head()
        print(first_entries)

        MAX_OUTPUT_TOKENS = 600
        TOP_K = 0
        TOP_P = 0.9
        END_STRINGS = "['</s>']"
        TEMPERATURE = 0.5
        rewritten_texts = []
        for text in first_entries["EN"]:
            llm_response = nemotron_cc.generate_diverse_qa(
                text,
                api_model_name,
                model_kwargs={
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    "stop": END_STRINGS,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": TEMPERATURE,
                },
            )
            rewritten_texts.append(llm_response[0])

        first_entries[llm_response_field] = rewritten_texts
        first_entries.to_json(output_path, orient="records", lines=True)
    else:
        first_entries = pd.read_json(output_path, lines=True)
        first_entries[llm_response_field] = first_entries[llm_response_field].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )

    print(first_entries)
    rephrased_dataset = DocumentDataset.from_pandas(first_entries)
    postprocessed_pipeline = build_diverse_qa_postprocessing_pipeline(
        tokenizer, text_field, llm_response_field
    )
    rephrased_dataset = postprocessed_pipeline(rephrased_dataset)
    rephrased_head = rephrased_dataset.df.head()
    print(rephrased_head)
    rephrased_head.to_json(final_output_path, orient="records", lines=True)


def build_distill_postprocessing_pipeline(
    tokenizer: AutoTokenizer, llm_response_field: str
):
    MAX_REPHRASED_TOKENS = 1598
    MIN_DOCUMENT_TOKENS = 50

    long_segment_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, max_tokens=MAX_REPHRASED_TOKENS
    )
    document_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, min_tokens=MIN_DOCUMENT_TOKENS
    )
    postprocessing_pipeline = Sequential(
        [
            # Filter by token count
            ScoreFilter(
                long_segment_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_segment_token_count",
                score_type=int,
            ),
            # Remove markdown formatting
            Modify(MarkdownRemover(), text_field=llm_response_field),
            # Remove documents not starting with the specified prefix
            ScoreFilter(
                SubstringFilter(substring="Paraphrased Text:", position="prefix"),
                text_field=llm_response_field,
                score_field="substring",
                score_type=int,
            ),
            # Remove the paraphrase prefix
            Modify(
                Slicer(
                    left="Paraphrased Text:",
                    include_left=False,
                    strip=True,
                ),
                text_field=llm_response_field,
            ),
            # Remove quotation marks
            Modify(QuotationRemover(), text_field=llm_response_field),
            # Filter out documents that are too short
            ScoreFilter(
                document_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_document_token_count",
                score_type=int,
            ),
        ]
    )

    return postprocessing_pipeline


def distill(
    input_data_path: str,
    output_path: str,
    final_output_path: str,
    text_field: str,
    llm_response_field: str,
    postprocess_only: bool,
    openai_client: OpenAI = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key="<insert-NV-api-key>"),
):
    client = OpenAIClient(openai_client)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    nemotron_cc = NemotronCCGenerator(client, tokenizer)
    api_model_name = "nvdev/nv-mistralai/mistral-nemo-12b-instruct"

    if not postprocess_only:
        dataset = DocumentDataset.read_json(input_data_path)

        MIN_DOCUMENT_TOKENS = 30
        MIN_SEGMENT_TOKENS = 10
        MAX_INPUT_TOKENS = 2000
        preprocessing_pipeline = build_preprocessing_pipeline(
            tokenizer,
            text_field,
            NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
            DISTILL_PROMPT_TEMPLATE,
            MIN_DOCUMENT_TOKENS,
            MIN_SEGMENT_TOKENS,
            MAX_INPUT_TOKENS,
        )

        dataset = preprocessing_pipeline(dataset)

        first_entries = dataset.df.head()
        print(first_entries)

        MAX_OUTPUT_TOKENS = 1600
        TOP_K = 0
        TOP_P = 0.9
        END_STRINGS = "['</s>']"
        TEMPERATURE = 0.5
        rewritten_texts = []
        for text in first_entries["EN"]:
            llm_response = nemotron_cc.distill(
                text,
                api_model_name,
                model_kwargs={
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    "stop": END_STRINGS,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": TEMPERATURE,
                },
            )
            rewritten_texts.append(llm_response[0])

        first_entries[llm_response_field] = rewritten_texts
        first_entries.to_json(output_path, orient="records", lines=True)
    else:
        first_entries = pd.read_json(output_path, lines=True)
        first_entries[llm_response_field] = first_entries[llm_response_field].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )

    print(first_entries)
    rephrased_dataset = DocumentDataset.from_pandas(first_entries)
    postprocessed_pipeline = build_distill_postprocessing_pipeline(
        tokenizer, llm_response_field
    )
    rephrased_dataset = postprocessed_pipeline(rephrased_dataset)
    rephrased_head = rephrased_dataset.df.head()
    print(rephrased_head)
    rephrased_head.to_json(final_output_path, orient="records", lines=True)


def build_extract_knowledge_postprocessing_pipeline(
    tokenizer: AutoTokenizer, llm_response_field: str
):
    MAX_REPHRASED_TOKENS = 1398
    MIN_DOCUMENT_TOKENS = 50

    long_segment_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, max_tokens=MAX_REPHRASED_TOKENS
    )
    document_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, min_tokens=MIN_DOCUMENT_TOKENS
    )
    postprocessing_pipeline = Sequential(
        [
            # Filter by token count
            ScoreFilter(
                long_segment_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_segment_token_count",
                score_type=int,
            ),
            # Remove markdown formatting
            Modify(MarkdownRemover(), text_field=llm_response_field),
            # Remove passage lines
            Modify(
                LineRemover(
                    patterns=["Passage:", "Passage 1:", "Passage 2:", "Passage 3:"]
                ),
                text_field=llm_response_field,
            ),
            # Filter out documents that are too short
            ScoreFilter(
                document_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_document_token_count",
                score_type=int,
            ),
        ]
    )

    return postprocessing_pipeline


def extract_knowledge(
    input_data_path: str,
    output_path: str,
    final_output_path: str,
    text_field: str,
    llm_response_field: str,
    postprocess_only: bool,
    openai_client: OpenAI = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="<insert-api-key>",  # NV DEV API KEY
    ),
):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    client = OpenAIClient(openai_client)
    nemotron_cc = NemotronCCGenerator(client, tokenizer)
    api_model_name = "nvdev/nv-mistralai/mistral-nemo-12b-instruct"

    if not postprocess_only:
        dataset = DocumentDataset.read_json(input_data_path)

        MIN_DOCUMENT_TOKENS = 30
        MIN_SEGMENT_TOKENS = 30
        MAX_INPUT_TOKENS = 1400
        preprocessing_pipeline = build_preprocessing_pipeline(
            tokenizer,
            text_field,
            NEMOTRON_CC_SYSTEM_PROMPT,
            EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
            MIN_DOCUMENT_TOKENS,
            MIN_SEGMENT_TOKENS,
            MAX_INPUT_TOKENS,
        )

        dataset = preprocessing_pipeline(dataset)

        first_entries = dataset.df.head()
        print(first_entries)

        MAX_OUTPUT_TOKENS = 1400
        TOP_K = 0
        TOP_P = 0.9
        END_STRINGS = "['</s>']"
        TEMPERATURE = 0.5
        rewritten_texts = []
        for text in first_entries["EN"]:
            llm_response = nemotron_cc.extract_knowledge(
                text,
                api_model_name,
                model_kwargs={
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    "stop": END_STRINGS,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": TEMPERATURE,
                },
            )
            rewritten_texts.append(llm_response[0])

        first_entries[llm_response_field] = rewritten_texts
        first_entries.to_json(output_path, orient="records", lines=True)
    else:
        first_entries = pd.read_json(output_path, lines=True)
        first_entries[llm_response_field] = first_entries[llm_response_field].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )

    print(first_entries)
    rephrased_dataset = DocumentDataset.from_pandas(first_entries)
    postprocessed_pipeline = build_extract_knowledge_postprocessing_pipeline(
        tokenizer, llm_response_field
    )
    rephrased_dataset = postprocessed_pipeline(rephrased_dataset)
    rephrased_head = rephrased_dataset.df.head()
    print(rephrased_head)
    rephrased_head.to_json(final_output_path, orient="records", lines=True)


def build_knowledge_list_postprocessing_pipeline(
    tokenizer: AutoTokenizer, llm_response_field: str
):
    MAX_REPHRASED_TOKENS = 598
    MIN_DOCUMENT_TOKENS = 50

    long_segment_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, max_tokens=MAX_REPHRASED_TOKENS
    )
    document_token_count_filter = TokenCountFilter(
        tokenizer=tokenizer, min_tokens=MIN_DOCUMENT_TOKENS
    )
    postprocessing_pipeline = Sequential(
        [
            # Filter by token count
            ScoreFilter(
                long_segment_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_segment_token_count",
                score_type=int,
            ),
            # Remove markdown formatting
            Modify(MarkdownRemover(), text_field=llm_response_field),
            NemotronCCKnowledgeListPostprocessor(text_field=llm_response_field),
            # Filter out documents that are too short
            ScoreFilter(
                document_token_count_filter,
                text_field=llm_response_field,
                score_field="rephrased_document_token_count",
                score_type=int,
            ),
        ]
    )

    return postprocessing_pipeline


def knowledge_list(
    input_data_path: str,
    output_path: str,
    final_output_path: str,
    text_field: str,
    llm_response_field: str,
    postprocess_only: bool,
    openai_client: OpenAI = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key="<insert-api-key>"),
):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    client = OpenAIClient(openai_client)
    nemotron_cc = NemotronCCGenerator(client, tokenizer)
    api_model_name = "nvdev/nv-mistralai/mistral-nemo-12b-instruct"

    if not postprocess_only:
        dataset = DocumentDataset.read_json(input_data_path)

        MIN_DOCUMENT_TOKENS = 30
        MIN_SEGMENT_TOKENS = 30
        MAX_INPUT_TOKENS = 1000
        preprocessing_pipeline = build_preprocessing_pipeline(
            tokenizer,
            text_field,
            NEMOTRON_CC_SYSTEM_PROMPT,
            KNOWLEDGE_LIST_PROMPT_TEMPLATE,
            MIN_DOCUMENT_TOKENS,
            MIN_SEGMENT_TOKENS,
            MAX_INPUT_TOKENS,
        )

        dataset = preprocessing_pipeline(dataset)

        first_entries = dataset.df.head()
        print(first_entries)

        MAX_OUTPUT_TOKENS = 600
        TOP_K = 0
        TOP_P = 0.9
        END_STRINGS = "['</s>']"
        TEMPERATURE = 0.5
        rewritten_texts = []
        for text in first_entries["EN"]:
            llm_response = nemotron_cc.generate_knowledge_list(
                text,
                api_model_name,
                model_kwargs={
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    "stop": END_STRINGS,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": TEMPERATURE,
                },
            )
            rewritten_texts.append(llm_response[0])

        first_entries[llm_response_field] = rewritten_texts
        first_entries.to_json(output_path, orient="records", lines=True)
    else:
        first_entries = pd.read_json(output_path, lines=True)
        first_entries[llm_response_field] = first_entries[llm_response_field].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )

    print(first_entries)
    rephrased_dataset = DocumentDataset.from_pandas(first_entries)
    postprocessed_pipeline = build_knowledge_list_postprocessing_pipeline(
        tokenizer, llm_response_field
    )
    rephrased_dataset = postprocessed_pipeline(rephrased_dataset)
    rephrased_head = rephrased_dataset.df.head()
    print(rephrased_head)
    rephrased_head.to_json(final_output_path, orient="records", lines=True)


if __name__ == "__main__":
    # wikipedia_rephraser()
    # diverse_qa()
    # distill()
    # extract_knowledge()
    knowledge_list()
