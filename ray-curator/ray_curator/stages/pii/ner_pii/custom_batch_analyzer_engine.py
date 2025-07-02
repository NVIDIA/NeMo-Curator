# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from presidio_analyzer import (
    AnalyzerEngine,
    BatchAnalyzerEngine,
    DictAnalyzerResult,
    EntityRecognizer,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpArtifacts

if TYPE_CHECKING:
    from ray_curator.stages.pii.ner_pii.custom_nlp_engine import CustomNlpEngine

logger = logging.getLogger("presidio-analyzer")


class CustomBatchAnalyzerEngine(BatchAnalyzerEngine):
    """
    Batch analysis of documents (tables, lists, dicts).

    Wrapper class to run Presidio Analyzer Engine on multiple values,
    either lists/iterators of strings, or dictionaries.

    :param: analyzer_engine: AnalyzerEngine instance to use
    for handling the values in those collections.
    """

    def __init__(self, analyzer_engine: AnalyzerEngine | None = None):
        super().__init__(analyzer_engine)

    def analyze_batch(  # noqa: PLR0913
        self,
        texts: Iterable[str],  # noqa: ARG002
        language: str,
        entities: list[str] | None = None,
        correlation_id: str | None = None,  # noqa: ARG002
        score_threshold: float | None = None,  # noqa: ARG002
        return_decision_process: bool | None = False,  # noqa: ARG002
        ad_hoc_recognizers: list[EntityRecognizer] | None = None,
        context: list[str] | None = None,  # noqa: ARG002
        allow_list: list[str] | None = None,  # noqa: ARG002
        nlp_artifacts_batch: Iterable[NlpArtifacts] | None = None,
    ) -> list[list[RecognizerResult]]:
        all_fields = not entities

        recognizers = self.analyzer_engine.registry.get_recognizers(
            language=language,
            entities=entities,
            all_fields=all_fields,
            ad_hoc_recognizers=ad_hoc_recognizers,
        )

        if all_fields:
            # Since all_fields=True, list all entities by iterating
            # over all recognizers
            entities = self.analyzer_engine.get_supported_entities(language=language)

        all_results = []

        for text, nlp_artifacts in nlp_artifacts_batch:
            results = []
            for recognizer in recognizers:
                current_results = recognizer.analyze(text=text, entities=entities, nlp_artifacts=nlp_artifacts)
                if current_results:
                    # add recognizer name to recognition metadata inside results
                    # if not exists
                    results.extend(current_results)

            all_results.append(results)

        return all_results

    def analyze_iterator(
        self,
        texts: Iterable[str | bool | float | int],
        language: str,
        batch_size: int = 32,
        **kwargs,
    ) -> list[list[RecognizerResult]]:
        """
        Analyze an iterable of strings.

        :param texts: An list containing strings to be analyzed.
        :param language: Input language
        :param kwargs: Additional parameters for the `AnalyzerEngine.analyze` method.
        :param batch_size
        """
        # Process the texts as batch for improved performance
        nlp_engine: CustomNlpEngine = self.analyzer_engine.nlp_engine
        nlp_artifacts_batch: Iterable[NlpArtifacts] = nlp_engine.process_batch(
            texts=texts,
            language=language,
            batch_size=batch_size,
            as_tuples=kwargs.get("as_tuples", False),
        )

        return self.analyze_batch(
            texts=texts,
            nlp_artifacts_batch=nlp_artifacts_batch,
            language=language,
            **kwargs,
        )

    def analyze_dict(
        self,
        input_dict: dict[str, Any | Iterable[Any]],
        language: str,
        keys_to_skip: list[str] | None = None,
        **kwargs,
    ) -> Iterator[DictAnalyzerResult]:
        """
        Analyze a dictionary of keys (strings) and values/iterable of values.

        Non-string values are returned as is.

        :param input_dict: The input dictionary for analysis
        :param language: Input language
        :param keys_to_skip: Keys to ignore during analysis
        :param kwargs: Additional keyword arguments
        for the `AnalyzerEngine.analyze` method.
        Use this to pass arguments to the analyze method,
        such as `ad_hoc_recognizers`, `context`, `return_decision_process`.
        See `AnalyzerEngine.analyze` for the full list.
        """

        context = []
        if "context" in kwargs:
            context = kwargs["context"]
            del kwargs["context"]

        if not keys_to_skip:
            keys_to_skip = []

        for key, value in input_dict.items():
            if not value or key in keys_to_skip:
                yield DictAnalyzerResult(key=key, value=value, recognizer_results=[])
                continue  # skip this key as requested

            # Add the key as an additional context
            specific_context = context[:]
            specific_context.append(key)

            if type(value) in (str, int, bool, float):
                results: list[RecognizerResult] = self.analyzer_engine.analyze(
                    text=str(value), language=language, context=[key], **kwargs
                )
            elif isinstance(value, dict):
                new_keys_to_skip = self._get_nested_keys_to_skip(key, keys_to_skip)
                results = self.analyze_dict(
                    input_dict=value,
                    language=language,
                    context=specific_context,
                    keys_to_skip=new_keys_to_skip,
                    **kwargs,
                )
            elif isinstance(value, Iterable):
                # Recursively iterate nested dicts

                results: list[list[RecognizerResult]] = self.analyze_iterator(
                    texts=value,
                    language=language,
                    context=specific_context,
                    **kwargs,
                )
            else:
                msg = f"type {type(value)} is unsupported."
                raise ValueError(msg)

            yield DictAnalyzerResult(key=key, value=value, recognizer_results=results)
