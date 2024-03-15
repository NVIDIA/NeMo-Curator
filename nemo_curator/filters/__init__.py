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

from .doc_filter import DocumentFilter, import_filter
from .classifier_filter import FastTextLangId, FastTextQualityFilter, BatchedFastTextQualityFilter
from .heuristic_filter import NonAlphaNumericFilter, SymbolsToWordsFilter, NumbersFilter, UrlsFilter, BulletsFilter, WhiteSpaceFilter, ParenthesesFilter, LongWordFilter, WordCountFilter, BoilerPlateStringFilter, MeanWordLengthFilter, RepeatedLinesFilter, RepeatedParagraphsFilter, RepeatedLinesByCharFilter, RepeatedParagraphsByCharFilter, RepeatingTopNGramsFilter, RepeatingDuplicateNGramsFilter, PunctuationFilter, EllipsisFilter, CommonEnglishWordsFilter, WordsWithoutAlphabetsFilter, PornographicUrlsFilter
from .code import PythonCommentToCodeFilter, GeneralCommentToCodeFilter, NumberOfLinesOfCodeFilter, TokenizerFertilityFilter, XMLHeaderFilter, AlphaFilter, HTMLBoilerplateFilter, PerExtensionFilter

__all__ = ["BatchedFastTextQualityFilter", "DocumentFilter", "import_filter", "FastTextLangId", "FastTextQualityFilter", "NonAlphaNumericFilter", "SymbolsToWordsFilter", "NumbersFilter", "UrlsFilter", "BulletsFilter", "WhiteSpaceFilter", "ParenthesesFilter", "LongWordFilter", "WordCountFilter", "BoilerPlateStringFilter", "MeanWordLengthFilter", "RepeatedLinesFilter", "RepeatedParagraphsFilter", "RepeatedLinesByCharFilter", "RepeatedParagraphsByCharFilter", "RepeatingTopNGramsFilter", "RepeatingDuplicateNGramsFilter", "PunctuationFilter", "EllipsisFilter", "CommonEnglishWordsFilter", "WordsWithoutAlphabetsFilter", "PornographicUrlsFilter", "PythonCommentToCodeFilter", "GeneralCommentToCodeFilter", "NumberOfLinesOfCodeFilter", "TokenizerFertilityFilter", "XMLHeaderFilter", "AlphaFilter", "HTMLBoilerplateFilter", "PerExtensionFilter"]

