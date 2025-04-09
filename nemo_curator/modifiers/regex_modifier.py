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

import re
from typing import Dict, List

from nemo_curator.modifiers import DocumentModifier

__all__ = ["RegexModifier"]


class RegexModifier(DocumentModifier):
    """
    A class for modifying documents using regular expressions.

    This class applies a series of regex-based substitutions to an input text.
    Each substitution rule is defined by a dictionary containing at least a
    'pattern' and a 'repl' key, and optionally a 'count' key to limit the number
    of substitutions.
    """

    def __init__(self, regex_params_list: List[Dict]):
        """
        Initialize the RegexModifier with a list of regex parameter dictionaries.

        Args:
            regex_params_list (List[Dict]): List of dictionaries where each dictionary
                contains keys 'pattern' and 'repl' (and optionally 'count'). Each
                dictionary defines a regex substitution rule.

        Raises:
            ValueError: If any dictionary in the list is missing the 'pattern' or 'repl' key.
        """
        super().__init__()
        self.regex_params_list = regex_params_list

        # verify all dicts in regex_params_list have "pattern" and "repl" keys
        for regex_params_dict in self.regex_params_list:
            if not "pattern" in regex_params_dict.keys():
                raise ValueError(
                    f"Need to have key 'pattern' in all entries of `regex_params_list`: {self.regex_params_list}"
                )
            if not "repl" in regex_params_dict.keys():
                raise ValueError(
                    f"Need to have key 'repl' in all entries of `regex_params_list`: {self.regex_params_list}"
                )

    def modify_document(self, text: str) -> str:
        """
        Modify the given text by applying regex substitutions as defined in regex_params_list.

        The process includes:
            1. Adding a space at the beginning and end of the text to help match whole words.
            2. Iteratively applying each regex substitution using the rules in regex_params_list.
            3. Removing any extra spaces that might result from the substitutions.

        Args:
            text (str): The input text to modify.

        Returns:
            str: The modified text after all regex substitutions have been applied.
        """
        text_in = RegexModifier._add_start_end_spaces(text)
        for regex_params in self.regex_params_list:
            text_out = re.sub(
                pattern=regex_params["pattern"],
                repl=regex_params["repl"],
                string=text_in,
                # note: this count param is the maximum number of pattern occurrences to be replaced.
                count=regex_params.get("count", 0),
            )
            text_in = text_out

        text_out = RegexModifier._remove_extra_spaces(text_out)

        return text_out

    @staticmethod
    def _remove_extra_spaces(input_string):
        """
        Remove extra spaces between words and trim spaces at the start and end of the string.

        This method splits the input string by any whitespace and rejoins the tokens with
        a single space, effectively collapsing multiple spaces into one and removing any
        leading or trailing spaces.

        Args:
            input_string (str): The string from which extra spaces should be removed.

        Returns:
            str: The cleaned string with extra spaces removed.

        Examples:
            >>> RegexModifier.remove_extra_spaces("abc  xyz   abc xyz")
            'abc xyz abc xyz'
            >>> RegexModifier.remove_extra_spaces(" abc xyz ")
            'abc xyz'
        """
        output_string = " ".join(input_string.split())
        return output_string

    @staticmethod
    def _add_start_end_spaces(input_string):
        """
        Add a single space at the beginning and end of the input string.

        This is useful when specifying regex patterns that require a word to have spaces
        on both sides, ensuring that words at the boundaries of the string are correctly matched.
        The method first normalizes the string by removing extra spaces, then pads it with a
        space at both the start and the end.

        Args:
            input_string (str): The original string to pad.

        Returns:
            str: The padded string with a leading and trailing space.

        Examples:
            >>> RegexModifier.add_start_end_spaces("abc xyz")
            ' abc xyz '
            >>> RegexModifier.add_start_end_spaces("  abc  xyz  ")
            ' abc xyz '
        """
        # ensure no extra spaces
        no_extra_spaces_string = RegexModifier._remove_extra_spaces(input_string)
        output_string = f" {no_extra_spaces_string} "

        return output_string
