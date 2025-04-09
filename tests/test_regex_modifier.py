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

from nemo_curator.modifiers.regex_modifier import RegexModifier


class TestRegexModifier:

    def test_modify_document(self):
        # Test a simple replacement: "ö" should be normalized into "o",
        # then remove anything other than alphanumeric characters and punctuations.
        regex_params = [
            {"pattern": "ö", "repl": "o"},
            {
                "pattern": "[^ !$%',-.0123456789;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/:]",
                "repl": "",
            },
        ]
        modifier = RegexModifier(regex_params)
        # "Nein, es ist möglich🙃 " -> replacement ö ->
        # "Nein, es ist moglich🙃 " -> remove anything other than alphanumeric characters and punctuations
        # "Nein, es ist moglich " -> remove extra spaces -> "Nein, es ist moglich"
        assert (
            modifier.modify_document("Nein, es ist möglich🙃 ")
            == "Nein, es ist moglich"
        )

    def test_modify_document_with_count(self):
        # Test using the count parameter: only one occurrence should be replaced.
        regex_params = [{"pattern": r"\ba\b", "repl": "b", "count": 1}]
        modifier = RegexModifier(regex_params)
        # Input "a a a" becomes " a a a " with extra spaces.
        # With count=1, only the first "a" is replaced: " b a a "
        # Removing extra spaces yields "b a a".
        assert modifier.modify_document("a a a") == "b a a"
