# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# from https://gitlab-master.nvidia.com/dl/JoC/prospector-lm/-/blob/main/prospector/download/utils.py

import unicodedata

import cchardet as chardet
import justext
import lxml
import pycld2 as cld2

# These functions are modified versions of those originally written
# by EluetherAI for downloading and extracting Pile-CC
# https://github.com/leogao2/commoncrawl_downloader/blob/master/download_commoncrawl.py


def decode_html(html):
    # Convert from bytes to text using utf-8 encoding
    try:
        html = html.decode("utf-8")
    except UnicodeDecodeError:
        # try to figure out encoding if not urf-8
        guess = chardet.detect(html)["encoding"]
        if not guess or guess == "UTF-8":
            return
        try:
            html = html.decode(guess)
        except (UnicodeDecodeError, LookupError):
            # still cant figure out encoding, give up
            return
    return html


def lang_detect(html):
    try:
        _, _, details = cld2.detect(html)
    except Exception:
        html_no_ctrl_chars = "".join(
            [
                i
                for i in html
                if unicodedata.category(i)[0]
                not in [
                    "C",
                ]
            ]
        )
        _, _, details = cld2.detect(html_no_ctrl_chars)

    return details[0][0].upper()


def extract_text(
    html,
    stop_words,
    length_low=70,
    length_high=200,
    stopwords_low=0.30,
    stopwords_high=0.32,
    max_link_density=0.2,
    max_heading_distance=200,
    no_headings=False,
    logger=None,
):
    # Segment the HTML into paragraphs
    try:
        # Form the DOM tree
        dom = justext.core.html_to_dom(html)
        cleaned_dom = justext.core.preprocessor(dom)
        # Get the paragraphs from the DOM
        handler = justext.core.ParagraphMaker()
        lxml.sax.saxify(cleaned_dom, handler)
    except (lxml.etree.ParserError, ValueError, Exception):
        # Return nothing when we cannot segment the document
        if logger is not None:
            logger.info("Could not segment paragaphs in the document")
        return
    paragraphs = handler.paragraphs

    # Context free classification
    justext.core.classify_paragraphs(
        paragraphs,
        stop_words,
        length_low,
        length_high,
        stopwords_low,
        stopwords_high,
        max_link_density,
        no_headings,
    )

    # Copy the context free class to the class_style
    # This handles the headings as described in the
    # documentation
    for paragraph in paragraphs:
        paragraph.class_type = paragraph.cf_class

    # Context sensitive classification
    justext.core.revise_paragraph_classification(
        paragraphs,
        max_heading_distance,
    )

    return [p.text for p in paragraphs if not p.is_boilerplate]


def get_stop_list_dict(languages=[]):

    # Name mapping for language names from CLD2 (values)
    # and jusText (keys)
    lang_map = {
        "Haitian": "HAITIAN_CREOLE",
        "Norwegian_Bokmal": "NORWEGIAN",
        "Norwegian_Nynorsk": "NORWEGIAN_N",
        "Waray_Waray": "WARAY_PHILIPPINES",
    }
    if len(languages) == 0:
        languages = justext.get_stoplists()
        # Remove latin as it yields a lot of low quality documents
        languages_no_latin = list(languages)
        languages_no_latin.remove("Latin")
        languages = frozenset(languages_no_latin)

    stop_list_dict = {}
    for language in languages:
        if language in lang_map:
            lang_key = lang_map[language]
        else:
            lang_key = language.upper()
        stop_list_dict[lang_key] = justext.get_stoplist(language)

    return stop_list_dict


def get_all_stop_words():
    stop_words = set()
    for language in justext.get_stoplists():
        stop_words.update(justext.get_stoplist(language))

    return frozenset(stop_words)
