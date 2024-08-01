============================
Filters
============================

------------------------------
Base Class
------------------------------

.. autoclass:: nemo_curator.filters.DocumentFilter
    :members:
    :member-order: bysource

.. autofunction:: nemo_curator.filters.import_filter

------------------------------
Modules
------------------------------

.. autoclass:: nemo_curator.ScoreFilter
    :members:

.. autoclass:: nemo_curator.Score
    :members:

.. autoclass:: nemo_curator.Filter
    :members:

------------------------------
FastText Filters
------------------------------

.. autoclass:: nemo_curator.filters.FastTextLangId
    :members:

.. autoclass:: nemo_curator.filters.FastTextQualityFilter
    :members:

------------------------------
Heuristic Filters
------------------------------

.. autoclass:: nemo_curator.filters.NonAlphaNumericFilter
    :members:

.. autoclass:: nemo_curator.filters.SymbolsToWordsFilter
    :members:

.. autoclass:: nemo_curator.filters.NumbersFilter
    :members:

.. autoclass:: nemo_curator.filters.UrlsFilter
    :members:

.. autoclass:: nemo_curator.filters.BulletsFilter
    :members:

.. autoclass:: nemo_curator.filters.WhiteSpaceFilter
    :members:

.. autoclass:: nemo_curator.filters.ParenthesesFilter
    :members:

.. autoclass:: nemo_curator.filters.LongWordFilter
    :members:

.. autoclass:: nemo_curator.filters.WordCountFilter
    :members:

.. autoclass:: nemo_curator.filters.BoilerPlateStringFilter
    :members:

.. autoclass:: nemo_curator.filters.MeanWordLengthFilter
    :members:

.. autoclass:: nemo_curator.filters.RepeatedLinesFilter
    :members:

.. autoclass:: nemo_curator.filters.RepeatedParagraphsFilter
    :members:

.. autoclass:: nemo_curator.filters.RepeatedLinesByCharFilter
    :members:

.. autoclass:: nemo_curator.filters.RepeatedParagraphsByCharFilter
    :members:

.. autoclass:: nemo_curator.filters.RepeatingTopNGramsFilter
    :members:

.. autoclass:: nemo_curator.filters.RepeatingDuplicateNGramsFilter
    :members:

.. autoclass:: nemo_curator.filters.PunctuationFilter
    :members:

.. autoclass:: nemo_curator.filters.EllipsisFilter
    :members:

.. autoclass:: nemo_curator.filters.CommonEnglishWordsFilter
    :members:

.. autoclass:: nemo_curator.filters.WordsWithoutAlphabetsFilter
    :members:

.. autoclass:: nemo_curator.filters.PornographicUrlsFilter
    :members:

------------------------------
Code Filters
------------------------------

.. autoclass:: nemo_curator.filters.PythonCommentToCodeFilter
    :members:

.. autoclass:: nemo_curator.filters.GeneralCommentToCodeFilter
    :members:

.. autoclass:: nemo_curator.filters.NumberOfLinesOfCodeFilter
    :members:

.. autoclass:: nemo_curator.filters.TokenizerFertilityFilter
    :members:

.. autoclass:: nemo_curator.filters.XMLHeaderFilter
    :members:

.. autoclass:: nemo_curator.filters.AlphaFilter
    :members:

.. autoclass:: nemo_curator.filters.HTMLBoilerplateFilter
    :members:

.. autoclass:: nemo_curator.filters.PerExtensionFilter
    :members: