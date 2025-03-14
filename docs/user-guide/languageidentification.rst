
.. _data-curator-languageidentification:

#######################################################
Language Identification and Unicode Fixing
#######################################################

-----------------------------------------
Background
-----------------------------------------
Large unlabeled text corpora often contain a variety of languages.
However, data curation usually includes steps that are language specific (e.g. using language-tuned heuristics for quality filtering)
and many curators are only interested in curating a monolingual dataset.

NeMo Curator provides utilities to identify languages using `fastText <https://fasttext.cc/docs/en/language-identification.html>`_.
Even though a preliminary language identification may have been performed on the unextracted text (as is the case in our Common Crawl pipeline
using pyCLD2), `fastText <https://fasttext.cc/docs/en/language-identification.html>`_ is more accurate so it can be used for a second pass.

-----------------------------------------
Usage
-----------------------------------------

We provide an example of how to use the language identification and unicode reformatting utility at ``examples/identify_languages.py``.
At a high level, the module first identifies the languages of the documents and removes any documents for which it has high uncertainty about the language.

-----------------------------------------
Related Scripts
-----------------------------------------

To perform the language identification, we can use the config file provided in the `config` directory
and provide the path to a local copy of the `lid.176.bin` language identification fastText model. Then, with the general purpose
:code:`filter_documents` tool, we can compute language scores and codes for each document in the corpus as follows

.. code-block:: bash

    filter_documents \
      --input-data-dir=<Path to directory containing jsonl files> \
      --filter-config-file=./config/fasttext_langid.yaml \
      --log-scores \
      --log-dir=./log/lang_id


This will apply the fastText model, compute the score and obtain the language class, and then write this
information as additonal keys within each json document.

With the language information present within the keys of each json, the :code:`separate_by_metadata`, will first construct
a count of the documents by language within the corpus and then from that information, split each file across all the languages
within that file. Below is an example run command for :code:`separate_by_metadata`

.. code-block:: bash

    separate_by_metadata \
     --input-data-dir=<Path to the input directory containing jsonl files> \
     --input-metadata-field=language \
     --output-data-dir=<Output directory containing language sub-directories> \
     --output-metadata-distribution=./data/lang_distro.json

After running this module, the output directory will consist of one directory per language present within the corpus and all documents
within those directories will contain text that originates from the same language.
