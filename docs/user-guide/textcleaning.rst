.. _data-curator-text-cleaning:

=========================
Text Cleaning
=========================

--------------------
Overview
--------------------
Documents in datasets may contain improperly decoded characters (e.g. "The Mona Lisa doesn't have eyebrows." decoding as "The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows."), inconsistent line spacing, and many urls.
NeMo Curator provides a few modules that can help remove undesirable text from within individual documents.

--------------------
Use Cases
--------------------
* Fixing improperly decoded unicode characters from webpages.
* Standardizing document layout by removing excessive newlines.
* Removing URLs in documents.

--------------------
Modules
--------------------
NeMo Curator provides a collection of easy to use modules for cleaning text.

.. code-block:: python

    from nemo_curator import Sequential, Modify, get_client
    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.modifiers import UnicodeReformatter, UrlRemover, NewlineNormalizer

    def main():
        client = get_client(cluster_type="cpu")

        dataset = DocumentDataset.read_json("books.jsonl")
        cleaning_pipeline = Sequential([
            Modify(UnicodeReformatter()),
            Modify(NewlineNormalizer()),
            Modify(UrlRemover()),
        ])

        cleaned_dataset = cleaning_pipeline(dataset)

        cleaned_dataset.to_json("cleaned_books.jsonl")

    if __name__ == "__main__":
        main()

Here, we load a dataset and perform all of the cleaning operations that NeMo Curator supports.
* ``Modify(UnicodeReformatter())``: Uses `ftfy <https://ftfy.readthedocs.io/en/latest/>`_ to fix broken Unicode characters. Modifies the `"text"` field of the datset by default. This can be changed by setting ``Modify(UnicodeReformatter(), text_field="my_field")``.
* ``Modify(NewlineNormalizer())``: Uses regex to replace 3 or more consecutive newline characters in each document with only 2 newline characters.
* ``Modify(UrlRemover())``: Uses regex to remove all urls in each document

Any subset of these steps can be run at a time.

Additionally, NeMo Curator has the ``text_cleaning`` CLI command that can perform the same functions:

.. code-block:: bash

    text_cleaning \
      --input-data-dir=/path/to/input/ \
      --output-clean-dir=/path/to/output/ \
      --normalize-newlines \
      --remove-urls

By default, the CLI will only perform unicode reformatting. Adding the ``--normalize-newlines`` and ``--remove-urls`` options add the other text cleaning options.

------------------------
Custom Text Cleaner
------------------------
It's easy to write your own custom text cleaner. The implementation of ``UnicodeReformatter`` can be used as an example.

.. code-block:: python
    import ftfy

    from nemo_curator.modifiers import DocumentModifier


    class UnicodeReformatter(DocumentModifier):
        def __init__(self):
            super().__init__()

        def modify_document(self, text: str) -> str:
            return ftfy.fix_text(text)

Simply define a new class that inherits from ``DocumentModifier`` and define the constructor and ``modify_text`` method.
Also, like the ``DocumentFilter`` class, ``modify_document`` can be annotated with ``batched`` to take in a pandas series of documents instead of a single document.
See the :ref:`document filtering page <data-curator-qualityfiltering>` for more information.

---------------------------
Additional Resources
---------------------------
* `Single GPU Tutorial <https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/single_node_tutorial/single_gpu_tutorial.ipynb>`_
* `ftfy <https://ftfy.readthedocs.io/en/latest/>`_
* `Refined Web Paper <https://arxiv.org/abs/2306.01116>`_
* `Nemotron-CC Paper <https://arxiv.org/abs/2412.02595>`_