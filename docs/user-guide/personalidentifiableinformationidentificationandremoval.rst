
.. _data-curator-pii:

======================================
PII Identification and Removal
======================================

--------------------------------------
Background
--------------------------------------

The purpose of a personal identifiable information (PII) de-identification tool is to help scrub sensitive data out of datasets.
Some examples of sensitive data types that are currently supported by the tool include:

- Name
- Email Address
- Physical Address
- Phone Numbers
- IP Address
- Credit/Debit Card Numbers
- US Social Security Numbers
- Dates

NeMo Curator provides several tools for PII identification and removal.

The ``PiiModifier`` class utilizes the `Presidio <https://microsoft.github.io/presidio/>`_ library to identify and redact PII in text.
`Dask <https://dask.org>`_ is used to parallelize tasks and hence it can be used to scale up to terabytes of data easily.
Although Dask can be deployed on various distributed compute environments, such as HPC clusters, Kubernetes, and other cloud offerings
(such as Amazon EKS, Google Cloud, etc.), the current implementation only supports Dask on HPC clusters that use Slurm as the resource manager.

The ``LLMPiiModifier`` and ``AsyncLLMPiiModifier`` classes utilize LLM models to identify PII in text.
Using `NVIDIA NIM <https://developer.nvidia.com/nim>`_, the ``LLMPiiModifier`` and ``AsyncLLMPiiModifier`` classes can submit prompts to LLMs, such as Meta's `Llama 3.1 <https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct>`_, for PII identification.
The LLM's reponse is then parsed and used to redact the PII in the text.

-----------------------------------------
Usage
-----------------------------------------
############################
PII redaction using Presidio
############################

Imagine we have a "books" dataset stored in the following structure:
::

    books_dataset/
        books_00.jsonl
        books_01.jsonl
        books_02.jsonl

You could read, de-identify the dataset, and write it to an output directory using the following approach:

.. code-block:: python

    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.utils.distributed_utils import read_data, write_to_disk, get_client
    from nemo_curator.utils.file_utils import get_batched_files
    from nemo_curator.modules.modify import Modify
    from nemo_curator.modifiers.pii_modifier import PiiModifier

    client = get_client(cluster_type="cpu")

    modifier = PiiModifier(
        language="en",
        supported_entities=["PERSON", "EMAIL_ADDRESS"],
        anonymize_action="replace",
        batch_size=1000,
        device="gpu",
    )

    for file_names in get_batched_files(
        "book_dataset",
        "output_directory",
        "jsonl",
        32
    ):
        source_data = read_data(
            file_names, file_type="jsonl", backend="pandas", add_filename=True
        )
        dataset = DocumentDataset(source_data)
        print(f"Dataset has {source_data.npartitions} partitions")

        modify = Modify(modifier)
        modified_dataset = modify(dataset)
        write_to_disk(
            modified_dataset.df,
            "output_directory",
            write_to_filename=True,
            output_type="jsonl",
        )

Let's walk through this code line by line:

* ``modifier = PiiModifier(...)`` creates an instance of ``PiiModifier`` class that is responsible for PII de-identification.
* ``supported_entities=["PERSON", "EMAIL_ADDRESS"]`` specifies the PII entities that the ``PiiModifier`` will identify and redact. By default, the ``PiiModifier`` will identify and redact the following entities:
::

    [
        "ADDRESS",
        "CREDIT_CARD",
        "EMAIL_ADDRESS",
        "DATE_TIME",
        "IP_ADDRESS",
        "LOCATION",
        "PERSON",
        "URL",
        "US_SSN",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
        "PHONE_NUMBER",
    ]

* ``for file_names in get_batched_files`` retrieves a batch of 32 documents from the ``book_dataset`` directory.
* ``source_data = read_data(...)`` reads the data from all the files using Dask using Pandas as the backend. The ``add_filename`` argument ensures that the output files have the same filename as the input files.
* ``dataset = DocumentDataset(source_data)``  creates an instance of ``DocumentDataset`` using the batch files. ``DocumentDataset`` is the standard format for text datasets in NeMo Curator.
* ``modify = Modify(modifier)`` creates an instance of the ``Modify`` class. This class can take any modifier as an argument.
* ``modified_dataset = modify(dataset)`` modifies the data in the dataset by performing the PII de-identification based upon the passed parameters.
* ``write_to_disk(...)`` writes the de-identified documents to disk.

The ``PiiModifier`` module can be invoked via the ``nemo_curator/scripts/find_pii_and_deidentify.py`` script which provides a CLI-based interface. To see a complete list of options supported by the script, execute:

``find_pii_and_deidentify --help``

To launch the script from within a Slurm environment, the script ``examples/slurm/start-slurm.sh`` can be modified and used.

############################
LLM-based PII redaction
############################

Let's again consider the "books" dataset stored in the following structure:
::

    books_dataset/
        books_00.jsonl
        books_01.jsonl
        books_02.jsonl

In order to use the ``AsyncLLMPiiModifier`` class, you will need to set up a NIM endpoint with a ``base_url`` and ``api_key``.
For instructions on how to set up a NIM endpoint, please refer to the `NIM Getting Started page <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html>`_.

After setting up a NIM endpoint, you can read, de-identify the dataset, and write it to an output directory with:

.. code-block:: python

    from nemo_curator.datasets import DocumentDataset
    from nemo_curator.utils.distributed_utils import read_data, write_to_disk, get_client
    from nemo_curator.utils.file_utils import get_batched_files
    from nemo_curator.modules.modify import Modify
    from nemo_curator.modifiers.async_llm_pii_modifier import AsyncLLMPiiModifier

    client = get_client(cluster_type="cpu")

    modifier = AsyncLLMPiiModifier(
        # Endpoint for the user's NIM
        base_url="http://0.0.0.0:8000/v1",
        api_key="API KEY (if needed)",
        model="meta/llama-3.1-70b-instruct",
        # The user may provide custom labels for PII entities if desired
        pii_labels=["name", "email"],
        language="en",
        max_concurrent_requests=10,
    )

    for file_names in get_batched_files(
        "book_dataset",
        "output_directory",
        "jsonl",
        32
    ):
        source_data = read_data(
            file_names, file_type="jsonl", backend="pandas", add_filename=True
        )
        dataset = DocumentDataset(source_data)
        print(f"Dataset has {source_data.npartitions} partitions")

        modify = Modify(modifier)
        modified_dataset = modify(dataset)
        write_to_disk(
            modified_dataset.df,
            "output_directory",
            write_to_filename=True,
            output_type="jsonl",
        )

Let's walk through this code line by line:

* ``modifier = AsyncLLMPiiModifier(...)`` creates an instance of ``AsyncLLMPiiModifier`` class that is responsible for PII de-identification.
* ``pii_labels=["name", "email"]`` specifies the PII entities that the ``AsyncLLMPiiModifier`` will identify and redact. By default, the ``AsyncLLMPiiModifier`` will identify and redact the following entities:
::

    [
        "medical_record_number",
        "location",
        "address",
        "ssn",
        "date_of_birth",
        "date_time",
        "name",
        "email",
        "customer_id",
        "employee_id",
        "phone_number",
        "ip_address",
        "credit_card_number",
        "user_name",
        "device_identifier",
        "bank_routing_number",
        "company_name",
        "unique_identifier",
        "biometric_identifier",
        "account_number",
        "certificate_license_number",
        "license_plate",
        "vehicle_identifier",
        "api_key",
        "password",
        "health_plan_beneficiary_number",
        "national_id",
        "tax_id",
        "url",
        "swift_bic",
        "cvv",
        "pin",
    ]

* We recommend setting ``max_concurrent_requests=10`` to avoid overwhelming the NIM endpoint. However, the user can set this to a higher or lower value depending on their use case.
* ``for file_names in get_batched_files`` retrieves a batch of 32 documents from the ``book_dataset`` directory.
* ``source_data = read_data(...)`` reads the data from all the files using Dask using Pandas as the backend. The ``add_filename`` argument ensures that the output files have the same filename as the input files.
* ``dataset = DocumentDataset(source_data)``  creates an instance of ``DocumentDataset`` using the batch files. ``DocumentDataset`` is the standard format for text datasets in NeMo Curator.
* ``modify = Modify(modifier)`` creates an instance of the ``Modify`` class. This class can take any modifier as an argument.
* ``modified_dataset = modify(dataset)`` modifies the data in the dataset by performing the PII de-identification based upon the passed parameters.
* ``write_to_disk(...)`` writes the de-identified documents to disk.

The ``AsyncLLMPiiModifier`` module can be invoked via the ``nemo_curator/scripts/async_llm_pii_redaction.py`` script which provides a CLI-based interface. To see a complete list of options supported by the script, execute:

``async_llm_pii_redaction --help``

Above, we recommend using the ``AsyncLLMPiiModifier`` because it utilizes ``AsyncOpenAI`` to submit multiple concurrent requests to the NIM endpoint.
The higher the ``max_concurrent_requests`` is, the more faster the ``AsyncLLMPiiModifier`` will be, but the user should be mindful to avoid overwhelming the NIM endpoint.
Alternatively, the user can use the ``LLMPiiModifier`` class which does not utilize ``AsyncOpenAI`` and hence submits requests serially.
Use of the ``LLMPiiModifier`` class is the same as the ``AsyncLLMPiiModifier`` class except that the ``max_concurrent_requests`` parameter is not used.

For example:

.. code-block:: python

    from nemo_curator.modifiers.llm_pii_modifier import LLMPiiModifier

    modifier = LLMPiiModifier(
        # Endpoint for the user's NIM
        base_url="http://0.0.0.0:8000/v1",
        api_key="API KEY (if needed)",
        model="meta/llama-3.1-70b-instruct",
        # The user may provide custom labels for PII entities if desired
        pii_labels=["name", "email"],
        language="en",
    )

The ``LLMPiiModifier`` module can be invoked via the ``nemo_curator/scripts/llm_pii_redaction.py`` script which provides a CLI-based interface. To see a complete list of options supported by the script, execute:

``llm_pii_redaction --help``

############################
Resuming from interruptions
############################

It can be helpful to track which documents in a dataset have already been processed so that long curation jobs can be resumed if they are interrupted.
NeMo Curator provides a utility for easily tracking which dataset shards have already been processed.
A call to ``get_batched_files`` will return an iterator over the files that have yet to be processed by a modifier such as ``PiiModifier``.
When you re-run the code example provided above, NeMo Curator ensures that only unprocessed files are processed by the PII module.
