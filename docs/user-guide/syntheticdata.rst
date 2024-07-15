
.. _data-curator-syntheticdata:

======================================
Synthetic Data Generation
======================================
--------------------------------------
Background
--------------------------------------
Synthetic data generation has become increasing useful in large language model training.
It is used in pretraining, fine-tuning, and evalutation.
Synthetically generated data can be useful for adapting an LLM to low resource languages/domains, or performing knowledge distillation from other models among other purposes.
There are a variety of ways to construct synthetic data generation pipelines, with numerous LLM and classical filters.

NeMo Curator has a simple, easy-to-use set of tools that allow you to use prebuilt synthetic generation pipelines or build your own.
Any model inference service that uses the OpenAI API is compatible with the synthetic data generation module, allowing you to generate your data from any model.
Furthermore, NeMo Curator also can interface with `NeMo's Export and Deploy <https://docs.nvidia.com/nemo-framework/user-guide/latest/deployingthenemoframeworkmodel.html#use-nemo-export-and-deploy-module-apis-to-run-inference>`_
module which allows you to host your own model for LLM inference.
NeMo Curator has prebuilt synthetic data generation pipelines for supervised fine-tuning (SFT) and preference data that were used to generate data for the training of `Nemotron-4 340B <https://research.nvidia.com/publication/2024-06_nemotron-4-340b>`_.
And, you can easily interweave filtering and deduplication steps in your synthetic data pipeline with the other modules in NeMo Curator.

--------------------------------------
Connecting to an LLM Service
--------------------------------------
NeMo Curator supports connecting to `OpenAI API <https://github.com/openai/openai-python?tab=readme-ov-file#openai-python-api-library>`_ compatible services and `NeMo Deploy <https://docs.nvidia.com/nemo-framework/user-guide/latest/deployingthenemoframeworkmodel.html#use-nemo-export-and-deploy-module-apis-to-run-inference>`_ services.
Despite its name, the OpenAI API is used for querying models across different platforms beyond OpenAI's own models.
Here is how we can connect to `build.nvidia.com <https://build.nvidia.com/explore/discover>`_ to query Gemma 2 9b-it using NeMo Curator and the OpenAI API.

.. code-block:: python
    from openai import OpenAI
    from nemo_curator import OpenAIClient

    openai_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="<insert API key>",
    )
    client = OpenAIClient(openai_client)
    responses = client.query_model(
        model="mistralai/mixtral-8x7b-instruct-v0.1",
        messages=[
            {
                "role": "user",
                "content": "Write a limerick about the wonders of GPU computing.",
            }
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    print(responses[0])
    # Output:
    # A GPU with numbers in flight, Brings joy to programmers late at night.
    # With parallel delight, Solving problems, so bright,
    # In the realm of computing, it's quite a sight!

As you can see, ``OpenAIClient.query_model`` has a nearly identical function signature and behavior to the OpenAI chat completion API.
``client.query_model`` returns a list of respones, and the list is only greater than length 1 if ``n > 1`` is specified in the arugments.

The OpenAI API is great for accessing models that are hosted externally through a simple API.
However, these services are often rate limited, and if you are generating lots of synthetic data you may run into these limits.
An alternative to accessing externally hosted models is to deploy an LLM inference service yourself.
If you want to self-host models, we recommend using `NeMo's Export and Deploy <https://docs.nvidia.com/nemo-framework/user-guide/latest/deployingthenemoframeworkmodel.html#use-nemo-export-and-deploy-module-apis-to-run-inference>`_ module to ensure that you get the best performance.

Assuming you deploy a model named "mistralai/mixtral-8x7b-instruct-v0.1" on your local machine following `this NeMo Deploy guide <https://docs.nvidia.com/nemo-framework/user-guide/latest/deployingthenemoframeworkmodel.html#deploy-a-llm-model-to-tensorrt-llm>`_,
you can run the same query using the following code.

.. code-block:: python
    from nemo.deploy.nlp import NemoQueryLLM
    from nemo_curator import NemoDeployClient
    from nemo_curator.synthetic import Mixtral8x7BFormatter

    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    nemo_client = NemoQueryLLM(url="localhost:8000", model_name=model)
    client = NemoDeployClient(nemo_client)
    repsonses = client.query_model(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Write a limerick about the wonders of GPU computing.",
            }
        ],
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        conversation_formatter=Mixtral8x7BFormatter(),
    )
    print(repsonses[0])
    # Output:
    # A GPU with numbers in flight, Brings joy to programmers late at night.
    # With parallel delight, Solving problems, so bright,
    # In the realm of computing, it's quite a sight!

Let's focus on the main differences here.
* ``nemo_client = NemoQueryLLM(url="localhost:8000", model_name=model)``. This initialization requires you to specify the model name. NeMoQueryLLM is primarily built for querying a single LLM, but NeMo Curator allows you to change the model you are querying on your local server for each request.
* ``conversation_formatter=Mixtral8x7BFormatter()``. LLMs take a tokenized string of text as input, not a list of conversation turns. Therefore, during the alignment process each LLM uses a conversation format to turn the conversation into a single string. For Mixtral-8x7B-Instruct-v0.1, the format looks like this:

    .. code-block::
        <s> [INST] Instruction [/INST] Model answer</s> [INST] Follow-up instruction [/INST]

    Services that use the OpenAI API perform this formatting on the backend. In contrast, since NeMo Deploy allows you to run any model you want, you need to specify what conversation format you should use on when making the request.
    NeMo Curator provides prebuilt conversation formatters for Mixtral-8x7B-Instruct-v0.1 and Nemotron-4 340B named ``Mixtral8x7BFormatter``and ``NemotronFormatter`` respectively.

.. important::
    OpenAI API backends likely format the conversation for you automatically. Depending on your synthetic data generation process, this may lead to incorrect results. Please refer to your service's documentation to see what kind of prompt formatting they follow.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Querying a Reward Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Reward models can be used to score conversations between a user and assistant.
Instead of responding to a user prompt with text follow up as an assistant, a reward model will return a mapping of category to score.
These scores can then be used to filter the dataset to be higher quality.
Here is how we can query the Nemotron-4 340b reward model in NeMo Curator:

.. code-block:: python
    from openai import OpenAI
    from nemo_curator import OpenAIClient

    openai_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="<insert API key>",
    )
    client = OpenAIClient(openai_client)

    model = "nvidia/nemotron-4-340b-reward"

    messages = [
        {"role": "user", "content": "I am going to Paris, what should I see?"},
        {
            "role": "assistant",
            "content": "Ah, Paris, the City of Light! There are so many amazing things to see and do in this beautiful city ...",
        },
    ]

    rewards = client.query_reward_model(messages=messages, model=model)
    print(rewards)
    # {
    # "helpfulness": 1.6171875
    # "correctness": 1.6484375
    # "coherence": 3.3125
    # "complexity": 0.546875
    # "verbosity": 0.515625
    # }

For more details on the reward categories, please see the `Nemotron-4 340B Technical Report <https://arxiv.org/abs/2406.11704v1>`_.

--------------------------------------
Nemotron-4 340B Pipeline
--------------------------------------
Nemotron-4 340B is an LLM released by NVIDIA that synthetically generated 98% of the data used for its supervised fine-tuning and preference fine-tuning.
NeMo Curator contains prebuilt functions that allow you to follow the same process using the same prompt templates, and you can customize the pipelines to fit your usecase.

############################
Synthetic Prompt Generation
############################
Prompt generation is the process of synthetically generating the first line of a dialogue between a user and assistant.
This is also called "openline" generation.
Nemotron-4 340B used four different pipelines based on the generation of the `UltraChat dataset <https://arxiv.org/abs/2305.14233>`_ for generating open Q&A, writing, closed Q&A, and math & coding prompts.
NeMo Curator encapsulates all the synthetic data generation methods for Nemotron-4 340B in ``nemo_curator.synthetic.NemotronGenerator``.

We'll dive into all the methods it provides in the following sections, but here is a small example that establishes a pattern you will see with all of the functions.

.. code-block:: python
    from openai import OpenAI
    from nemo_curator import OpenAIClient
    from nemo_curator.synthetic import NemotronGenerator

    openai_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="<insert API key>"
    )
    client = OpenAIClient(openai_client)
    generator = NemotronGenerator(client)

    n_macro_topics = 20
    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    model_kwargs = {
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
    }

    responses = generator.generate_macro_topics(
        n_macro_topics=n_macro_topics, model=model, model_kwargs=model_kwargs
    )

    print(responses[0])
    # Output:
    # 1. Climate Change and Sustainable Living
    # 2. Space Exploration and the Universe
    # ...

This example should seem very similar to the ``OpenAIClient.query_model``.
We specify the model we are using just like before, along with additional keyword arguments to control the model's generation.
``generator.generate_macro_topics`` queries the LLM and asks it to generate a list of topics about the world.
There is an additional ``prompt_template`` parameter that is defaulted to the one used in Nemotron-4 340B, but it can be changed if desired.
``responses`` will be a list of responses. There will be only one response unless ``n > 1`` is specified in ``model_kwargs``.

The output of the above snippet will be a string response that contains a list of topics.
Many LLM responses in the Nemotron pipeline will contain a list.
Therefore, ``NemotronGenerator`` provides a helper function that will attempt to convert an LLM response into a Python list of strings

.. code-block:: python
    responses = generator.generate_macro_topics(
        n_macro_topics=n_macro_topics, model=model, model_kwargs=model_kwargs
    )

    topic_list = generator.convert_response_to_yaml_list(
        responses[0], model=model, model_kwargs=model_kwargs
    )
    print(topic_list[0])
    # Output:
    # Climate Change and Sustainable Living

This helper function prompts an LLM to convert the previous response into a yaml format, then attempts to parse the yaml format.
If the parsing fails, it will throw a ``YamlConversionError``.
``topic_list`` is not guaranteed to have a length of 20.
In our end to end pipelines that you will see later, NeMo Curator will raise a ``YamlConversionError`` if there is a mismatch between desired length of list and the received length of list, but this function does not check for it.

With these examples out of the way, let's look at exactly how to replicate the Nemotron-4 340B synthetic data generation pipeline in NeMo Curator.
For a more in-depth explanation of each of the steps, please refer to the `Nemotron-4 340B Technical Report <https://arxiv.org/abs/2406.11704v1>`_.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Open Q&A Prompt Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Open Q&A prompt generation follows these steps:
1. Generate a list of macro topics about the world
2. Generate a list of subtopics related to each macro topic
3. Create a list of questions relating to the previously generated topics
4. Revise the questions to be more detailed

Using NeMo Curator, each step can be performed as follows:

.. code-block:: python
    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    macro_topic_responses = generator.generate_macro_topics(
        n_macro_topics=20, model=model
    )
    macro_topics_list = ... # Parse responses manually or with convert_response_to_yaml_list

    subtopic_responses = generator.generate_subtopics(
        macro_topic=macro_topics_list[0], n_subtopics=5, model=model
    )
    subtopic_list = ... # Parse responses manually or with convert_response_to_yaml_list

    topics = macro_topics_list + subtopic_list

    question_responses = generator.generate_open_qa_from_topic(
        topic=topics[0], n_openlines=10, model=model
    )
    questions = ... # Parse responses manually or with convert_response_to_yaml_list

    revised_questions_responses = generator.revise_open_qa(
        openline=questions[0], n_revisions=5, model=model
    )
    revised_questions = ... # Parse responses manually or with convert_response_to_yaml_list

An end-to-end pipeline that composes all of these steps can be run with the ``NemotronGenerator.run_open_qa_pipeline``

.. code-block:: python
    open_qa_questions = generator.run_open_qa_pipeline(
        n_macro_topics=20,
        n_subtopics=5,
        n_openlines=10,
        n_revisions=5,
        model=model,
        ignore_conversion_failure=True,
    )

    print(open_qa_questions[0])
    # Output:
    # What are some effective sources of renewable energy?

This function runs all the previous steps together.
In order to do so, it tries to automatically convert the LLM responses to Python lists using ``convert_response_to_yaml_list``.
``ignore_conversion_failure=True`` will cause responses that cannot be automatically converted to be discarded instead of raising an error.
However, an error will still be thrown if the first step of the pipeline cannot be parsed successfully.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Writing Prompt Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Writing prompt generation follows these steps:
1. Generate tasks to write an email, essay, etc. about a topic
2. Revise the tasks to be more detailed

Using NeMo Curator, each step can be performed as follows:

.. code-block:: python
    model = "mistralai/mixtral-8x7b-instruct-v0.1"
    writing_tasks_responses = generator.generate_writing_tasks(
        topic="Climate Change and Sustainable Living",
        text_material_type="Poems",
        n_openlines=5,
        model=model,
    )
    writing_tasks_list = ... # Parse responses manually or with convert_response_to_yaml_list

    revised_writing_tasks_responses = generator.revise_writing_tasks(
        openline=writing_tasks_list[0], n_revisions=5, model=model
    )
    revised_writing_tasks = ...  # Parse responses manually or with convert_response_to_yaml_list

An end-to-end pipeline that composes all of these steps can be run with the ``NemotronGenerator.run_writing_pipeline``

.. code-block:: python
    writing_tasks = generator.run_writing_pipeline(
        topics=[
            "Climate Change and Sustainable Living",
            "Space Exploration and the Universe",
            ...,
        ],
        text_material_types=["Poems", "Essays", ...],
    )

    print(writing_tasks[0])
    # Output:
    # Write a poem about the most effective sources of renewable energy.

This function runs all the previous steps together.
In order to do so, it tries to automatically convert the LLM responses to Python lists using ``convert_response_to_yaml_list``.
``ignore_conversion_failure=True`` will cause responses that cannot be automatically converted to be discarded instead of raising an error.
However, an error will still be thrown if the first step of the pipeline cannot be parsed successfully.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Closed Q&A Prompt Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Math & Coding Prompt Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Automatic Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

###################################
Asynchronous Generation
###################################