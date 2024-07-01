DEFAULT_YAML_CONVERSION_PROMPT_TEMPLATE = "Can you convert this list of items into a yaml format? {llm_response} \n\n Your answer should only be a parsable yaml format."

DEFAULT_MACRO_TOPICS_PROMPT_TEMPLATE = "Can you generate {n_macro_topics} comprehensive topics that encompass various aspects of our daily life, the world, and science? Your answer should be a list of topics. Make the topics as diverse as possible.For example, 1. Food and drinks. \n2. Technology.\n"

DEFAULT_SUBTOPICS_PROMPT_TEMPLATE = "Can you generate {n_subtopics} comprehensive topics that encompass various aspects of {macro_topic}? Your answer should be a list of topics. Make the topics as diverse as possible."

DEFAULT_OPEN_QA_FROM_TOPICS_PROMPT_TEMPLATE = "Can you generate {n_openlines} questions or requests related to {topic}? The questions and requests should be as diverse possible. Your answer should be a list."
