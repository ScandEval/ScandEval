"""Constants used throughout the project."""

# This is used as input to generative models; it cannot be a special token
DUMMY_FILL_VALUE = 100


GENERATIVE_MODEL_TASKS = ["text-generation", "conversational", "text2text-generation"]


GENERATIVE_DATASET_TASKS = [
    "knowledge",
    "common-sense-reasoning",
    "multiple-choice-reading-comprehension",
]


GENERATIVE_DATASET_SUPERTASKS = ["text-to-text", "text-modelling"]


TASKS_USING_JSON = ["ner"]


SUPERTASKS_USING_LOGPROBS = ["sequence-classification"]


METRIC_ATTRIBUTES_TAKING_UP_MEMORY = ["cached_bertscorer"]


GENERATIVE_TAGS = [
    "trl",
    "mistral",
    "text-generation-inference",
    "unsloth",
    "text-generation",
    "llama",
]


MAX_LOGPROBS = 10
