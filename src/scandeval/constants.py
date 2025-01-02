"""Constants used throughout the project."""

# This is used as input to generative models; it cannot be a special token
from .enums import TaskGroup
from .tasks import COMMON_SENSE, KNOW, MCRC, NER

DUMMY_FILL_VALUE = 100


GENERATIVE_MODEL_TASKS = ["text-generation", "conversational", "text2text-generation"]


GENERATIVE_DATASET_TASKS = [KNOW, COMMON_SENSE, MCRC]


GENERATIVE_DATASET_TASK_GROUPS = [TaskGroup.TEXT_TO_TEXT]


TASKS_USING_JSON = [NER]


TASK_GROUPS_USING_LOGPROBS = [
    TaskGroup.SEQUENCE_CLASSIFICATION,
    TaskGroup.MULTIPLE_CHOICE_CLASSIFICATION,
]


METRIC_ATTRIBUTES_TAKING_UP_MEMORY = ["cached_bertscorer"]


GENERATIVE_TAGS = [
    "trl",
    "mistral",
    "text-generation-inference",
    "unsloth",
    "text-generation",
    "gemma",
    "gemma2",
    "llama",
]


MAX_LOGPROBS = 10
