"""Constants used in the dataset creation scripts."""

# Bounds on the size of texts in question answering datasets
MIN_NUM_CHARS_IN_CONTEXT = 30
MAX_NUM_CHARS_IN_CONTEXT = 5000
MIN_NUM_CHARS_IN_QUESTION = 10
MAX_NUM_CHARS_IN_QUESTION = 100


# Bounds on the size of texts in summarisation datasets
MIN_NUM_CHARS_IN_ARTICLE = 30
MAX_NUM_CHARS_IN_ARTICLE = 6000


# Bounds on the size of texts in multiple choice datasets
MIN_NUM_CHARS_IN_INSTRUCTION = 30
MAX_NUM_CHARS_IN_INSTRUCTION = 2000
MIN_NUM_CHARS_IN_OPTION = 1
MAX_NUM_CHARS_IN_OPTION = 1000
MAX_REPETITIONS = 50
