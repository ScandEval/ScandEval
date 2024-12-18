# 🇬🇧 English

This is an overview of all the datasets used in the English part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### SST-5

[description]

[size-info]

Here are a few examples from the training split:

```json
[example-1]
```
```json
[example-2]
```
```json
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  The following are texts and their sentiment, which can be 'positive', 'neutral' or 'negative'.
  ```
- Base prompt template:
  ```
  Text: {text}
  Sentiment: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Text: {text}

  Classify the sentiment in the text. Answer with 'positive', 'neutral' or 'negative'.
  ```
- Label mapping:
    - `positive` ➡️ `positive`
    - `neutral` ➡️ `neutral`
    - `negative` ➡️ `negative`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset sst5
```


## Named Entity Recognition

### CoNLL-2003-En

[description]

[size-info]

Here are a few examples from the training split:

```json
[example-1]
```
```json
[example-2]
```
```json
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Below are sentences and JSON dictionaries with the named entities that occur in the given sentence.
  ```
- Base prompt template:
  ```
  Sentence: {text}
  Named entities: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Sentence: {text}

  Identify the named entities in the sentence. You should output this as a JSON dictionary with the keys being 'person', 'location', 'organization' and 'miscellaneous'. The values should be lists of the named entities of that type, exactly as they appear in the sentence.
  ```
- Label mapping:
    - `B-PER` ➡️ `person`
    - `I-PER` ➡️ `person`
    - `B-LOC` ➡️ `location`
    - `I-LOC` ➡️ `location`
    - `B-ORG` ➡️ `organization`
    - `I-ORG` ➡️ `organization`
    - `B-MISC` ➡️ `miscellaneous`
    - `I-MISC` ➡️ `miscellaneous`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset conll-en
```


## Linguistic Acceptability

### ScaLA-En

[description]

[size-info]

Here are a few examples from the training split:

```json
[example-1]
```
```json
[example-2]
```
```json
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  The following are sentences and whether they are grammatically correct.
  ```
- Base prompt template:
  ```
  Sentence: {text}
  Grammatically correct: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Sentence: {text}

  Determine whether the sentence is grammatically correct or not. Reply with 'yes' if the sentence is correct and 'no' if it is not.
  ```
- Label mapping:
    - `correct` ➡️ `yes`
    - `incorrect` ➡️ `no`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scala-en
```


## Reading Comprehension

### SQuAD

[description]

[size-info]

Here are a few examples from the training split:

```json
[example-1]
```
```json
[example-2]
```
```json
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  The following are texts with accompanying questions and answers.
  ```
- Base prompt template:
  ```
  Text: {text}
  Question: {question}
  Answer in max 3 words:
  ```
- Instruction-tuned prompt template:
  ```
  Text: {text}

  Answer the following question about the above text in at most 3 words.

  Question: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset squad
```


## Knowledge

### MMLU

[description]

[size-info]

Here are a few examples from the training split:

```json
[example-1]
```
```json
[example-2]
```
```json
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  The following are multiple choice questions (with answers).
  ```
- Base prompt template:
  ```
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Answer: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Answer the above question by replying with 'a', 'b', 'c' or 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset mmlu
```


### Unofficial: ARC

[description]

[size-info]

Here are a few examples from the training split:

```json
[example-1]
```
```json
[example-2]
```
```json
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  The following are multiple choice questions (with answers).
  ```
- Base prompt template:
  ```
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Answer: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Answer the above question by replying with 'a', 'b', 'c' or 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset arc
```


## Common-sense Reasoning

### HellaSwag

[description]

[size-info]

Here are a few examples from the training split:

```json
[example-1]
```
```json
[example-2]
```
```json
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  The following are multiple choice questions (with answers).
  ```
- Base prompt template:
  ```
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Answer: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Question: {text}
  Options:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Answer the above question by replying with 'a', 'b', 'c' or 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset hellaswag
```


## Summarization

### CNN/DailyMail

[description]

[size-info]

Here are a few examples from the training split:

```json
[example-1]
```
```json
[example-2]
```
```json
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  The following are articles with accompanying summaries.
  ```
- Base prompt template:
  ```
  News article: {text}
  Summary: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  News article: {text}

  Write a summary of the above article.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset cnn-dailymail
```
