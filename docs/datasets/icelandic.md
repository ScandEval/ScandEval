# 🇮🇸 Icelandic

This is an overview of all the datasets used in the Icelandic part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Hotter and Colder Sentiment

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```
- Label mapping:
    - `positive` ➡️ `X`
    - `neutral` ➡️ `X`
    - `negative` ➡️ `X`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```


## Named Entity Recognition

### MIM-GOLD-NER

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```
- Label mapping:
    - `B-PER` ➡️ `X`
    - `I-PER` ➡️ `X`
    - `B-LOC` ➡️ `X`
    - `I-LOC` ➡️ `X`
    - `B-ORG` ➡️ `X`
    - `I-ORG` ➡️ `X`
    - `B-MISC` ➡️ `X`
    - `I-MISC` ➡️ `X`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```


## Linguistic Acceptability

### ScaLA-is

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```
- Label mapping:
    - `correct` ➡️ `X`
    - `incorrect` ➡️ `X`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```


### Unofficial: IceEC

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```
- Label mapping:
    - `correct` ➡️ `X`
    - `incorrect` ➡️ `X`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```


### Unofficial: IceLinguistic

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```
- Label mapping:
    - `correct` ➡️ `X`
    - `incorrect` ➡️ `X`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```


## Reading Comprehension

### NQiI

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```
- Label mapping:
    - `X` ➡️ `Y`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```


### Unofficial: IcelandicQA

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```
- Label mapping:
    - `X` ➡️ `Y`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```


## Knowledge

### ARC-is

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```


### Unofficial: MMLU-is

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```


## Common-sense Reasoning

### Winogrande-is

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```


### Unofficial: HellaSwag-is

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```


## Summarization

### RNN

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```


### Unofficial: IceSum

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

- Number of few-shot examples: XX
- Prefix prompt:
  ```
  [prefix-prompt]
  ```
- Base prompt template:
  ```
  [base-prompt]
  ```
- Instruction-tuned prompt template:
  ```
  [instruction-tuned-prompt]
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset X
```
