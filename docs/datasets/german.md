# 🇩🇪 German

This is an overview of all the datasets used in the German part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### SB10k

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

### GermEval

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

### ScaLA-de

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

### GermanQuAD

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

### MMLU-de

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


### Unofficial: ARC-de

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

### HellaSwag-de

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

### MLSum

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
