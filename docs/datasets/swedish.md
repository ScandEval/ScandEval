# 🇸🇪 Swedish

This is an overview of all the datasets used in the Swedish part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### SweReC

[description]

[size-info]

Here are a few examples from the training split:

```
[example-1]
```
```
[example-2]
```
```
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Följande är recensioner och deras sentiment, som kan vara 'positiv', 'neutral' eller 'negativ'.
  ```
- Base prompt template:
  ```
  Recension: {text}
  Sentiment: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Recension: {text}

  Klassificera sentimentet i recensionen. Svara med 'positiv', 'neutral' eller 'negativ'.
  ```
- Label mapping:
    - `positive` ➡️ `positiv`
    - `neutral` ➡️ `neutral`
    - `negative` ➡️ `negativ`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset swerec
```


## Named Entity Recognition

### SUC 3.0

[description]

[size-info]

Here are a few examples from the training split:

```
[example-1]
```
```
[example-2]
```
```
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Följande är meningar och JSON-ordböcker med de namngivna enheter som förekommer i den givna meningen.
  ```
- Base prompt template:
  ```
  Mening: {text}
  Namngivna entiteter: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Mening: {text}

  Identifiera de namngivna enheterna i meningen. Du ska outputta detta som en JSON-ordbok med nycklarna 'person', 'plats', 'organisation' och 'diverse'. Värdena ska vara listor över de namngivna enheter av den typen, precis som de förekommer i meningen.
  ```
- Label mapping:
    - `B-PER` ➡️ `person`
    - `I-PER` ➡️ `person`
    - `B-LOC` ➡️ `plats`
    - `I-LOC` ➡️ `plats`
    - `B-ORG` ➡️ `organisation`
    - `I-ORG` ➡️ `organisation`
    - `B-MISC` ➡️ `diverse`
    - `I-MISC` ➡️ `diverse`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset suc3
```


## Linguistic Acceptability

### ScaLA-sv

[description]

[size-info]

Here are a few examples from the training split:

```
[example-1]
```
```
[example-2]
```
```
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Följande är meningar och huruvida de är grammatiskt korrekta.
  ```
- Base prompt template:
  ```
  Mening: {text}
  Grammatisk korrekt: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Mening: {text}

  Bestäm om meningen är grammatiskt korrekt eller inte. Svara med 'ja' om meningen är korrekt och 'nej' om den inte är.
  ```
- Label mapping:
    - `correct` ➡️ `ja`
    - `incorrect` ➡️ `nej`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scala-sv
```


## Reading Comprehension

### ScandiQA-sv

[description]

[size-info]

Here are a few examples from the training split:

```
[example-1]
```
```
[example-2]
```
```
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Nedan följer texter med tillhörande frågor och svar.
  ```
- Base prompt template:
  ```
  Text: {text}
  Fråga: {question}
  Svar på max 3 ord: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Text: {text}

  Besvara följande fråga om texten ovan med högst 3 ord.

  Fråga: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scandiqa-sv
```


## Knowledge

### MMLU-sv

[description]

[size-info]

Here are a few examples from the training split:

```
[example-1]
```
```
[example-2]
```
```
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Följande är flervalsfrågor (med svar).
  ```
- Base prompt template:
  ```
  Fråga: {text}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Fråga: {text}

  Besvara följande fråga med 'a', 'b', 'c' eller 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset mmlu-sv
```


### ARC-sv

[description]

[size-info]

Here are a few examples from the training split:

```
[example-1]
```
```
[example-2]
```
```
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Följande är flervalsfrågor (med svar).
  ```
- Base prompt template:
  ```
  Fråga: {text}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Fråga: {text}

  Besvara följande fråga med 'a', 'b', 'c' eller 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset arc-sv
```


## Common-sense Reasoning

### HellaSwag-sv

[description]

[size-info]

Here are a few examples from the training split:

```
[example-1]
```
```
[example-2]
```
```
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Följande är flervalsfrågor (med svar).
  ```
- Base prompt template:
  ```
  Fråga: {text}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Fråga: {text}

  Besvara följande fråga med 'a', 'b', 'c' eller 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset hellaswag-sv
```


## Summarization

### SweDN

[description]

[size-info]

Here are a few examples from the training split:

```
[example-1]
```
```
[example-2]
```
```
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Nedan följer artiklar med tillhörande sammanfattningar.
  ```
- Base prompt template:
  ```
  Artikel: {text}
  Sammanfattning: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Artikel: {text}

  Skriv en sammanfattning av artikeln ovan.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset swedn
```


### Unofficial: Schibsted-Sv

[description]

[size-info]

Here are a few examples from the training split:

```
[example-1]
```
```
[example-2]
```
```
[example-3]
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Nedan följer artiklar med tillhörande sammanfattningar.
  ```
- Base prompt template:
  ```
  Artikel: {text}
  Sammanfattning: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Artikel: {text}

  Skriv en sammanfattning av artikeln ovan.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset schibsted-sv
```
