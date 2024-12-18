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

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Im Folgenden sind Tweets und ihre Stimmung aufgeführt, die 'positiv', 'neutral' oder 'negativ' sein kann.
  ```
- Base prompt template:
  ```
  Tweet: {text}
  Stimmungslage: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tweet: {text}

  Klassifizieren Sie die Stimmung im Tweet. Antworten Sie mit 'positiv', 'neutral' oder 'negativ'.
  ```
- Label mapping:
    - `positive` ➡️ `positiv`
    - `neutral` ➡️ `neutral`
    - `negative` ➡️ `negativ`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset sb10k
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

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Es folgen Sätze und JSON-Wörterbücher mit den benannten Entitäten, die in der angegebenen Phrase vorkommen.
  ```
- Base prompt template:
  ```
  Satz: {text}
  Benannte Entitäten: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Satz: {text}

  Identifizieren Sie die benannten Entitäten im Satz. Sie sollten dies als JSON-Wörterbuch mit den Schlüsseln 'person', 'ort', 'organisation' und 'verschiedenes' ausgeben. Die Werte sollten Listen der benannten Entitäten dieses Typs sein, genau wie sie im Satz erscheinen.
  ```
- Label mapping:
    - `B-PER` ➡️ `person`
    - `I-PER` ➡️ `person`
    - `B-LOC` ➡️ `ort`
    - `I-LOC` ➡️ `ort`
    - `B-ORG` ➡️ `organisation`
    - `I-ORG` ➡️ `organisation`
    - `B-MISC` ➡️ `verschiedenes`
    - `I-MISC` ➡️ `verschiedenes`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset germeval
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

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Die folgenden Sätze und ob sie grammatikalisch korrekt sind.
  ```
- Base prompt template:
  ```
  Satz: {text}
  Grammatikalisch richtig: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Satz: {text}

  Bestimmen Sie, ob der Satz grammatikalisch korrekt ist oder nicht. Antworten Sie mit 'ja', wenn der Satz korrekt ist und 'nein', wenn er es nicht ist.
  ```
- Label mapping:
    - `correct` ➡️ `ja`
    - `incorrect` ➡️ `nein`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scala-de
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

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Im Folgenden finden Sie Texte mit den dazugehörigen Fragen und Antworten.
  ```
- Base prompt template:
  ```
  Text: {text}
  Fragen: {question}
  Fragen Antwort in maximal 3 Wörtern: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Text: {text}

  Beantworten Sie die folgende Frage zum obigen Text in höchstens 3 Wörtern.

  Frage: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset germanquad
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

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).
  ```
- Base prompt template:
  ```
  Frage: {text}
  Antwort: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Beantworten Sie die obige Frage mit 'a', 'b', 'c' oder 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset mmlu-de
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

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).
  ```
- Base prompt template:
  ```
  Frage: {text}
  Antwort: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Beantworten Sie die obige Frage mit 'a', 'b', 'c' oder 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset arc-de
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

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Die folgenden Fragen sind Multiple-Choice-Fragen (mit Antworten).
  ```
- Base prompt template:
  ```
  Frage: {text}
  Antwort: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Frage: {text}
  Antwortmöglichkeiten:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Beantworten Sie die obige Frage mit 'a', 'b', 'c' oder 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset hellaswag-de
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

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Im Folgenden finden Sie Nachrichtenartikel mit den dazugehörigen Zusammenfassungen.
  ```
- Base prompt template:
  ```
  Nachrichtenartikel: {text}
  Zusammenfassung: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Nachrichtenartikel: {text}

  Schreiben Sie eine Zusammenfassung des obigen Artikels.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset mlsum
```
