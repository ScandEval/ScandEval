# 🇳🇴 Norwegian

This is an overview of all the datasets used in the Norwegian part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### NoReC

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
  Følgende er anmeldelser og deres sentiment, som kan være 'positiv', 'nøytral' eller 'negativ'.
  ```
- Base prompt template:
  ```
  Anmeldelse: {text}
  Sentiment: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Anmeldelse: {text}

  Klassifiser sentimentet i anmeldelsen. Svar med 'positiv', 'nøytral' eller 'negativ'.
  ```
- Label mapping:
    - `positive` ➡️ `positiv`
    - `neutral` ➡️ `nøytral`
    - `negative` ➡️ `negativ`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset norec
```


## Named Entity Recognition

### NorNE-nb

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
  Følgende er fraser og JSON-ordbøker med de navngitte enhetene som forekommer i den gitte frasen.
  ```
- Base prompt template:
  ```
  Frase: {text}
  Navngitte enheter: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Frase: {text}

  Identifiser de navngitte enhetene i frasen. Du bør outputte dette som en JSON-ordbok med nøklene 'person', 'sted', 'organisasjon' og 'diverse'. Verdiene skal være lister over de navngitte enhetene av den typen, akkurat som de vises i frasen.
  ```
- Label mapping:
    - `B-PER` ➡️ `person`
    - `I-PER` ➡️ `person`
    - `B-LOC` ➡️ `sted`
    - `I-LOC` ➡️ `sted`
    - `B-ORG` ➡️ `organisasjon`
    - `I-ORG` ➡️ `organisasjon`
    - `B-MISC` ➡️ `diverse`
    - `I-MISC` ➡️ `diverse`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset norne-nb
```


### NorNE-nb

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
  Følgende er fraser og JSON-ordbøker med de navngitte enhetene som forekommer i den gitte frasen.
  ```
- Base prompt template:
  ```
  Frase: {text}
  Navngitte enheter: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Frase: {text}

  Identifiser de navngitte enhetene i frasen. Du bør outputte dette som en JSON-ordbok med nøklene 'person', 'sted', 'organisasjon' og 'diverse'. Verdiene skal være lister over de navngitte enhetene av den typen, akkurat som de vises i frasen.
  ```
- Label mapping:
    - `B-PER` ➡️ `person`
    - `I-PER` ➡️ `person`
    - `B-LOC` ➡️ `sted`
    - `I-LOC` ➡️ `sted`
    - `B-ORG` ➡️ `organisasjon`
    - `I-ORG` ➡️ `organisasjon`
    - `B-MISC` ➡️ `diverse`
    - `I-MISC` ➡️ `diverse`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset norne-nn
```


## Linguistic Acceptability

### ScaLA-nb

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
  Følgende er setninger og hvorvidt de er grammatisk korrekte.
  ```
- Base prompt template:
  ```
  Setning: {text}
  Grammatisk korrekt: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Setning: {text}

  Bestem om setningen er grammatisk korrekt eller ikke. Svar med 'ja' hvis setningen er korrekt og 'nei' hvis den ikke er.
  ```
- Label mapping:
    - `correct` ➡️ `ja`
    - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scala-nb
```


### ScaLA-nn

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
  Følgende er setninger og hvorvidt de er grammatisk korrekte.
  ```
- Base prompt template:
  ```
  Setning: {text}
  Grammatisk korrekt: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Setning: {text}

  Bestem om setningen er grammatisk korrekt eller ikke. Svar med 'ja' hvis setningen er korrekt og 'nei' hvis den ikke er.
  ```
- Label mapping:
    - `correct` ➡️ `ja`
    - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scala-nn
```


## Reading Comprehension

### NorQuAD

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

- Number of few-shot examples: 2
- Prefix prompt:
  ```
  Her følger tekster med tilhørende spørsmål og svar.
  ```
- Base prompt template:
  ```
  Tekst: {text}
  Spørsmål: {question}
  Svar på maks 3 ord: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekst: {text}

  Besvar følgende spørsmål om teksten ovenfor med maks 3 ord.

  Spørsmål: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset norquad
```


### Unofficial: NorGLM Multi QA

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

- Number of few-shot examples: 2
- Prefix prompt:
  ```
  Her følger tekster med tilhørende spørsmål og svar.
  ```
- Base prompt template:
  ```
  Tekst: {text}
  Spørsmål: {question}
  Svar på maks 3 ord: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekst: {text}

  Besvar følgende spørsmål om teksten ovenfor med maks 3 ord.

  Spørsmål: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset norglm-multi-qa
```


### Unofficial: NorGLM Multi QA

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

- Number of few-shot examples: 2
- Prefix prompt:
  ```
  Her følger tekster med tilhørende spørsmål og svar.
  ```
- Base prompt template:
  ```
  Tekst: {text}
  Spørsmål: {question}
  Svar på maks 3 ord: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekst: {text}

  Besvar følgende spørsmål om teksten ovenfor med maks 3 ord.

  Spørsmål: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset norglm-multi-qa
```


## Knowledge

### MMLU-no

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
  Følgende er flervalgsspørsmål (med svar).
  ```
- Base prompt template:
  ```
  Spørsmål: {text}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spørsmål: {text}

  Besvar følgende spørsmål med 'a', 'b', 'c' eller 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset mmlu-no
```


### ARC-no

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
  Følgende er flervalgsspørsmål (med svar).
  ```
- Base prompt template:
  ```
  Spørsmål: {text}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spørsmål: {text}

  Besvar følgende spørsmål med 'a', 'b', 'c' eller 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset arc-no
```


## Common-sense Reasoning

### HellaSwag-no

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
  Følgende er flervalgsspørsmål (med svar).
  ```
- Base prompt template:
  ```
  Spørsmål: {text}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spørsmål: {text}

  Besvar følgende spørsmål med 'a', 'b', 'c' eller 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset hellaswag-no
```


## Summarization

### NoSammendrag

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
  Her følger nyhetsartikler med tilhørende sammendrag.
  ```
- Base prompt template:
  ```
  Nyhetsartikkel: {text}
  Sammendrag: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Nyhetsartikkel: {text}

  Skriv et sammendrag av den ovennevnte artikkelen.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset no-sammendrag
```


### Unofficial: NorGLM Multi Sum

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
  Her følger nyhetsartikler med tilhørende sammendrag.
  ```
- Base prompt template:
  ```
  Nyhetsartikkel: {text}
  Sammendrag: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Nyhetsartikkel: {text}

  Skriv et sammendrag av den ovennevnte artikkelen.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset norglm-multi-sum
```


### Unofficial: Schibsted-No

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
  Her følger nyhetsartikler med tilhørende sammendrag.
  ```
- Base prompt template:
  ```
  Nyhetsartikkel: {text}
  Sammendrag: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Nyhetsartikkel: {text}

  Skriv et sammendrag av den ovennevnte artikkelen.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset schibsted-no
```
