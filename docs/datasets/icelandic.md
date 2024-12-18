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

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Eftirfarandi eru yfirferðir ásamt lyndisgildi þeirra, sem getur verið 'jákvætt', 'hlutlaust' eða 'neikvætt'.
  ```
- Base prompt template:
  ```
  Yfirferð: {text}
  Lyndi: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Texti: {text}

  Flokkaðu tilfinninguna í textanum. Svaraðu með 'jákvætt', 'hlutlaust' eða 'neikvætt'.
  ```
- Label mapping:
    - `positive` ➡️ `jákvætt`
    - `neutral` ➡️ `hlutlaust`
    - `negative` ➡️ `neikvætt`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset hotter-and-colder-sentiment
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

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Eftirfarandi eru setningar ásamt JSON lyklum með nefndum einingum sem koma fyrir í setningunum.
  ```
- Base prompt template:
  ```
  Setning: {text}
  Nefndar einingar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Setning: {text}

  Greinið nefndu einingarnar í setningunni. Þú ættir að skila þessu sem JSON orðabók með lyklunum 'einstaklingur', 'staðsetning', 'stofnun' og 'ýmislegt'. Gildin ættu að vera listi yfir nefndu einingarnar af þeirri gerð, nákvæmlega eins og þær koma fram í setningunni.
  ```
- Label mapping:
    - `B-PER` ➡️ `einstaklingur`
    - `I-PER` ➡️ `einstaklingur`
    - `B-LOC` ➡️ `staðsetning`
    - `I-LOC` ➡️ `staðsetning`
    - `B-ORG` ➡️ `stofnun`
    - `I-ORG` ➡️ `stofnun`
    - `B-MISC` ➡️ `ýmislegt`
    - `I-MISC` ➡️ `ýmislegt`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset mim-gold-ner
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

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Eftirfarandi eru setningar og hvort þær eru málfræðilega réttar.
  ```
- Base prompt template:
  ```
  Setning: {text}
  Málfræðilega rétt: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Setning: {text}

  Greinið hvort setningin er málfræðilega rétt eða ekki. Svarið skal vera 'já' ef setningin er rétt og 'nei' ef hún er ekki.
  ```
- Label mapping:
    - `correct` ➡️ `já`
    - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scala-is
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

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Eftirfarandi eru setningar og hvort þær eru málfræðilega réttar.
  ```
- Base prompt template:
  ```
  Setning: {text}
  Málfræðilega rétt: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Setning: {text}

  Greinið hvort setningin er málfræðilega rétt eða ekki. Svarið skal vera 'já' ef setningin er rétt og 'nei' ef hún er ekki.
  ```
- Label mapping:
    - `correct` ➡️ `já`
    - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset ice-ec
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

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Eftirfarandi eru setningar og hvort þær eru málfræðilega réttar.
  ```
- Base prompt template:
  ```
  Setning: {text}
  Málfræðilega rétt: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Setning: {text}

  Greinið hvort setningin er málfræðilega rétt eða ekki. Svarið skal vera 'já' ef setningin er rétt og 'nei' ef hún er ekki.
  ```
- Label mapping:
    - `correct` ➡️ `já`
    - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset ice-linguistic
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

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Eftirfarandi eru textar með tilheyrandi spurningum og svörum.
  ```
- Base prompt template:
  ```
  Texti: {text}
  Spurning: {question}
  Svaraðu með að hámarki 3 orðum: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Texti: {text}

  Svaraðu eftirfarandi spurningu um textann að hámarki í 3 orðum.

  Spurning: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset nqii
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

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Eftirfarandi eru textar með tilheyrandi spurningum og svörum.
  ```
- Base prompt template:
  ```
  Texti: {text}
  Spurning: {question}
  Svaraðu með að hámarki 3 orðum: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Texti: {text}

  Svaraðu eftirfarandi spurningu um textann að hámarki í 3 orðum.

  Spurning: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset icelandic-qa
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

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```
- Base prompt template:
  ```
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svara: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c' eða 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset arc-is
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

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```
- Base prompt template:
  ```
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svara: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c' eða 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset mmlu-is
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

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```
- Base prompt template:
  ```
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svara: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c' eða 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset winogrande-is
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

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Eftirfarandi eru fjölvalsspurningar (með svörum).
  ```
- Base prompt template:
  ```
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svara: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spurningar: {text}
  Svarmöguleikar:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Svaraðu eftirfarandi spurningum með 'a', 'b', 'c' eða 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset hellaswag-is
```


## Summarization

### RRN

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
  Eftirfarandi eru fréttagreinar með tilheyrandi samantektum.
  ```
- Base prompt template:
  ```
  Fréttagrein: {text}
  Samantekt: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Fréttagrein: {text}

  Skrifaðu samantekt um ofangreindu grein.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset rrn
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

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Eftirfarandi eru fréttagreinar með tilheyrandi samantektum.
  ```
- Base prompt template:
  ```
  Fréttagrein: {text}
  Samantekt: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Fréttagrein: {text}

  Skrifaðu samantekt um ofangreindu grein.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset icesum
```
