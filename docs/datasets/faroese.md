# 🇫🇴 Faroese

This is an overview of all the datasets used in the Faroese part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### FoSent

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
  Her eru nakrir tekstir flokkaðir eftir lyndi, sum kann vera 'positivt', 'neutralt' ella 'negativt'.
  ```
- Base prompt template:
  ```
  Text: {text}
  Lyndi: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekstur: {text}

  Flokka lyndið í tekstinum. Svara við 'positivt', 'neutralt' ella 'negativt'.
  ```
- Label mapping:
    - `positive` ➡️ `positivt`
    - `neutral` ➡️ `neutralt`
    - `negative` ➡️ `negativt`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset fosent
```


## Named Entity Recognition

### FoNE

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
  Her eru nakrir setningar og nakrar JSON orðabøkur við nevndar eindir, sum eru í setningunum.
  ```
- Base prompt template:
  ```
  Setningur: {text}
  Nevndar eindir: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Setningur: {text}

  Greinið nevndu einingarnar í setningunni. Þú ættir að skila þessu sem JSON orðabók með lyklunum 'persónur', 'staður', 'felagsskapur' og 'ymiskt'. Gildin ættu að vera listi yfir nevndu einingarnar af þeirri gerð, nákvæmlega eins og þær koma fram í setningunni.
  ```
- Label mapping:
    - `B-PER` ➡️ `persónur`
    - `I-PER` ➡️ `persónur`
    - `B-LOC` ➡️ `staður`
    - `I-LOC` ➡️ `staður`
    - `B-ORG` ➡️ `felagsskapur`
    - `I-ORG` ➡️ `felagsskapur`
    - `B-MISC` ➡️ `ymiskt`
    - `I-MISC` ➡️ `ymiskt`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset fone
```


### Unofficial: WikiANN-fo

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
  Her eru nakrir setningar og nakrar JSON orðabøkur við nevndar eindir, sum eru í setningunum.
  ```
- Base prompt template:
  ```
  Setningur: {text}
  Nevndar eindir: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Setningur: {text}

  Greinið nevndu einingarnar í setningunni. Þú ættir að skila þessu sem JSON orðabók með lyklunum 'persónur', 'staður', 'felagsskapur' og 'ymiskt'. Gildin ættu að vera listi yfir nevndu einingarnar af þeirri gerð, nákvæmlega eins og þær koma fram í setningunni.
  ```
- Label mapping:
    - `B-PER` ➡️ `persónur`
    - `I-PER` ➡️ `persónur`
    - `B-LOC` ➡️ `staður`
    - `I-LOC` ➡️ `staður`
    - `B-ORG` ➡️ `felagsskapur`
    - `I-ORG` ➡️ `felagsskapur`
    - `B-MISC` ➡️ `ymiskt`
    - `I-MISC` ➡️ `ymiskt`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset wikiann-fo
```


## Linguistic Acceptability

### ScaLA-fo

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
  Hetta eru nakrir setningar og um teir eru mállæruliga rættir.
  ```
- Base prompt template:
  ```
  Setningur: {text}
  Mállæruliga rættur: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Setningur: {text}

  Greinið hvort setningurin er mállæruliga rættur ella ikki. Svarið skal vera 'ja' um setningurin er rættur og 'nei' um hann ikki er.
  ```
- Label mapping:
    - `correct` ➡️ `ja`
    - `incorrect` ➡️ `nei`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scala-fo
```


## Reading Comprehension

### FoQA

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
  Hetta eru tekstir saman við spurningum og svar.
  ```
- Base prompt template:
  ```
  Tekstur: {text}
  Spurningur: {question}
  Svara við í mesta lagi trimum orðum: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekstur: {text}

  Svara hesum spurninginum um tekstin uppiyvir við í mesta lagi trimum orðum.

  Spurningur: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset foqa
```
