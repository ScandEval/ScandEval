# 🇩🇰 Danish

This is an overview of all the datasets used in the Danish part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Angry Tweeets

This dataset was published in [this
paper](https://aclanthology.org/2021.nodalida-main.53/) and was a crowd-sourcing effort
to annotate sentiment of Danish tweets. The original full dataset consists of 3,458
samples, and we are using a split of 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total).

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Prefix prompt:
  ```
  Følgende er tweets og deres sentiment, som kan være 'positiv', 'neutral'
  eller 'negativ'.
  ```
- Base prompt template:
  ```
  Tweet: {text}\nSentiment: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tweet: {text}\n\nKlassificer sentimentet i tweetet. Svar kun med 'positiv',
  'neutral' eller 'negativ'.
  ```
- Number of few-shot examples: 12
- Label mapping:
    - `positiv` ➡️ `positive`
    - `neutral` ➡️ `neutral`
    - `negativ` ➡️ `negative`


## Named Entity Recognition

### DANSK

Missing.

### Unofficial: DaNE

Missing.


## Linguistic Acceptability

### ScaLA-da

Missing.


## Reading Comprehension

### ScandiQA-da

Missing.


## Knowledge

### Danske Talemåder

Missing.

### Danish Citizen Tests

Missing.

### Unofficial: MMLU-da

Missing.

### Unofficial: ARC-da

Missing.


## Common-sense Reasoning

### HellaSwag-da

Missing.


## Summarization

### Nordjylland News

Missing.
