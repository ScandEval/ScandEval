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
validation and testing, respectively (so 3,328 samples used in total). All the samples
in the original test set are included in our test set, but our test set is furthermore
using a subset of the original training set as test samples as well. The original
dataset did not have a validation split, so we have created one by sampling from the
training set.

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Følgende er tweets og deres sentiment, som kan være 'positiv', 'neutral'
  eller 'negativ'.
  ```
- Base prompt template:
  ```
  Tweet: {text}
  Sentiment: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tweet: {text}

  Klassificer sentimentet i tweetet. Svar kun med 'positiv', 'neutral' eller 'negativ'.
  ```
- Label mapping:
    - `positive` ➡️ `positiv`
    - `neutral` ➡️ `neutral`
    - `negative` ➡️ `negativ`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset angry-tweeets
```


## Named Entity Recognition

### DANSK

This dataset was published in [this
paper](https://doi.org/10.3384/nejlt.2000-1533.2024.5249) and is a manually annotated
subset of [Danish Gigaword](https://aclanthology.org/2021.nodalida-main.46/) with the 18
different named entities, following the OntoNotes 5.0 scheme. It was annotated by 10
different annotators.

The original full dataset consists of 15,062 samples, and we are using a split of 1,024
/ 256 / 1,024 samples for training, validation and testing, respectively (so 2,304
samples used in total). All samples in the validation and test sets of our version also
belong to the original validation and test set, respectively.

We have furthermore converted the OntoNotes 5.0 labelling scheme to the CoNLL-2003
labelling scheme, which is more common in the NER literature. The mapping is as follows:

- `PERSON` ➡️ `PER`
- `LOCATION` ➡️ `LOC`
- `FACILITY` ➡️ `LOC`
- `GPE` ➡️ `LOC`
- `ORGANIZATION` ➡️ `PER`
- `EVENT` ➡️ `MISC`
- `LANGUAGE` ➡️ `MISC`
- `PRODUCT` ➡️ `MISC`
- `WORK OF ART` ➡️ `MISC`
- `NORP` ➡️ `MISC`
- `CARDINAL` ➡️ `O`
- `DATE` ➡️ `O`
- `LAW` ➡️ `O`
- `MONEY` ➡️ `O`
- `ORDINAL` ➡️ `O`
- `PERCENT` ➡️ `O`
- `QUANTITY` ➡️ `O`
- `TIME` ➡️ `O`

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Følgende er sætninger og JSON-ordbøger med de navngivne enheder, som
  forekommer i den givne sætning.
  ```
- Base prompt template:
  ```
  Sætning: {text}
  Navngivne enheder: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Sætning: {text}

  Identificér de navngivne enheder i sætningen. Du skal outputte dette som en JSON-ordbog med nøglerne 'person', 'sted', 'organisation' og 'diverse'. Værdierne skal være lister over de navngivne enheder af den type, præcis som de forekommer i sætningen.
  ```
- Label mapping:
    - `B-PER` ➡️ `person`
    - `I-PER` ➡️ `person`
    - `B-LOC` ➡️ `sted`
    - `I-LOC` ➡️ `sted`
    - `B-ORG` ➡️ `organisation`
    - `I-ORG` ➡️ `organisation`
    - `B-MISC` ➡️ `diverse`
    - `I-MISC` ➡️ `diverse`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset dansk
```


### Unofficial: DaNE

Coming soon!


## Linguistic Acceptability

### ScaLA-da

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Danish Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Danish-DDT/tree/master) by
assuming that the documents in the treebank are correct, and corrupting the samples to
create grammatically incorrect samples. The corruptions were done by either removing a
word from a sentence, or by swapping two neighbouring words in a sentence. To ensure
that this does indeed break the grammaticality of the sentence, a set of rules were used
on the part-of-speech tags of the words in the sentence.

The original full dataset consists of 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Følgende er sætninger og om de er grammatisk korrekte.
  ```
- Base prompt template:
  ```
  Sætning: {text}
  Grammatisk korrekt: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Sætning: {text}

  Bestem om sætningen er grammatisk korrekt eller ej. Svar med 'ja', hvis sætningen er korrekt, og 'nej', hvis den ikke er.
  ```
- Label mapping:
    - `correct` ➡️ `ja`
    - `incorrect` ➡️ `nej`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scala-da
```


## Reading Comprehension

### ScandiQA-da

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the Danish part of the [MKQA
dataset](https://aclanthology.org/2021.tacl-1.82/). The MKQA dataset is based on the
English [Natural Questions dataset](https://aclanthology.org/Q19-1026/), based on search
queries from the Google search engine. The questions and answers were manually
translated to Danish (and other languages) as part of MKQA, and the contexts were in
ScandiQA-da machine translated using the [DeepL translation
API](https://www.deepl.com/en/pro-api/). A rule-based approach was used to ensure that
the translated contexts still contained the answer to the question, potentially by
changing the answers slightly.

The original full dataset consists of 6,810 / 500 / 500 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). All validation
samples in our version also belong to the original validation set, and all original test
samples are included in our test set. The remaining 1,548 test samples in our version
was sampled from the original training set.

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Følgende er tekster med tilhørende spørgsmål og svar.
  ```
- Base prompt template:
  ```
  Tekst: {text}
  Spørgsmål: {question}
  Svar med maks. 3 ord: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekst: {text}

  Besvar følgende spørgsmål om teksten ovenfor med maks. 3 ord.

  Spørgsmål: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scandiqa-da
```


## Knowledge

### Danske Talemåder

This dataset was created by The Danish Language and Literature Society, published
[here](https://sprogteknologi.dk/dataset/1000-talemader-evalueringsdatasaet). The
dataset features Danish idioms along with their official meaning. For each idiom, three
negative samples were created: (a) a random idiom, (b) a concrete made-up idiom, and (c)
an abstract made-up idiom. The dataset was created to evaluate the ability of language
models to understand Danish idioms.

The original full dataset consists of 1,000 samples. We use a 128 / 64 / 808 split for
training, validation and testing, respectively (so 1,000 samples used in total).

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Følgende er multiple choice spørgsmål (med svar).
  ```
- Base prompt template:
  ```
  Hvad er betydningen af følgende talemåde: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Hvad er betydningen af følgende talemåde: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Besvar ovenstående spørgsmål ved at svare med 'a', 'b', 'c' eller 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset danske-talemaader
```


### Danish Citizen Tests

This dataset was created by the Alexandra Institute by scraping the Danish citizenship
tests (indfødsretsprøven) and permanent residency tests (medborgerskabsprøven) from 2016
to 2023. These are available on the [official website of the Danish Ministry of
International Recruitment and Integration](https://danskogproever.dk/).

The original full dataset consists of 720 samples. We use an 80 / 128 / 512 split for
training, validation and testing, respectively (so 720 samples used in total).

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Følgende er multiple choice spørgsmål (med svar).
  ```
- Base prompt template:
  ```
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Spørgsmål: {text}
  Svarmuligheder:
  a. {option_a}
  b. {option_b}
  c. {option_c}

  Besvar ovenstående spørgsmål ved at svare med 'a', 'b' eller 'c'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset danish-citizen-tests
```


### Unofficial: MMLU-da

Coming soon!


### Unofficial: ARC-da

Coming soon!


## Common-sense Reasoning

### HellaSwag-da

Coming soon!


## Summarization

### Nordjylland News

Coming soon!
