# 🇳🇱 Dutch

This is an overview of all the datasets used in the Dutch part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Dutch Social

Description coming soon!

Here are a few examples from the training split:

```
{
  "text": "via @NYTimes  https://t.co/IjbCWIwYvR",
  "label": "neutral"
}
```
```
{
  "text": "Novak Djokovic positief getest op coronavirus na eigen tennistoernooi\n\nhttps://t.co/U7VOcjANh9",
  "label": "positive"
}
```
```
{
  "text": "RT @BryanRoyAjax: Dit is echt zo gigantisch groot nieuws. 👇🏿Twee van mijn moeders zusters aan kanker overleden. Is toch heel erg dit. Kan t…",
  "label": "neutral"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Hieronder staan tweets en hun sentiment, dat 'positief', 'neutraal' of 'negatief' kan zijn.
  ```
- Base prompt template:
  ```
  Tweet: {text}
  Sentiment: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tweet: {text}

  Classificeer het sentiment in de tweet. Antwoord met 'positief', 'neutraal' of 'negatief'.
  ```
- Label mapping:
    - `positive` ➡️ `positief`
    - `neutral` ➡️ `neutraal`
    - `negative` ➡️ `negatief`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset dutch-social
```


## Named Entity Recognition

### CoNLL-2002 Dutch

Description coming soon!

Here are a few examples from the training split:

```
{
  "tokens": [
    "Te",
    "verdienen",
    "bonificaties"
  ],
  "labels": [
    "O",
    "O",
    "O"
  ],
  "text": "Te verdienen bonificaties"
}
```
```
{
  "tokens": [
    "Algemeen"
  ],
  "labels": [
    "O"
  ],
  "text": "Algemeen"
}
```
```
{
  "tokens": [
    "Kan",
    "ook",
    "moeilijk",
    "met",
    "de",
    "Giro",
    "die",
    "nog",
    "bezig",
    "is",
    "."
  ],
  "labels": [
    "O",
    "O",
    "O",
    "O",
    "O",
    "B-MISC",
    "O",
    "O",
    "O",
    "O",
    "O"
  ],
  "text": "Kan ook moeilijk met de Giro die nog bezig is ."
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Hieronder staan zinnen en JSON woordenboeken met de genoemde entiteiten die voorkomen in de gegeven zin.
  ```
- Base prompt template:
  ```
  Zin: {text}
  Genoemde entiteiten: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Zin: {text}

  Identificeer de genoemde entiteiten in de zin. Je moet dit uitvoeren als een JSON-woordenboek met de sleutels 'persoon', 'locatie', 'organisatie' en 'diversen'. De waarden moeten lijsten zijn van de genoemde entiteiten van dat type, precies zoals ze voorkomen in de zin.
  ```
- Label mapping:
    - `B-PER` ➡️ `persoon`
    - `I-PER` ➡️ `persoon`
    - `B-LOC` ➡️ `locatie`
    - `I-LOC` ➡️ `locatie`
    - `B-ORG` ➡️ `organisatie`
    - `I-ORG` ➡️ `organisatie`
    - `B-MISC` ➡️ `diversen`
    - `I-MISC` ➡️ `diversen`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset conll-nl
```


## Linguistic Acceptability

### ScaLA-nl

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Dutch Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Dutch-Alpino/) by assuming that
the documents in the treebank are correct, and corrupting the samples to create
grammatically incorrect samples. The corruptions were done by either removing a word
from a sentence, or by swapping two neighbouring words in a sentence. To ensure that
this does indeed break the grammaticality of the sentence, a set of rules were used on
the part-of-speech tags of the words in the sentence.

The original full dataset consists of 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```
{
  "text": "Met het toepassen van zelfbestuur wordt ook al op de lagere school begonnen.",
  "corruption_type": null,
  "label": "correct"
}
```
```
{
  "text": "Van vergeving kan alleen gesproken worden, als een mens zo iets onmenselijks bedreven heeft, dat er geen straf, zelfs de doodstraf niet, de schuld kan vereffenen, wanneer de schuld zo mateloos is, dat ze nog alleen vergeven kan worden.",
  "corruption_type": null,
  "label": "correct"
}
```
```
{
  "text": "We volstonden met het inleggen - een werphengel met aan het nylon snoer drie haken, zonder er wat aan! - gingen wat verpozen in de messroom - waarna we ophaalden en:",
  "corruption_type": null,
  "label": "correct"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Hieronder staan zinnen en of ze grammaticaal correct zijn.
  ```
- Base prompt template:
  ```
  Zin: {text}
  Grammaticaal correct: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Zin: {text}

  Bepaal of de zin grammaticaal correct is of niet. Antwoord met 'ja' als de zin correct is en 'nee' als dat niet het geval is.
  ```
- Label mapping:
    - `correct` ➡️ `ja`
    - `incorrect` ➡️ `nee`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset scala-nl
```


### Unofficial: Dutch CoLA

Coming soon!

Here are a few examples from the training split:

```
{
  "text": "Dat is de op de eend geschoten man.",
  "label": "incorrect"
}
```
```
{
  "text": "Jan zag 't Els 'm aanbieden.",
  "label": "incorrect"
}
```
```
{
  "text": "Hij heeft hun allemaal gisteren een uitnodiging gestuurd.",
  "label": "incorrect"
}
```


## Reading Comprehension

### SQuAD-nl

Description coming soon!

Here are a few examples from the training split:

```
Missing
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Hieronder volgen teksten met bijbehorende vragen en antwoorden.
  ```
- Base prompt template:
  ```
  Tekst: {text}
  Vraag: {question}
  Antwoord in max 3 woorden: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Tekst: {text}

  Beantwoord de volgende vraag over de bovenstaande tekst in maximaal 3 woorden.

  Besvar følgende spørgsmål om teksten ovenfor med maks. 3 ord.

  Vraag: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset squad-nl
```


## Knowledge

### MMLU-nl

Description coming soon!

Here are a few examples from the training split:

```
Missing
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Hieronder staan meerkeuzevragen (met antwoorden).
  ```
- Base prompt template:
  ```
  Vraag: {text}
  Antwoordopties:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Antwoord: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Vraag: {text}
  Antwoordopties:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Beantwoord de bovenstaande vraag met 'a', 'b', 'c' of 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset mmlu-nl
```


### Unofficial: ARC-nl

Description coming soon!

Here are a few examples from the training split:

```
Missing
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Hieronder staan meerkeuzevragen (met antwoorden).
  ```
- Base prompt template:
  ```
  Vraag: {text}
  Antwoordopties:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Antwoord: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Vraag: {text}
  Antwoordopties:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Beantwoord de bovenstaande vraag met 'a', 'b', 'c' of 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset arc-nl
```


## Common-sense Reasoning

### HellaSwag-nl

This dataset is a machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/). The original dataset was based on both
video descriptions from ActivityNet as well as how-to articles from WikiHow. The dataset
was translated by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 9,310 samples. We use an 1,024 / 256 / 2,048 split
for training, validation and testing, respectively (so 3,328 samples used in total).

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 5
- Prefix prompt:
  ```
  Hieronder staan meerkeuzevragen (met antwoorden).
  ```
- Base prompt template:
  ```
  Vraag: {text}
  Antwoordopties:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Antwoord: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Vraag: {text}
  Antwoordopties:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Beantwoord de bovenstaande vraag met 'a', 'b', 'c' of 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset hellaswag-nl
```


## Summarization

### WikiLingua-nl

Description coming soon!

Here are a few examples from the training split:

```
Missing
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Hieronder volgen artikelen met bijbehorende samenvattingen.
  ```
- Base prompt template:
  ```
  Artikel: {text}
  Samenvatting: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Artikel: {text}

  Schrijf een samenvatting van het bovenstaande artikel.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset wiki-lingua-nl
```
