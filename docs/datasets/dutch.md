# 🇳🇱 Dutch

This is an overview of all the datasets used in the Dutch part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Dutch Social

This dataset consists of Dutch tweets annotated with sentiment labels. It is not sure
how the sentiment labels were assigned, this information is pending from the authors.

The original full dataset consists of 162,805 / 54,269 / 54,268 samples for training,
validation and testing, respectively (so 271,342 samples used in total). We use a 1,024
/ 256 / 1,024 split for training, validation and testing, respectively. All the new
splits are subsets of the original splits.

Here are a few examples from the training split:

```
{
  'text': 'Novak Djokovic positief getest op coronavirus na eigen tennistoernooi\n\nhttps://t.co/U7VOcjANh9',
  'label': 'positive'
}
```
```
{
  "text": "via @NYTimes  https://t.co/IjbCWIwYvR",
  "label": "neutral"
}
```
```
{
  "text": "@backinflow 30 min Corona tijd....",
  "label": "negative"
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

This dataset was published in [this paper](https://aclanthology.org/W02-2024/) and
consists of named entity recognition annotations of the Belgian newspaper "De Morgen" of
2000.

The original full dataset consists of 8,324 / 1,916 / 1,518 samples for training,
validation and testing, respectively (so 11,758 samples used in total). We use a 1,024 /
256 / 1,024 split for training, validation and testing, respectively. All the new splits
are subsets of the original splits.

Here are a few examples from the training split:

```
{
  'tokens': array(['Puitstraat', '6', ',', '8890', 'Moorslede', '.'], dtype=object),
  'labels': array(['B-LOC', 'O', 'O', 'O', 'B-LOC', 'O'], dtype=object),
}
```
```
{
  'tokens': array(['Monami-Van', 'Roost', 'had', 'nochtans', 'verloren', '.'], dtype=object),
  'labels': array(['B-PER', 'I-PER', 'O', 'O', 'O', 'O'], dtype=object),
}
```
```
{
  'tokens': array(['Het', 'overwicht', 'lag', 'op', 'nieuw', 'nummers', 'als', "'", 'Maria', 'Maria', "'", ',', "'", 'Put', 'Your', 'Lights', 'On', "'", 'en', "'", 'Smooth', "'", ',', 'stuk', 'voor', 'stuk', 'knappe', 'songs', 'die', 'zich', 'op', 'de', 'koop', 'toe', 'in', 'korte', ',', 'krachtige', 'versies', 'lieten', 'bewonderen', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'B-PER', 'O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object),
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
  "label": "correct"
}
```
```
{
  "text": "Vragen, die door een leek niet zo eenvoudig te zijn.",
  "label": "incorrect"
}
```
```
{
  "text": "U ziet een soort eng nachtclubomgeving, waar een groepje schertsaristocraten glazig zit te lachen om haar zouteloze tussenteksten, waarin ze wanhopig probeert een intelligent ondertoontje te leggen.",
  "label": "incorrect"
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

This dataset is published [here](https://huggingface.co/datasets/GroNLP/dutch-cola) and
is a manually annotated linguistic acceptability dataset, with documents coming from
descriptions of Dutch syntax.

The original full dataset consists of 19,900 / 2,400 / 2,400 samples for training,
validation and testing, respectively (so 24,700 samples used in total). We use a 1,024 /
256 / 1,024 split for training, validation and testing, respectively. The original
splits were imbalanced, so we ensure a 50/50 split of correct/incorrect samples in the
new splits. All new splits are subsets of the original splits.

Here are a few examples from the training split:

```
{
  "text": "Tasman heeft geen Maori gezien.",
  "label": "correct"
}
```
```
{
  "text": "Jan is vrij bang voor honden en ik ben het zeer erg voor spinnen.",
  "label": "incorrect"
}
```
```
{
  "text": "Wat is het duidelijk dat Jan zal krijgen?",
  "label": "incorrect"
}
```


## Reading Comprehension

### SQuAD-nl

This dataset is published
[here](https://huggingface.co/datasets/yhavinga/squad_v2_dutch) and is a machine
translated dataset of the English [SQuAD dataset](https://aclanthology.org/D16-1264/).
These are based on English Wikipedia articles and the questions and answers are written
by crowdworkers. It is not clear how the translations were done, this information is
pending from the authors.

Here are a few examples from the training split:

```
{
  'context': 'Zoals in zijn keizerlijke edicten bleek, was de Hongwu-keizer op de hoogte van de boeddhistische band tussen Tibet en China en wilde hij die versterken. Rolpe Dorje, 4de Karmapa Lama (1340-1383), wees de uitnodiging van de Hongwu-keizer af, hoewel hij wel een paar discipelen als afgezanten naar het hof in Nanjing stuurde. De Hongwu-keizer vertrouwde ook zijn goeroe Zongluo, een van de vele boeddhistische monniken aan het hof, toe om in 1378-1382 een religieuze missie naar Tibet te leiden om boeddhistische teksten te bemachtigen.',
  'question': 'Wie verwierp een uitnodiging van de Hongwu-keizer?',
  'answers': {
    'answer_start': array([153]),
    'answer_start_en': array([139]),
    'text': array(['Rolpe Dorje'], dtype=object),
    'text_en': array(['Rolpe Dorje'], dtype=object)
  }
}
```
```
{
  'context': "De term Ashkenazi verwijst ook naar de nusach Ashkenaz (Hebreeuwse, 'liturgische traditie', of rite) die de Ashkenazische joden gebruiken in hun Siddur (gebedenboek). Een nusach wordt bepaald door de keuze van gebeden, de volgorde van gebeden, de tekst van gebeden en melodieën die wordt gebruikt bij het zingen van gebeden. Twee andere belangrijke vormen van nusach onder de Ashkenazische joden zijn Nusach Sefard (niet te verwarren met het Sefardische ritueel), de Poolse Hasidische nusach, en Nusach Ari, zoals gebruikt door Lubavitch Hasidim.",
  'question': 'Nusach Ari wordt door wie gebruikt?',
  'answers': {
    'answer_start': array([528]),
    'answer_start_en': array([487]),
    'text': array(['Lubavitch Hasidim'], dtype=object),
    'text_en': array(['Lubavitch Hasidim'], dtype=object)
  }
}
```
```
{
  'context': "Op dit moment bestaat er verwarring over de naam van de oudste Vinaccia-luthier die de winkel voor het eerst beheerde. Zijn naam is genoemd als Gennaro Vinaccia (actief van 1710 tot 1788) en Nic. Vinaccia. Zijn zoon Antonio Vinaccia was actief van 1734 tot 1796. Een van de vroegste exemplaren van een mandoline is die van Antonio Vinaccia, die in 1759 aan de Universiteit van Edinburgh is gebouwd. Een andere is van Giuseppe Vinaccia, gebouwd in 1893, die ook aan de Universiteit van Edinburgh is gebouwd. De oudste nog bestaande mandoline werd in 1744 gebouwd door Antonio's zoon, Gaetano Vinaccia. Hij verblijft in het Conservatoire Royal de Musique in Brussel, België.",
  'question': 'Waar woont de mandoline waarin Gaetano Vinaccia woont?',
  'answers': {
    'answer_start': array([622]),
    'answer_start_en': array([565]),
    'text': array(['Conservatoire Royal de Musique in Brussel, België'], dtype=object),
    'text_en': array(['Conservatoire Royal de Musique in Brussels, Belgium.'], dtype=object)
  }
}
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

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Dutch was done by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 269 / 1,410 / 13,200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```
{
  "text": "Polarisatie is een eigenschap van\nAntwoordopties:\na. transversale golven.\nb. longitudinale golven.\nc. alle golven.\nd. Geen van deze.",
  "label": "a",
}
```
```
{
  "text": "Welk internetbedrijf gaat onder de afkorting AOL?\nAntwoordopties:\na. Amerika Over Lijnen\nb. Amerika Online\nc. Amerikanen op Links\nd. Amerikanen op LOR",
  "label": "b",
}
```
```
{
  "text": "Deze vraag verwijst naar de volgende informatie. Lees het volgende fragment. Nooit waren talenten van het hoogste genie van de meest verheven soort overvloediger geschonken aan een mens. Het genie van Napoleon is verbazingwekkend. Alle takken van menselijke kennis leken even vertrouwd voor zijn gigantische geest. Zijn conversaties op St. Helena, verspreid over de talloze en omvangrijke herdenkingsstukken van degenen die ze verzamelden, zijn gevuld met de grootste interesse. Tijdens de lange doodsstrijd van zijn gevangenschap en zijn dood, sprak hij met volledige vrijheid over de gebeurtenissen van zijn wonderbaarlijke carri\u00e8re, en over al die onderwerpen van moralen, politiek en religie, die het meest diep de welvaart van ons ras betreffen. Er is geen geest die niet zal worden versterkt door bekendheid met deze diepzinnige gedachten, uitgedrukt met zoveel gloed van gevoel en energie van dictie. \u2014 John S. C. Abbott, historicus, Napoleon op St. Helena, 1855 Napoleon hielp de Franse Revolutie tot een internationale beweging te maken in de gebieden die hij veroverde.\nAntwoordopties:\na. Door een universele valuta op basis van de Franse frank op te leggen\nb. Door de brute onderdrukking van guerrilla-verzet\nc. Door het afschaffen van feodalisme en herenboerderijen\nd. Door het aanmoedigen van het gebruik van Frans als universele taal",
  "label": "c",
}
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

This dataset is a machine translated version of the English [ARC
dataset](https://doi.org/10.48550/arXiv.1803.05457) and features US grade-school science
questions. The translation to Dutch was done by the University of Oregon as part of
[this paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 1,110 / 297 / 1,170 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 1,024 split for training,
validation and testing, respectively (so 2,304 samples used in total). All new splits
are subsets of the original splits.

Here are a few examples from the training split:

```
{
  "text": "In een graslandecosysteem, als de populatie van adelaars plotseling afneemt, wat zal waarschijnlijk het effect zijn op de rest van het ecosysteem?\nAntwoordopties:\na. Het ecosysteem zal overbevolkt worden met slangen.\nb. Er zal een afname zijn in de populatie van slangen in het ecosysteem.\nc. De voedingswaarde van de bodem zal afnemen in het ecosysteem.\nd. Er zullen meer soorten planten beginnen te groeien in het ecosysteem.",
  "label": "a"
}
```
```
{
  "text": "Ptolemeus was een oude astronoom die dacht dat de Aarde het centrum van het universum was. Toen hij observaties deed die hiermee niet overeenkwamen, stelde hij een verschijnsel genaamd \"epicycli\" voor om de observaties te verklaren. Hoe was Ptolemeus' proces vergelijkbaar met het moderne wetenschappelijke proces?\nAntwoordopties:\na. Ptolemeus baseerde zijn model deels op een geloofssysteem.\nb. Observaties inspireerden Ptolemeus om zijn verklaringen aan te passen.\nc. Ptolemeus probeerde het universum te beschrijven in plaats van het te verklaren.\nd. Experimenten vormden de basis van Ptolemeus' model van het universum.",
  "label": "b"
}
```
```
{
  "text": "Wat onderscheidt de organismen in het rijk Fungi van andere eukaryotische organismen?\nAntwoordopties:\na. Fungi zijn eencellig.\nb. Fungi reproduceren seksueel.\nc. Fungi verkrijgen voedingsstoffen door middel van absorptie.\nd. Fungi maken voedsel door middel van fotosynthese.",
  "label": "c"
}
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

Here are a few examples from the training split:

```
{
  "text": "[header] Hoe maak je organische babydoekjes? [title] Kies een rol organische papieren handdoeken. [step] Deze dienen als de eigenlijke doekjes. Experimenteer met verschillende merken en texturen totdat je degene vindt die het beste werkt voor de huid van je baby.\nAntwoordopties:\na. Het is belangrijk om organische papieren handdoeken te gebruiken, omdat niet-organische papieren handdoeken bleekmiddel, verf en andere chemicali\u00ebn kunnen bevatten die vaak worden gebruikt bij de productie van papierproducten. [substeps] Over het algemeen maken bekende merken van papieren handdoeken betere doekjes dan de goedkopere, generieke versies.\nb. Je kunt een papieren handdoek gebruiken die gebruikt wordt voor luierdoekjes, maar je kunt dezelfde ook gebruiken voor andere doekjes. [substeps] Je kunt drie- of vierzijdige doekjes gebruiken om je te helpen bij het mengen van alle melk, yoghurt en water die je in \u00e9\u00e9n container hebt gemengd.\nc. Als je zelfgemaakte lotion gebruikt, gebruik dan geen papieren handdoeken; deze moeten ook van niet-papier zijn. Rol een grote rol kleine papieren handdoeken uit en houd rekening met de algehele geur van de pad.\nd. [substeps] Spreid het droge doekje uit over het hele oppervlak van de huid van je baby en vermijd contact met het droge doekje (tondeuse, kam of puimsteen). [title] Plaats de fles boven een kom met warm water gedurende 10 minuten.",
  "label": "a",
}
```
```
{
  "text": "[header] Hoe maak je een jurk zonder patroon [title] Koop een jurkmodel. [step] Je hebt een verstelbaar jurkmodel nodig om ervoor te zorgen dat je jurkontwerpen op exact de maat worden gemaakt die je nodig hebt. Verstelbare jurkmodellen zijn verkrijgbaar voor ongeveer $ 250 nieuw.\nAntwoordopties:\na. [substeps] Je kunt een schoenmakersstof, bedrukte binnenbekleding of bedrukt behang gebruiken om je jurkmodel te maken. Kies het patroon en knip het patroon zelf uit.\nb. [title] Stel je jurkmodel af op de hoogte-, taille- en torso-maten die je gaat gebruiken voor je prototypejurk. [title] Maak een schets van de jurk die je wilt maken.\nc. Als je van plan bent om strapless jurken te dragen, wil je misschien een jurkmodel kopen met een grotere voor-achter-maat. [title] Plaats je jurkmodel op de tafel.\nd. Je kunt ook een jurkmodel in de supermarkt kopen. [substeps] Als je een strapless jurk wilt, kies dan voor een mouwloze jurk.",
  "label": "b",
}
```
```
{
  "text": "[header] Hoe citrusvruchten te raspen [title] Was de citrusvrucht. [step] Voordat je begint, spoel de vrucht af onder stromend koel water en wrijf het vervolgens zachtjes schoon met een schone doek of papieren handdoek. Een lichte spoeling helpt bij het verwijderen van het natuurlijke wasachtige residu aan de buitenkant van de vrucht.\nAntwoordopties:\na. [substeps] Zorg ervoor dat de vrucht volledig is afgespoeld voordat je doorgaat naar de volgende stap. De meeste citrusvruchten hebben het beschadigde deel verwijderd, maar met het middenstuk kun je afwisselen tussen het opfrissen van de schil met water en het verwijderen van de schil.\nb. [substeps] Het werk kan het beste ook laat in de avond worden gedaan, nadat de suiker is verdampt. [title] Maak een zure citrus door een kom met zout in het water te dompelen.\nc. Je kunt de citrusvrucht ook kort laten weken in een ondiepe kom met water. [substeps] Het is belangrijk om citrusvruchten altijd te wassen wanneer je ze raspt, omdat de buitenkant het deel is dat daadwerkelijk in je voedsel terechtkomt.\nd. [title] Doe het mengsel van rasp in een druppelaar. [step] Commercieel verkrijgbare rasp komt van de schil van de citrusboom.",
  "label": "c",
}
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
$ scandeval --model <model-id> --dataset hellaswag-nl
```


## Summarization

### WikiLingua-nl

This dataset was published [here](https://aclanthology.org/2020.findings-emnlp.360/) and
consists of Dutch WikiHow articles and their summaries, where a summary consists of the
first sentence of each "how-to" step in the article (and this first sentence is not
included in the article text).

The original full dataset consists of 21,345 / 3,058 / 6,105 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 1,024 split for training,
validation and testing, respectively (so 2,304 samples used in total). All new splits
are subsets of the original splits.

Here are a few examples from the training split:

```
{
  "text": "Je gaat de ham ongeveer 15 tot 20 minuten glaceren voordat hij klaar is met koken. Om het glazuur op tijd klaar te hebben, begin je met de bereiding ervan ongeveer 45 tot 60 minuten voordat je verwacht dat de ham klaar zal zijn. Snelle glazuren zijn in een paar minuten klaar, en zelfs de glazuren die op het fornuis moeten sudderen, nemen minder dan 15 minuten in beslag. Voor de eenvoudigste optie zonder te koken, klop je gewoon 270 g donkerbruine suiker met 60 ml sinaasappelsap, rode wijn of cognac. Meng de ingredi\u00ebnten in een kleine kom totdat de suiker volledig is opgelost. Als alternatief, combineer je 270 g lichtbruine suiker, 160 ml sojasaus, en twee gehakte knoflookteentjes in een kleine steelpan -- breng dan de ingredi\u00ebnten aan de kook op gemiddeld vuur. Zet  de temperatuur lager zodra het mengsel aan de kook is. Roer het af en toe door en laat het 3-5 minuten sudderen, of tot het iets is ingedikt. Zet dan het vuur uit en laat het glazuur minstens 10 tot 15 minuten afkoelen alvorens het over de ham te strijken. Klop 320 ml melasse, 160 ml bourbon en \u00bd theelepel (1 g) gemalen kruidnagel in een kleine steelpan. Breng de ingredi\u00ebnten aan de kook op middelmatig vuur, zet het vuur dan laag en laat het onder af en toe roeren, sudderen gedurende 3-5 minuten. Op het moment dat het mengsel iets verdikt is, zet je het vuur uit en laat je het 10 tot 15 minuten afkoelen. Combineer 180 ml ahornsiroop, 120 ml sinaasappelmarmelade, 2 eetlepels (30 g) ongezouten boter, 1 eetlepel (16 g) Dijon-mosterd, 1 theelepel (2 g) gemalen zwarte peper, en \u00bc theelepel gemalen kaneel in een kleine steelpan. Laat het mengsel op matig vuur sudderen, onder af en toe roeren, gedurende 5-10 minuten, of totdat het stroperig is en is ingedikt tot 240 ml. Laat het glazuur minstens 10 tot 15 minuten afkoelen alvorens het over de ham te strijken. Er zijn talloze recepten voor glazuren te vinden, maar het bedenken van een eigen glazuur is eenvoudig. Experimenteer met ingredi\u00ebnten tot je de zoete, zure en hartige smaken in balans hebt gebracht. Streef naar ongeveer 240 tot 500 ml glazuur, en reserveer ongeveer een derde ervan voor op de eettafel. De basisingredi\u00ebnten van een glazuur zijn een zoetstof (zoals bruine suiker of melasse), een zuur (zoals azijn of sinaasappelsap), en kruiden of specerijen (zoals tijm of kruidnagel).",
  "target_text": "Bereid het glazuur voor nadat je de ham in de oven hebt gezet. Klop een glazuur van bruine suiker voor een eenvoudige klassieker. Sudder een sojasausglazuur voor een hartige smaak. Combineer bourbon, melasse en kruidnagel voor een diep, warm glazuur. Maak een esdoorn-sinaasappelglazuur voor een pittige, opvallende smaakcombinatie. Bedenk je eigen aangepaste glazuur."
}
```
```
{
  "text": "Je koplampen zijn je meest belangrijke levenslijn tijdens het rijden in het donker. Als ze niet in goede conditie zijn, vergroot je onnodig het risico op een ongeval. Houd je koplampen schoon door ze om de paar weken te wassen -- dit houdt de helderheid en scherpte van de lichtbundel hoog. Als een koplamp opbrandt, vervang deze dan zo snel mogelijk en rijd niet in het donker totdat de lamp hersteld is. Het is daarnaast overigens ook verboden om auto te rijden zonder goed werkende koplampen. Bovendien moet je voor de meeste zichtbaarheid je voorruit, ramen en spiegels zo helder en schoon maken als je kunt. Veeg deze belangrijke onderdelen van je auto niet schoon met je hand -- de natuurlijke olie van je huid kan vlekken op de spiegel achterlaten. Gebruik in plaats daarvan een krant of microvezeldoekje. De verstralerlichten van je auto kunnen je veiligheid significant vergroten wanneer je 's nachts rijdt, maar alleen als je ze correct gebruikt. Verstralers gebruik je bij het rijden door zeer donkere gebieden met weinig zicht, waar er niet veel verkeer is. In deze gevallen kunnen verstralers je gezichtsbereik veel breder en langer maken, dus gebruik ze waar nodig.  Zorg dat je verstralers uitschakelt wanneer je achter een andere auto rijdt of als er tegenliggers zijn. In deze gevallen kan het heldere licht van de verstralers andere automobilisten verblinden, waardoor het moeilijker voor hen wordt om veilig te rijden. Als je afslaat bij een bocht of over een heuveltop gaat en de zwakke gloed ziet van de koplampen van een andere auto, zet je verstralers dan voor alle zekerheid uit, zodat de andere bestuurder niet plotseling wordt verblind. Soms, zijn de koplampen van een auto schuiner naar de grond gericht dan nodig is, of zijn ze niet perfect symmetrisch uitgelijnd. De helderste koplampen in de wereld zijn niet nuttig als de weg voor je niet naar behoren verlichten. Dus als je merkt dat het moeilijk is om de weg voor je te zien tijdens het rijden in het donker, dan kun je overwegen om je koplampen opnieuw bij te stellen. Bij een professionele garage is deze procedure meestal heel snel en goedkoop geregeld. Het is ook mogelijk om zelf je koplampen bij te stellen. Aangezien iedere auto anders is, zal je de handleiding van je auto moeten raadplegen. Wees geduldig, want het kan even duren om koplampen perfect uitgelijnd te krijgen. In een perfecte wereld zouden andere bestuurders altijd hun verstralers dimmen als ze je zien, net zoals jij voor hen zou doen. Helaas willen automobilisten dit nog wel eens vergeten. Als een tegemoetkomende auto verstralers aan heeft staan, kijk daar dan niet naar, want het felle licht kan je tijdelijk verblinden. Kijk in plaats daarvan naar de rechterkant van je rijbaan (of in landen waar je aan de linkerkant van de weg rijdt, naar links), terwijl je vanuit je perifere zicht op gevaren let. Dit houdt je zo opmerkzaam mogelijk op de gevaren om je heen, met behoud van je zicht. Als een auto achter je verstralers aan heeft staan, probeer dan je achteruitkijkspiegel te verstellen om het licht uit je ogen te houden. Je kunt zelfs de spiegel zo instellen dat het licht weerkaatst naar de bestuurder van die auto, om hem te wijzen op zijn fout. Als je verwacht dat je veel 's nachts gaat rijden en onder mistige omstandigheden, dan kun je overwegen om te investeren in een set mistlampen. Vaak zijn deze lichten laag gemonteerd op de voorbumper om zoveel mogelijk wegdek te verlichten (mist is het dunst tot op een halve meter of zo boven het wegdek). Niet alle aftermarket lichten zijn even goed gemaakt, dus praat met je autodealer alvorens deze aanschaf te doen. Gebruik nooit je standaard verstralers in de mist. De reflecterende waterdeeltjes waaruit mist bestaat kunnen het heldere licht naar je terugkaatsen, waardoor je nog minder van de weg kunt zien dan zonder licht. De koplampen van andere auto's (en vooral verstralers) kunnen unieke uitdagingen vormen voor chauffeurs met een bril. Glazen kunnen soms tegemoetkomend licht op manieren reflecteren die een verduisterende schittering vormt voor de brildrager. Om dit te voorkomen kun je contactlenzen proberen of een brilglazen kopen met een anti-reflecterende coating, om deze effecten te minimaliseren. Als je een paar speciale brilglazen koopt, leg die dan in je auto zodat je ze altijd bij de hand hebt wanneer je de weg op gaat.",
  "target_text": "Houd je koplampen, spiegels en voorruit in topconditie. Gebruik je verstraler voor situaties met weinig licht. Pas eventueel je koplampen aan. Ga op de juiste manier om met verstralers van andere weggebruikers door naar de kant van de weg te kijken. Overweeg om lage mistlampen te installeren. Draag je een bril, gebruik dan een anti-reflecterende coating."
}
```
```
{
  "text": "Over het algemeen hebben raszuivere Cavaliers voorspelbare eigenschappen. Als je een raszuivere Cavalier koopt, kun je verwachten dat ze energieke, knuffelbare huisdieren zijn met een redelijk te onderhouden vacht. Genetisch bepaald hebben Cavaliers een neiging tot zorgeloosheid. Als je een rashond koopt, kun je een dergelijk karakter verwachten. Niet raszuivere Cavaliers kunnen sommige van de biologische eigenschappen overnemen van om het even welk ander ras waar ze mee gekruist zijn. Als ze zijn gekruist met een jachthond, dan kunnen ze een sterker jachtinstinct hebben, op dezelfde manier kunnen ze, als ze met een ras zijn gekruist met minder energie, zoals de shih tzu, dat energieke enthousiasme kwijtraken waar je in de eerste plaats op gevallen bent. Mensen hebben hun zinnen gezet op raszuivere Cavaliers. Dit betekent dat ze uit een beperkte genenpoel gefokt zijn. Om aangeduid te worden als raszuiver, wordt er op veel plaatsen inteelt gedaan met hun honden, en anderen hebben onwetend gefokt met een genenpoel die te klein is. Dit heeft heel realistische en bijzonder ongewenste consequenties. Raszuivere Cavaliers hebben een verhoogd risico op hartklachten, hernia en/of ernstige neurologische aandoeningen.   Hartziekte: in Engeland heeft 59% van de Cavaliers ouder dan 4 jaar een hartruis. Zijnde bijna tweederden van de populatie Cavaliers in Engeland is dit een uitzonderlijk statistisch gegeven.  Chiari misvorming en Syringomyelia: Kort gezegd betekent deze aandoening dat de schedel van de hond te klein is voor zijn hersenen. Dit veroorzaakt afschuwelijke zenuwpijn. Het diergeneeskundige leerboek \"Breed Predispositions to Disease in the Dogs and Cats\" bestempelt deze aandoening als \"veel voorkomend\" met tekenen die zich ontwikkelen tussen de leeftijd van 5 maanden tot 3 jaar.   Epilepsie: Honden kunnen op elk moment aanvallen ontwikkelen, maar tussen de 6 maanden en 6 jaar is de meest voorkomende periode.  Hernia:  Dit is een andere \"veelvoorkomende\" afwijking, vooral als Cavaliers ouder worden.  In de meeste gevallen zul je niet weten dat je Cavalier gevoelig is voor een hernia, tot je hem stijf ziet lopen of zijn hoofd met tegenzin naar beneden brengt naar zijn voerbak of waterbak.",
  "target_text": "Overweeg de voordelen als je kiest voor een raszuivere Cavalier. Stel vast wat de schaduwzijden zijn van het kopen van een rashond. Houd algemene gezondheidsproblemen van de Cavalier in gedachten."
}
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