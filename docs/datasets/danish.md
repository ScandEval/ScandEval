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

Here are a few examples from the training split:

```
{
  "text": "Jeg tror, det der var kampen. Goff virker lost",
  "label": "negative"
}
{
  "text": "@USER Véd ikke hvor gammel du er, men på min "Glad-liste" er Stig Møllers "Sikke´n dejlig dag det er i dag" - også Gnags "Lav sol over Århus", "Safari", "Slingrer ned af Vestergade", "Sensommer på Strøget" plus mange andre.",
  "label": "positive"
}
{
  "text": "Næste gang nogen kalder EU for "fredens projekt", kommer jeg til at eksplosiv-ørle!! #eudk #ep19dk #dkpol #daxitNU [LINK]",
  "label": "negative"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 12
- Prefix prompt:
  ```
  Følgende er tweets og deres sentiment, som kan være 'positiv', 'neutral' eller 'negativ'.
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

Here are a few examples from the training split:

```
{
  'text': 'Klik på linket i den e-mail vi har sendt dig',
  'tokens': array(['Klik', 'på', 'linket', 'i', 'den', 'e-mail', 'vi', 'har', 'sendt', 'dig'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
{
  'text': 'Space Invaders Testområde Artikler 2 Trivia Quiz Udrykninger Klanmedlemmer Server Information Round_n_Navigate Lan Party',
  'tokens': array(['Space', 'Invaders', 'Testområde', 'Artikler', '2', 'Trivia', 'Quiz', 'Udrykninger', 'Klanmedlemmer', 'Server', 'Information', 'Round_n_Navigate', 'Lan', 'Party'], dtype=object),
  'labels': array(['B-MISC', 'I-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
{
  'text': '"The Beast" kunne kun købes i sin tid, ved forudbestilling på selve destilleriet, hvilket min mand tog over og gjorde, og derfor kan vi nu udbyde 4 flasker til salg med flg',
  'tokens': array(['"', 'The', 'Beast', '"', 'kunne', 'kun', 'købes', 'i', 'sin', 'tid', ',', 'ved', 'forudbestilling', 'på', 'selve', 'destilleriet', ',', 'hvilket', 'min', 'mand', 'tog', 'over', 'og', 'gjorde', ',', 'og', 'derfor', 'kan', 'vi', 'nu', 'udbyde', '4', 'flasker', 'til', 'salg', 'med', 'flg'], dtype=object),
  'labels': array(['O', 'B-MISC', 'I-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Følgende er sætninger og JSON-ordbøger med de navngivne enheder, som forekommer i den givne sætning.
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

Here are a few examples from the training split:

```
{
  'text': 'Men han gjorde Viborg i en symaskine-spindende og venlig Citroën B 12 fra 1926.',
  'tokens': array(['Men', 'han', 'gjorde', 'Viborg', 'i', 'en', 'symaskine-spindende', 'og', 'venlig', 'Citroën', 'B', '12', 'fra', '1926', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'O', 'O'], dtype=object)
}
{
  'text': 'Jeg fik min Secumar-vest på i en rasende fart, mens skipper Tom Christiansen vendte skibet.',
  'tokens': array(['Jeg', 'fik', 'min', 'Secumar-vest', 'på', 'i', 'en', 'rasende', 'fart', ',', 'mens', 'skipper', 'Tom', 'Christiansen', 'vendte', 'skibet', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O'], dtype=object)},
{
  'text': 'Når nøglen i en "tilholderlås" er drejet til låst stilling, bør riglen/fallen være i sin yderste stilling med mindst lOmm\'s indgreb.',
  'tokens': array(['Når', 'nøglen', 'i', 'en', '"', 'tilholderlås', '"', 'er', 'drejet', 'til', 'låst', 'stilling', ',', 'bør', 'riglen/fallen', 'være', 'i', 'sin', 'yderste', 'stilling', 'med', 'mindst', "lOmm's", 'indgreb', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```


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

Here are a few examples from the training split:

```
{
  "text": "Et kort øjeblik frygtede han, at bedstefar Jonas var ved at dø for alvor, men anfaldet var allerede på vej væk og hånden blev slap.",
  "corruption_type": null,
  "label": "correct"
}
{
  "text": "Robert brugte sin frokostpause, som han plejede at bruge den.",
  "corruption_type": null,
  "label": "correct"
}
{
  "text": "Hvis der overhovedet var energi nogen tilbage i dig.",
  "corruption_type": "flip_neighbours",
  "label": "incorrect"
}
```

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

Here are a few examples from the training split:

```
{
  'id': '2010287291290882376',
  'question': 'Hvem er kendt som kongen af rock and rock?',
  'answers': {
    'answer_start': array([514]),
    'text': array(['Elvis Presley'], dtype=object)
  },
  'context': 'The King of Rock and Roll er Little Richards andet album for Reprise Records, et opfølgningsalbum, der indeholdt en original Little Richard-sang, gospelrocken "In the Name", og en ny sang, "Green Power", skrevet i samarbejde med produceren H. B. Barnum, som blev udgivet som single, samt versioner af numre af så forskellige kunstnere som Hank Williams, The Temptations, Martha and the Vandellas, Three Dog Night og The Rolling Stones. Titelnummeret, en spøgelsesagtig praleri, der bl.a. refererede til Tom Jones, Elvis Presley, Ike & Tina Turner, Sly and the Family Stone og Aretha Franklin, forstyrrede nogle fans, selv om albummets titelmelodi fik god airplay i New York - en jump blues i 1950\'ernes stil med en enestående Little Richard-sang! Men fans og kritikere var yderligere oprørte over, at albummet ikke indeholdt akustisk klaver, og at de fleste numre var dårligt mixet med et påtrængende pigegruppekor.',
  'answers_en': {
    'answer_start': array([474]),
    'text': array(['Elvis Presley'], dtype=object)
  },
  'context_en': 'The King of Rock and Roll is Little Richard\'s second album for Reprise Records, a follow-up album that contained one original Little Richard song, the gospel rock "In the Name" and a new song co-written by Producer H. B. Barnum, "Green Power", the single release; and versions of tracks by artists as diverse as Hank Williams, The Temptations, Martha and the Vandellas, Three Dog Night, and The Rolling Stones. The title track, a mock braggadocio that referenced Tom Jones, Elvis Presley, Ike & Tina Turner, Sly and the Family Stone, and Aretha Franklin, amongst others, upset some fans, although the album\'s title tune got good airplay in New York - a 1950s style jump blues, with an exceptional Little Richard shouting vocal! But fans and critics were further upset that the album did not feature acoustic piano and that most tracks were badly mixed, with an intrusive girl group chorus.',
  'title_en': 'The King of Rock and Roll'
}
{
  'id': '6235822902962606890',
  'question': 'Hvem ejer the boston red sox baseball hold?',
  'answers': {
    'answer_start': array([115]),
    'text': array(['John W. Henry'], dtype=object)
  },
  'context': 'John William Henry II (født den 13. september 1949) er en amerikansk forretningsmand og investor og grundlægger af John W. Henry & Company, et investeringsforvaltningsselskab. Han er hovedindehaver af The Boston Globe, Boston Red Sox og Liverpool Football Club og medejer af Roush Fenway Racing. I marts 2006 anslog Boston Magazine Henrys nettoformue til 1,1 mia. dollars, men bemærkede, at hans virksomhed for nylig havde haft problemer. I november 2012 meddelte firmaet, at det ville stoppe med at forvalte kundernes penge ved årets udgang, og Henry bekræftede, at de samlede aktiver under firmaets forvaltning var faldet fra 2,5 mia. dollar i 2006 til under 100 mio. dollar i slutningen af 2012. I juli 2017 anslog Forbes hans nettoformue til at være 2,6 milliarder dollars.',
  'answers_en': {
    'answer_start': array([107]),
    'text': array(['John W. Henry'], dtype=object)
  },
  'context_en': "John William Henry II (born September 13, 1949) is an American businessman and investor and the founder of John W. Henry & Company, an investment management firm. He is the principal owner of The Boston Globe, the Boston Red Sox and Liverpool Football Club and co-owner of Roush Fenway Racing. In March 2006, Boston Magazine estimated Henry's net worth at $1.1 billion but noted that his company had recently experienced difficulties. In November 2012, the company announced that it would stop managing clients' money by the end of the year, and Henry confirmed that total assets under the firm's management had fallen from $2.5 billion in 2006 to less than $100 million as of late 2012. As of July 2017, Forbes estimated his net worth to be $2.6 billion.",
  'title_en': 'John W. Henry'},
{
  'id': '6981008936931722768',
  'question': 'Der grundlagde den første baptistkirke i amerika?',
  'answers': {
    'answer_start': array([222]),
    'text': array(['Roger Williams'], dtype=object)
  },
  'context': "First Baptist Church in America\nDen første baptistkirke i Amerika er First Baptist Church of Providence, Rhode Island, også kendt som First Baptist Meetinghouse. Det er den ældste baptistkirke i USA, som blev grundlagt af Roger Williams i Providence, Rhode Island i 1638. Den nuværende kirkebygning blev opført i 1774-75 og holdt sine første møder i maj 1775. Den ligger på 75 North Main Street i Providence's College Hill-kvarter og er et nationalt historisk vartegn.",
  'answers_en': {
    'answer_start': array([217]),
    'text': array(['Roger Williams'], dtype=object)
  },
  'context_en': "The First Baptist Church in America is the First Baptist Church of Providence, Rhode Island, also known as the First Baptist Meetinghouse. It is the oldest Baptist church congregation in the United States, founded by Roger Williams in Providence, Rhode Island in 1638. The present church building was erected in 1774–75 and held its first meetings in May 1775. It is located at 75 North Main Street in Providence's College Hill neighborhood and is a National Historic Landmark.",
  'title_en': 'First Baptist Church in America'
}
```

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

Here are a few examples from the training split:

```
{
  "text": "Sprog som en havnearbejder\nSvarmuligheder:\na. Grimt sprog\nb. Fortryde\nc. Ikke reagere på noget bestemt\nd. Være presset af en opgave",
  "label": "a"
},
{
  "text": "Være i gode hænder\nSvarmuligheder:\na. Hård modstand\nb. Være i sikkerhed hos venlige mennesker\nc. Gå meget tidligt i seng\nd. Ødelægge en god stemning",
  "label": "b"
},
{
  "text": "Korthuset falder sammen\nSvarmuligheder:\na. Ødelægge noget\nb. Sige ja til noget uden at ville det\nc. Det går galt\nd. Se ned på noget",
  "label": "c"
}
```

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

Here are a few examples from the training split:

```
{
  "text": "Hvornår blev protestantismen indført i Danmark?\nSvarmuligheder:\na. 1100 -tallet\nb. 1300 -tallet\nc. 1500 -tallet",
  "label": "c",
  "test_type": "medborgerskabsprøven"
}
{
  "text": "Hvad hedder farvandet mellem København og Sverige?\nSvarmuligheder:\na. Øresund\nb. Kattegat\nc. Lillebælt",
  "label": "a",
  "test_type": "medborgerskabsprøven"
}
{
  "text": "Hvem bestemmer, hvem der skal danne regering efter et valg?\nSvarmuligheder:\na. Dronningen.\nb. Folketinget.\nc. Domstolene.",
  "label": "b",
  "test_type": "medborgerskabsprøven"
}
```

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

Here are a few examples from the training split:

```
{
  "text": "Ældre voksne yder generelt en fremragende præstation når deres _____ hukommelse testes.\nSvarmuligheder:\na. Episodisk\nb. Arbejds-\nc. Retrospektiv\nd. Semantisk",
  "label": "d",
  "category": "human_aging"
},
{
  "text": "Nipah er en zoonotisk paramyxovirus. Hvor stammer den fra?\nSvarmuligheder:\na. Den stammer fra grise.\nb. Den stammer fra flagermus.\nc. Den stammer fra mennesker.\nd. Den stammer fra heste.",
  "label": "c",
  "category": "virology"
},
{
  "text": "Et firma er interesseret i at sammenligne den gennemsnitlige salgsindtægt pr. sælger på to forskellige steder. Chefen tager en tilfældig stikprøve på 10 sælgere fra hver placering uafhængigt af hinanden og registrerer salgsindtægterne, som hver person har genereret i de sidste fire uger. Han beslutter sig for at bruge en t-test til at sammenligne den gennemsnitlige salgsindtægt på de to placeringer. Hvilket af følgende antagelser er nødvendigt for gyldigheden af t-testen?\nSvarmuligheder:\na. De populationsstandardafvigelser på begge placeringer er ens.\nb. De populationsstandardafvigelser på begge placeringer er ikke ens.\nc. De populationsstandardafvigelser på begge placeringer er kendte.\nd. Populationerne af salgsregistreringer på hver placering er normalt fordelt.",
  "label": "d",
  "category": "high_school_statistics"
}
```


### Unofficial: ARC-da

Coming soon!

Here are a few examples from the training split:

```
{
  "text": "Hvilket begreb bruges til at beskrive en fysisk egenskab af et mineral?\nSvarmuligheder:\na. organisk\nb. fast\nc. gasformig\nd. fossilholdigt",
  "label": "b"
},
{
  "text": "Hvad forårsager DEN STØRSTE forandring i en græsmark over tid?\nSvarmuligheder:\na. Dagens tidspunkt\nb. Mængde af årlig nedbør\nc. Antal fugle, der bygger rede\nd. Årlige dyr bevægelser",
  "label": "b"
},
{
  "text": "Nogle elever brugte en varmeplade til at opvarme 1 L vand fra 20°C til kogepunktet for vand. Eleverne registrerede temperaturen på vandet hvert minut, indtil det begyndte at koge. Hvad er den mest hensigtsmæssige måde at repræsentere data på?\nSvarmuligheder:\na. en søjlediagram med temperatur på y-aksen og tid på x-aksen\nb. en søjlediagram med tid på y-aksen og temperatur på x-aksen\nc. en linjediagram med temperatur på y-aksen og tid på x-aksen\nd. en linjediagram med tid på y-aksen og temperatur på x-aksen",
  "label": "c"
}
```


## Common-sense Reasoning

### HellaSwag-da

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
  "text": "[header] Sådan rengøres ruskindshandsker [title] Tjek dine handskers plejemærkater. [step] Før du begynder at rengøre dine ruskindshandsker, skal du tjekke plejemærkaterne. Der er mange forskellige typer af ruskind og ruskindafslutninger, og hver type har forskellige plejebehov.\nSvarmuligheder:\na. Lær om hvert enkelt ruskindprodukt og dets plejeskilte for at sikre, at du bruger den rette rensemiddel. [substeps] Plejeskiltene for handsker skal være nummer 1 eller den seneste mængde og niveau for brug.\nb. Mærkaterne skal fortælle dig, hvordan du bedst kan rengøre din type ruskind. [substeps] For eksempel bør du ikke bruge vand på meget fint ruskind, da det sandsynligvis vil misfarve det.\nc. Du bør tjekke disse plejemærkater for at se, om din vare (eller produkt) specifikt skal renses eller presses, inden du tager vare på den. [substeps] Du kan tjekke dit plejemærkat ved at trække det ud af en papkasse eller vaske det i din vaskemaskine.\nd. Genstande som ruskindspander, -panner og -kasser vil have forskellige plejeskilte, som du skal følge. [substeps] Bed forhandleren om at give dig et opkald eller sende dig dit grundlæggende rengøringskit.",
  "label": "b",
  "activity_label": "Home and Garden"
}
{
  "text": "En kværnemaskine vises på en terrasse. en mand\nSvarmuligheder:\na. taler, mens han viser maskindelene.\nb. begynder at sprøjte jorden på terrassen med kværnen.\nc. arbejder på en boliggræsplæne.\nd. vises med at skrabe sne af landeren, efterfulgt af at fjerne tøj.",
  "label": "a",
  "activity_label": "Cutting the grass"
}
{
  "text": "En stor gruppe mennesker ses spille en fodboldkamp på en sandet mark, mens mange ser på fra sidelinjen. kameraet\nSvarmuligheder:\na. fortsætter med at følge gruppen og viser mange, der kaster bolden til hinanden, mens spillet bliver spillet for tilskuere.\nb. fanger kampen fra alle vinkler og ser på, mens en scorer et mål i målet.\nc. følger de modsatte hold og fører ind i dem, der sparkes og trækker i hinandens arme.\nd. viser nærbilleder af spillere samt ketsjere og bolden, der bliver ramt.",
  "label": "b",
  "activity_label": "Beach soccer"
}
```

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
  d. {option_d}

  Besvar ovenstående spørgsmål ved at svare med 'a', 'b', 'c' eller 'd'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset hellaswag-da
```


## Summarization

### Nordjylland News

This dataset is based on news articles from the Danish news site [TV2
Nord](https://www.tv2nord.dk/), where the summaries are taken as the introductory
paragraphs of the articles.

The original full dataset consists of 75,200 samples. We use an 1,024 / 256 / 2,048
split for training, validation and testing, respectively (so 3,328 samples used in
total).

Here are a few examples from the training split:

```
{
  "text": "Lørdag er en dag med masser af vind og blæst. Men for 35 år siden, var blæsevejret endnu voldsommere. I disse dage er det nemlig 35 år siden Danmark blev ramt af en af de kraftigste storme gennem tiderne. Stormen anrettede store skader og væltede hele skove, mens den gav en stormflod, der med rekordhøj vandstand sørgede for omfattende oversvømmelser. Eluf Harring, som dengang var otte år og boede på landet, husker stadig orkanen, som var det i går. - Jeg kan huske, at jeg vågnede hele tiden. Der var ingen af os, som kunne sove. Også havde vi ingen strøm i halvandet døgn, siger Eluf Harring. Også Mette Henriksen fra Aars, der dengang var 23 år, husker tydeligt stormen. Hun var på sin fødselsdag på vej hjem fra Skørping i sin bil, da et træ væltede og spærrede vejen i skoven ved Torstedlund. Herefter ville Mette Henriksen vende om og køre tilbage, men også her var flere træer væltet. Hun var fanget og måtte vente på, der kom nogen for at fjerne træerne. - Jeg sad fanget i min bil i halvanden time, og jeg var så bange for, der ville vælte et træ lige nede i mig, siger Mette Henriksen. Stormen udmærkede sig ved at ramme hele Danmark - og især i den nordlige halvdel af landet. Det var også her man oplevede det kraftigste vindstød overhovedet registeret under stormen. I Thisted blev der om eftermiddagen den 24. november målt vindstød på 43 m/s. Mange steder i landet betød stormen blandt andet, at hele skovområder væltede. Særdeles gik det hårdt ud over Rold Skov i Himmerland. Stormen kostede to mennesker livet. En fisker omkom, da en kutter sank i Nordsøen, mens en skovarbejder omkom, da han blev ramt af et af de mange væltede træer, skriver TV2.dk.",
  "target_text": "Den 24. november i 1981 blev Danmark ramt af en historisk voldsom storm. Eluf Harring har sendt arkivbilleder til TV2 Nord fra dengang.",
  "text_len": 1671,
  "summary_len": 135
}
{
  "text": "Der blev sendt flere vogne til Jomfru Ane Gade lørdag eftermiddag, efter der blev meldt om \"anspændt stemning\" i mellem fangrupperinger i anledning af dagens fodboldopgør på Aalborg Portland Park. AaB tager 17.30 imod Århus, hvor de skal forsøge at hive sæsonens første sejr hjem. Selvom der ikke blev rapporteret om slåskampe, valgte Nordjyllands Politi alligevel at sende betjente til Gaden for at holde de to fangrupper adskilt og undgå uroligheder. De to grupperinger befandt sig på hver deres beværtning, og politiet dannede en kæde, så de to fangrupper ikke kom i kontakt med hinanden, da de forlod gaden. Efterfølgende blev de to grupperinger sendt i hver sin retning mod stadion, hvor alt foregik i ro og orden.",
  "target_text": "Nordjyllands Politi holder et godt øje med fodboldfansene, efter der blev rapporteret om anspændt stemning mellem de grupperinger.",
  "text_len": 719,
  "summary_len": 130
}
{
  "text": "- Uanset hvilket niveau du spiller fodbold på, så er det jo også for kammeratskabets skyld. Sådan lyder det fra cheftræner hos AaB, Jacob Friis. - Seks uger er lang tid for en fodboldspiller, det er længere tid end en normal sommerferie og en normal vinterpause, så det var pludselig ude, hvor vi ikke kunne bunde, men heldigvis kan vi nu samles igen, siger Jacob Friis, til TV2 Nord. Siden d. 8. marts har Superligaen ligget stille. Spillertrupperne har i en periode været sendt hjem, og ellers har den stået på træning i små grupper. Men nu har Divisionsforeningen meldt ud, at Superligaen genoptages i slutningen af maj. Det sociale ved at komme i gang igen betyder meget for spillertruppen. - Nu får vi lov til at gå til den og spille noget rigtig fodbold igen. Det har vi savnet rigtig meget. Det er fedt at se hele truppen igen og at kunne være sammen en lille smule socialt, inden vi sætter os i bilerne og køre hjem igen, siger AaB-spiller, Jakob Ahlmann, til TV2 Nord. Hårdt prøvet økonomi I Hobro IK er spillertruppen også tilbage i fuld vigør. Nedlukningen af Danmark var i sidste ende ved at komme til at koste klubben rigtig dyrt. - Vi var på vej hen mod en konkurs, men man ved det selvfølgelig aldrig, før man sidder i skifteretten, for der kan jo komme en rig mand med en pose penge, men vi var vildt pressede, hvis vi ikke var kommet i gang igen, siger bestyrelsesformand i Hobro IK, Lars Kühnel, til TV2 Nord. Hobro IK skal en tur til Randers d. 1. juni, mens AaB dagen før gæster Esbjerg. Hobro IK indtager i øjeblikket 12.-pladsen og derfor venter en spændende periode, hvor der skal kæmpes for livet i Superligaen, hvor der er hele tre direkte nedrykkere i denne sæson. Derfor er klubben også særligt glad for, at man selv får indflydelse på sin skæbne. - Det skal ikke afgøres ved et skrivebord, det skal afgøres på en fodboldbane. Det har vi altid sagt, og det bliver vi ved med at sige, så at vi kommer igang igen, det betyder alt for fodbolden, siger Lars Kühnel. Sidste runde af grundspillet bliver spillet d. 7. juni og herefter venter enten et medaljeslutspil eller en kamp for overlevelse i en af to nedrykningspuljer.",
  "target_text": "Både spillerne i Hobro IK og i AaB er mere end klar til igen at komme i gang med at spille Superliga.",
  "text_len": 2147,
  "summary_len": 101
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 1
- Prefix prompt:
  ```
  Følgende er nyhedsartikler med tilhørende resuméer.
  ```
- Base prompt template:
  ```
  Nyhedsartikel: {text}
  Resumé: {target_text}
  ```
- Instruction-tuned prompt template:
  ```
  Nyhedsartikel: {text}

  Skriv et resumé af ovenstående artikel.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset nordjylland-news
```
