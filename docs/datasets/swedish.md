# 🇸🇪 Swedish

This is an overview of all the datasets used in the Swedish part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### SweReC

This dataset was published [in this B.Sc.
thesis](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1105494&dswid=3392) and
is a manually annotated dataset of Swedish reviews from both Trustpilot and Reco.se.

The original dataset contains 10,757 reviews. We use a split of 1,024 / 256 / 2,048
samples for training, validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
  "text": "Jättebra och rekommenderas till alla",
  "label": "positive"
}
```
```json
{
  "text": "Lugnt och trevlig stämning, inte för bullrigt. god mat, lite mer variation hade önskats på de varma rätterna. trevlig personal, dock missade de att ta dryckesbeställningar från oss vilket var ett litet minus. överlag trevlig ställe.",
  "label": "neutral"
}
```
```json
{
  "text": "Extremt dålig mottagning - både gsm och 3g? samtalen bryts hela tiden och så tar dom betalt för en ny uppkopplingsavgift varje gång.",
  "label": "negative"
}
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

This dataset, also known as the Stockholm-Umeå Corpus 3.0, was published
[here](https://doi.org/10.23695%2Fwy84-ar30) and is a manually NER-annotated dataset,
based on Swedish texts from the 1990s. The dataset does not follow the CONLL format, so
we convert it into that format using the following mapping:

- `animal` ➡️ `MISC`
- `event` ➡️ `MISC`
- `inst` ➡️ `ORG`
- `myth` ➡️ `MISC`
- `other` ➡️ `MISC`
- `person` ➡️ `PER`
- `place` ➡️ `LOC`
- `product` ➡️ `MISC`
- `work` ➡️ `MISC`

The dataset consists of 74,245 samples, which we split into 1,024 / 256 / 2,048 samples
for training, validation, and testing, respectively.

Here are a few examples from the training split:

```json
{
  "tokens": array(['Det', 'låter', 'som', 'en', 'västanfläkt', 'jämfört', 'med', 'den', 'i', 'filmen', 'förkättrade', 'biljätten', 'General', 'Motors', ',', 'som', 'friställt', '35000', 'jobbare', 'i', 'staden', 'Flint', ',', 'Michigan', '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['En', 'liknande', 'kunskapsteoretisk', 'grundfråga', ',', 'fast', 'i', 'mer', 'modernt', 'sofistikerad', 'form', ',', 'når', 'oss', 'nu', 'från', 'Paris', ':'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O'], dtype=object)
}
```
```json
{
  "tokens": array(['-', 'Dessvärre', ',', 'sa', 'man', ',', 'vi', 'har', 'ingen', 'Björn', 'Eriksson', 'på', 'passagerarlistan', '.'], dtype=object),
  "labels": array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O'], dtype=object)
}
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

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [Swedish Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_Swedish-Talbanken) by assuming
that the documents in the treebank are correct, and corrupting the samples to create
grammatically incorrect samples. The corruptions were done by either removing a word
from a sentence, or by swapping two neighbouring words in a sentence. To ensure that
this does indeed break the grammaticality of the sentence, a set of rules were used on
the part-of-speech tags of the words in the sentence.

The original full dataset consists of 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "U-länderna måste ta en genväg för att komma i fatt.",
  "label": "correct"
}
```
```json
{
  "text": "Undra att vi blev lite undandragna.",
  "label": "incorrect"
}
```
```json
{
  "text": "Det är också att viktigt ha tillräckligt korta dubbar.",
  "label": "incorrect"
}
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

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the Swedish part of the [MKQA
dataset](https://aclanthology.org/2021.tacl-1.82/). The MKQA dataset is based on the
English [Natural Questions dataset](https://aclanthology.org/Q19-1026/), based on search
queries from the Google search engine. The questions and answers were manually
translated to Swedish (and other languages) as part of MKQA, and the contexts were in
ScandiQA-sv machine translated using the [DeepL translation
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

```json
{
  "context": "I Freedom Cry får spelaren ta rollen som Adéwalé, en frigiven slav från Trinidad som blev Edward Kenways kvartermästare och senare medlem av Assassin Order. Berättelseläget utspelar sig 15 år efter händelserna i Assassin's Creed IV: Black Flag där Adéwalé har blivit en tränad lönnmördare och finner sig själv skeppsbruten i Saint-Domingue, där han ställs öga mot öga med något av det mest brutala slaveriet i Västindien. DLC:n är skriven av Jill Murray, som skrev Liberation och Aveline-innehållet för Black Flag. I februari 2014 meddelades att Freedom Cry skulle släppas som en fristående titel till PlayStation 4 och PlayStation 3 den 18 februari 2014 för Nordamerika och den 19 februari 2014 för Europa. Det släpptes för PC den 25 februari 2014.",
  "question": "När släpptes assassin's creed freedom cry?",
  "answers": {
    "answer_start": array([637]),
    "text": array(['18 februari 2014'], dtype=object)
  }
}
```
```json
{
  "context": 'Political history of the United Kingdom (1945–present)\nÅr 1950 orsakade Koreakriget ett nytt tungt tryck på statskassan för militära utgifter. Detta orsakade en bitter splittring inom Labourpartiet.  De konservativa gjorde åtstramningspolitiken till en viktig fråga i parlamentsvalet 1950. Labour förlorade det mesta av sin stora majoritet. Svängningen var 3,6 % mot dem och de förlorade 78 platser, vilket gav Attlee en knapp majoritet i parlamentet. Ett år senare förlorade Labour dock parlamentsvalet 1951 trots att det fick fler röster än i valet 1945, och faktiskt fler röster än det konservativa partiet.',
  "question": 'Hur många år har det varit sen 1940?',
  "answers": {
    "answer_start": array([388]),
    "text": array(['78'], dtype=object)
  }
}
```
```json
{
  "context": 'Data link layer\nOSI-modellen\nper skikt\n\n\n\n\n7.  Applikationslager[visa]\n\n\nNNTP\nSIP\nSSI\nDNS\nFTP\nGopher\nHTTP\nNFS\nNTP\nSMPP\nSMTP\nSNMP\nTelnet\nDHCP\nNetconf\nmer....\n\n\n\n\n\n\n\n\n6.  Presentationslager[visa]\n\n\nMIME\nXDR\n\n\n\n\n\n\n\n\n5.  Sessionsskikt[visa]\n\n\nNamngiven pipe\nNetBIOS\nSAP\nPPTP\nRTP\nSOCKS\nSPDY\n\n\n\n\n\n\n\n\n4.  Transportlager[visa]\n\n\nTCP\nUDP\nSCTP\nDCCP\nSPX\n\n\n\n\n\n\n\n\n3.  Nätverksskikt[visa]\n\n\nIP\n\nIPv4\nIPv6\n\n\nICMP\nIPsec\nIGMP\nIPX\nAppleTalk\nX.25 PLP\n\n\n\n\n\n\n\n\n2.  Datalänkskiktet[visa]\n\n\nATM\nARP\nIS-IS\nSDLC\nHDLC\nCSLIP\nSLIP\nGFP\nPLIP\nIEEE 802.2\nLLC\nMAC\nL2TP\nIEEE 802.3\nFrame Relay\nITU-T G.hn DLL\nPPP\nX.25 LAPB\nQ.921 LAPD\nQ.922 LAPF\n\n\n\n\n\n\n\n\n1.  Fysiskt lager[visa]\n\n\nEIA/TIA-232\nEIA/TIA-449\nITU-T V-serien\nI.430\nI.431\nPDH\nSONET/SDH\nPON\nOTN\nDSL\nIEEE 802.3\nIEEE 802.11\nIEEE 802.15\nIEEE 1394\nITU-T G.hn PHY\nUSB\nBluetooth\nRS-232\nRS-449\n\n\n\n\n\n\n\n\n\nv\nt\ne',
  "question": 'Vilket lager av osi-modellen är uppdelad i två delskikt?',
  "answers": {
    "answer_start": array([0]),
    "text": array(['Data link layer'], dtype=object)
  }
}
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


### Belebele

This dataset was published in [this paper](https://aclanthology.org/2024.acl-long.44/) and is a large-scale multilingual reading comprehension dataset covering 122 languages. The questions are generated from Wikipedia articles and are designed to test various aspects of reading comprehension, including factual understanding, inference, and numerical reasoning.

The dataset provides training, validation, and test splits with human-verified question-answer pairs. The questions are generated to be answerable from the given context and cover diverse topics from Wikipedia articles.

When evaluating generative models, we use the following setup (see the [methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Nedan följer texter med tillhörande frågor och svar.
  ```
- Base prompt template:
  ```
  Text: {text}
  Fråga: {question}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Text: {text}

  Besvara följande fråga om texten ovan.

  Fråga: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset belebele-sv
```


## Knowledge

### MMLU-sv

This dataset is a machine translated version of the English [MMLU
dataset](https://openreview.net/forum?id=d7KBjmI3GmQ) and features questions within 57
different topics, such as elementary mathematics, US history and law. The translation to
Swedish was done by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 269 / 1,410 / 13,200 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 2,048 split for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
new and there can thus be some overlap between the original validation and test sets and
our validation and test sets.

Here are a few examples from the training split:

```json
{
  "text": "Varför är tidpunkten för monumental byggnation vid Ceibal signifikant?\nSvarsalternativ:\na. Det motsäger hypotesen att den monumental byggnationen av Maya i huvudsak inspirerades av Olmekerna.\nb. Det bekräftar att invånarna i Ceibal inspirerades av Olmekerna för att bygga stora plattformar.\nc. Det motsäger hypotesen att utvecklingen av monumental byggnation bland Maya var en intern process.\nd. Det bekräftar att Olmekerna, som byggde de flesta Maya-monumenten, inspirerades av egyptierna.",
  "label": "a"
}
```
```json
{
  "text": "Vilken populationsstatistik visar födelsetalet vid vilket en befolkning precis får tillräckligt med födslar för att ersätta föräldrarna och kompensera för tidiga dödsfall?\nSvarsalternativ:\na. Rå födelsetal\nb. Ersättningstal\nc. Dödlighetstal\nd. Total fertilitetstal",
  "label": "b"
}
```
```json
{
  "text": "En subenhet av DNA och protein som består av 134-baspar långa sträckor av DNA som omger en proteinoktomer kallas (a)\nSvarsalternativ:\na. histon\nb. kromatin\nc. nukleosom\nd. solenoid",
  "label": "c"
}
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

  Besvara följande fråga med 'a', 'b', 'c' eller 'd', och inget annat.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset mmlu-sv
```


### Unofficial: ARC-sv

This dataset is a machine translated version of the English [ARC
dataset](https://doi.org/10.48550/arXiv.1803.05457) and features US grade-school science
questions. The translation to Swedish was done by the University of Oregon as part of
[this paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 1,110 / 297 / 1,170 samples for training,
validation and testing, respectively. We use a 1,024 / 256 / 1,024 split for training,
validation and testing, respectively (so 2,304 samples used in total). All new splits
are subsets of the original splits.

Here are a few examples from the training split:

```json
{
  "text": "En typ av fågel i Afrika äter blodsugande insekter från stora däggdjur. Vilket ord beskriver bäst relationen mellan fågeln och däggdjuren?\nSvarsalternativ:\na. mutualism\nb. parasitism\nc. neutralism\nd. kommensalism",
  "label": "a"
}
```
```json
{
  "text": "Mr. Pratt gör en vetenskaplig demonstration. Han blåser upp en ballong, placerar den i en frys och tar sedan ut den efter 10 minuter. Vilket alternativ beskriver bäst ballongens volym när den är i frysen och efter att den har tagits ut och åter tillåtits att värmas upp?\nSvarsalternativ:\na. expanderar i frysen och kontraherar sedan när den blir varmare igen\nb. kontraherar i frysen och expanderar sedan när den blir varmare igen\nc. expanderar i frysen och håller sedan den volymen när den värms upp\nd. kontraherar i frysen och håller sedan den volymen när den värms upp",
  "label": "b"
}
```
```json
{
  "text": "En elev tillsätter vatten och rengöringsmedel till en kopp med jord. Blandningen skakas och tillåts sätta sig. Eleven observerar att silt-partiklar förblir uppsuspenderade långt efter att de andra partiklarna bildar lager på botten av behållaren. Den mest troliga förklaringen är att silt-partiklarna är\nSvarsalternativ:\na. organiska.\nb. upplösta.\nc. mindre tätt packade.\nd. rör sig snabbare.",
  "label": "c"
}
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
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Besvara följande fråga med 'a', 'b', 'c' eller 'd', och inget annat.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset arc-sv
```


## Common-sense Reasoning

### HellaSwag-sv

This dataset is a machine translated version of the English [HellaSwag
dataset](https://aclanthology.org/P19-1472/). The original dataset was based on both
video descriptions from ActivityNet as well as how-to articles from WikiHow. The dataset
was translated by the University of Oregon as part of [this
paper](https://aclanthology.org/2023.emnlp-demo.28/), using GPT-3.5-turbo.

The original full dataset consists of 9,310 samples. We use a 1,024 / 256 / 2,048 split
for training, validation and testing, respectively (so 3,328 samples used in total).

Here are a few examples from the training split:

```json
{
  "text": "[header] Hur man hittar de perfekta brudtärneklänningarna [title] Internet är en underbar resurs för att hitta brudtärneklänningar. [step] Vi rekommenderar också att bläddra genom populära bröllopstidningar, såsom brudens och moderna brudtärnets tidningar. Rekommenderat är att bruden går och handlar med en eller två av sina brudtärnor och ser vilka stilar de gillar.\nSvarsalternativ:\na. När du har begränsat urvalet kan du sedan få input från dina andra brudtärnor om du önskar det. [title] Vilka är de senaste trenderna i brudtärneklänningar? [title] A-linje klänningar som ser bra ut på alla olika kroppsformer och storlekar är mycket populära.\nb. Tyvärr kan du inte handla lika ofta som om du letade efter matchade brudtärnor. [title] När du väljer din brud, välj tre olika stilar: [step] Klipp längd, klipp tjocklek och från de flesta \"för-skjutna\" stilarna till de grundläggande.\nc. Medan varje brud är annorlunda, alla är både olika och har olika smaker. [title] Se om bruden har en favoritlook för sin bröllopsklänning.\nd. [title] Börja söka efter idéer eller allmänna åsikter om särskilda bröllopsklänningar. [step] Försök att inte bli för stel och sök bara efter några klänningar som du tror kan fungera bra tillsammans.",
  "label": "a"
}
```
```json
{
  "text": "[header] Hur man gör en pedikyr [title] Ta bort all befintlig färg med nagellacksborttagare. [step] Täck toppen på din nagellacksborttagare med en bomullstuss, vänd snabbt upp och ner den och omedelbart upp och ner igen för att applicera lite av produkten. Gnugga sedan nagellacksborttagaren över dina tånaglar för att ta bort färgen.\nSvarsalternativ:\na. [title] Låt dina tånaglar blötläggas i vatten i 10 till 20 minuter. [step] Vatten kan göra dina naglar vitare genom att lösa upp andra föreningar, särskilt syror.\nb. [substeps] Flytta bomullstussen i små, cirkulära rörelser om du har svårt att ta bort färgen. [title] Fyll en fotspa eller en balja med varmt vatten.\nc. [substeps] Om du inte har nagellacksborttagare kan du överväga att använda den vita nagellacksborttagaren från föregående steg för en enklare applikation. [title] Täck dina händer med bandage eller tejp med canvas-lining.\nd. [title] Använd aceton på dina tånaglar. [step] Aceton kan verkligen hjälpa till att ta bort gammalt nagellack från dina naglar.",
  "label": "b"
}
```
```json
{
  "text": "Han fortsätter att klippa gräset. Kameran fokuserar på det rinnande vattnet igen. Den går tillbaka till mannen som klipper gräset. sedan\nSvarsalternativ:\na. den går tillbaka till filmen av mannen som klipper jord.\nb. återvänder till honom och dem som pratar igen.\nc. växlar tillbaka till det rinnande vattnet.\nd. mörk himmel igen.",
  "label": "c"
}
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
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}
  Svar: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Fråga: {text}
  Svarsalternativ:
  a. {option_a}
  b. {option_b}
  c. {option_c}
  d. {option_d}

  Besvara följande fråga med 'a', 'b', 'c' eller 'd', och inget annat.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset swedn
```
