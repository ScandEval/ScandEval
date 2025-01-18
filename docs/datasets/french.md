# 🇫🇷 French

This is an overview of all the datasets used in the French part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

### Allocine

This dataset was published in [this Github
repository](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert) and
features reviews from the French movie review website Allocine. The reviews range from
0.5 to 5 (inclusive), with steps of 0.5. The negative samples are reviews with a rating
of at most 2, and the positive ones are reviews with a rating of at least 4. The reviews
in between were discarded.

The original full dataset consists of 160,000 / 20,000 / 20,000 samples for training,
validation, and testing, respectively. We use 1,024 / 256 / 2,048 samples for training,
validation, and testing, respectively. All our splits are subsets of the original ones.

Here are a few examples from the training split:

```json
{
  "text": "Ce 7ème volet ne mérite pas de notre part une grande attention, au vu du précédent New Police Story. À la limite du huis clos, Jackie évolue dans une boîte de nuit, sorte de piège du méchant cherchant à se venger, ou du moins à découvrir la vérité sur la mort de sa sœur. Notre cascadeur acteur ne bénéficie pas d'un décors à la hauteur de son potentiel acrobatique et le film d'un scénario à la hauteur d'une production, et cette production d'une large distribution, ce qui explique son arrivée direct tout étagère.",
  "label": "negative"
}
```
```json
{
  "text": "Meme pour ceux qui n'aime pas les Chevaliers du Fiel allez voir. 1 il est meilleur que le 1 et cela est rare de voir une suite qui est meilleur que le 1. Des scènes qui peuvent faire rire les petit et les grands. On ne s'ennuie pas. Super film allez le voir. L'interpretation des acteurs sont super. Bonne journée",
  "label": "positive"
}
```
```json
{
  "text": "Une ambiance envoûtante, un récit où se mélangent sorcellerie, croyances indiennes, enquête policière sur fond de trafic de drogue, tout est conforme au livre de Tony Hillerman, même si ce dernier a \"renié\" le film. Personnellement j'adore. Hélas introuvable en France et diffusé seulement sur canal , il y a ..... un certain temps.",
  "label": "positive"
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Voici des textes et leur sentiment, qui peut être 'positif' ou 'négatif'.
  ```
- Base prompt template:
  ```
  Texte: {text}
  Sentiment: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Texte: {text}

  Classez le sentiment dans le texte. Répondez par ‘positif' ou ‘négatif'.
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset allocine
```


## Named Entity Recognition

### ELTeC

This dataset was published in [this paper](https://doi.org/10.3828/mlo.v0i0.364) and
consists of sentences from 100 novels in French during the period 1840-1920, all of
which are in the public domain. These novels were automatically labelled with named
entities using Stanza-NER, and then manually corrected.

The original dataset consists of 100 samples, one for each novel. We split the novels
into sentences using the French NLTK sentence splitter, resulting in 4,815 samples. We
use 1,024 / 256 / 2,048 samples for training, validation, and testing, respectively.

We have furthermore converted the OntoNotes 5.0 labelling scheme to the CoNLL-2003
labelling scheme, which is more common in the NER literature. The mapping is as follows:

- `PERS` ➡️ `PER`
- `LOC` ➡️ `LOC`
- `ORG` ➡️ `ORG`
- `OTHER` ➡️ `MISC`
- `DEMO` ➡️ `O`
- `ROLE` ➡️ `O`
- `EVENT` ➡️ `O`

Here are a few examples from the training split:

```json
{
  'tokens': array(['Jamais', 'ils', 'ne', 'firent', 'de', 'provisions', ',', 'excepté', 'quelques', 'bottes', "d'ail", 'ou', "d'oignons", 'qui', 'ne', 'craignaient', 'rien', 'et', 'ne', 'coûtaient', 'pas', "grand'chose", ';', 'le', 'peu', 'de', 'bois', "qu'ils", 'consommaient', 'en', 'hiver', ',', 'la', 'Sauviat', "l'achetait", 'aux', 'fagotteurs', 'qui', 'passaient', ',', 'et', 'au', 'jour', 'le', 'jour', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  'tokens': array(['I', 'Il', 'y', 'avait', 'plus', 'de', 'soixante', 'ans', 'que', "l'empereur", 'Napoléon', ',', 'pressé', "d'argent", ',', 'avait', 'vendu', 'les', 'provinces', 'de', 'la', 'Louisiane', 'à', 'la', 'République', 'des', 'États-Unis', ';', 'mais', ',', 'en', 'dépit', 'de', "l'infiltration", 'yankee', ',', 'les', 'traditions', 'des', 'créoles', 'français', 'se', 'perpétuaient', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```
```json
{
  'tokens': array(['Les', 'fenêtres', 'de', 'la', 'vieille', 'demeure', 'royale', ',', 'ordinairement', 'si', 'sombres', ',', 'étaient', 'ardemment', 'éclairées', ';', 'les', 'places', 'et', 'les', 'rues', 'attenantes', ',', 'habituellement', 'si', 'solitaires', ',', 'dès', 'que', 'neuf', 'heures', 'sonnaient', 'à', "Saint-Germain-l'Auxerrois", ',', 'étaient', ',', "quoiqu'il", 'fût', 'minuit', ',', 'encombrées', 'de', 'populaire', '.'], dtype=object),
  'labels': array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], dtype=object)
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 8
- Prefix prompt:
  ```
  Vous trouverez ci-dessous des phrases et des dictionnaires JSON avec les entités nommées qui apparaissent dans la phrase donnée.
  ```
- Base prompt template:
  ```
  Sentence: {text}
  Entités nommées: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Sentence: {text}

  Identifiez les entités nommées dans la phrase. Vous devez produire ceci sous forme de dictionnaire JSON avec les clés 'personne', 'lieu', 'organisation' et 'divers'. Les valeurs doivent être des listes des entités nommées de ce type, exactement comme elles apparaissent dans la phrase.
  ```

- Label mapping:
    - `B-PER` ➡️ `personne`
    - `I-PER` ➡️ `personne`
    - `B-LOC` ➡️ `lieu`
    - `I-LOC` ➡️ `lieu`
    - `B-ORG` ➡️ `organisation`
    - `I-ORG` ➡️ `organisation`
    - `B-MISC` ➡️ `divers`
    - `I-MISC` ➡️ `divers`

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset eltec
```


## Linguistic Acceptability

### ScaLA-fr

This dataset was published in [this paper](https://aclanthology.org/2023.nodalida-1.20/)
and was automatically created from the [French Universal Dependencies
treebank](https://github.com/UniversalDependencies/UD_French-GSD/tree/master) by
assuming that the documents in the treebank are correct, and corrupting the samples to
create grammatically incorrect samples. The corruptions were done by either removing a
word from a sentence, or by swapping two neighbouring words in a sentence. To ensure
that this does indeed break the grammaticality of the sentence, a set of rules were used
on the part-of-speech tags of the words in the sentence.

The original full dataset consists of 1,024 / 256 / 2,048 samples for training,
validation and testing, respectively (so 3,328 samples used in total). These splits are
used as-is in the framework.

Here are a few examples from the training split:

```json
{
  "text": "Le dessert est une part minuscule de gâteau.",
  "label": "correct"
}
```
```json
{
  "text": "Le trafic international sera normal vendredi sur Eurostar, Thalys, et sur les trains à grande vitesse à destination de l', a indiqué la SNCF dans un communiqué.",
  "label": "incorrect"
}
```
```json
{
  "text": "Certains craignent qu' un avantage compétitif trop net et trop durable favorise les positions dominantes, monopoles et oligopoles, qui limitent la et concurrence finissent par peser sur le consommateur.",
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

### FQuAD

This dataset was published in [this
paper](https://aclanthology.org/2020.findings-emnlp.107/), and is a manually annotated
dataset of questions and answers from the French Wikipedia.

The original full dataset consists of 20,731 / 3,188 / 2,189 samples for training,
validation and testing, respectively. Note that the testing split is not publicly
accessible, however, so we only use the training and validation split. We use 1,024 /
256 / 2,048 samples for training, validation, and testing, respectively. Our training
split is a subset of the original training split, and our validation and testing splits
are subsets of the original validation split.

Here are a few examples from the training split:

```json
{
  'context': "Parmi leurs thèmes récurrents, on en trouve qui sont communs à beaucoup d'autres groupes contemporains ou plus anciens : les Stranglers ont décrit, à plusieurs reprises, la vie d'un groupe de rock dans toutes ses dimensions (fans, autres groupes, vie en tournée). Le thème rebattu - chez les groupes des années 1960-1970 - de la drogue, est abordée sur une demi-douzaine de chansons (Don't Bring Harry), tandis que la vision angoissée du futur, dans le contexte de la guerre froide ou en lien avec les avancées de la science, a donné lieu à plusieurs titres (Curfew). On retrouve également chez eux des préoccupations écologiques (Dreamtime) ou sociales. La guerre, notamment les deux guerres mondiales (Northwinds), mais aussi les guerres contemporaines (I Don't Agree), sont à l'origine de divers textes. Mais le thème qui les a le plus inspirés, c'est de loin les femmes (The Man They Love to Hate).",
  'question': 'Sur combien de chanson le thème de la drogue est il abordé ?',
  'answers': {
    'answer_start': array([353]),
    'text': array(['une demi-douzaine'], dtype=object)
  }
}
```
```json
{
  'context': "Au cours de cette période, Cavour se distingue par son talent de financier. Il contribue de manière prépondérante à la fusion de la Banque de Gênes et de la nouvelle Banque de Turin au sein de la Banque Nationale des États sardes (Banca Nazionale degli Stati Sardi). Après le succès électoral de décembre 1849, Cavour devient également une des figures dominantes de la politique piémontaise et il prend la fonction de porte-parole de la majorité modérée qui vient de se créer. Fort de cette position, il fait valoir que le moment des réformes est arrivé, favorisé par le Statut albertin qui a créé de réelles perspectives de progrès. Le Piémont peut ainsi s'éloigner du front catholique et réactionnaire, qui triomphe dans le reste de l'Italie. ",
  'question': "En quel année sort-il vainqueur d'une élection ?",
  'answers': {
    'answer_start': array([305]),
    'text': array(['1849'], dtype=object)
  }
}
```
```json
{
  'context': "Pour autant, le phénomène météorologique se décline sous d'autres variantes : ocelles du paon, évoquant les cent yeux d'Argus, fleurs champêtres et ornant les jardins où s'établit l'osmose entre couleurs complémentaires. La poésie tient en main la palette du peintre,, celle de Claude Gellée ou de Poussin. Pour autant, il ne s'agit pas là d'une posture habituelle chez lui, qui privilégie les paysages quasi-monochromes.",
  'question': "Qu'est ce que l'auteur préfère décrire ?",
  'answers': {
    'answer_start': array([394]),
    'text': array(['paysages'], dtype=object)
  }
}
```

When evaluating generative models, we use the following setup (see the
[methodology](/methodology) for more information on how these are used):

- Number of few-shot examples: 4
- Prefix prompt:
  ```
  Les textes suivants sont accompagnés de questions et de réponses.
  ```
- Base prompt template:
  ```
  Texte: {text}
  Question: {question}
  Réponse en 3 mots maximum: {label}
  ```
- Instruction-tuned prompt template:
  ```
  Texte: {text}

  Répondez à la question suivante sur le texte ci-dessus en 3 mots maximum.

  Question: {question}
  ```

You can evaluate this dataset directly as follows:

```bash
$ scandeval --model <model-id> --dataset fquad
```


## Knowledge

Missing!


## Common-sense Reasoning

Missing!


## Summarization

Missing!
