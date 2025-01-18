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

Missing!


## Linguistic Acceptability

Missing!


## Reading Comprehension

### FQuAD

[description]

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
