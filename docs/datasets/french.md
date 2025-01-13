# 🇫🇷 French

This is an overview of all the datasets used in the French part of ScandEval. The
datasets are grouped by their task - see the [task overview](/tasks) for more
information about what these constitute.


## Sentiment Classification

Missing!


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
