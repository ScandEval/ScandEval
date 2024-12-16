---
hide:
    - toc
---
# Datasets

👈 Choose a language on the left to see all the evaluation datasets available for that
language.


_Notes:_

Classification datasets:

```
print(json.dumps(df[['text', 'label']].query('label == @label').sample(n=1).iloc[0].to_dict(), indent=2, ensure_ascii=False))
```

NER datasets:

```
df[df.labels.map(lambda x: len(set(x)) > 1)][['tokens', 'labels']].sample(n=1).iloc[0].to_dict()
```

RC datasets:

```
df[['context', 'question', 'answers']].sample(n=1).iloc[0].to_dict()
```

SUMM datasets:

```
print(json.dumps(df[['text', 'target_text']].sample(n=1).iloc[0].to_dict(), indent=2, ensure_ascii=False))
```
