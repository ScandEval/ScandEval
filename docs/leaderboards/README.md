---
hide:
    - toc
---
# Leaderboards

👈 Choose a leaderboard on the left to see the results.


## 📊 How to Read the Leaderboards

The main score columns are the `Reading` and `Writing` columns, showing the [mean rank
score](/methodology) of the model across all [NLU and NLG tasks](/tasks/#overview),
respectively. The lower the rank, the better the model.

The columns that follow the rank columns are metadata about the model:

- `Parameters`: The total number of parameters in the model, in millions.
- `Vocabulary`: The size of the model's vocabulary, in thousands.
- `Context`: The maximum number of tokens that the model can process at a time.
- `Speed`: The inference time of the model - see more [here](/tasks/speed).
- `Commercial`: Whether the model can be used for commercial purposes. See [here](/faq)
  for more information.
- `Merge`: Whether the model is a merge of other models.

After these metadata columns, the individual scores for each dataset is shown. Each
dataset has a primary and secondary score - see what these are on the [task
page](/tasks). Lastly, the final columns show the ScandEval version used to benchmark
the given model on each of the datasets.

To read more about the individual datasets, see the [datasets](/datasets) page. Uf
you're interested in the methodology behind the benchmark, see the
[methodology](/methodology) page.
