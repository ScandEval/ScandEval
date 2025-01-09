---
hide:
    - toc
---
# Leaderboards

👈 Choose a leaderboard on the left to see the results.


## 🏷️ Types of Leaderboards

Each language has two leaderboards:

- **Generative Leaderboard**: This leaderboard shows the performance of models that can
  generate text. These models have been evaluated on _all_ [tasks](/tasks), both NLU and
  NLG.
- **NLU Leaderboard**: This leaderboard shows the performance of models that can only
  understand text, and not generate text themselves. These models have been evaluated on
  the NLU tasks only.


## 📊 How to Read the Leaderboards

The main score column is the `Rank`, showing the [mean rank score](/methodology) of the
model across all the tasks in the leaderboard. The lower the rank, the better the model.

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
