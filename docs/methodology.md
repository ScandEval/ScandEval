---
hide:
    - navigation
---
# Evaluation Methodology

## Robust Evaluations

When working with language models, the smallest change in the data can often lead to
large changes in the model's performance. For this reason, all models in ScandEval are
evaluated 10 times on bootstrapped (i.e., sampling with replacement) training/prompt
sets and test sets, and the mean and 95% confidence interval of these 10 runs are
reported in the leaderboard. This ensures that the reported scores are robust and not
just a result of random fluctuations in the data.

## Formulating NLU Tasks as Generative Tasks

Coming soon!

## Score Aggregation

Coming soon!

## Papers

Check out more in-depth descriptions of the methodology in the associated research
papers:

- [Encoder vs Decoder: Comparative Analysis of Encoder and Decoder Language Models on
  Multilingual NLU Tasks](https://doi.org/10.48550/arXiv.2406.13469)
- [ScandEval: A Benchmark for Scandinavian Natural Language
  Processing](https://aclanthology.org/2023.nodalida-1.20/)
