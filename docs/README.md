---
hide:
    - navigation
---
#
<div align='center'>
<img src="https://raw.githubusercontent.com/ScandEval/ScandEval/main/gfx/scandeval.png" width="517" height="217">
<h3>A Robust Multilingual Evaluation Framework for Language Models</h3>
</div>

--------------------------

ScandEval is a language model benchmarking framework that supports evaluating all types
of language models out there: encoders, decoders, encoder-decoders, base models, and
instruction tuned models. ScandEval has been battle-tested for more than three years and
are the standard evaluation benchmark for many companies, universities and organisations
around Europe.

All models on the [Hugging Face Hub](https://huggingface.co/models) can be evaluated
using ScandEval, as well as models accessible through 100+ different APIs, including
models you are hosting yourself via, e.g., [Ollama](https://ollama.com/) or [LM
Studio](https://lmstudio.ai/).

When working with language models, the smallest change in the data can often lead to
large changes in the model's performance. For this reason, all models in ScandEval are
evaluated 10 times on bootstrapped training/prompt sets and test sets, and the mean and
95% confidence interval of these 10 runs are reported in the leaderboard. This ensures
that the reported scores are robust and not just a result of random fluctuations in the
data.

All benchmark results have been computed using the associated [ScandEval Python
package](https://github.com/ScandEval/ScandEval), which you can use to replicate all the
results. The methodology of the benchmark can be found in the associated research
papers:

- [Encoder vs Decoder: Comparative Analysis of Encoder and Decoder Language Models on
  Multilingual NLU Tasks](https://doi.org/10.48550/arXiv.2406.13469)
- [ScandEval: A Benchmark for Scandinavian Natural Language
  Processing](https://aclanthology.org/2023.nodalida-1.20/)

ScandEval is maintained by researchers at [Alexandra
Institute](https://alexandra.dk) and [Aarhus University](https://au.dk), and is funded
by the EU project [TrustLLM](https://trustllm.eu/).
