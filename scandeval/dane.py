'''Evaluation of a language model on Scandinavian NER tasks'''

from danlp.datasets import DDT
from transformers import (AutoConfig,
                          AutoTokenizer,
                          AutoModelForTokenClassification,
                          DataCollatorForTokenClassification,
                          TrainingArguments,
                          Trainer)
from datasets import Dataset, load_metric
from functools import partial
import numpy as np
from typing import Tuple, Dict, List
import warnings

from .evaluator import Evaluator


class DaneEvaluator(Evaluator):
    def __init__(self,
                 prefer_flax: bool = False,
                 cache_dir: str = '~/.cache/huggingface',
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 50,
                 batch_size: int = 16):
        self._metric = load_metric("seqeval")
        label2id = {'B-LOC': 0,
                    'I-LOC': 1,
                    'B-ORG': 2,
                    'I-ORG': 3,
                    'B-PER': 4,
                    'I-PER': 5,
                    'B-MISC': 6,
                    'I-MISC': 7,
                    'O': 8}
        super().__init__(num_labels=9,
                         label2id=label2id,
                         prefer_flax=prefer_flax,
                         cache_dir=cache_dir,
                         learning_rate=learning_rate,
                         epochs=5,
                         warmup_steps=warmup_steps,
                         batch_size=batch_size)

    def _get_model_class(self) -> type:
        return AutoModelForTokenClassification

    def _tokenize_and_align_labels(self, examples: dict, tokenizer):
        '''Tokenize all texts and align the labels with them'''
        tokenized_inputs = tokenizer(
            examples['docs'],
            # We use this argument because the texts in our dataset are lists
            # of words (with a label for each word)
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples['orig_labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label
                # to -100 so they are automatically ignored in the loss
                # function
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id[label[word_idx]])
                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def _preprocess_data(self, dataset: Dataset, tokenizer) -> Dataset:
        return (dataset.map(partial(self._tokenize_and_align_labels,
                                    tokenizer=tokenizer),
                            batched=True,
                            num_proc=4)
                       .remove_columns(['docs', 'orig_labels']))

    def _load_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        # Load the DaNE data
        train, val, test = DDT().load_as_simple_ner(predefined_splits=True)

        # Split docs and labels
        train_docs, train_labels = train
        val_docs, val_labels = val
        test_docs, test_labels = test

        # Convert dataset to the HuggingFace format
        train_dataset = Dataset.from_dict(dict(docs=train_docs,
                                               orig_labels=train_labels))
        val_dataset = Dataset.from_dict(dict(docs=val_docs,
                                             orig_labels=val_labels))
        test_dataset = Dataset.from_dict(dict(docs=test_docs,
                                              orig_labels=test_labels))

        return train_dataset, val_dataset, test_dataset

    def _load_data_collator(self, tokenizer):
        return DataCollatorForTokenClassification(tokenizer)

    def _compute_metrics(self,
                         predictions_and_labels: tuple) -> Dict[str, float]:
        '''Helper function for computing metrics'''

        # Get the predictions from the model
        predictions, labels = predictions_and_labels
        predictions = np.argmax(predictions, axis=-1)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for _, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self._metric.compute(predictions=true_predictions,
                                       references=true_labels)
        return dict(micro_f1=results["overall_f1"])

    def _log_metrics(self, metrics: Dict[str, List[dict]]):
        '''Log the metrics.

        Args:
            metrics (dict):
                The metrics that are to be logged. This is a dict with keys
                'train', 'val' and 'split', with values being lists of
                dictionaries full of metrics.
        '''
        def get_stats(split: str) -> Tuple[float, float]:
            '''Helper function to compute the mean with confidence intervals.

            Args:
                split (str):
                    The dataset split we are calculating statistics of.

            Returns:
                pair of floats:
                    The mean micro-average F1-score and the radius of its 95%
                    confidence interval.
            '''
            metric_list = metrics[split]
            micro_f1s = [dct[f'{split}_micro_f1'] for dct in metric_list]
            mean_micro_f1 = np.mean(micro_f1s)
            std_err = np.std(micro_f1s, ddof=1) / np.sqrt(len(micro_f1s))
            return 100 * mean_micro_f1, 196 * std_err

        train_mean, train_std_err = get_stats('train')
        val_mean, val_std_err = get_stats('val')
        test_mean, test_std_err = get_stats('test')

        print('Mean micro-average F1-scores on DaNE:')
        print(f'  Train: {train_mean:.2f} +- {train_std_err:.2f}')
        print(f'  Validation: {val_mean:.2f} +- {val_std_err:.2f}')
        print(f'  Test: {test_mean:.2f} +- {test_std_err:.2f}')
