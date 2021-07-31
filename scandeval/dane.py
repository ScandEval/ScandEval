'''NER evaluation of a language model on the DaNE dataset'''

from danlp.datasets import DDT
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, load_metric
from functools import partial
import numpy as np
from typing import Tuple, Dict, List
from tqdm.auto import tqdm

from .evaluator import Evaluator
from .utils import doc_inherit


class DaneEvaluator(Evaluator):
    '''Evaluator of language models on the DaNE dataset.

    Args:
        cache_dir (str, optional):
            Where the downloaded models will be stored. Defaults to
            '~/.cache/huggingface', which is also the default HuggingFace cache
            directory.
        learning_rate (float, optional):
            What learning rate to use when finetuning the models. Defaults to
            2e-5.
        warmup_steps (int, optional):
            The number of training steps in which the learning rate will be
            warmed up, meaning starting from nearly 0 and progressing up to
            `learning_rate` after `warmup_steps` many steps. Defaults to 50.
        batch_size (int, optional):
            The batch size used while finetuning. Defaults to 16.

    Attributes:
        cache_dir (str): Directory where models are cached
        learning_rate (float): Learning rate used while finetuning
        warmup_steps (int): Number of steps used to warm up the learning rate
        batch_size (int): The batch size used while finetuning
        epochs (int): The number of epochs to finetune
        num_labels (int): The number of NER labels in the dataset
        label2id (dict): Conversion dict from NER labels to their indices
        id2label (dict): Conversion dict from NER label indices to the labels
    '''
    def __init__(self,
                 cache_dir: str = '~/.cache/huggingface',
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 50,
                 batch_size: int = 16,
                 include_misc_tags: bool = True):
        self._metric = load_metric("seqeval")
        self.include_misc_tags = include_misc_tags
        label2id = {'B-LOC': 0,
                    'I-LOC': 1,
                    'B-ORG': 2,
                    'I-ORG': 3,
                    'B-PER': 4,
                    'I-PER': 5,
                    'B-MISC': 6,
                    'I-MISC': 7,
                    'O': 8}
        super().__init__(task='token-classification',
                         num_labels=9,
                         label2id=label2id,
                         cache_dir=cache_dir,
                         learning_rate=learning_rate,
                         epochs=5,
                         warmup_steps=warmup_steps,
                         batch_size=batch_size)

    def _tokenize_and_align_labels(self,
                                   examples: dict,
                                   tokenizer,
                                   label2id: dict):
        '''Tokenise all texts and align the labels with them.

        Args:
            examples (dict):
                The examples to be tokenised.
            tokenizer (HuggingFace tokenizer):
                A pretrained tokenizer.
            label2id (dict):
                A dictionary that converts NER tags to IDs.

        Returns:
            dict:
                A dictionary containing the tokenized data as well as labels.
        '''
        tokenized_inputs = tokenizer(
            examples['docs'],
            # We use this argument because the texts in our dataset are lists
            # of words (with a label for each word)
            is_split_into_words=True,
        )
        all_labels = []
        for i, labels in enumerate(examples['orig_labels']):
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
                    label = labels[word_idx]
                    if not self.include_misc_tags and label[-4:] == 'MISC':
                        label = 'O'
                    try:
                        label_id = label2id[label]
                    except KeyError:
                        err_msg = (f'The label {label} was not found in '
                                   f'the model\'s config.')
                        if label[-4:] == 'MISC':
                            err_msg += (' You need to initialise this '
                                        'Evaluator with `include_misc_tags` '
                                        'set to False.')
                        raise IndexError(err_msg)
                    label_ids.append(label_id)

                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            all_labels.append(label_ids)
        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    @staticmethod
    def _collect_docs(examples: dict) -> str:
        examples['doc'] = [' '.join(toks) for toks in examples['docs']]
        return examples

    @doc_inherit
    def _preprocess_data(self,
                         dataset: Dataset,
                         framework: str,
                         **kwargs) -> Dataset:
        if framework in ['pytorch', 'tensorflow', 'jax']:
            map_fn = partial(self._tokenize_and_align_labels,
                             tokenizer=kwargs['tokenizer'],
                             label2id=kwargs['config'].label2id)
            tokenised_dataset = dataset.map(map_fn, batched=True, num_proc=8)
            return tokenised_dataset.remove_columns(['docs', 'orig_labels'])
        elif framework == 'spacy':
            return dataset.map(self._collect_docs, batched=True, num_proc=8)

    @doc_inherit
    def _load_data(self) -> Tuple[Dataset, Dataset]:

        # Load the DaNE data
        train, _, test = DDT().load_as_simple_ner(predefined_splits=True)

        # Split docs and labels
        train_docs, train_labels = train
        test_docs, test_labels = test

        # Convert dataset to the HuggingFace format
        train_dataset = Dataset.from_dict(dict(docs=train_docs,
                                               orig_labels=train_labels))
        test_dataset = Dataset.from_dict(dict(docs=test_docs,
                                              orig_labels=test_labels))

        return train_dataset, test_dataset

    @doc_inherit
    def _load_data_collator(self, tokenizer):
        return DataCollatorForTokenClassification(tokenizer)

    @doc_inherit
    def _compute_metrics(self,
                         predictions_and_labels: tuple) -> Dict[str, float]:
        # Get the predictions from the model
        predictions, labels = predictions_and_labels

        if isinstance(predictions, np.ndarray):
            raw_predictions = np.argmax(predictions, axis=-1)

            # Remove ignored index (special tokens)
            predictions = [
                [self.id2label[p] for p, l in zip(prediction, label)
                                  if l != -100]
                for prediction, label in zip(raw_predictions, labels)
            ]
            labels = [
                [self.id2label[l] for _, l in zip(prediction, label)
                                  if l != -100]
                for prediction, label in zip(raw_predictions, labels)
            ]

        results = self._metric.compute(predictions=predictions,
                                       references=labels)
        return dict(micro_f1=results["overall_f1"])

    @doc_inherit
    def _log_metrics(self,
                     metrics: Dict[str, List[Dict[str, float]]],
                     model_id: str):
        kwargs = dict(metrics=metrics, metric_name='micro_f1')
        train_mean, train_std_err = self._get_stats(split='train', **kwargs)
        test_mean, test_std_err = self._get_stats(split='test', **kwargs)

        if not np.isnan(train_std_err):
            print(f'Mean micro-average F1-scores on DaNE for {model_id}:')
            print(f'  Train: {train_mean:.2f} +- {train_std_err:.2f}')
            print(f'  Test: {test_mean:.2f} +- {test_std_err:.2f}')
        else:
            print(f'Micro-average F1-scores on DaNE for {model_id}:')
            print(f'  Train: {train_mean:.2f}')
            print(f'  Test: {test_mean:.2f}')

    @staticmethod
    def _extract_spacy_predictions(tokens_processed: tuple) -> dict:
        tokens, processed = tokens_processed

        # Get the model's named entity predictions
        ner_tags = {ent.text: ent.label_ for ent in processed.ents}

        # Organise the predictions to make them comparable to the labels
        preds = list()
        for token in tokens:
            for ner_tag in ner_tags.keys():
                if ner_tag.startswith(token):
                    preds.append('B-' + ner_tags[ner_tag])
                    break
                elif token in ner_tag:
                    preds.append('I-' + ner_tags[ner_tag])
                    break
            else:
                preds.append('O')

        return preds

    @doc_inherit
    def _get_spacy_predictions_and_labels(self,
                                          model,
                                          dataset: Dataset,
                                          progress_bar: bool) -> tuple:
        # Initialise progress bar
        if progress_bar:
            itr = tqdm(dataset['doc'], desc='Evaluating')
        else:
            itr = dataset['doc']

        disable = ['tok2vec', 'tagger', 'parser',
                   'attribute_ruler', 'lemmatizer']
        processed = model.pipe(itr, disable=disable, batch_size=256)

        map_fn = self._extract_spacy_predictions
        predictions = map(map_fn, zip(dataset['docs'], processed))

        return list(predictions), dataset['orig_labels']
