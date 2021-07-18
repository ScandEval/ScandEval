'''Abstract base class for evaluating models'''

from abc import ABC, abstractmethod
from datasets import Dataset
from transformers import (PreTrainedTokenizerBase,
                          AutoTokenizer,
                          AutoConfig,
                          RobertaPreTrainedModel,
                          TrainingArguments,
                          Trainer,
                          PrinterCallback)
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import warnings
import datasets.utils.logging as ds_logging
import transformers.utils.logging as tf_logging
from tqdm.auto import tqdm


warnings.filterwarnings(
    'ignore',
    module='torch.nn.parallel*',
    message=('Was asked to gather along dimension 0, but all input '
             'tensors were scalars; will instead unsqueeze and return '
             'a vector.')
)
warnings.filterwarnings('ignore', module='seqeval*')

# This disables the tokenizer progress bars
ds_logging.get_verbosity = lambda: ds_logging.NOTSET

# This disables most of the `transformers` logging
tf_logging.set_verbosity_error()


class Evaluator(ABC):
    '''Abstract base class for evaluating models'''
    def __init__(self,
                 num_labels: int,
                 label2id: Dict[str, int],
                 prefer_flax: bool = False,
                 cache_dir: str = '~/.cache/huggingface',
                 learning_rate: float = 2e-5,
                 epochs: int = 5,
                 warmup_steps: int = 50,
                 batch_size: int = 16):
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = {id: label for label, id in label2id.items()}
        self.prefer_flax = prefer_flax
        self.cache_dir = cache_dir
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size


    @abstractmethod
    def _get_model_class(self) -> type:
        pass

    def _load_model(self,
                   transformer: str,
                   prefer_flax: Optional[bool] = None) -> tuple:
        '''Load the model with its tokenizer.

        Args:
            transformer (str):
                The full HuggingFace Hub path to the pretrained transformer
                model.
            prefer_flax (bool, optional):
                Whether to prefer to load the pretrained Flax model from the
                HuggingFace model repository. Defaults to False.

        Returns:
            tuple: The model and the tokenizer.
        '''
        if prefer_flax is None:
            prefer_flax = self.prefer_flax

        config = AutoConfig.from_pretrained(transformer,
                                            num_labels=self.num_labels,
                                            label2id=self.label2id,
                                            id2label=self.id2label)

        try:
            model = self._get_model_class().from_pretrained(transformer,
                                                     config=config,
                                                     from_flax=prefer_flax,
                                                     cache_dir=self.cache_dir)

        # Loading of model failed, due to the Flax/PyTorch version not being
        # available. Trying the other one.
        except OSError:
            prefer_flax = not prefer_flax
            model = self._get_model_class().from_pretrained(transformer,
                                                     config=config,
                                                     from_flax=prefer_flax,
                                                     cache_dir=self.cache_dir)

        # If the model is a subclass of `RobertaPreTrainedModel` then we have
        # to add a prefix space to the tokens, by the way the model is
        # constructed.
        prefix = isinstance(model, RobertaPreTrainedModel)
        tokenizer = AutoTokenizer.from_pretrained(transformer, use_fast=True,
                                                  add_prefix_space=prefix)

        return model, tokenizer

    @abstractmethod
    def _load_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        pass

    def _preprocess_data(self, dataset: Dataset, tokenizer) -> Dataset:
        return dataset

    @abstractmethod
    def _load_data_collator(self,
                           tokenizer: Optional[PreTrainedTokenizerBase] = None):
        pass

    @abstractmethod
    def _compute_metrics(self,
                         predictions_and_labels: tuple) -> Dict[str, float]:
        pass

    @abstractmethod
    def _log_metrics(self, metrics: Dict[str, List[dict]]):
        pass

    def evaluate(self,
                 transformer: str,
                 num_finetunings: int = 10,
                 progress_bar: bool = True):

        # Set up progress bar
        if progress_bar:
            desc = 'Finetuning and evaluating'
            itr = tqdm(range(num_finetunings), desc=desc)
        else:
            itr = range(num_finetunings)

        # Load the tokenizer and model
        model, tokenizer = self._load_model(transformer)

        # Load the dataset
        train, val, test = self._load_data()

        # Preprocess the datasets
        preprocessed_train = self._preprocess_data(train, tokenizer)
        preprocessed_val = self._preprocess_data(val, tokenizer)
        preprocessed_test = self._preprocess_data(test, tokenizer)

        # Load the data collator
        data_collator = self._load_data_collator(tokenizer)

        # Initialise training arguments
        training_args = TrainingArguments(
            output_dir='.',
            evaluation_strategy='no',
            logging_strategy='no',
            save_strategy='no',
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            report_to='all',
            save_total_limit=0,
            log_level='error',  # Separate logging levels for Trainer
            log_level_replica='error'  # Separate logging levels for Trainer
        )

        metrics = defaultdict(list)
        for _ in itr:

            # Initialise Trainer
            trainer = Trainer(model=model,
                              args=training_args,
                              train_dataset=preprocessed_train,
                              eval_dataset=preprocessed_val,
                              tokenizer=tokenizer,
                              data_collator=data_collator,
                              compute_metrics=self._compute_metrics)

            # Remove the callback which prints the metrics after each
            # evaluation
            trainer.remove_callback(PrinterCallback)

            # Finetune the model
            train_result = trainer.train()

            # Log training metrics and save the state
            train_metrics = trainer.evaluate(preprocessed_train,
                                             metric_key_prefix='train')
            metrics['train'].append(train_metrics)

            # Log validation metrics
            val_metrics = trainer.evaluate(preprocessed_val,
                                           metric_key_prefix='val')
            metrics['val'].append(val_metrics)

            # Log test metrics
            test_metrics = trainer.evaluate(preprocessed_test,
                                            metric_key_prefix='test')
            metrics['test'].append(test_metrics)

        self._log_metrics(metrics)
        return metrics
