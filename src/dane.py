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


def evaluate_dane(transformer: str, prefer_flax: bool = False):
    '''Finetune and evaluate a pretrained model on the DaNE dataset.

    Args:
        transformer (str):
            The full HuggingFace Hub path to the pretrained transformer model.
        prefer_flax (bool, optional):
            Whether to prefer to load the pretrained Flax model from the
            HuggingFace model repository. Defaults to False.
    '''
    # Load the DaNE data
    train, val, test = DDT().load_as_simple_ner(predefined_splits=True)

    # Split docs and labels
    train_docs, train_labels = train
    val_docs, val_labels = val
    test_docs, test_labels = test

    # Get the set of all unique labels in the dataset
    unique_labels = list({lbl for lbl_list in train_labels for lbl in lbl_list})

    # Set up a numeric representation of the labels
    label2id = {unique_labels[id]: id for id in range(len(unique_labels))}
    id2label = {id: unique_labels[id] for id in range(len(unique_labels))}

    # Load config of pretrained model
    config = AutoConfig.from_pretrained(transformer,
                                        num_labels=len(unique_labels),
                                        label2id=label2id,
                                        id2label=id2label,
                                        finetuning_task='ner')

    # Load tokenizer of pretrained model
    tokenizer = AutoTokenizer.from_pretrained(transformer,
                                              use_fast=True,
                                              add_prefix_space=True)

    # Load pretrained models
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            transformer,
            config=config,
            from_flax=prefer_flax
        )
    except OSError:
        prefer_flax = not prefer_flax
        model = AutoModelForTokenClassification.from_pretrained(
            transformer,
            config=config,
            from_flax=prefer_flax)

    # Convert dataset to the HuggingFace format
    train_dataset = Dataset.from_dict(dict(docs=train_docs,
                                           orig_labels=train_labels))
    val_dataset = Dataset.from_dict(dict(docs=val_docs,
                                         orig_labels=val_labels))
    test_dataset = Dataset.from_dict(dict(docs=test_docs,
                                          orig_labels=test_labels))

    def tokenize_and_align_labels(examples: dict, tokenizer) -> dict:
        '''Tokenize all texts and align the labels with them'''
        tokenized_inputs = tokenizer(
            examples['docs'],
            # We use this argument because the texts in our dataset are lists of
            # words (with a label for each word)
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples['orig_labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to
                # -100 so they are automatically ignored in the loss function
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
        return dataset.map(partial(tokenize_and_align_labels,
                                   tokenizer=tokenizer),
                           batched=True,
                           num_proc=4,
                           desc="Running tokenizer on dataset")

    # Tokenize the datasets
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_val = tokenize_dataset(val_dataset, tokenizer)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer)

    # Initialise the data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Initialise training arguments
    training_args = TrainingArguments(output_dir='.',
                                      evaluation_strategy='epoch',
                                      logging_strategy='epoch',
                                      save_strategy='no',
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16,
                                      gradient_accumulation_steps=1,
                                      learning_rate=2e-5,
                                      num_train_epochs=3,
                                      warmup_steps=50,
                                      report_to='all',
                                      save_total_limit=1,
                                      load_best_model_at_end=True)

    def compute_metrics(p):
        '''Helper function for computing metrics'''
        # Initialise metric
        metric = load_metric("seqeval")

        # Get the predictions from the model
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)

        # Remove ignored index (special tokens)
        true_predictions = [
            [id2label[p] for p, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for _, l in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions,
                                 references=true_labels)
        return dict(precision=results["overall_precision"],
                    recall=results["overall_recall"],
                    f1=results["overall_f1"],
                    accuracy=results["overall_accuracy"])

    # Initialise Trainer
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=tokenized_train,
                      eval_dataset=tokenized_val,
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics)

    # Finetune the model
    train_result = trainer.train()

    # Log training metrics and save the state
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Log validation metrics
    metrics = trainer.evaluate(tokenized_val)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # Log test metrics
    metrics = trainer.evaluate(tokenized_test)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


if __name__ == '__main__':
    evaluate_dane('flax-community/roberta-base-danish')
