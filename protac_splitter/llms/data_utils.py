import os
import torch
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from typing import Optional


def process_data_to_model_inputs(
    batch,
    tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
    encoder_max_length: int = 512,
    decoder_max_length: int = 512,
):
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    # tokenize the inputs and labels
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["labels"], padding="max_length", truncation=True, max_length=decoder_max_length)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()
    # Because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored when calculating the loss.
    # NOTE: Check the `ignore_index` argument in nn.CrossEntropyLoss.
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
    return batch


def load_tokenized_dataset(
    daset_dir: str,
    dataset_config: str = 'default',
    tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
    batch_size: int = 512,
    encoder_max_length:int = 512,
    decoder_max_length:int = 512,
    token: Optional[str] = None,
) -> Dataset:
    """ Load dataset and tokenize it.
    
    Args:
        daset_dir: Dataset directory.
        dataset_config: Dataset configuration.
        tokenizer: Tokenizer.
        batch_size: Batch size.
        encoder_max_length: Encoder max length.
        decoder_max_length: Decoder max length.
        token: Token.
    """
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    dataset = load_dataset(daset_dir, dataset_config, token=token)
    dataset_tokenized = dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        fn_kwargs={
            "tokenizer": tokenizer,
            "encoder_max_length": encoder_max_length,
            "decoder_max_length": decoder_max_length,
        },
    )
    dataset_tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return dataset_tokenized


def tokenize(sample, tokenizer, max_length=512):
    input_ids = tokenizer.encode(sample["query"], padding="max_length", max_length=max_length)
    return {"input_ids": input_ids, "query": sample["query"]}


def load_trl_dataset(
    tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",  
    token: Optional[str] = None,
    max_length: int = 512,
    dataset_name: str = "ailab-bio/PROTAC-Splitter-Dataset",
    ds_config: str = "standard",
    ds_unalabeled: Optional[str] = None,
) -> Dataset:
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    # Load training data
    train_dataset = load_dataset(
        dataset_name,
        ds_config,
        split="train",
        token=token,
    )
    train_dataset = train_dataset.rename_column("text", "query")
    train_dataset = train_dataset.remove_columns(["labels"])

    if ds_unalabeled is not None:
        # Load un-labelled data
        unlabeled_dataset = load_dataset(
            dataset_name,
            ds_unalabeled,
            split="train",
            token=token,
        )
        unlabeled_dataset = unlabeled_dataset.rename_column("text", "query")
        unlabeled_dataset = unlabeled_dataset.remove_columns(["labels"])
        # Concatenate datasets row-wise
        dataset = concatenate_datasets([train_dataset, unlabeled_dataset])
    else:
        dataset = train_dataset
    return dataset.map(lambda x: tokenize(x, tokenizer, max_length), batched=False)


def data_collator_for_trl(batch):
    return {
        "input_ids": [torch.tensor(x["input_ids"]) for x in batch],
        "query": [x["query"] for x in batch],
    }