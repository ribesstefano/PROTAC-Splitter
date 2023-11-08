import pandas as pd
import os
from datasets import load_dataset
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
    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
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
):
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