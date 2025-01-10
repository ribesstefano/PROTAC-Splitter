import random
import logging
from typing import Optional

import torch
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoTokenizer
from rdkit import Chem


def randomize_smiles_dataset(
    batch: dict,
    repeat: int = 1,
    prob: float = 0.5,
    apply_to_text: bool = True,
    apply_to_labels: bool = False,
) -> dict:
    """ Randomize SMILES in a batch of data.
    
    Args:
        batch (dict): Batch of data with "text" and "labels" keys.
        repeat (int, optional): Number of times to repeat the randomization. Defaults to 1.
        prob (float, optional): Probability of randomizing SMILES. Defaults to 0.5.
        apply_to_text (bool, optional): Whether to apply randomization to text. Defaults to True.
        apply_to_labels (bool, optional): Whether to apply randomization to labels. Defaults to False.

    Returns:
        dict: Randomized batch of data.
    """
    new_texts, new_labels = [], []
    for text, label in zip(batch["text"], batch["labels"]):
        try:
            mol_text = Chem.MolFromSmiles(text)
            mol_label = Chem.MolFromSmiles(label)
        except Exception:
            logging.error("Failed to convert SMILES to Mol!")
            new_texts.append(text)
            new_labels.append(label)
            continue

        if random.random() < prob:
            if apply_to_text:
                rand_texts = [Chem.MolToSmiles(mol_text, canonical=False, doRandom=True) for _ in range(repeat)]
            else:
                rand_texts = [text] * repeat

            if apply_to_labels:
                rand_labels = [Chem.MolToSmiles(mol_label, canonical=False, doRandom=True) for _ in range(repeat)]
            else:
                rand_labels = [label] * repeat

            new_texts.extend(rand_texts)
            new_labels.extend(rand_labels)
        else:
            new_texts.append(text)
            new_labels.append(label)

    return {"text": new_texts, "labels": new_labels}


def process_data_to_model_inputs(
        batch,
        tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
        encoder_max_length: int = 512,
        decoder_max_length: int = 512,
):
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    # tokenize the inputs and labels
    inputs = tokenizer(batch["text"], truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["labels"], truncation=True, max_length=decoder_max_length)
    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # batch["input_ids"] = batch["input_ids"].to(device)
    # batch["attention_mask"] = batch["attention_mask"].to(device)
    # batch["labels"] = batch["labels"].to(device)

    # Because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored when calculating the loss.
    # NOTE: Check the `ignore_index` argument in nn.CrossEntropyLoss.
    # NOTE: The following is already done in the DataCollatorForSeq2Seq
    # batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
    return batch


def load_tokenized_dataset(
        daset_dir: str,
        dataset_config: str = 'default',
        tokenizer: AutoTokenizer | str = "seyonec/ChemBERTa-zinc-base-v1",
        batch_size: int = 512,
        encoder_max_length:int = 512,
        decoder_max_length:int = 512,
        token: Optional[str] = None,
        num_proc_map: int = 1,
        randomize_smiles: bool = False,
        randomize_smiles_prob: float = 0.5,
        randomize_smiles_repeat: int = 1,
        randomize_text: bool = True,
        randomize_labels: bool = False,
        cache_dir: Optional[str] = None,
) -> Dataset:
    """ Load dataset and tokenize it.
    
    Args:
        daset_dir (str): The directory of the dataset or the name of the data on the Hugging Face Hub.
        dataset_config (str, optional): The configuration of the dataset. Defaults to 'default'.
        tokenizer (AutoTokenizer | str, optional): The tokenizer to use for tokenization. If a string, the tokenizer will be loaded using `AutoTokenizer.from_pretrained(tokenizer)`. Defaults to "seyonec/ChemBERTa-zinc-base-v1".
        batch_size (int, optional): The batch size for tokenization. Defaults to 512.
        encoder_max_length (int, optional): The maximum length of the encoder input sequence. Defaults to 512.
        decoder_max_length (int, optional): The maximum length of the decoder input sequence. Defaults to 512.
        token (Optional[str], optional): The Hugging Face API token. Defaults to None.
        num_proc_map (int, optional): The number of processes to use for mapping. Defaults to 1.
        randomize_smiles (bool, optional): Whether to randomize SMILES. Defaults to False.
        randomize_smiles_prob (float, optional): The probability of randomizing SMILES. Defaults to 0.5.
        randomize_smiles_repeat (int, optional): The number of times to repeat the randomization. Defaults to 1.
        randomize_text (bool, optional): Whether to randomize text. Defaults to True.
        randomize_labels (bool, optional): Whether to randomize labels. Defaults to False.

    Returns:
        Dataset: The tokenized dataset.
    """
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    dataset = load_dataset(
        daset_dir,
        dataset_config,
        token=token,
        cache_dir=cache_dir,
    )
    print(f"Loaded dataset. Length: {dataset.num_rows}")
    if randomize_smiles:
        dataset = dataset.map(
            randomize_smiles_dataset,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={
                "repeat": randomize_smiles_repeat,
                "prob": randomize_smiles_prob,
                "apply_to_text": randomize_text,
                "apply_to_labels": randomize_labels,
            },
            num_proc=num_proc_map,
            load_from_cache_file=True,
            desc="Randomizing SMILES",
        )
        print(f"Randomized SMILES in dataset. Length: {dataset.num_rows}")
    dataset = dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        fn_kwargs={
            "tokenizer": tokenizer,
            "encoder_max_length": encoder_max_length,
            "decoder_max_length": decoder_max_length,
        },
        num_proc=num_proc_map,
        load_from_cache_file=True,
        desc="Tokenizing dataset",
    )
    print(f"Tokenized dataset. Length: {dataset.num_rows}")

    # Move dataset to torch format and move to device
    # dataset.set_format(type="torch")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset = dataset.map(
    #     lambda batch: {key: x.to(device) for key in batch for x in batch[key]},
    #     batched=True,
    #     num_proc=num_proc_map,
    #     load_from_cache_file=True,
    #     desc="Moving dataset to device",
    # )
    # print(f"Moved dataset to device: {device}")

    return dataset


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

    def tokenize(sample, tokenizer, max_length=512):
        input_ids = tokenizer.encode(sample["query"], padding="max_length", max_length=max_length)
        return {"input_ids": input_ids, "query": sample["query"]}

    return dataset.map(lambda x: tokenize(x, tokenizer, max_length), batched=False)


def data_collator_for_trl(batch):
    return {
        "input_ids": [torch.tensor(x["input_ids"]) for x in batch],
        "query": [x["query"] for x in batch],
    }