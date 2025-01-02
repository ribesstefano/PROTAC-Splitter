import pytest
from unittest.mock import patch
from transformers import AutoTokenizer
from datasets import Dataset
from protac_splitter.llms.data_utils import load_tokenized_dataset

# FILE: protac_splitter/llms/test_data_utils.py

texts = ["CCO", "CCN", "CCC", "CCF", "CCCl", "CCBr", "CCI", "CCOCC", "CCNCC", "CCCC"]

@pytest.fixture
def sample_dataset():
    return Dataset.from_dict({
        "text": texts,
        "labels": ["OCC", "NCC", "CCC", "FCC", "ClCC", "BrCC", "ICC", "CCOCC", "CCNCC", "CCCC"],
    })

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

@patch("protac_splitter.llms.data_utils.load_dataset")
def test_load_tokenized_dataset(mock_load_dataset, sample_dataset, tokenizer):
    mock_load_dataset.return_value = sample_dataset

    dataset = load_tokenized_dataset(
        daset_dir="dummy_dir",
        dataset_config="default",
        tokenizer=tokenizer,
        batch_size=2,
        encoder_max_length=10,
        decoder_max_length=10,
        token=None,
        num_proc_map=1,
        randomize_smiles=True,
        randomize_smiles_prob=1,
        randomize_smiles_repeat=10,
    )

    assert "input_ids" in dataset.column_names
    assert "attention_mask" in dataset.column_names
    assert "labels" in dataset.column_names
    assert len(dataset) == len(texts) * 10


    dataset = load_tokenized_dataset(
        daset_dir="dummy_dir",
        dataset_config="default",
        tokenizer=tokenizer,
        batch_size=2,
        encoder_max_length=10,
        decoder_max_length=10,
        token=None,
        num_proc_map=1,
        randomize_smiles=True,
        randomize_smiles_prob=0.5,
        randomize_smiles_repeat=10,
    )

    assert "input_ids" in dataset.column_names
    assert "attention_mask" in dataset.column_names
    assert "labels" in dataset.column_names
    # Check that the lenght of labels is the same as the length of input_ids
    assert len(dataset["input_ids"]) == len(dataset["labels"])