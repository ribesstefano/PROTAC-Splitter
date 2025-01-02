import pytest
from unittest.mock import patch
from datasets import Dataset, DatasetDict
from protac_splitter.llms.training_mlm_model import train_mlm_model

@pytest.fixture
def sample_dataset():
    train_ds = Dataset.from_dict({
        "text": ["CCO", "CCN", "CCC", "CCF", "CCCl", "CCBr", "CCI", "CCOCC", "CCNCC", "CCCC"],
        "labels": ["OCC", "NCC", "CCC", "FCC", "ClCC", "BrCC", "ICC", "CCOCC", "CCNCC", "CCCC"]
    })
    val_ds = Dataset.from_dict({
        "text": ["CCO", "CCN", "CCC", "CCF", "CCCl", "CCBr", "CCI", "CCOCC", "CCNCC", "CCCC"],
        "labels": ["OCC", "NCC", "CCC", "FCC", "ClCC", "BrCC", "ICC", "CCOCC", "CCNCC", "CCCC"]
    })
    return DatasetDict({"train": train_ds, "validation": val_ds})


@patch("protac_splitter.llms.data_utils.load_dataset")
def test_train_mlm_model(mock_load_dataset, sample_dataset):
    # Mock the dataset
    mock_load_dataset.return_value = sample_dataset

    # Call the function
    train_mlm_model(
        model_id="test_model",
        ds_name="test_dataset",
        ds_config="default",
        learning_rate=5e-5,
        max_steps=-1,
        num_train_epochs=1,  # Training for one epoch
        batch_size=16,
        batch_size_tokenizer=512,
        gradient_accumulation_steps=1,
        hub_token=None,
        organization=None,
        output_dir="./models/",
        tokenizer="seyonec/ChemBERTa-zinc-base-v1",
        randomize_smiles=True,
        randomize_smiles_prob=0.5,
        randomize_smiles_repeat=2,
    )

    # Assertions
    mock_load_dataset.assert_called_once_with("test_dataset", "default", token=None)

if __name__ == "__main__":
    pytest.main()