from typing import List, Tuple

import pytest
from unittest.mock import patch
from transformers import AutoTokenizer
from datasets import Dataset
from protac_splitter.llms.data_utils import load_tokenized_dataset

# FILE: protac_splitter/llms/test_data_utils.py

@pytest.fixture
def protac_examples() -> List[Tuple[str, str]]:
    return [
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(N4CCN(CCCCCNc5ccc6c(c5)C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]CCCCCN2CCN(c1ccc(C([*:1])=O)cc1)CC2.[*:2]Nc3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'CN(c1ccc(C#N)c(Cl)c1)[C@H]1CC[C@H](NC(=O)c2ccc(N3CC(CN4CCN(c5ccc6c(c5)C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)C3)cc2)CC1',
            '[*:1]N[C@@H]2CC[C@@H](N(C)c1ccc(C#N)c(Cl)c1)CC2.[*:1]C(=O)c3ccc(N2CC(CN1CCN([*:2])CC1)C2)cc3.[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'CN1C(=O)CCc2cc3cc(c21)OCCOCC1CN(C(=O)CCC(=O)NCCCOCCOCCOc2cccc4c2C(=O)N(C2CCC(=O)NC2=O)C4=O)CCN1c1ncc(Cl)c(n1)N3',
            '[*:1]N5CCN4c1ncc(Cl)c(n1)Nc3cc2CCC(=O)N(C)c2c(c3)OCCOCC4C5.[*:2]OCCOCCOCCCNC(=O)CCC([*:1])=O.[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'N#CC1(CNc2cccc(-c3cc(N[C@H]4CC[C@H](NCC(=O)NCCOCCOCCOCCNc5cccc6c5C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)ncc3Cl)n2)CCOCC1',
            '[*:1]C(=O)CN[C@@H]4CC[C@@H](Nc3cc(c2cccc(NCC1(C#N)CCOCC1)n2)c(Cl)cn3)CC4.[*:2]NCCOCCOCCOCCN[*:1].[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'O=C1CCC(N2C(=O)c3ccc(OCCOCCOCCOCCN4CCN(Cc5ccc6nc(NC(=O)c7cccc(C(F)(F)F)c7)n([C@H]7CC[C@@H](CO)CC7)c6c5)CC4)cc3C2=O)C(=O)N1',
            '[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3.[*:1]CN1CCN(CCOCCOCCOCCO[*:2])CC1.[*:1]c4ccc3nc(NC(=O)c1cccc(C(F)(F)F)c1)n([C@@H]2CC[C@H](CO)CC2)c3c4',
        ],
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(N4CCN(CCCCNc5ccc6c(c5)C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]NCCCCN2CCN(c1ccc(C([*:1])=O)cc1)CC2.[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)COCCOCCOCCNC(=O)CCC(=O)N2CCN([C@H]3CC[C@@H](Nc4ncnn5ccc(C(C)C)c45)CC3)CC2)C(C)(C)C)cc1',
            '[*:2]N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)NCc3ccc(c2scnc2C)cc3)C(C)(C)C.[*:1]C(=O)CCC(=O)NCCOCCOCCOCC([*:2])=O.[*:1]N4CCN([C@@H]3CC[C@H](Nc1ncnn2ccc(C(C)C)c12)CC3)CC4',
        ],
        [
            'Cc1ncsc1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(NC(=O)CNC(=O)c2cccc(-c3ccc(N4CCN(C)CC4)c(NC(=O)c4c[nH]c(=O)cc4C(F)(F)F)c3)c2)C(C)(C)C)cc1',
            '[*:2]NC(C(=O)N1CC(O)CC1C(=O)NCc3ccc(c2scnc2C)cc3)C(C)(C)C.[*:1]NCC([*:2])=O.[*:1]C(=O)c4cccc(c3ccc(N1CCN(C)CC1)c(NC(=O)c2c[nH]c(=O)cc2C(F)(F)F)c3)c4',
        ],
        [
            'CN1C(=O)CCc2cc3cc(c21)OCCOC[C@H]1CN(C(=O)CCC(=O)NCCCOCCOCCOc2cccc4c2C(=O)N(C2CCC(=O)NC2=O)C4=O)CCN1c1ncc(Cl)c(n1)N3',
            '[*:1]N5CCN4c1ncc(Cl)c(n1)Nc3cc2CCC(=O)N(C)c2c(c3)OCCOC[C@H]4C5.[*:2]OCCOCCOCCCNC(=O)CCC([*:1])=O.[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'CC(C)Nc1cc(-n2ccc3cc(C#N)cnc32)ncc1C(=O)N[C@H]1CC[C@H](C(=O)NCCOCCOCCOCCNC(=O)COc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1',
            '[*:1]C(=O)[C@@H]4CC[C@@H](NC(=O)c3cnc(n2ccc1cc(C#N)cnc12)cc3NC(C)C)CC4.[*:1]NCCOCCOCCOCCNC(=O)CO[*:2].[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],

        [
            'CC(C)Nc1cc(-n2ccc3cc(C#N)cnc32)ncc1C(=O)N[C@H]1CC[C@H](C(=O)NCCOCCOCCNC(=O)COc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1',
            '[*:1]C(=O)[C@@H]4CC[C@@H](NC(=O)c3cnc(n2ccc1cc(C#N)cnc12)cc3NC(C)C)CC4.[*:1]NCCOCCOCCNC(=O)CO[*:2].[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)CCCNC(=O)CCC(=O)N2CCN([C@H]3CC[C@@H](Nc4ncnn5ccc(C(C)C)c45)CC3)CC2)C(C)(C)C)cc1',
            '[*:2]N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)NCc3ccc(c2scnc2C)cc3)C(C)(C)C.[*:2]C(=O)CCCNC(=O)CCC([*:1])=O.[*:1]N4CCN([C@@H]3CC[C@H](Nc1ncnn2ccc(C(C)C)c12)CC3)CC4',
        ],
        [
            'CC(C)Nc1cc(-n2ccc3cc(C#N)cnc32)ncc1C(=O)N[C@H]1CC[C@H](C(=O)NCCOCCNC(=O)COc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1',
            '[*:1]C(=O)c3cnc(n2ccc1cc(C#N)cnc12)cc3NC(C)C.[*:1]N[C@@H]1CC[C@@H](C(=O)NCCOCCNC(=O)CO[*:2])CC1.[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'O=C1CCC(N2C(=O)c3ccc(OCCOCCOCCN4CCN(Cc5ccc6nc(NC(=O)c7cccc(C(F)(F)F)c7)n([C@H]7CC[C@@H](CO)CC7)c6c5)CC4)cc3C2=O)C(=O)N1',
            '[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3.[*:1]CCOCCOCCO[*:2].[*:1]N5CCN(Cc4ccc3nc(NC(=O)c1cccc(C(F)(F)F)c1)n([C@@H]2CC[C@H](CO)CC2)c3c4)CC5',
        ],
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(NCCCNc4ccc5c(c4)C(=O)N(C4CCC(=O)NC4=O)C5=O)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]NCCCNc1ccc(C([*:1])=O)cc1.[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'CC(C)Nc1cc(-n2ccc3cc(C#N)cnc32)ncc1C(=O)N[C@H]1CC[C@H](C(=O)NCCCCNC(=O)COc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1',
            '[*:1]C(=O)c3cnc(n2ccc1cc(C#N)cnc12)cc3NC(C)C.[*:1]N[C@@H]1CC[C@@H](C(=O)NCCCCNC(=O)CO[*:2])CC1.[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'Cc1ncsc1-c1ccc([C@H](C)NC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)CN2CCN(CCN3CCC(O[C@H]4C[C@H](Oc5ccc6c(c5)Sc5cc([N+](=O)[O-])ccc5N6)C4)CC3)CC2)C(C)(C)C)cc1',
            '[*:2]N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)N[C@@H](C)c3ccc(c2scnc2C)cc3)C(C)(C)C.[*:1]O[C@@H]3C[C@@H](OC2CCN(CCN1CCN(CC([*:2])=O)CC1)CC2)C3.[*:1]c3ccc2[nH]c1ccc(N(=O)=O)cc1sc2c3',
        ],
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(NCCCCCCCCCNc4ccc5c(c4)C(=O)N(C4CCC(=O)NC4=O)C5=O)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]NCCCCCCCCCNc1ccc(C([*:1])=O)cc1.[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
    ]

texts = ["CCO", "CCN", "CCC", "CCF", "CCCl", "CCBr", "CCI", "CCOCC", "CCNCC", "CCCC"]

@pytest.fixture
def sample_dataset(protac_examples) -> Dataset:

    return Dataset.from_dict({
        "text": [x[0] for x in protac_examples],
        "labels": [x[1] for x in protac_examples],
    })

    # return Dataset.from_dict({
    #     "text": texts,
    #     "labels": ["OCC", "NCC", "CCC", "FCC", "ClCC", "BrCC", "ICC", "CCOCC", "CCNCC", "CCCC"],
    # })

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


@patch("protac_splitter.llms.data_utils.load_dataset")
def test_load_tokenized_dataset_with_fragments_and_linkers(mock_load_dataset, sample_dataset, tokenizer):
    mock_load_dataset.return_value = sample_dataset

    dataset = load_tokenized_dataset(
        daset_dir="dummy_dir",
        dataset_config="default",
        tokenizer=tokenizer,
        batch_size=2,
        encoder_max_length=512,
        decoder_max_length=512,
        token=None,
        num_proc_map=1,
        randomize_smiles=False,
        all_fragments_as_labels=False,
        linkers_only_as_labels=True,
    )

    assert "input_ids" in dataset.column_names
    assert "attention_mask" in dataset.column_names
    assert "labels" in dataset.column_names

    # Use the tokenizer to decode the labels
    labels = tokenizer.batch_decode(dataset["labels"], skip_special_tokens=True)
    # Print the labels
    for label in labels:
        print(label)

    assert all(label.count(".") == 0 for label in labels)
    assert all(label.count("[*:1]") == 1 for label in labels)
    assert all(label.count("[*:2]") == 1 for label in labels)


    dataset = load_tokenized_dataset(
        daset_dir="dummy_dir",
        dataset_config="default",
        tokenizer=tokenizer,
        batch_size=2,
        encoder_max_length=512,
        decoder_max_length=512,
        token=None,
        num_proc_map=1,
        randomize_smiles=False,
        all_fragments_as_labels=False,
        linkers_only_as_labels=False,
    )

    assert "input_ids" in dataset.column_names
    assert "attention_mask" in dataset.column_names
    assert "labels" in dataset.column_names

    # Use the tokenizer to decode the labels
    labels = tokenizer.batch_decode(dataset["labels"], skip_special_tokens=True)

    # Print the labels
    for label in labels:
        print(label)

    assert all(label.count(".") == 1 for label in labels)
    assert all(label.count("[*:1]") == 1 for label in labels)
    assert all(label.count("[*:2]") == 1 for label in labels)