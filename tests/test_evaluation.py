import os
import time
from typing import List, Tuple

import pytest
from rdkit import Chem
import datasets
from transformers import AutoTokenizer, EvalPrediction
import numpy as np

from protac_splitter.evaluation import score_prediction
from protac_splitter.llms.evaluation import decode_and_get_metrics
from protac_splitter.llms.data_utils import load_tokenized_dataset

# FILE: protac_splitter/pytest/test_evaluation.py

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
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(NCCCCCCCCCNc4ccc5c(c4)C(=O)N(C4CCC(=O)NC4=O)C5=O)cc3)CC2)cc1Cl',
            'a_wrong_smiles.[*:2]NCCCCCCCCCNc1ccc(C([*:1])=O)cc1.[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
    ]


def test_score_prediction_basic(protac_examples):
    protac_smiles, label_smiles = protac_examples[0]
    
    # Test with identical prediction (should get perfect scores)
    scores = score_prediction(
        protac_smiles=protac_smiles,
        label_smiles=label_smiles,
        pred_smiles=label_smiles
    )
    
    assert scores['valid'] == True
    assert scores['has_three_substructures'] == True
    assert scores['has_all_attachment_points'] == True
    assert scores['reassembly'] == True


def test_score_prediction_corrupted(protac_examples):
    # Test with invalid SMILES
    protac_smiles, label_smiles = protac_examples[0]
    
    scores = score_prediction(
        protac_smiles=protac_smiles,
        label_smiles=label_smiles,
        pred_smiles="invalid_smiles",
    )
    
    assert scores['valid'] == False
    assert scores['has_three_substructures'] == False
    assert scores['has_all_attachment_points'] == False

    pred_smiles = label_smiles.split('.')
    pred_smiles[1] = 'invalid_smiles'
    pred_smiles = '.'.join(pred_smiles)

    scores = score_prediction(
        protac_smiles=protac_smiles,
        label_smiles=label_smiles,
        pred_smiles=pred_smiles,
    )
    
    assert scores['valid'] == False
    assert scores['has_three_substructures'] == True
    assert scores['has_all_attachment_points'] == False
    assert scores['reassembly'] == False
    assert scores['reassembly_nostereo'] == False

    pred_smiles = label_smiles.split('.')
    pred_smiles[1] = 'invalid_smiles'
    pred_smiles[2] = 'very_invalid_smiles'
    pred_smiles = '.'.join(pred_smiles)

    scores = score_prediction(
        protac_smiles=protac_smiles,
        label_smiles=label_smiles,
        pred_smiles=pred_smiles,
    )
    
    assert scores['valid'] == False
    assert scores['has_three_substructures'] == True
    assert scores['has_all_attachment_points'] == False
    assert scores['reassembly'] == False
    assert scores['reassembly_nostereo'] == False


def test_score_prediction_with_rdkit_metrics(protac_examples):
    protac_smiles, label_smiles = protac_examples[0]
    
    scores = score_prediction(
        protac_smiles=protac_smiles,
        label_smiles=label_smiles,
        pred_smiles=label_smiles,
        compute_rdkit_metrics=True
    )
    
    assert 'tanimoto_similarity' in scores
    assert isinstance(scores['tanimoto_similarity'], float)
    assert 0 <= scores['tanimoto_similarity'] <= 1

def test_score_prediction_with_graph_metrics(protac_examples):
    protac_smiles, label_smiles = protac_examples[0]
    
    scores = score_prediction(
        protac_smiles=protac_smiles,
        label_smiles=label_smiles,
        pred_smiles=label_smiles,
        compute_graph_metrics=True
    )
    
    for substr in ['e3', 'poi', 'linker']:
        assert f'{substr}_graph_edit_distance' in scores
        assert f'{substr}_graph_edit_distance_norm' in scores

def test_score_prediction_substructure_metrics(protac_examples):
    protac_smiles, label_smiles = protac_examples[0]
    
    # Corrupt one of the attachment points
    corrupted_pred = label_smiles.replace('[*:1]', '[*:3]')
    
    scores = score_prediction(
        protac_smiles=protac_smiles,
        label_smiles=label_smiles,
        pred_smiles=corrupted_pred
    )
    
    # Check that substructure-specific metrics exist
    for substr in ['e3', 'poi', 'linker']:
        assert f'{substr}_valid' in scores
        assert f'{substr}_equal' in scores
        assert f'{substr}_has_attachment_point(s)' in scores

# @pytest.mark.benchmark
def test_score_prediction_all_metrics_types():

    try:
        dataset = datasets.load_dataset(
            "ailab-bio/PROTAC-Splitter-Dataset",
            "standard",
            split="validation",
            token=os.environ.get("HF_TOKEN", None)
        )
        print(f"Loaded dataset. Length: {dataset.num_rows:,}")
    except Exception as e:
        pytest.skip(f"Could not load dataset: {str(e)}")

    num_samples = 2048
    max_allowed_time = 60.0

    # Take first 100 examples to keep test runtime reasonable
    examples = dataset.select(range(num_samples))
    
    elapsed_time = 0
    
    for i, example in enumerate(examples):
        protac = example['text']
        label = example['labels'] 
        
        start_time = time.time()

        # Score prediction using label as prediction (perfect case)
        scores = score_prediction(
            protac_smiles=protac,
            label_smiles=label, 
            pred_smiles=label,
            compute_rdkit_metrics=True,
            compute_graph_metrics=True,
            graph_edit_kwargs={"timeout": 0.5} # "Good" timeout: 0.05
        )

        stop_time = time.time()
        elapsed_time += stop_time - start_time
        
        # Basic validation of scores
        assert isinstance(scores, dict), f"Expected dict, got {type(scores)}"
        assert scores['valid'] == True, f"Invalid PROTAC SMILES: {protac}"
        assert scores['has_three_substructures'] == True, f"Missing substructures: {label}"
        assert elapsed_time < max_allowed_time, f"Score prediction took {elapsed_time:.2f}s at sample N.{i+1} ({i/num_samples:.1%} done), should be under {max_allowed_time}s"

    avg_time = elapsed_time / num_samples
        
    # Assert reasonable performance - should process 100 examples in under 30 seconds
    assert elapsed_time < max_allowed_time, f"Score prediction took {elapsed_time:.2f}s (avg: {avg_time:.2f}s), should be under {max_allowed_time}s"
    
    # Print timing info
    print(f"\nProcessed 100 examples in {elapsed_time:.2f}s")
    print(f"Average time per example: {(elapsed_time/num_samples)*1000:.1f}ms")


def test_decode_and_get_metrics(protac_examples):
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    
    pred = {
        'inputs': tokenizer([x[0] for x in protac_examples], padding=True, return_tensors='np').input_ids,
        'label_ids': tokenizer([x[1] for x in protac_examples], padding=True, return_tensors='np').input_ids,
    }

    # Set the PAD tokens to -100
    pred['inputs'][pred['inputs'] == tokenizer.pad_token_id] == -100
    pred['label_ids'][pred['label_ids'] == tokenizer.pad_token_id] == -100

    print(f"Inputs: {pred['inputs']}")
    print(f"Labels: {pred['label_ids']}")

    pred = EvalPrediction(
        predictions=pred['label_ids'],
        label_ids=pred['label_ids'],
        inputs=pred['inputs'],
    )

    scores = decode_and_get_metrics(pred, tokenizer=tokenizer, num_proc=1, batch_size=128, compute_graph_metrics=False)
    scores_parallel = decode_and_get_metrics(pred, tokenizer=tokenizer, num_proc=4, batch_size=128, compute_graph_metrics=False)

    for k in scores.keys():
        print(f"{k}: {scores[k]}")
        print(f"{k}: {scores_parallel[k]} [parallel]")

        if np.isnan(scores[k]) or np.isnan(scores_parallel[k]):
            assert np.isnan(scores[k]) and np.isnan(scores_parallel[k]), f"Score mismatch for key: {k}"
            continue
        assert scores[k] == scores_parallel[k], f"Score mismatch for key: {k}"

    assert scores == scores_parallel


def test_decode_and_get_metrics(protac_examples):
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    
    pred = {
        'inputs': tokenizer([x[0] for x in protac_examples], padding=True, return_tensors='np').input_ids,
        'label_ids': tokenizer([x[1] for x in protac_examples], padding=True, return_tensors='np').input_ids,
    }

    # Set the PAD tokens to -100
    pred['inputs'][pred['inputs'] == tokenizer.pad_token_id] == -100
    pred['label_ids'][pred['label_ids'] == tokenizer.pad_token_id] == -100

    print(f"Inputs: {pred['inputs']}")
    print(f"Labels: {pred['label_ids']}")

    pred = EvalPrediction(
        predictions=pred['label_ids'],
        label_ids=pred['label_ids'],
        inputs=pred['inputs'],
    )

    scores = decode_and_get_metrics(pred, tokenizer=tokenizer, num_proc=1, batch_size=128, compute_graph_metrics=False)
    scores_parallel = decode_and_get_metrics(pred, tokenizer=tokenizer, num_proc=4, batch_size=128, compute_graph_metrics=False)

    for k in scores.keys():
        print(f"{k}: {scores[k]}")
        print(f"{k}: {scores_parallel[k]} [parallel]")

        if np.isnan(scores[k]) or np.isnan(scores_parallel[k]):
            assert np.isnan(scores[k]) and np.isnan(scores_parallel[k]), f"Score mismatch for key: {k}"
            continue
        assert scores[k] == scores_parallel[k], f"Score mismatch for key: {k}"

    assert scores == scores_parallel


def test_time_decode_and_get_metrics():
    # try:
    #     dataset = datasets.load_dataset(
    #         "ailab-bio/PROTAC-Splitter-Dataset",
    #         "large",
    #         split="validation",
    #         token=os.environ.get("HF_TOKEN", None)
    #     )
    #     print(f"Loaded dataset. Length: {dataset.num_rows:,}")
    # except Exception as e:
    #     pytest.skip(f"Could not load dataset: {str(e)}")

    num_proc = 4
    tokenizer_batch_size = 1024
    decode_batch_size = 128

    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    dataset = load_tokenized_dataset(
        dataset_dir="ailab-bio/PROTAC-Splitter-Dataset",
        dataset_config="large",
        tokenizer=tokenizer,
        batch_size=tokenizer_batch_size,
        encoder_max_length=512,
        decoder_max_length=512,
        token=os.environ.get("HF_TOKEN", None),
        num_proc_map=num_proc,
        randomize_smiles=False,
        all_fragments_as_labels=True,
        linkers_only_as_labels=False,
    )

    print(f"Loaded dataset. Length: {dataset.num_rows}")

    for ds_size in [1024, 2048, 4096, -1]:
        if ds_size > 0:
            ds = dataset['validation'].select(range(ds_size))
        else:
            ds = dataset['validation']
        
        print('-'*80)
        print(f"Decoding {len(ds)} examples")
        print('-'*80)

        pred = EvalPrediction(
            predictions=ds['labels'],
            label_ids=ds['labels'],
            inputs=ds['input_ids'],
        )

        start_time = time.time()
        scores_parallel = decode_and_get_metrics(
            pred,
            tokenizer=tokenizer,
            num_proc=num_proc,
            batch_size=decode_batch_size,
            compute_graph_metrics=False,
        )
        elapsed_time = time.time() - start_time

        print(f"Decoding and scoring took {elapsed_time:.2f}s")