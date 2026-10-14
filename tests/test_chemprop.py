"""Tests for the ChemProp splitting model (skipped without the [chemprop] extra)."""
import numpy as np
import pytest
from rdkit import Chem

pytest.importorskip("chemprop")

from protac_splitter import protac_splitter as ps
from protac_splitter.chemprop import ChemPropSplitter
from protac_splitter.chemprop.inference import decode_pair, parse_protac
from protac_splitter.evaluation import check_reassembly

EXAMPLE_SMILES = (
    "N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(N4CCN(CCCCCNc5ccc6c(c5)"
    "C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)cc3)CC2)cc1Cl"
)

# The smallest architecture the default ChemProp featurizer accepts.
TINY_CONFIG = {
    "message_passing": {"d_v": 72, "d_e": 14, "d_h": 8, "bias": False, "depth": 2,
                        "dropout": 0.0, "activation": "relu", "undirected": False},
    "bond_predictor": {"n_tasks": 1, "n_classes": 3, "input_dim": 16, "hidden_dim": 8,
                       "n_layers": 1, "dropout": 0.0, "activation": "relu"},
    "class_weights": [1.0, 1.0, 1.0],
}


@pytest.fixture(scope="module")
def tiny_splitter():
    from protac_splitter.chemprop.inference import build_model
    return ChemPropSplitter.from_state_dict(TINY_CONFIG, build_model(TINY_CONFIG).state_dict(), device="cpu")


def test_decode_pair_follows_probabilities():
    mol = Chem.MolFromSmiles("c1ccccc1CCOCCc1ccncc1")
    bridges = [b.GetIdx() for b in mol.GetBonds() if not b.IsInRing()]
    probs = np.tile([0.98, 0.01, 0.01], (mol.GetNumBonds(), 1))
    probs[bridges[0]] = [0.01, 0.98, 0.01]
    probs[bridges[-1]] = [0.01, 0.01, 0.98]
    pair = decode_pair(mol, probs)
    assert (pair["bond_1"], pair["bond_2"]) == (bridges[0], bridges[-1])
    assert set(pair["parts"]) == {"part_1", "linker", "part_2"}


def test_decode_pair_none_without_two_cuttable_bonds():
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert decode_pair(mol, np.full((mol.GetNumBonds(), 3), 1 / 3)) is None


def test_parse_protac_rejects_invalid_inputs():
    assert parse_protac("not a smiles") is None
    assert parse_protac("CCO.CC") is None
    assert parse_protac("CC[*:1]") is None
    assert parse_protac("CCO") is not None


def test_split_reassembles(tiny_splitter):
    pred, invalid = tiny_splitter.split([EXAMPLE_SMILES, "CCO.CC"])
    assert invalid is None
    assert check_reassembly(EXAMPLE_SMILES, pred)


def test_save_and_load_roundtrip(tiny_splitter, tmp_path):
    tiny_splitter.save_pretrained(tmp_path)
    loaded = ChemPropSplitter.from_pretrained(str(tmp_path), device="cpu")
    assert loaded.config == TINY_CONFIG
    np.testing.assert_array_equal(
        tiny_splitter.predict_proba([EXAMPLE_SMILES])[0], loaded.predict_proba([EXAMPLE_SMILES])[0]
    )


def test_split_protac_chemprop(tiny_splitter, monkeypatch):
    monkeypatch.setattr(ps, "_cached_chemprop_splitter", lambda device: tiny_splitter)
    result = ps.split_protac([EXAMPLE_SMILES, EXAMPLE_SMILES], model="chemprop", batch_size=2)
    assert [r["model_name"] for r in result] == ["ChemProp", "ChemProp"]
    assert check_reassembly(result[0]["SMILES"], result[0]["default_pred_n0"])


def test_pretrained_model_from_hub():
    """Downloads the published model on first run."""
    result = ps.split_protac(EXAMPLE_SMILES, model="chemprop", device="cpu")
    assert result["model_name"] == "ChemProp"
    assert check_reassembly(result["SMILES"], result["default_pred_n0"])
