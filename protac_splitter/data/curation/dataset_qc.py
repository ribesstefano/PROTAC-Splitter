"""Row-level QC checks for a PROTAC-Splitter dataset (or prediction CSV).

Three independent problems are checked for, per row:

1. Structural validity of the SMILES / prediction (parseable, reassembles).
2. Chemical plausibility of the molecule and its fragments (BRENK unstable
   substructures, synthesis-artefact leaving groups, implausible linker
   topology, out-of-range fragment sizes).
3. Plausibility of the *split itself* — independent of (1) and (2), since a
   structurally valid reassembly can still cut the molecule at the wrong
   bonds. Checked via similarity to curated E3/warhead reference ligands and
   agreement with the betweenness-centrality heuristic splitter (and,
   optionally, the XGBoost edge classifier's own decision margin).

Entry point is `qc_row()`, which returns a flat dict of metrics and
`flag_*` booleans for a single (protac_smiles, prediction) pair. Nothing is
deleted or auto-corrected here — this module only labels rows for review.
"""
from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from protac_splitter.chemoinformatics import canonize_smiles
from protac_splitter.evaluation import count_flags, score_split, split_prediction
from protac_splitter.graphs.clustering import get_representative_e3s_fp, get_representative_whs_fp
from protac_splitter.graphs.splitting_algorithms import split_protac_graph_based
from protac_splitter.graphs.utils import get_fp
from protac_splitter.protac_splitter import load_graph_edge_classifier_from_cache

# ---------------------------------------------------------------------------
# Chemical-plausibility reference data
# ---------------------------------------------------------------------------

# Common synthesis handles / protecting groups that should not survive into a
# final PROTAC. A hit anywhere in the molecule usually means the scraper
# picked up a synthetic intermediate rather than the reported final compound.
LEAVING_GROUP_SMARTS: Dict[str, str] = {
    "boronic_acid_or_ester": "[#6][B]([OX2])[OX2,OX1H]",
    "silyl_ether": "[Si]([CH3])([CH3])[CH3]",
    "boc_carbamate": "[NX3]C(=O)OC(C)(C)C",
    "cbz_carbamate": "[NX3]C(=O)OCc1ccccc1",
    "fmoc_carbamate": "[NX3]C(=O)OCC1c2ccccc2-c2ccccc21",
    "tosylate": "c1ccc(cc1)S(=O)(=O)O[#6]",
    "mesylate": "[CH3]S(=O)(=O)O[#6]",
    "azide": "[NX2-]=[NX2+]=[NX1-]",
    "diazo": "[CX3]=[N+]=[N-]",
    "free_halide_sp3": "[CX4][Cl,Br,I]",
}

# BRENK categories that fire on structural motifs legitimately common in real
# PROTACs (long PEG/alkyl linkers, glutarimide/phthalimide-type CRBN
# degrons) rather than on genuine synthesis/stability hazards. Excluded from
# `flag_unstable` so the gate isn't triggered on most of the dataset by
# design; the full, unfiltered hit list is still reported for inspection.
BRENK_ALLOWLIST_CATEGORIES = frozenset({
    "Aliphatic_long_chain",
    "phthalimide",
    "aniline",
})


@functools.lru_cache(maxsize=1)
def _leaving_group_queries() -> Dict[str, Chem.Mol]:
    return {name: Chem.MolFromSmarts(smarts) for name, smarts in LEAVING_GROUP_SMARTS.items()}


@functools.lru_cache(maxsize=1)
def _brenk_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)


@functools.lru_cache(maxsize=1)
def _e3_reference_fp():
    return get_representative_e3s_fp()


@functools.lru_cache(maxsize=1)
def _wh_reference_fp():
    return get_representative_whs_fp()


@functools.lru_cache(maxsize=1)
def _xgboost_classifier():
    return load_graph_edge_classifier_from_cache()


def warm_caches(load_xgboost: bool = True) -> None:
    """Force-populate the module-level caches in the current process.

    Call this once before fanning work out to a process pool so each worker
    inherits already-built state (via fork copy-on-write) instead of
    redundantly downloading the XGBoost model / rebuilding fingerprints.
    """
    _e3_reference_fp()
    _wh_reference_fp()
    _brenk_catalog()
    if load_xgboost:
        _xgboost_classifier()


# ---------------------------------------------------------------------------
# Per-fragment checks
# ---------------------------------------------------------------------------

def _brenk_hits(mol: Chem.Mol) -> List[str]:
    if mol is None:
        return []
    return [match.GetDescription() for match in _brenk_catalog().GetMatches(mol)]


def _instability_scan(protac_smiles: str) -> Dict[str, Any]:
    """Scan the *intact* molecule for BRENK unstable/reactive substructures.

    Run on the whole PROTAC rather than per-fragment so cut-point valence
    capping (see `_fragment_descriptors`) can't manufacture spurious hits.
    """
    mol = Chem.MolFromSmiles(protac_smiles)
    hits = _brenk_hits(mol)
    gating_hits = [h for h in hits if h not in BRENK_ALLOWLIST_CATEGORIES]
    return {
        "brenk_hits": ";".join(hits),
        "flag_unstable": len(gating_hits) > 0,
    }


def _leaving_group_scan(protac_smiles: str) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(protac_smiles)
    hits = []
    if mol is not None:
        for name, query in _leaving_group_queries().items():
            if query is not None and mol.HasSubstructMatch(query):
                hits.append(name)
    return {
        "leaving_group_hits": ";".join(hits),
        "flag_leaving_group": len(hits) > 0,
    }


def _fragment_similarity(smi1: Optional[str], smi2: Optional[str]) -> Optional[float]:
    if smi1 is None or smi2 is None:
        return None
    fp1 = get_fp(smi1, return_np=False)
    fp2 = get_fp(smi2, return_np=False)
    if fp1 is None or fp2 is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def _heuristic_agreement(
    protac_smiles: str,
    pred_frags: Dict[str, Optional[str]],
    betweenness_threshold: float,
    agreement_similarity_threshold: float,
) -> Dict[str, Any]:
    out = {
        "heuristic_e3": None,
        "heuristic_linker": None,
        "heuristic_poi": None,
        "heuristic_min_similarity": None,
        "flag_method_disagreement": False,
    }
    try:
        heuristic = split_protac_graph_based(
            protac_smiles=protac_smiles,
            use_classifier=False,
            representative_e3s_fp=_e3_reference_fp(),
            betweenness_threshold=betweenness_threshold,
        )
    except Exception:
        return out

    if any(v is None for v in heuristic.values()):
        return out

    out["heuristic_e3"] = heuristic["e3"]
    out["heuristic_linker"] = heuristic["linker"]
    out["heuristic_poi"] = heuristic["poi"]

    # Exact-SMILES agreement is too strict: the heuristic and the classifier
    # can legitimately pick cut bonds one atom apart on an otherwise-correct
    # split. Compare fragment-by-fragment similarity instead, and only flag
    # when the *worst* of the three roles diverges substantially — that's
    # the signature of a genuinely different (not just off-by-one-bond) cut.
    similarities = [
        _fragment_similarity(pred_frags.get(role), heuristic[role])
        for role in ("e3", "linker", "poi")
        if pred_frags.get(role) is not None
    ]
    similarities = [s for s in similarities if s is not None]
    if similarities:
        out["heuristic_min_similarity"] = round(min(similarities), 3)
        out["flag_method_disagreement"] = out["heuristic_min_similarity"] < agreement_similarity_threshold
    return out


def _xgboost_confidence(protac_smiles: str, margin_threshold: float) -> Dict[str, Any]:
    out = {"xgb_top1_proba": None, "xgb_margin": None, "flag_low_confidence": False}
    try:
        clf = _xgboost_classifier()
        features = clf.extract_graph_features(
            protac_smiles, wh_smiles=None, lk_smiles=None, e3_smiles=None,
            n_bits=clf.n_bits, radius=clf.radius, descriptor_names=clf.descriptor_names,
        )
        Xf = clf._ensure_features(features)
        proba = clf.pipeline.predict_proba(Xf)
        split_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        sorted_proba = np.sort(split_proba)[::-1]
        if len(sorted_proba) > 0:
            out["xgb_top1_proba"] = round(float(sorted_proba[0]), 3)
        if len(sorted_proba) > 1:
            margin = float(sorted_proba[0] - sorted_proba[1])
            out["xgb_margin"] = round(margin, 3)
            out["flag_low_confidence"] = margin < margin_threshold
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Row-level orchestration
# ---------------------------------------------------------------------------

def qc_row(
    protac_smiles: str,
    pred: Optional[str],
    e3_sim_threshold: float = 0.2,
    poi_sim_threshold: float = 0.2,
    betweenness_threshold: float = 0.4,
    agreement_similarity_threshold: float = 0.6,
    xgb_margin_threshold: float = 0.15,
    run_heuristic_agreement: bool = True,
    run_xgboost_confidence: bool = True,
) -> Dict[str, Any]:
    """Run all QC checks on a single (protac_smiles, e3.linker.poi prediction) pair."""
    pred = None if pred is None or (isinstance(pred, float) and np.isnan(pred)) else pred

    canon = canonize_smiles(protac_smiles)
    valid_protac_smiles = canon is not None
    protac = canon or protac_smiles

    # The split-dependent checks (structural validity, fragment size, linker topology,
    # known-ligand similarity) live in evaluation.score_split, shared with the
    # QC-gated `model="adaptive"` escalation in protac_splitter.py.
    result: Dict[str, Any] = {"valid_protac_smiles": valid_protac_smiles}
    result.update(score_split(
        protac, pred,
        e3_sim_threshold=e3_sim_threshold,
        poi_sim_threshold=poi_sim_threshold,
        representative_e3s_fp=_e3_reference_fp(),
        representative_whs_fp=_wh_reference_fp(),
    ))
    # score_split assumes a valid input molecule; fold in the input-level check too, so
    # an unparseable protac_smiles still surfaces as a structural failure here.
    result["flag_structural"] = result["flag_structural"] or not valid_protac_smiles

    frags = split_prediction(pred) if pred else {"e3": None, "linker": None, "poi": None}
    result.update(_instability_scan(protac))
    result.update(_leaving_group_scan(protac))

    if run_heuristic_agreement:
        result.update(_heuristic_agreement(protac, frags, betweenness_threshold, agreement_similarity_threshold))
    if run_xgboost_confidence:
        result.update(_xgboost_confidence(protac, xgb_margin_threshold))

    n_flags, review_reasons = count_flags(result)
    result["n_flags"] = n_flags
    result["review_reasons"] = review_reasons
    return result
