"""Descriptive structural/complexity metrics for a PROTAC split, for dataset analysis.

Complements `dataset_qc.qc_row` (structural validity, chemical plausibility, split
correctness proxies -- all pass/fail gates): this module adds *descriptive* metrics
meant to be plotted and explored rather than thresholded -- molecular size and shape,
ring/macrocycle content, linker rigidity, and uncommon-element content -- for the
intact PROTAC and for each of its three fragments (E3, linker, POI; POI == "warhead"
== WH, attachment `[*:1]`, E3 attachment `[*:2]`, matching `evaluation.split_prediction`).

Entry point is `split_metrics_row()`, which returns a flat dict of `{scope}_*` columns
for one (protac_smiles, prediction) pair, ready to `pd.concat` onto a DataFrame
alongside `dataset_qc.qc_row()`'s output.
"""
from __future__ import annotations

import functools
from typing import Any, Dict, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from protac_splitter.chemoinformatics import remove_dummy_atoms
from protac_splitter.evaluation import split_prediction

# Elements outside the "common organic PROTAC" set {H, C, N, O, F, Cl, Br, S, I}: a hit
# usually signals a boron-based warhead, a silicon isostere, or a phosphorus-containing
# linker -- all real, but uncommon enough to be worth surfacing for inspection rather
# than silently averaging into the rest of the distribution.
COMMON_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "Cl", "Br", "S", "I"})

# Smallest ring size counted as a macrocycle (common med-chem convention, e.g. Driggers
# et al. 2008 / Giordanetto & Kihlberg 2014 macrocycle reviews use >=12 heavy atoms).
MACROCYCLE_MIN_RING_SIZE = 12

# RDKit's own "Default" rotatable-bond definition (rdMolDescriptors.CalcNumRotatableBonds
# with NumRotatableBondsOptions.Default): a non-ring single bond between two non-terminal
# atoms, neither of which is part of a triple bond. Matched explicitly (rather than just
# calling CalcNumRotatableBonds) so the *same* bonds can be intersected with the linker
# backbone path in `linker_backbone_metrics`.
_ROTATABLE_BOND_SMARTS = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"


@functools.lru_cache(maxsize=1)
def _brenk_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)


@functools.lru_cache(maxsize=1)
def _rotatable_bond_query() -> Chem.Mol:
    return Chem.MolFromSmarts(_ROTATABLE_BOND_SMARTS)


def _rotatable_bond_idx(mol: Chem.Mol) -> set:
    return {
        mol.GetBondBetweenAtoms(a, b).GetIdx()
        for a, b in mol.GetSubstructMatches(_rotatable_bond_query())
    }


def _ring_descriptors(mol: Chem.Mol, scope: str) -> Dict[str, Any]:
    ring_sizes = sorted(len(r) for r in mol.GetRingInfo().AtomRings())
    macrocycle_sizes = [s for s in ring_sizes if s >= MACROCYCLE_MIN_RING_SIZE]
    return {
        f"{scope}_num_rings": rdMolDescriptors.CalcNumRings(mol),
        f"{scope}_num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        f"{scope}_num_aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        f"{scope}_num_saturated_rings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        f"{scope}_max_ring_size": ring_sizes[-1] if ring_sizes else 0,
        f"{scope}_ring_sizes": ";".join(map(str, ring_sizes)),
        f"{scope}_num_macrocycles": len(macrocycle_sizes),
        f"{scope}_macrocycle_sizes": ";".join(map(str, macrocycle_sizes)),
        f"{scope}_num_spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        f"{scope}_num_bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
    }


def _shape_descriptors(mol: Chem.Mol, scope: str) -> Dict[str, Any]:
    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms >= 2:
        # Topological "maximum graph length": the longest shortest-path between any two
        # heavy atoms (graph diameter), i.e. how stretched-out vs. globular the fragment is.
        dmat = Chem.GetDistanceMatrix(mol)
        finite = dmat[np.isfinite(dmat)]
        graph_diameter = int(finite.max()) if finite.size else 0
    else:
        graph_diameter = 0
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    num_bonds = mol.GetNumBonds()
    return {
        f"{scope}_heavy_atoms": heavy_atoms,
        f"{scope}_mw": round(Descriptors.MolWt(mol), 1),
        f"{scope}_graph_diameter": graph_diameter,
        f"{scope}_num_bonds": num_bonds,
        f"{scope}_num_rotatable_bonds": rot_bonds,
        f"{scope}_frac_rotatable_bonds": round(rot_bonds / heavy_atoms, 3) if heavy_atoms else None,
        f"{scope}_rotatable_bond_ratio": round(rot_bonds / num_bonds, 3) if num_bonds else None,
        f"{scope}_frac_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        f"{scope}_tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
        f"{scope}_logp": round(Descriptors.MolLogP(mol), 2),
        f"{scope}_num_stereocenters": (
            rdMolDescriptors.CalcNumAtomStereoCenters(mol)
            + rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
        ),
    }


def _uncommon_moiety_descriptors(mol: Chem.Mol, scope: str) -> Dict[str, Any]:
    rare_elements = sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() not in COMMON_ELEMENTS})
    brenk_hits = [match.GetDescription() for match in _brenk_catalog().GetMatches(mol)]
    return {
        f"{scope}_rare_elements": ";".join(rare_elements),
        f"{scope}_flag_rare_element": len(rare_elements) > 0,
        f"{scope}_brenk_hits": ";".join(brenk_hits),
        f"{scope}_num_brenk_hits": len(brenk_hits),
    }


_EMPTY_KEYS = (
    "heavy_atoms", "mw", "graph_diameter", "num_bonds", "num_rotatable_bonds",
    "frac_rotatable_bonds", "rotatable_bond_ratio",
    "frac_csp3", "tpsa", "logp", "num_stereocenters",
    "num_rings", "num_aromatic_rings", "num_aliphatic_rings", "num_saturated_rings",
    "max_ring_size", "num_macrocycles", "num_spiro_atoms", "num_bridgehead_atoms",
    "num_brenk_hits",
)
_EMPTY_STR_KEYS = ("ring_sizes", "macrocycle_sizes", "rare_elements", "brenk_hits")


def molecule_metrics(smiles: Optional[str], scope: str, strip_dummies: bool = False) -> Dict[str, Any]:
    """Compute all structural/complexity metrics for a single SMILES under `{scope}_*` columns.

    Args:
        smiles: SMILES to analyze. May contain `[*:n]` attachment points (a fragment) or not
            (the intact PROTAC).
        scope: Column-name prefix, e.g. "protac", "e3", "linker", "poi".
        strip_dummies: If True, remove `[*:n]` attachment atoms before computing descriptors --
            appropriate for fragments, whose dummy atoms were never real atoms in the parent
            molecule and would otherwise inflate heavy-atom/ring counts and register as a rare
            element. Leave False for the intact PROTAC (which has none) and for callers that
            need the dummy atoms, e.g. `linker_backbone_metrics`.

    Returns:
        A flat dict of `{scope}_*` metrics, all None/""/False (as appropriate to the field) when
        `smiles` is None or fails to parse.
    """
    out: Dict[str, Any] = {f"{scope}_{k}": None for k in _EMPTY_KEYS}
    out.update({f"{scope}_{k}": "" for k in _EMPTY_STR_KEYS})
    out[f"{scope}_flag_rare_element"] = False
    if smiles is None:
        return out

    work_smiles = remove_dummy_atoms(smiles, canonical=True) if strip_dummies else smiles
    if work_smiles is None:
        return out

    mol = Chem.MolFromSmiles(work_smiles)
    if mol is None:
        return out

    out.update(_shape_descriptors(mol, scope))
    out.update(_ring_descriptors(mol, scope))
    out.update(_uncommon_moiety_descriptors(mol, scope))
    return out


def linker_backbone_metrics(linker_smiles: Optional[str]) -> Dict[str, Any]:
    """Rigidity and stretch of the linker's *backbone* -- the shortest path between its two
    attachment points -- versus the fragment as a whole.

    Backbone rigidity (`linker_backbone_frac_rotatable`) is a more targeted rigidity measure
    than `linker_frac_rotatable_bonds` (from `molecule_metrics(..., scope="linker",
    strip_dummies=True)`): a substituent hanging off the backbone can be rigid or flexible
    without changing how far apart, or how freely, the backbone itself lets the E3 and POI
    ligands move relative to each other.

    `linker_graph_length_ratio` catches the complementary case: a linker whose longest path
    runs *through a branch or ring*, not along the direct attachment-to-attachment route. Both
    the numerator (`linker_max_graph_length`, the fragment's graph diameter) and denominator
    (`linker_effective_length`, the direct attachment-point-to-attachment-point path) are
    computed on the same raw (dummy-atom-included) linker graph so the ratio is exactly 1.0
    for a simple unbranched chain and grows only when some other pair of atoms is topologically
    farther apart than the two attachment points themselves.
    """
    out = {
        "linker_backbone_length": None,
        "linker_backbone_rotatable_bonds": None,
        "linker_backbone_frac_rotatable": None,
        "linker_effective_length": None,
        "linker_max_graph_length": None,
        "linker_graph_length_ratio": None,
    }
    if linker_smiles is None:
        return out
    mol = Chem.MolFromSmiles(linker_smiles)
    if mol is None:
        return out

    dummy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy_idx) != 2:
        return out
    path = list(Chem.GetShortestPath(mol, dummy_idx[0], dummy_idx[1]))
    if len(path) < 2:
        return out

    backbone_bond_idx = {
        mol.GetBondBetweenAtoms(path[i], path[i + 1]).GetIdx()
        for i in range(len(path) - 1)
    }
    backbone_rotatable = len(backbone_bond_idx & _rotatable_bond_idx(mol))

    out["linker_backbone_length"] = len(path) - 2  # heavy atoms strictly between the two dummies
    out["linker_backbone_rotatable_bonds"] = backbone_rotatable
    out["linker_backbone_frac_rotatable"] = (
        round(backbone_rotatable / len(backbone_bond_idx), 3) if backbone_bond_idx else None
    )

    effective_length = len(path) - 1  # bonds directly connecting the two attachment points
    dmat = Chem.GetDistanceMatrix(mol)
    finite = dmat[np.isfinite(dmat)]
    max_graph_length = int(finite.max()) if finite.size else 0

    out["linker_effective_length"] = effective_length
    out["linker_max_graph_length"] = max_graph_length
    out["linker_graph_length_ratio"] = (
        round(max_graph_length / effective_length, 3) if effective_length else None
    )
    return out


def split_metrics_row(
    protac_smiles: str,
    pred: Optional[str],
    poi_attachment_id: int = 1,
    e3_attachment_id: int = 2,
) -> Dict[str, Any]:
    """Compute descriptive structural metrics for the intact PROTAC and each predicted
    fragment (E3, linker, POI) of one (protac_smiles, prediction) pair.

    Unlike `dataset_qc.qc_row`, nothing here is a pass/fail gate -- it's meant to feed
    distribution plots in a downstream notebook, not to flag rows for review.
    """
    pred = None if pred is None or (isinstance(pred, float) and np.isnan(pred)) else pred
    frags = split_prediction(pred, poi_attachment_id, e3_attachment_id) if pred else {"e3": None, "linker": None, "poi": None}

    result: Dict[str, Any] = {}
    result.update(molecule_metrics(protac_smiles, "protac", strip_dummies=False))
    result.update(molecule_metrics(frags.get("e3"), "e3", strip_dummies=True))
    result.update(molecule_metrics(frags.get("linker"), "linker", strip_dummies=True))
    result.update(molecule_metrics(frags.get("poi"), "poi", strip_dummies=True))
    result.update(linker_backbone_metrics(frags.get("linker")))
    return result
