"""Extract graph edge features from the PROTAC-Splitter dataset and push to HF Hub.

Usage:
    python scripts/get_graph_dataset.py --help
    python scripts/get_graph_dataset.py --num-proc 4 --hf-token <token>
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import tyro
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm

from protac_splitter.chemoinformatics import get_atom_idx_at_attachment
from scripts.common import get_hub_token


@dataclasses.dataclass
class Args:
    """Extract graph features from PROTAC-Splitter and push to HuggingFace Hub."""

    num_proc: int = 8
    hf_token: Optional[str] = None
    """HuggingFace token (defaults to HF_TOKEN in .env)."""


def extract_edge_features(
    protac_smiles: str,
    e3_split_pair: Optional[Tuple[int, int]] = None,
    wh_split_pair: Optional[Tuple[int, int]] = None,
    n_bits: int = 512,
    radius: int = 6,
    descriptor_names: Optional[List[str]] = None,
    fp_as_string: bool = False,
) -> pd.DataFrame:
    mol = Chem.MolFromSmiles(protac_smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {protac_smiles}")
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))

    G = nx.Graph()
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    LG = nx.line_graph(G)
    line_betweenness = nx.betweenness_centrality(LG, endpoints=True)
    betweenness = nx.betweenness_centrality(G, endpoints=True)
    line_degree = dict(LG.degree())
    line_degree_r2, line_degree_r3 = {}, {}
    for node in LG.nodes():
        n2 = nx.single_source_shortest_path_length(LG, node, cutoff=2)
        n3 = nx.single_source_shortest_path_length(LG, node, cutoff=3)
        line_degree_r2[node] = len([n for n, d in n2.items() if d == 2])
        line_degree_r3[node] = len([n for n, d in n3.items() if d == 3])
    degree = dict(G.degree())
    degree_r2, degree_r3 = {}, {}
    for node in G.nodes():
        n2 = nx.single_source_shortest_path_length(G, node, cutoff=2)
        n3 = nx.single_source_shortest_path_length(G, node, cutoff=3)
        degree_r2[node] = len([n for n, d in n2.items() if d == 2])
        degree_r3[node] = len([n for n, d in n3.items() if d == 3])

    if e3_split_pair is not None and wh_split_pair is not None:
        true_split_edges = {frozenset(e3_split_pair), frozenset(wh_split_pair)}

    fp_bitvec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    fp_arr = np.zeros((n_bits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp_bitvec, fp_arr)
    if fp_as_string:
        fp = {"chem_mol_fp": "".join([str(int(b)) for b in fp_arr])}
    else:
        fp = {f"chem_mol_fp_{i}": bool(fp_arr[i]) for i in range(n_bits)}

    descriptor_func_names = descriptor_names or ["MolWt", "HeavyAtomCount", "NumHAcceptors", "NumHDonors", "TPSA", "NumRotatableBonds", "RingCount", "MolLogP"]
    descriptors = {f"chem_mol_desc_{n}": getattr(Descriptors, n)(mol) for n in descriptor_func_names}

    edge_features = []
    for (u, v) in nx.bridges(G):
        bond = mol.GetBondBetweenAtoms(u, v)
        node = (u, v) if (u, v) in LG else (v, u)
        node_key = node if node in line_betweenness else (v, u)
        feat = {
            "graph_bond_idx": bond.GetIdx(),
            "graph_num_nodes": num_nodes,
            "graph_num_edges": num_edges,
            "graph_betweenness": line_betweenness.get(node_key, 0.0),
            "graph_degree": line_degree.get(node_key, 0),
            "graph_degree_r2": line_degree_r2.get(node_key, 0),
            "graph_degree_r3": line_degree_r3.get(node_key, 0),
            "graph_node_u_degree": degree.get(u, 0),
            "graph_node_u_degree_r2": degree_r2.get(u, 0),
            "graph_node_u_degree_r3": degree_r3.get(u, 0),
            "graph_node_v_degree": degree.get(v, 0),
            "graph_node_v_degree_r2": degree_r2.get(v, 0),
            "graph_node_v_degree_r3": degree_r3.get(v, 0),
            "graph_node_u_betweenness": betweenness.get(u, 0.0),
            "graph_node_v_betweenness": betweenness.get(v, 0.0),
            "chem_bond_type": str(bond.GetBondType()),
            "chem_atom_u": mol.GetAtomWithIdx(u).GetSymbol(),
            "chem_atom_v": mol.GetAtomWithIdx(v).GetSymbol(),
            "chem_is_aromatic": bond.GetIsAromatic(),
            "chem_is_in_ring": bond.IsInRing(),
            "chem_mol_smiles": protac_smiles,
            "chem_mol_n_bits": n_bits,
            "chem_mol_radius": radius,
        }
        feat.update(fp)
        feat.update(descriptors)
        if e3_split_pair is not None and wh_split_pair is not None:
            feat.update({
                "label_is_split": frozenset([u, v]) in true_split_edges,
                "label_e3_split": frozenset([u, v]) == frozenset(e3_split_pair),
                "label_wh_split": frozenset([u, v]) == frozenset(wh_split_pair),
            })
        edge_features.append(feat)

    df = pd.DataFrame(edge_features)
    int64_cols = df.select_dtypes(include=["int64"]).columns
    return df.astype({col: np.int32 for col in int64_cols})


def get_edge_features(
    protac_smiles: Union[str, List[str]],
    wh_smiles: Union[str, List[str]],
    lk_smiles: Union[str, List[str]],
    e3_smiles: Union[str, List[str]],
    n_bits: int = 512,
    radius: int = 6,
    descriptor_names: Optional[List[str]] = None,
    fp_as_string: bool = False,
    verbose: int = 0,
) -> pd.DataFrame:
    if isinstance(protac_smiles, str):
        protac_smiles, wh_smiles, lk_smiles, e3_smiles = [protac_smiles], [wh_smiles], [lk_smiles], [e3_smiles]
    iterables = tqdm(zip(protac_smiles, wh_smiles, lk_smiles, e3_smiles), total=len(protac_smiles), disable=verbose == 0)
    features = []
    for p_smi, w_smi, l_smi, e_smi in iterables:
        p, w, l, e = [Chem.MolFromSmiles(s) for s in (p_smi, w_smi, l_smi, e_smi)]
        if any(m is None for m in (p, w, l, e)):
            raise ValueError(f"Invalid SMILES: {p_smi}, {w_smi}, {l_smi}, {e_smi}")
        wh_edge = get_atom_idx_at_attachment(p, w, l)
        e3_edge = get_atom_idx_at_attachment(p, e, l)
        features.append(extract_edge_features(p_smi, e3_split_pair=e3_edge, wh_split_pair=wh_edge,
                                               n_bits=n_bits, radius=radius,
                                               descriptor_names=descriptor_names, fp_as_string=fp_as_string))
    return pd.concat(features, ignore_index=True)


def main(args: Args) -> None:
    from datasets import load_dataset
    token = get_hub_token(args.hf_token)

    def get_substructs(row):
        return {
            "PROTAC SMILES": row["text"],
            "POI Ligand SMILES with direction": row["labels"].split(".")[2],
            "Linker SMILES with direction": row["labels"].split(".")[1],
            "E3 Binder SMILES with direction": row["labels"].split(".")[0],
        }

    ds = load_dataset("ailab-bio/PROTAC-Splitter-Dataset", "clustered")
    ds = ds.map(get_substructs, num_proc=args.num_proc, remove_columns=["text", "labels"])
    ds = ds.map(
        lambda x: get_edge_features(
            protac_smiles=x["PROTAC SMILES"],
            wh_smiles=x["POI Ligand SMILES with direction"],
            lk_smiles=x["Linker SMILES with direction"],
            e3_smiles=x["E3 Binder SMILES with direction"],
            verbose=0,
        ),
        num_proc=args.num_proc,
        batched=True,
        remove_columns=["PROTAC SMILES", "POI Ligand SMILES with direction", "Linker SMILES with direction", "E3 Binder SMILES with direction"],
    )
    ds.push_to_hub("ailab-bio/PROTAC-Splitter-Graph-Dataset", token=token, private=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
