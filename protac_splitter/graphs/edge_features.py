from typing import Tuple, List, Union

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm

from protac_splitter.chemoinformatics import get_atom_idx_at_attachment
from protac_splitter.display_utils import safe_display, get_mapped_protac_img


def bond_capacity(bond: Chem.Bond) -> int:
    """ Calculate the capacity of a bond based on its type and properties.
    Parameters:
        bond (Chem.Bond): The bond object from RDKit.
    Returns:
        int: The capacity of the bond, where higher values indicate less preference for cutting.
    """
    # High capacity for aromatic and ring bonds to avoid cutting them
    if bond.GetIsAromatic() or bond.IsInRing():
        return 1000  # very high capacity: avoid cutting aromatic bonds
    elif bond.GetBondType() == Chem.BondType.SINGLE:
        return 1     # low capacity: prefer to cut here
    elif bond.GetBondType() == Chem.BondType.DOUBLE:
        return 10    # medium penalty
    elif bond.GetBondType() == Chem.BondType.TRIPLE:
        return 20    # stronger penalty
    else:
        return 50    # fallback for unknown/rare types

def smiles_to_nx(
    smiles_or_mol: Union[str, Chem.Mol],
    use_capacity: bool = False,
) -> nx.Graph:
    """ Convert a SMILES string (or pre-parsed RDKit Mol) to a NetworkX graph.
    Parameters:
        smiles_or_mol: SMILES string or already-parsed RDKit Mol.
        use_capacity (bool): Whether to use bond capacity as edge weights.
    Returns:
        nx.Graph: The NetworkX graph representation of the molecule.
    """
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            raise ValueError(f"Input SMILES could not be parsed: {smiles_or_mol}")
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
        if mol is None:
            raise ValueError(f"Input SMILES could not be canonicalized: {smiles_or_mol}")
    else:
        mol = smiles_or_mol

    G = nx.Graph()
    if use_capacity:
        for bond in mol.GetBonds():
            capacity = bond_capacity(bond)
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), capacity=capacity)
    else:
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G

def extract_edge_features(
    protac_smiles: str,
    e3_split_pair: Tuple[int, int] = None,
    wh_split_pair: Tuple[int, int] = None,
    n_bits: int = 512,
    radius: int = 6,
    descriptor_names: List[str] = None,
    fp_as_string: bool = False,
    betweenness_approx_frac: float = None,
) -> pd.DataFrame:
    """Extract features from the edges of a PROTAC molecule represented as a SMILES string.

    Parameters:
        protac_smiles (str): SMILES representation of the PROTAC molecule.
        e3_split_pair (Tuple[int, int]): Indices of the E3 split pair.
        wh_split_pair (Tuple[int, int]): Indices of the warhead split pair.
        n_bits (int): Number of bits for Morgan fingerprints.
        radius (int): Radius for Morgan fingerprints.
        descriptor_names (List[str]): List of RDKit descriptor names to compute.

    Returns:
        pd.DataFrame: DataFrame containing edge features.
    """
    # Parse and canonicalize mol ONCE — re-used for graph and fingerprint
    mol = Chem.MolFromSmiles(protac_smiles)
    if mol is None:
        raise ValueError(f"Input SMILES could not be parsed: {protac_smiles}")
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True))
    if mol is None:
        raise ValueError(f"Input SMILES could not be canonicalized: {protac_smiles}")

    # Build graph directly from mol (no extra SMILES parsing)
    G = smiles_to_nx(mol, use_capacity=False)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Line graph and betweenness centrality
    LG = nx.line_graph(G)
    lg_k = max(1, round(betweenness_approx_frac * LG.number_of_nodes())) if betweenness_approx_frac is not None else None
    g_k  = max(1, round(betweenness_approx_frac * G.number_of_nodes()))  if betweenness_approx_frac is not None else None
    line_betweenness = nx.betweenness_centrality(LG, endpoints=True, k=lg_k)
    betweenness = nx.betweenness_centrality(G, endpoints=True, k=g_k)
    line_degree = dict(LG.degree())
    degree = dict(G.degree())

    # k-hop degrees via all_pairs_shortest_path_length (one BFS pass each, faster
    # than calling single_source_shortest_path_length per node)
    line_degree_r2 = {}
    line_degree_r3 = {}
    for node, paths in nx.all_pairs_shortest_path_length(LG, cutoff=3):
        line_degree_r2[node] = sum(1 for d in paths.values() if d == 2)
        line_degree_r3[node] = sum(1 for d in paths.values() if d == 3)

    degree_r2 = {}
    degree_r3 = {}
    for node, paths in nx.all_pairs_shortest_path_length(G, cutoff=3):
        degree_r2[node] = sum(1 for d in paths.values() if d == 2)
        degree_r3[node] = sum(1 for d in paths.values() if d == 3)

    if e3_split_pair is not None and wh_split_pair is not None:
        true_split_edges = {frozenset(e3_split_pair), frozenset(wh_split_pair)}

    # Molecular fingerprint — computed once, shared across all bridge-bond rows
    fp_bitvec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    fp_arr = np.zeros((n_bits,), dtype=np.uint8)
    AllChem.DataStructs.ConvertToNumpyArray(fp_bitvec, fp_arr)

    # RDKit descriptors — also molecular-level, computed once
    descriptor_func_names = descriptor_names or [
        "MolWt", "HeavyAtomCount", "NumHAcceptors", "NumHDonors",
        "TPSA", "NumRotatableBonds", "RingCount", "MolLogP"
    ]
    functions = [getattr(Descriptors, name) for name in descriptor_func_names]
    descriptors = {f"chem_mol_desc_{name}": func(mol) for name, func in zip(descriptor_func_names, functions)}

    # Per-bridge-bond scalar features (no fingerprint — added below as numpy block)
    bridges = list(nx.bridges(G))
    n_bridges = len(bridges)

    scalar_rows = []
    for (u, v) in bridges:
        bond = mol.GetBondBetweenAtoms(u, v)
        node = (u, v) if (u, v) in LG else (v, u)
        node_key = node if node in line_betweenness else (v, u)

        row = {
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
            "chem_bond_idx": bond.GetIdx(),
            "chem_bond_type": str(bond.GetBondType()),
            "chem_atom_u": mol.GetAtomWithIdx(u).GetSymbol(),
            "chem_atom_v": mol.GetAtomWithIdx(v).GetSymbol(),
            "chem_is_aromatic": bond.GetIsAromatic(),
            "chem_is_in_ring": bond.IsInRing(),
            "chem_mol_smiles": protac_smiles,
            "chem_mol_n_bits": n_bits,
            "chem_mol_radius": radius,
        }
        row.update(descriptors)

        if e3_split_pair is not None and wh_split_pair is not None:
            row["label_is_split"] = frozenset([u, v]) in true_split_edges
            row["label_e3_split"] = frozenset([u, v]) == frozenset(e3_split_pair)
            row["label_wh_split"] = frozenset([u, v]) == frozenset(wh_split_pair)

        scalar_rows.append(row)

    if not scalar_rows:
        return pd.DataFrame()

    scalar_df = pd.DataFrame(scalar_rows)

    # Add fingerprint as a contiguous numpy block — much faster than per-row dicts
    # with 512 individual bool entries
    if fp_as_string:
        scalar_df["chem_mol_fp"] = "".join(str(int(b)) for b in fp_arr)
    else:
        fp_cols = [f"chem_mol_fp_{i}" for i in range(n_bits)]
        fp_bool = fp_arr.astype(bool)
        # np.tile avoids n_bridges copies of a Python list; concat is cheaper than
        # updating 512-key dicts per row
        fp_df = pd.DataFrame(
            np.tile(fp_bool, (n_bridges, 1)),
            columns=fp_cols,
        )
        scalar_df = pd.concat([scalar_df.reset_index(drop=True), fp_df], axis=1)

    int64_cols = scalar_df.select_dtypes(include=["int64"]).columns
    scalar_df = scalar_df.astype({col: np.int32 for col in int64_cols})

    return scalar_df

def get_edge_features(
    protac_smiles: str | List[str],
    wh_smiles: str | List[str],
    lk_smiles: str | List[str],
    e3_smiles: str | List[str],
    n_bits: int = 512,
    radius: int = 6,
    descriptor_names: List[str] = None,
    fp_as_string: bool = False,
    verbose: int = 0,
) -> pd.DataFrame:
    """Get edge features for a given PROTAC molecule and its components.
    
    Parameters:
        protac_smiles (str | List[str]): SMILES representation of the PROTAC molecule.
        wh_smiles (str | List[str]): SMILES representation of the warhead.
        lk_smiles (str | List[str]): SMILES representation of the linker.
        e3_smiles (str | List[str]): SMILES representation of the E3 binder.
        n_bits (int): Number of bits for Morgan fingerprints.
        radius (int): Radius for Morgan fingerprints.
        descriptor_names (List[str]): List of RDKit descriptor names to compute.
        
    Returns:
        pd.DataFrame: DataFrame containing edge features.
    """
    if isinstance(protac_smiles, str):
        protac_smiles = [protac_smiles]
    if isinstance(wh_smiles, str):
        wh_smiles = [wh_smiles]
    if isinstance(lk_smiles, str):
        lk_smiles = [lk_smiles]
    if isinstance(e3_smiles, str):
        e3_smiles = [e3_smiles]

    iterables = zip(protac_smiles, wh_smiles, lk_smiles, e3_smiles)
    iterables = tqdm(iterables, desc="Extracting edge features", total=len(protac_smiles), disable=verbose == 0)
    features_list = []
    for protac_smi, wh_smi, lk_smi, e3_smi in iterables:
        if verbose > 1:
            get_mapped_protac_img(protac_smi, wh_smi, lk_smi, e3_smi, w=1500, h=600, display_image=True, useSVG=True)

        # Convert SMILES to RDKit molecules
        protac = Chem.MolFromSmiles(protac_smi)
        wh = Chem.MolFromSmiles(wh_smi)
        lk = Chem.MolFromSmiles(lk_smi)
        e3 = Chem.MolFromSmiles(e3_smi)
        if protac is None or wh is None or lk is None or e3 is None:
            raise ValueError(f"Invalid SMILES string: {protac}, {wh}, {lk}, {e3}")

        # Get the attachment points
        wh_edge = get_atom_idx_at_attachment(protac, wh, lk)
        e3_edge = get_atom_idx_at_attachment(protac, e3, lk)

        # Extract features
        features = extract_edge_features(
            protac_smi,
            e3_split_pair=e3_edge,
            wh_split_pair=wh_edge,
            n_bits=n_bits,
            radius=radius,
            descriptor_names=descriptor_names,
            fp_as_string=fp_as_string,
        )

        if verbose > 1:
            # Randomly sample and display 5 edges
            sample_edges = features.sample(n=5, random_state=42)
            # Display the sampled edges
            for _, row in sample_edges.iterrows():
                bond = protac.GetBondWithIdx(row['chem_bond_idx'])
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                safe_display(Draw.MolToImage(
                    protac,
                    size=(1500, 400),
                    highlightColor=(1, 0, 1, 0.3), # Light purple
                    highlightAtoms=[u, v], # Highlight the two atoms
                    legend=f"Graph nodes: {u}, {v} (Betweenness centrality: {row['graph_betweenness']:.3f})",
                ))
                # print(row[[c for c in features.columns if c.startswith('graph_')] + ['chem_atom_u', 'chem_atom_v', 'chem_is_in_ring']])
                print(row)

        # Append the features to the list
        features_list.append(features)

    return pd.concat(features_list, ignore_index=True)