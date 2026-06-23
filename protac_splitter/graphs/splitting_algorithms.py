import re
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from joblib import Parallel, delayed

import numpy as np
import networkx as nx
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .edge_classifier import GraphEdgeClassifier
from .e3_clustering import get_representative_e3s_fp
from .utils import max_tanimoto_similarity
from protac_splitter.data.curation.bond_adjustments import (
    adjust_amide_bonds_in_substructs,
    adjust_ester_bonds_in_substructs
)

def bond_capacity(bond: Chem.Bond) -> int:
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

def smiles_to_nx(smiles: str) -> nx.Graph:
    mol = Chem.MolFromSmiles(smiles)
    G = nx.Graph()
    for bond in mol.GetBonds():
        capacity = bond_capacity(bond)
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), capacity=capacity)
    return G

def extract_attachment_point(smiles):
    """
    Extracts the number X from the pattern [X*] in a SMILES string.

    Parameters:
        smiles (str): The SMILES string containing the attachment point.

    Returns:
        str or None: The extracted number as a string, or None if not found.
    """
    match = re.search(r'\[(\d+)\*\]', smiles)
    return match.group(1) if match else None

def split_protac_with_betweenness_centrality(
    protac_smiles: str,
    representative_e3s_fp: List[DataStructs.ExplicitBitVect] = None,
    morgan_fp_generator: Optional[Any] = None,
    use_capacity_weight: bool = False,
    betweenness_threshold: float = 0.4,
    betweenness_approx_frac: float = None,
    verbose: int = 0,
) -> Dict[str, str]:
    """
    Split the PROTAC molecule into two parts using the NetworkX library.
    
    Parameters:
        protac_smiles (str): The SMILES string of the PROTAC molecule.
        representative_e3s_fp (list): List of representative E3 ligands fingerprints.
        morgan_fp_generator: RDKit Morgan fingerprint generator (should be the same as the one that generated the E3 fingerprints).
        use_capacity_weight (bool): Whether to use bond capacity as weight for the graph.
        betweenness_threshold (float): Threshold for betweenness centrality to consider a node as a candidate for splitting.
        
    Returns:
        dict: A dictionary containing the E3 ligand, warhead, linker, top nodes, and max centrality score.
    """
    if morgan_fp_generator is None:
        # Create a default Morgan fingerprint generator
        morgan_fp_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=16,
            fpSize=1024,
            useBondTypes=True,
            includeChirality=True,
        )

    if representative_e3s_fp is None:
        # Get the representative E3 ligands fingerprints
        representative_e3s_fp = get_representative_e3s_fp(fp_generator=morgan_fp_generator)
    
    # -----------------------------------
    # Deterministic graph-based algorithm
    # -----------------------------------
    protac = Chem.MolFromSmiles(protac_smiles)
    if protac is None:
        raise ValueError(f"Invalid SMILES string: {protac_smiles}")

    G = smiles_to_nx(protac_smiles)

    # Compute betweenness centrality
    weight = 'capacity' if use_capacity_weight else None
    k = max(1, round(betweenness_approx_frac * G.number_of_nodes())) if betweenness_approx_frac is not None else None
    centrality = nx.betweenness_centrality(G, normalized=True, endpoints=True, weight=weight, k=k)

    # Get the two nodes with the highest betweenness centrality
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    # Get the list of bridges in the graph
    bridges = list(nx.bridges(G))

    # Get the top two nodes
    top_nodes = [n for n, _ in sorted_nodes if n in bridges][:2]

    # Get the top nodes with the highest betweenness centrality that are not in
    # a ring, but are adjacent to the top nodes or have a high betweenness
    for node, score in sorted_nodes:
        # Check if the node is in a ring in the protac molecule
        atom = protac.GetAtomWithIdx(node)
        if not atom.IsInRing():
            # Check if the atom is adjacent to any of the top nodes, if so, add it to the list
            for neighbor in G.neighbors(node):
                if neighbor in top_nodes:
                    top_nodes.append(node)
                    break
            if score > betweenness_threshold:
                top_nodes.append(node)

    # If a node as only top nodes as neighbors, add it to the list
    for node in G.nodes():
        if node not in top_nodes:
            neighbors = list(G.neighbors(node))
            if all(neighbor in top_nodes for neighbor in neighbors):
                top_nodes.append(node)

    # Get all paths between the top nodes, e.g., rings
    for i in range(len(top_nodes)):
        for j in range(i + 1, len(top_nodes)):
            node1 = top_nodes[i]
            node2 = top_nodes[j]

            for path in nx.all_simple_paths(G, node1, node2):
                for node in path:
                    if node not in top_nodes:
                        top_nodes.append(node)

    # Remove duplicates
    top_nodes = list(set(top_nodes))
    
    # Loop over the top nodes and find the nodes that have a neighbor outside
    # the top nodes
    edge_nodes = set()
    for top_node in top_nodes:
        for neighbor in G.neighbors(top_node):
            if neighbor not in top_nodes:
                edge_nodes.update([(top_node, neighbor)])
                break
            
    # Get molecule fragment from the top nodes
    bonds = [protac.GetBondBetweenAtoms(i, j) for (i, j) in edge_nodes]
    bonds_idx = [bond.GetIdx() for bond in bonds if bond is not None]
    
    # Try any pair of indexes, if the number of resulting fragments is not 3,
    # then do not consider them as candidates for splitting
    candidate_bonds = []
    for i in range(len(bonds_idx)):
        for j in range(i + 1, len(bonds_idx)):
            bond1 = bonds_idx[i]
            bond2 = bonds_idx[j]

            # Get the fragments
            fragments = Chem.FragmentOnBonds(protac, [bond1, bond2])
                
            # Check if there are 3 fragments
            if Chem.MolToSmiles(fragments).count(".") == 2:
                frag_lens = []
                avg_len = 0
                for frag in Chem.GetMolFrags(fragments, asMols=True):
                    frag_len = frag.GetNumAtoms()
                    frag_lens.append(frag_len)
                    avg_len += frag_len
                avg_len /= 3

                # Calculate the standard deviation of the fragment lengths
                len_std = 0
                for frag_len in frag_lens:
                    len_std += (frag_len - avg_len) ** 2
                len_std = (len_std / 3) ** 0.5
                candidate_bonds.append(((bond1, bond2), len_std))

    # Sort the candidate bonds by distance to average (smallest first)
    candidate_bonds = sorted(candidate_bonds, key=lambda x: x[1])

    ligands = None
    while ligands is None and len(candidate_bonds) > 0:
        bonds_idx = candidate_bonds[0][0]
        try:
            ligands = Chem.FragmentOnBonds(protac, bonds_idx, addDummies=True, dummyLabels=[(1, 1), (2, 2)])
        except Exception as e:
            if verbose > 0:
                print(f"Error fragmenting the molecule: {e}")
            candidate_bonds.pop(0)

    # If no candidate bonds were found, return None
    if ligands is None:
        if verbose > 0:
            print(f"No candidate bonds found for splitting PROTAC: {protac_smiles}")
        return {'e3': None, 'poi': None, 'linker': None, 'top_nodes': None, 'centrality': None}

    # Get the linker
    substructures = []
    for ligand in Chem.GetMolFrags(ligands, asMols=True):
        ligand_smiles = Chem.MolToSmiles(ligand, canonical=True)
        if ligand_smiles.count("*") == 2:
            linker_smiles = ligand_smiles
        else:
            substructures.append(ligand_smiles)

    sub1_sim = max_tanimoto_similarity(substructures[0], representative_e3s_fp, morgan_fp_generator)
    sub2_sim = max_tanimoto_similarity(substructures[1], representative_e3s_fp, morgan_fp_generator)
    if sub1_sim > sub2_sim:
        e3_smiles = substructures[0]
        wh_smiles = substructures[1]
    else:
        e3_smiles = substructures[1]
        wh_smiles = substructures[0]

    # Get the attachment point using a regex, e.g., should return 1 if [1*] is in the SMILES
    e3_attach_point = extract_attachment_point(e3_smiles)
    e3_smiles = e3_smiles.replace(f"[{e3_attach_point}*]", "[*:2]")
    linker_smiles = linker_smiles.replace(f"[{e3_attach_point}*]", "[*:2]")

    wh_attach_point = extract_attachment_point(wh_smiles)
    wh_smiles = wh_smiles.replace(f"[{wh_attach_point}*]", "[*:1]")
    linker_smiles = linker_smiles.replace(f"[{wh_attach_point}*]", "[*:1]")
    return {'e3': e3_smiles, 'poi': wh_smiles, 'linker': linker_smiles, 'top_nodes': top_nodes, 'centrality': centrality}


def split_protac_with_edge_classifier(
        protac_smiles: str,
        pipeline: Union[str, Path],
        representative_e3s_fp: Optional[List[np.array]] = None,
        morgan_fp_generator: Optional[Any] = None,
) -> Dict[str, str]:
    """ Split the PROTAC molecule into two parts using the pretrained edge classifier.
    
    Parameters:
        protac_smiles (str): The SMILES string of the PROTAC molecule.
        pipeline (Union[str, Path]): Path to the trained GraphEdgeClassifier model.
        representative_e3s_fp (Optional[List[np.array]]): Precomputed fingerprints of representative E3 ligands.
        morgan_fp_generator (Optional[Any]): RDKit Morgan fingerprint generator (should be the same as the one that generated the E3 fingerprints).
        
    Returns:
        dict: A dictionary containing the E3 ligand, warhead, linker, and bonds_idx
    """
    if morgan_fp_generator is None:
        # Create a default Morgan fingerprint generator
        morgan_fp_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=16,
            fpSize=1024,
            useBondTypes=True,
            includeChirality=True,
        )

    if representative_e3s_fp is None:
        # Get the representative E3 ligands fingerprints
        representative_e3s_fp = get_representative_e3s_fp(fp_generator=morgan_fp_generator)
    
    protac = Chem.MolFromSmiles(protac_smiles)
    if protac is None:
        raise ValueError(f"Invalid SMILES string: {protac_smiles}")

    if isinstance(pipeline, str):
        pipeline = GraphEdgeClassifier.load(pipeline)

    # TODO: Get the top-n bonds, if splitting results in more than 3 ligands,
    # test other pairs of bonds, then repeat until we get 3 ligands exactly.
    bonds_idx = pipeline.predict_from_smiles(
        protac_smiles,
        wh_smiles=None,
        lk_smiles=None,
        e3_smiles=None,
        top_n=2,
        return_array=True,
    ).flatten().tolist()

    ligands = Chem.FragmentOnBonds(protac, bonds_idx, addDummies=True, dummyLabels=[(1, 1), (2, 2)])

    # Get the linker, i.e., count the dummy atoms: if two, it's the linker
    substructures = []
    for ligand in Chem.GetMolFrags(ligands, asMols=True):
        ligand_smiles = Chem.MolToSmiles(ligand, canonical=True)
        if ligand_smiles.count("*") == 2:
            linker_smiles = ligand_smiles
        else:
            substructures.append(ligand_smiles)

    if not pipeline.binary:
        e3_smiles = substructures[0]
        wh_smiles = substructures[1]
        # NOTE: The classifier was trained on the following labels assignment:
        e3_attach_point = 1
        wh_attach_point = 2
    else:
        if representative_e3s_fp is None or morgan_fp_generator is None:
            raise ValueError("For pipeline trained on binary classification, representative_e3s_fp and morgan_fp_generator must be provided.")

        sub1_sim = max_tanimoto_similarity(substructures[0], representative_e3s_fp, morgan_fp_generator)
        sub2_sim = max_tanimoto_similarity(substructures[1], representative_e3s_fp, morgan_fp_generator)
        if sub1_sim > sub2_sim:
            e3_smiles = substructures[0]
            wh_smiles = substructures[1]
        else:
            e3_smiles = substructures[1]
            wh_smiles = substructures[0]
            
        # Get the attachment point using a regex, e.g., should return 1 if [1*] is in the SMILES
        e3_attach_point = extract_attachment_point(e3_smiles)
        wh_attach_point = extract_attachment_point(wh_smiles)

    e3_smiles = e3_smiles.replace(f"[{e3_attach_point}*]", "[*:2]")
    linker_smiles = linker_smiles.replace(f"[{e3_attach_point}*]", "[*:2]")

    wh_smiles = wh_smiles.replace(f"[{wh_attach_point}*]", "[*:1]")
    linker_smiles = linker_smiles.replace(f"[{wh_attach_point}*]", "[*:1]")
    return {'e3': e3_smiles, 'poi': wh_smiles, 'linker': linker_smiles, "bonds_idx": bonds_idx}

def split_protac_graph_based(
    protac_smiles: str,
    use_classifier: bool = False,
    classifier: Optional['GraphEdgeClassifier'] = None,
    representative_e3s_fp: Optional[List[Any]] = None,
    morgan_fp_generator: Optional[Any] = None,
    use_capacity_weight: bool = False,
    betweenness_threshold: float = 0.4,
    betweenness_approx_frac: float = None,
    verbose: int = 0,
) -> Dict[str, str]:
    """ Splits a PROTAC molecule using either ML classifier or deterministic betweenness centrality.
    Returns a dictionary with e3, poi, linker, bonds_idx.
    """

    if representative_e3s_fp is None:
        if morgan_fp_generator is None:
            # Create a default Morgan fingerprint generator
            morgan_fp_generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=16,
                fpSize=1024,
                useBondTypes=True,
                includeChirality=True,
            )
        # Get the representative E3 ligands fingerprints
        representative_e3s_fp = get_representative_e3s_fp(fp_generator=morgan_fp_generator)

    if use_classifier:
        split = split_protac_with_edge_classifier(
            protac_smiles=protac_smiles,
            pipeline=classifier,
            representative_e3s_fp=representative_e3s_fp,
            morgan_fp_generator=morgan_fp_generator,
        )
    else:
        split = split_protac_with_betweenness_centrality(
            protac_smiles=protac_smiles,
            representative_e3s_fp=representative_e3s_fp,
            morgan_fp_generator=morgan_fp_generator,
            use_capacity_weight=use_capacity_weight,
            betweenness_threshold=betweenness_threshold,
            betweenness_approx_frac=betweenness_approx_frac,
            verbose=verbose,
        )

        # Fallback to the classifier if the graph-based method fails
        if any(value is None for value in split.values()) and classifier is not None:
            split = split = split_protac_with_edge_classifier(
                protac_smiles=protac_smiles,
                pipeline=classifier,
                representative_e3s_fp=representative_e3s_fp,
                morgan_fp_generator=morgan_fp_generator,
            )

    substructs = {
        "e3": split["e3"],
        "poi": split["poi"],
        "linker": split["linker"],
    }

    # If all of the substructures are not None, fix the amide and ester bonds
    if all(x is not None for x in substructs.values()):
        substructs = adjust_amide_bonds_in_substructs(substructs, protac_smiles)
        substructs = adjust_ester_bonds_in_substructs(substructs, protac_smiles)
        split["e3"] = substructs["e3"]
        split["poi"] = substructs["poi"]
        split["linker"] = substructs["linker"]

    return split

def split_protac_with_graphs_wrapper(
    protac_smiles: List[str],
    use_classifier: bool = False,
    classifier: Optional['GraphEdgeClassifier'] = None,
    representative_e3s: Optional[List[Any]] = None,
    representative_e3s_fp: Optional[List[Any]] = None,
    morgan_fp_generator: Optional[Any] = None,
    use_capacity_weight: bool = False,
    betweenness_threshold: float = 0.4,
    betweenness_approx_frac: float = None,
) -> List[Dict[str, str]]:
    """ Wrapper function to apply split_protac_graph_based over a list of PROTAC SMILES.
    
    Parameters:
        protac_smiles (List[str]): List of SMILES strings of PROTAC molecules.
        use_classifier (bool): Whether to use a classifier for splitting.
        classifier (Optional[GraphEdgeClassifier]): Classifier to use if use_classifier is True.
        representative_e3s_fp (Optional[List[Any]]): Precomputed fingerprints of representative E3 ligands.
        morgan_fp_generator (Optional[Any]): RDKit Morgan fingerprint generator.
        use_capacity_weight (bool): Whether to use bond capacity as weight for the graph.
        betweenness_threshold (float): Threshold for betweenness centrality to consider a node as a candidate for splitting.
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing the split results for each PROTAC molecule.
    """
    if morgan_fp_generator is None and (representative_e3s is None or representative_e3s_fp is None):
        # Create a default Morgan fingerprint generator
        morgan_fp_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=16,
            fpSize=1024,
            useBondTypes=True,
            includeChirality=True,
        )

    if representative_e3s is None and representative_e3s_fp is None:
        # Get the representative E3 ligands fingerprints
        representative_e3s_fp = get_representative_e3s_fp(fp_generator=morgan_fp_generator)
    elif representative_e3s is not None and representative_e3s_fp is None:
        # Convert representative E3 ligands to fingerprints
        representative_e3s_fp = get_representative_e3s_fp(e3_list=representative_e3s, fp_generator=morgan_fp_generator)

    # Load the classifier if it is a string or Path
    if use_classifier and classifier is not None and isinstance(classifier, (str, Path)):
        classifier = GraphEdgeClassifier.load(classifier)

    return [
        split_protac_graph_based(
            protac_smiles=smi,
            use_classifier=use_classifier,
            classifier=classifier,
            representative_e3s_fp=representative_e3s_fp,
            morgan_fp_generator=morgan_fp_generator,
            use_capacity_weight=use_capacity_weight,
            betweenness_threshold=betweenness_threshold,
            betweenness_approx_frac=betweenness_approx_frac,
        ) for smi in protac_smiles
    ]


def split_protac_with_graphs_parallel(
    protac_smiles: List[str],
    use_classifier: bool = False,
    classifier: Optional['GraphEdgeClassifier'] = None,
    representative_e3s: Optional[List[Any]] = None,
    representative_e3s_fp: Optional[List[Any]] = None,
    morgan_fp_generator: Optional[Any] = None,
    use_capacity_weight: bool = False,
    betweenness_threshold: float = 0.4,
    betweenness_approx_frac: float = None,
    n_jobs: int = 1,
    batch_size: int = 1,
) -> List[Dict[str, str]]:
    """ Splits a list of PROTAC molecules using either ML classifier or deterministic betweenness centrality.
    
    Parameters:
        protac_smiles (List[str]): List of SMILES strings of PROTAC molecules.
        use_classifier (bool): Whether to use a classifier for splitting.
        classifier (Optional[GraphEdgeClassifier]): Classifier to use if use_classifier is True.
        representative_e3s (Optional[List[Any]]): List of representative E3 ligands. If None, uses precomputed fingerprints.
        representative_e3s_fp (Optional[List[Any]]): Precomputed fingerprints of representative E3 ligands.
        morgan_fp_generator (Optional[Any]): RDKit Morgan fingerprint generator.
        use_capacity_weight (bool): Whether to use bond capacity as weight for the graph.
        betweenness_threshold (float): Threshold for betweenness centrality to consider a node as a candidate for splitting.
        n_jobs (int): Number of parallel jobs to run. If 1, runs sequentially.
        batch_size (int): Size of each batch for parallel processing.
    """
    # Load the classifier if it is a string or Path
    if use_classifier and classifier is not None and isinstance(classifier, (str, Path)):
        classifier = GraphEdgeClassifier.load(classifier)

    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer.")
    if n_jobs == 1:
        # If n_jobs is 1, run the function sequentially
        return split_protac_with_graphs_wrapper(
            protac_smiles=protac_smiles,
            use_classifier=use_classifier,
            classifier=classifier,
            representative_e3s=representative_e3s,
            representative_e3s_fp=representative_e3s_fp,
            morgan_fp_generator=morgan_fp_generator,
            use_capacity_weight=use_capacity_weight,
            betweenness_threshold=betweenness_threshold,
            betweenness_approx_frac=betweenness_approx_frac,
        )

    # Raise a warning if the n_jobs > 1 and the fingerprint generator is provided
    if morgan_fp_generator is not None:
        print("Warning: Using a custom Morgan fingerprint generator with n_jobs > 1 may be un-pickleable.")

    # Split the SMILES list into batches
    smiles_batches = [protac_smiles[i:i + batch_size] for i in range(0, len(protac_smiles), batch_size)]

    # Ensure all SMILES are processed, even if the last batch is smaller than batch_size
    smiles_batches = [protac_smiles[i:i + batch_size] for i in range(0, len(protac_smiles), batch_size)]
    # Remove any empty batches (shouldn't happen, but for safety)
    smiles_batches = [batch for batch in smiles_batches if batch]

    # Run each batch in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(split_protac_with_graphs_wrapper)(
            protac_smiles=batch,
            use_classifier=use_classifier,
            classifier=classifier,
            representative_e3s=representative_e3s,
            representative_e3s_fp=representative_e3s_fp,
            morgan_fp_generator=morgan_fp_generator,
            use_capacity_weight=use_capacity_weight,
            betweenness_threshold=betweenness_threshold,
            betweenness_approx_frac=betweenness_approx_frac,
        ) for batch in smiles_batches
    )

    # Flatten the list of lists into a single list
    return [item for batch_result in results for item in batch_result]