from .gnn.chemoinformatics import (
    get_murcko,
    get_anonymous_murcko,
    compute_countMorgFP,
    identify_bad_substructure_match,
)

from collections import Counter
from typing import Tuple, Dict, List, Callable, Any

import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def get_unique_perserve_order(seq: list) -> list:
    """
    Returns a list of unique elements from the input sequence while preserving the order.

    Args:
        seq (list): The input sequence.

    Returns:
        list: A new list containing the unique elements from the input sequence while preserving the order.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def count_unique_SMILES_and_MurckoScaffolds(smiles_list: List[str]) -> Dict[str, int]:
    """
    Count the number of unique SMILES, Murcko scaffolds, and frameworks in a given list of SMILES.

    Args:
        smiles_list (List[str]): A list of SMILES strings.

    Returns:
        count_dict (Dict[str, int]): A dictionary containing the counts of unique SMILES, Murcko scaffolds,
            and frameworks. The keys are "SMILES Count", "MurckoScaffold Count", and "Framework Count",
            respectively.
    """
    count_dict = {}
    smiles_list_unique = get_unique_perserve_order(smiles_list)
    count_dict["SMILES Count"] = len(smiles_list_unique)
    murcko_list_unique = get_unique_perserve_order(
        [get_murcko(smi) for smi in smiles_list_unique])
    count_dict["MurckoScaffold Count"] = len(murcko_list_unique)
    framework_list_unique = get_unique_perserve_order(
        [get_anonymous_murcko(smi) for smi in smiles_list_unique])
    count_dict["Framework Count"] = len(framework_list_unique)
    return count_dict


def create_group_index_mapping(unique_scaffolds):
    """
    Create a mapping from scaffolds to group indices.

    Args:
    unique_scaffolds (list): A list of unique scaffolds.

    Returns:
    dict: A dictionary mapping scaffolds to group indices.
    """
    return {scaffold: idx for idx, scaffold in enumerate(unique_scaffolds)}


def collect_unique_substructures(df, group_col, substructure_col):
    """
    Collects unique substructures within each group and returns them as lists.

    Args:
    df (pd.DataFrame): The DataFrame to process.
    group_col (str): The name of the column containing group indices.
    substructure_col (str): The name of the column containing substructures.

    Returns:
    dict: A dictionary with groups as keys and lists of unique substructures as values.
    """
    unique_substructures = {}
    for group, group_df in df.groupby(group_col):
        # Convert the set of unique substructures to a list
        unique_substructures[group] = list(
            group_df[substructure_col].dropna().unique())
    return unique_substructures


def make_graph_with_pos(smile: str) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    """
    Create a graph representation of a molecule with 2D coordinates.

    Args:
        smile (str): The SMILES representation of the molecule.

    Returns:
        Tuple[nx.Graph, Dict[int, Tuple[float, float]]]: A tuple containing the graph representation of the molecule
        and a dictionary mapping node indices to their 2D coordinates.
    """
    mol = Chem.MolFromSmiles(smile)
    AllChem.Compute2DCoords(mol)
    Graph = nx.Graph()
    for atom in mol.GetAtoms():
        Graph.add_node(atom.GetIdx(),
                       atomic_num=atom.GetAtomicNum(),)
    for bond in mol.GetBonds():
        Graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                       bond_type=bond.GetBondType())

    # Assign 2D coordinates to nodes
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        Graph.nodes[atom.GetIdx()]['pos'] = (pos.x, pos.y)
    pos = nx.get_node_attributes(Graph, 'pos')
    return Graph, pos


def remove_multiple_substrucmathes(
        df: pd.DataFrame,
        p_col: str,
        poi_col: str,
        e3_col: str,
) -> pd.DataFrame:
    """
    Remove rows from the test set that have bad substructure matches.

    Args:
        df (pd.DataFrame): The test set containing the data.
        p_col (str): The name of the column containing the PROTAC SMILES.
        poi_col (str): The name of the column containing the POI SMILES.
        e3_col (str): The name of the column containing the E3 SMILES.

    Returns:
        pd.DataFrame: The modified test set with bad substructure matches removed.
    """
    tmp = df.copy()

    bad_substructure_match_idx = []
    for idx, row in tmp.iterrows():
        protac_smile = row[p_col]
        poi_smile = row[poi_col]
        e3_smile = row[e3_col]

        bad_match = identify_bad_substructure_match(
            protac_smile, poi_smile, e3_smile)

        if bad_match:
            bad_substructure_match_idx.append(idx)

    tmp = tmp.drop(bad_substructure_match_idx)
    tmp = tmp.reset_index(drop=True)
    return tmp


def prepare_data_set(
        df: pd.DataFrame,
        p_col: str,
        poi_col: str,
        linker_col: str,
        e3_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the data set for further processing.

    Args:
        df (pd.DataFrame): The input test set.
        p_col (str): The name of the column containing the protac data.
        poi_col (str): The name of the column containing the poi data.
        linker_col (str): The name of the column containing the linker data.
        e3_col (str): The name of the column containing the e3 data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two dataframes:
            - df_protacs: A dataframe containing the protac data.
            - df_substructures: A dataframe containing the substructures data.
    """
    df['substructures'] = df.apply(lambda row: '.'.join(
        [str(row[poi_col]), str(row[linker_col]), str(row[e3_col])]), axis=1)

    df = remove_multiple_substrucmathes(df, p_col, poi_col, e3_col)

    df_substructures = df[['substructures']].copy()

    df_protacs = df[[p_col]].copy()
    df_protacs.rename(columns={p_col: 'Smiles'}, inplace=True)

    return df_protacs, df_substructures


def compute_FP_substructures(
        df: pd.DataFrame,
        columns: List[str],
        fp_function: Callable = compute_countMorgFP,
        return_unique: bool = True,
        convert_to_numpyarray: bool = False,
) -> List[Any]:
    """
    Compute fingerprint substructures for the given columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (List[str]): The columns in the DataFrame to compute the fingerprint substructures for.
        fp_function (Callable, optional): The function to use for computing the fingerprint substructures. Defaults to compute_countMorgFP.
        return_unique (bool, optional): Whether to return only unique substructures. Defaults to True.
        convert_to_numpyarray (bool, optional): Whether to convert the fingerprint substructures to numpy arrays. Defaults to False.

    Returns:
        List[Any]: A list of fingerprint substructures computed for each column.
    """
    out = []
    for c in columns:
        if return_unique:
            smi_list = df.loc[:, c].unique().tolist()
        else:
            smi_list = df.loc[:, c].tolist()
        countMorgFP = fp_function(smi_list)
        if convert_to_numpyarray:
            fp_numpy = []
            for fp in countMorgFP:
                arr = np.zeros((0,), dtype=int)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fp_numpy.append(arr)
            countMorgFP = fp_numpy
        out.append(countMorgFP)
    return out


def generate_murcko_in_df(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """
    Generate Murcko scaffolds for the molecules in a df.

    Args:
        df (pd.DataFrame): The input df containing the molecules.
        smiles_col (str): The name of the column in the df that contains the SMILES strings.

    Returns:
        pd.DataFrame: The input df with an additional column containing the Murcko scaffolds.
    """
    df[smiles_col + '_MS'] = df[smiles_col].apply(get_murcko)
    return df


def generate_anonymous_murcko_in_df(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """
    Generates anonymous Murcko scaffolds for the molecules in the given df.

    Args:
        df (pd.DataFrame): The input df containing the molecules.
        smiles_col (str): The name of the column in the df that contains the SMILES strings.

    Returns:
        pd.DataFrame: The df with an additional column containing the anonymous Murcko scaffolds.
    """
    df[smiles_col + '_AnonMS'] = df[smiles_col].apply(get_anonymous_murcko)
    return df
