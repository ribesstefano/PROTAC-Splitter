from numba import njit
import numpy as np
import networkx as nx
from rdkit import Chem


def mol2graph(mol: Chem.Mol) -> nx.Graph:
    # NOTE: https://github.com/maxhodak/keras-molecules/pull/32/files
    G = nx.Graph()
    for atom in mol.GetAtoms():
        # Skip non-heavy atoms
        if atom.GetAtomicNum() != 0:
            G.add_node(atom.GetIdx(), label=atom.GetSymbol())
    for bond in mol.GetBonds():
        # Skip bonds to non-heavy atoms
        if bond.GetBeginAtom().GetAtomicNum() == 0 or bond.GetEndAtom().GetAtomicNum() == 0:
            continue
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label=bond.GetBondType())
    return G

def smiles2graph(smiles: str) -> nx.Graph:
    return mol2graph(Chem.MolFromSmiles(smiles))

def get_smiles2graph_edit_distance(smi1: str, smi2: str, **kwargs) -> float:
    return nx.graph_edit_distance(smiles2graph(smi1), smiles2graph(smi2), **kwargs)

def get_mol2graph_edit_distance(mol1: str, mol2: str, **kwargs) -> float:
    return nx.graph_edit_distance(mol2graph(mol1), mol2graph(mol2), **kwargs)

def get_smiles2graph_edit_distance_norm(smi1: str, smi2: str, ged_G1_G2: None, **kwargs) -> float:
    G1 = smiles2graph(smi1)
    G2 = smiles2graph(smi2)
    G0 = nx.empty_graph()
    ged_G1_G2 = ged_G1_G2 if ged_G1_G2 is not None else nx.graph_edit_distance(G1, G2, **kwargs)
    ged_G1_G0 = nx.graph_edit_distance(G1, G0, **kwargs)
    ged_G2_G0 = nx.graph_edit_distance(G2, G0, **kwargs)
    return ged_G1_G2 / (ged_G1_G0 + ged_G2_G0)

def smiles2adjacency_matrix(smiles: str) -> np.ndarray:
    return nx.adjacency_matrix(smiles2graph(smiles)).todense()

def build_label_mapping(G1, G2):
    labels = set()
    for G in [G1, G2]:
        for node in G.nodes():
            labels.add(G.nodes[node]['label'])
    label_to_int = {label: idx for idx, label in enumerate(sorted(labels))}
    return label_to_int

def preprocess_graph(G, label_to_int):
    n = G.number_of_nodes()
    adj = np.zeros((n, n), dtype=np.int32)
    labels = np.zeros(n, dtype=np.int32)
    node_id_to_idx = {}
    for idx, node in enumerate(G.nodes()):
        node_id_to_idx[node] = idx
        label = G.nodes[node]['label']
        labels[idx] = label_to_int[label]
    for u, v in G.edges():
        idx_u = node_id_to_idx[u]
        idx_v = node_id_to_idx[v]
        adj[idx_u, idx_v] = 1
        adj[idx_v, idx_u] = 1  # Assuming undirected graph
    return adj, labels

@njit
def compute_cost_matrix(labels1, labels2, degrees1, degrees2):
    n1 = labels1.shape[0]
    n2 = labels2.shape[0]
    C = np.zeros((n1, n2), dtype=np.float64)
    for i in range(n1):
        for j in range(n2):
            label_cost = 0.0 if labels1[i] == labels2[j] else 1.0
            neighborhood_cost = abs(degrees1[i] - degrees2[j])
            C[i, j] = label_cost + neighborhood_cost
    return C

@njit
def greedy_assignment(C):
    n1, n2 = C.shape
    assigned_cols = np.full(n2, False)
    row_ind = np.full(n1, -1, dtype=np.int32)
    for i in range(n1):
        min_cost = np.inf
        min_j = -1
        for j in range(n2):
            if not assigned_cols[j] and C[i, j] < min_cost:
                min_cost = C[i, j]
                min_j = j
        if min_j != -1:
            row_ind[i] = min_j
            assigned_cols[min_j] = True
    return row_ind

@njit
def compute_total_cost(C, row_ind, n1, n2, c_node_del, c_node_ins):
    total_cost = 0.0
    assigned_cols = np.full(n2, False)
    for i in range(n1):
        j = row_ind[i]
        if j != -1:
            total_cost += C[i, j]
            assigned_cols[j] = True
        else:
            total_cost += c_node_del
    for j in range(n2):
        if not assigned_cols[j]:
            total_cost += c_node_ins
    return total_cost

def approximate_graph_edit_distance(adj1, labels1, adj2, labels2, c_node_del=1.0, c_node_ins=1.0):
    degrees1 = adj1.sum(axis=1)
    degrees2 = adj2.sum(axis=1)
    C = compute_cost_matrix(labels1, labels2, degrees1, degrees2)
    row_ind = greedy_assignment(C)
    total_cost = compute_total_cost(C, row_ind, labels1.shape[0], labels2.shape[0], c_node_del, c_node_ins)
    return total_cost

def get_approximate_ged(G1, G2):
    label_to_int = build_label_mapping(G1, G2)
    adj1, labels1 = preprocess_graph(G1, label_to_int)
    adj2, labels2 = preprocess_graph(G2, label_to_int)
    cost = approximate_graph_edit_distance(adj1, labels1, adj2, labels2)
    return cost

def get_smiles2graph_edit_distance_approx(smi1: str, smi2: str) -> float:
    G1 = smiles2graph(smi1)
    G2 = smiles2graph(smi2)
    return get_approximate_ged(G1, G2)
