# Standard library imports
import os
import sys
import io

# Data handling and scientific computing
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.linalg import inv

# PyTorch and related libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

# RDKit for cheminformatics
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, GetPeriodicTable
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolHash
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import RDKFingerprint

# NetworkX for network analysis
import networkx as nx

# Matplotlib for plotting and visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Other utilities
from tqdm import tqdm
from PIL import Image

# External datasets
from datasets import load_dataset

from collections import defaultdict


import re
import random


from IPython.display import display

sys.path.append('./Code/TestingGround/src/models/')

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.rdmolops.RemoveStereochemistry(mol)
    return mol





def compute_countMorgFP(smiles, radius=2):
    if smiles is None:
        return None
    if isinstance(smiles[0], str):
        mols = [get_mol(smi) for smi in smiles]
    else:
        mols = smiles #assume mols were fed instead
    fpgen = AllChem.GetMorganGenerator(radius=radius)
    fps = [fpgen.GetCountFingerprint(mol) for mol in mols]
    return fps

def compute_MorgFP(smiles, radius=2):
    if isinstance(smiles[0], str):
        mols = [get_mol(smi) for smi in smiles]
    else:
        mols = smiles #assume mols were fed instead
    fpgen = AllChem.GetMorganGenerator(radius=radius)
    fps = [fpgen.GetFingerprint(mol) for mol in mols]
    return fps

def compute_RDKitFP(smiles, minpath=1, maxPath=7, fpSize=2048):
    if isinstance(smiles[0], str):
        mols = [get_mol(smi) for smi in smiles]
    else:
        mols = smiles #assume mols were fed instead
    rdgen = rdFingerprintGenerator.GetRDKitFPGenerator(minPath=minpath, maxPath=maxPath, fpSize=fpSize)
    fps = [rdgen.GetCountFingerprint(mol) for mol in mols]
    return fps

def compute_TopologicalTorsionFP(smiles, countSimulation=True, fpSize=2048):
    if isinstance(smiles[0], str):
        mols = [get_mol(smi) for smi in smiles] 
    else:
        mols = smiles #assume mols were fed instead
    rdgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fpSize, countSimulation=countSimulation)
    fps = [rdgen.GetFingerprint(mol) for mol in mols]
    return fps

from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP
def compute_AvalonFP(smiles):
    if isinstance(smiles[0], str):
        mols = [get_mol(smi) for smi in smiles] 
    else:
        mols = smiles #assume mols were fed instead
    fps = [GetAvalonCountFP(mol) for mol in mols]
    return fps

def compute_FP_substructures(df, columns, fp_function=compute_countMorgFP, return_unique = True, convert_to_numpyarray = False):
    out = []
    for c in columns:
        
        if return_unique:
            smi_list = df.loc[:,c].unique().tolist()
        else:
            smi_list = df.loc[:,c].tolist()
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


def validate_test_set(test_set, training_set, cutoff, fp_function=compute_countMorgFP):                                                            #OBS! Does not take into aaccount if cutoff and cutoff_MS are different!
    """
    Validates that no molecule in the test set has a Tanimoto similarity above the cutoff
    with any molecule in the training set. Returns a list of maximum similarities for test set molecules.
    """
  
    test_fps = fp_function(test_set)
    train_fps = fp_function(training_set)
    max_similarities = []

    below_cutoff = True
    for test_fp in test_fps:
        similarities = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        max_similarity = max(similarities)
        max_similarities.append(max_similarity)
        if max_similarity > cutoff:
            below_cutoff = False


    return below_cutoff, max_similarities




def substructure_split_sort(substructure_smiles):
    if isinstance(substructure_smiles, str):
        substructure_smiles = substructure_smiles.split(".")
    for smile in substructure_smiles:
        if '[*:1]' in smile:
            if '[*:2]' in smile:
                linker_smile = smile
            else:
                poi_smile = smile
        elif '[*:2]' in smile:
            e3_smile = smile
        else:
            raise ValueError(f'[*:1] and [*:2] was not found in smile: {smile}')
    return poi_smile, linker_smile, e3_smile



def EdgeLabler(protac_smiles, substructure_smiles, print_smiles=False):
    """
    Draws a molecule with highlighted bonds where the substructures attach to the rest of the molecule.
    Additionally, it outputs a NetworkX graph and the indices of the corresponding bonds.
    
    Parameters:
    protac_smiles (str): The SMILES string of the molecule.
    substruct_smiles (list): A list of SMILES strings of the substructures with dummy atoms.
    
    Returns:
    G (NetworkX graph): The graph representation of the molecule.
    edge_indices (list): The indices of the atoms forming the corresponding bonds in the graph.
    """
    
    poi_smile, linker_smile, e3_smile = substructure_split_sort(substructure_smiles)

    if print_smiles is True:
        print(f'PROTAC: {protac_smiles}')
        print(f'poi_smile: {poi_smile}')
        print(f'linker_smile: {linker_smile}')
        print(f'e3_smile: {e3_smile}')
    
    mol = Chem.MolFromSmiles(protac_smiles)
    edge_indices = []
    
    ligand_smiles_list = [poi_smile, e3_smile]   
    for ligand_smile in ligand_smiles_list:
        substruct_mol = Chem.MolFromSmiles(ligand_smile)
        matches = mol.GetSubstructMatches(Chem.DeleteSubstructs(substruct_mol, Chem.MolFromSmiles('*')))                 #GetSubstructMatches vs GetSubstructMatch

        if not matches:
            continue  # If no match is found, skip to the next substructure
        match = matches[0]  # Take the first match                                         ##########################OBS!

        # Find the corresponding bond
        for bond in mol.GetBonds():
            begin_atom_label = int(bond.GetBeginAtomIdx() in match)
            end_atom_label = int(bond.GetEndAtomIdx() in match)
            if begin_atom_label != end_atom_label:
                edge_indices.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    sorted_edge_indices = [tuple(sorted(t)) for t in edge_indices]    #Create an unique set of edges
    unique_edge_set = set(sorted_edge_indices) # Step 2: Convert the list of sorted tuples to a set to remove duplicates
    unique_edge_indices = list(unique_edge_set) # Step 3: Convert the set back to a list

    if not edge_indices:
        print("No corresponding bonds were found.")
        return None

    return unique_edge_indices


def boundary_ligand_nodes(protac_smiles, substructure_smiles):
    poi_smile, _, e3_smile = substructure_split_sort(substructure_smiles)

    mol = Chem.MolFromSmiles(protac_smiles)

    boundary_POI_node_index = []
    boundary_E3_node_index = []
    ligand_smiles_list = [poi_smile, e3_smile]   
    for i, ligand_smile in enumerate(ligand_smiles_list):
        substruct_mol = Chem.MolFromSmiles(ligand_smile)
        matches = mol.GetSubstructMatches(Chem.DeleteSubstructs(substruct_mol, Chem.MolFromSmiles('*')))

        if not matches:
            continue  # If no match is found, skip to the next substructure
        match = matches[0]  # Take the first match                                         ##########################OBS!

        # Find boundary nodes for the POI and E3
        for bond in mol.GetBonds():
            begin_atom_label = int(bond.GetBeginAtomIdx() in match)
            end_atom_label = int(bond.GetEndAtomIdx() in match)
            if begin_atom_label != end_atom_label:
                if bond.GetBeginAtomIdx() in match and i == 0:
                    boundary_POI_node_index.append(bond.GetBeginAtomIdx())
                elif bond.GetEndAtomIdx() in match and i ==0:
                    boundary_POI_node_index.append(bond.GetEndAtomIdx())
                elif bond.GetBeginAtomIdx() in match and i == 1:
                    boundary_E3_node_index.append(bond.GetBeginAtomIdx())
                elif bond.GetEndAtomIdx() in match and i ==1:
                    boundary_E3_node_index.append(bond.GetEndAtomIdx())
                else:
                    raise ValueError(f'Edge labler error - Problem with substructure matches')

    if len(boundary_POI_node_index) == 0 or len(boundary_E3_node_index) == 0:
        display(Chem.MolFromSmiles(protac_smiles))
        display(Chem.MolFromSmiles(substructure_smiles))
        display(Chem.MolFromSmiles(poi_smile))
        display(Chem.MolFromSmiles(_))
        display(Chem.MolFromSmiles(e3_smile))
        print(f'boundary_POI_node_index: {boundary_POI_node_index}. boundary_E3_node_index: {boundary_E3_node_index}')
        raise ValueError("Failed to assign boundary index")

    return boundary_POI_node_index, boundary_E3_node_index

def boundary_ligand_nodes_v2(protac_smiles, poi_smile, e3_smile):

    mol = Chem.MolFromSmiles(protac_smiles)

    boundary_POI_node_index = -1
    boundary_E3_node_index = -1
    ligand_smiles_list = [poi_smile, e3_smile]   
    for i, ligand_smile in enumerate(ligand_smiles_list):
        substruct_mol = Chem.MolFromSmiles(ligand_smile)
        matches = mol.GetSubstructMatches(Chem.DeleteSubstructs(substruct_mol, Chem.MolFromSmiles('*')))

        if not matches:
            continue  # If no match is found, skip to the next substructure
        match = matches[0]  # Take the first match                                         ##########################OBS!

        # Find boundary nodes for the POI and E3
        for bond in mol.GetBonds():
            begin_atom_label = int(bond.GetBeginAtomIdx() in match)
            end_atom_label = int(bond.GetEndAtomIdx() in match)
            if begin_atom_label != end_atom_label:
                if bond.GetBeginAtomIdx() in match and i == 0:
                    boundary_POI_node_index = bond.GetBeginAtomIdx()
                    break
                elif bond.GetEndAtomIdx() in match and i ==0:
                    boundary_POI_node_index = bond.GetEndAtomIdx()
                    break
                elif bond.GetBeginAtomIdx() in match and i == 1:
                    boundary_E3_node_index = bond.GetBeginAtomIdx()
                    break
                elif bond.GetEndAtomIdx() in match and i ==1:
                    boundary_E3_node_index = bond.GetEndAtomIdx()
                    break
                else:
                    raise ValueError(f'Problem with substructure matches')

    if boundary_POI_node_index == -1 or boundary_E3_node_index == -1:
        display(Chem.MolFromSmiles(protac_smiles))
        display(Chem.MolFromSmiles(poi_smile))
        display(Chem.MolFromSmiles(e3_smile))
        print(f'boundary_POI_node_index: {boundary_POI_node_index}. boundary_E3_node_index: {boundary_E3_node_index}')
        raise ValueError("Failed to assign boundary index")

    return boundary_POI_node_index, boundary_E3_node_index


def get_boundary_bonds(protac_smiles, poi_smile, e3_smile):
    mol = Chem.MolFromSmiles(protac_smiles)

    boundary_POI_bond = -1
    boundary_E3_bond = -1
    ligand_smiles_list = [poi_smile, e3_smile]   
    for i, ligand_smile in enumerate(ligand_smiles_list):
        substruct_mol = Chem.MolFromSmiles(ligand_smile)
        matches = mol.GetSubstructMatches(Chem.DeleteSubstructs(substruct_mol, Chem.MolFromSmiles('*')))

        if not matches:
            continue  # If no match is found, skip to the next substructure
        match = matches[0]  # Take the first match                                         ##########################OBS!

        # Find boundary nodes for the POI and E3
        for bond in mol.GetBonds():
            begin_atom_label = int(bond.GetBeginAtomIdx() in match)
            end_atom_label = int(bond.GetEndAtomIdx() in match)
            if begin_atom_label != end_atom_label:
                if i == 0: #(bond.GetBeginAtomIdx() in match or bond.GetEndAtomIdx() in match) and i == 0:
                    boundary_POI_bond = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
                    break
                elif i == 1: # (bond.GetBeginAtomIdx() in match or bond.GetEndAtomIdx() in match) and i == 1:
                    boundary_E3_bond = (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
                    break
                else:
                    raise ValueError(f'Problem with substructure matches')
    
    if boundary_POI_bond == -1 or boundary_E3_bond == -1:
        display(Chem.MolFromSmiles(protac_smiles))
        display(Chem.MolFromSmiles(poi_smile))
        display(Chem.MolFromSmiles(e3_smile))
        print(f'boundary_POI_bond: {boundary_POI_bond}. boundary_E3_bond: {boundary_E3_bond}')
        raise ValueError("Failed to assign boundary index")
    
    return boundary_POI_bond, boundary_E3_bond


def get_bond_labels(splittable_bonds, boundary_bonds, poi_label=1, e3_label = 2, linker_label = 0):
    #choose labels with the forward method architecture in mind. If cosine angle => both may need to be equal to 1, negative values I guess will give an "unstable" prediction if it isnt perfectly confident. If feed the pair of nodes to a neural network, then I can choose anything

    #bond_labels = [torch.zeros(len(splittable_bonds_i), dtype=torch.int64)+linker_label for splittable_bonds_i in splittable_bonds]
    bond_labels = [[linker_label]*len(splittable_bonds_i) for splittable_bonds_i in splittable_bonds]

    for protac_idx, (splittable_bonds_protac, boundary_bonds_protac) in enumerate(zip(splittable_bonds, boundary_bonds)):
        poi_bond = boundary_bonds_protac[0]
        poi_bond_mirrored = (poi_bond[1], poi_bond[0])
        e3_bond = boundary_bonds_protac[1]
        e3_bond_mirrored = (e3_bond[1], e3_bond[0])

        for bond_idx, splittable_bond in enumerate(splittable_bonds_protac):
            if poi_bond == splittable_bond or poi_bond_mirrored == splittable_bond:
                bond_labels[protac_idx][bond_idx] = poi_label
            if e3_bond == splittable_bond or e3_bond_mirrored == splittable_bond:
                bond_labels[protac_idx][bond_idx] = e3_label

    return bond_labels


def get_node_labels(protac_smiles, poi_smile, e3_smile):        #Returns np.array
    idx = boundary_ligand_nodes_v2(protac_smiles, poi_smile, e3_smile)
    boundary_POI_node_index, boundary_E3_node_index = idx
    num_atoms = Chem.MolFromSmiles(protac_smiles).GetNumAtoms()
    #node_labels = np.zeros((num_atoms, 1)).astype(np.int64)
    node_labels = torch.ones([num_atoms], dtype=torch.int64)
    POI_LABEL = 0
    E3_LABEL = 2
    node_labels[boundary_POI_node_index] = POI_LABEL
    node_labels[boundary_E3_node_index] = E3_LABEL
    return node_labels

def get_substructure_smiles_function(protac_smiles, class_predictions, boundary_bonds = None):
    #SMILES -> Graph
    #Graph + Class predictions -> Substructure graphs
    #Graph -> SMILES
    protac_mol = Chem.MolFromSmiles(protac_smiles)
    protac_graph = mol_to_simple_graph(protac_mol)
    #add a [*:1] to each atom in the 
    substructure_smiles = []
    substructure_smiles_with_attatchmentpoints = []
    node_indices_substructure = [[], [], []]
    class_predictions_list = class_predictions.tolist()
    for substructure_idx in [0, 1, 2]:
        
        #OBS! Alternative strategy that may work
        #add 
        #substructure_smiles = Chem.MolFragmentToSmiles(mol, [2,3,4,5,6,7], kekuleSmiles=True)
        
        #substructure_mol_validHs = Chem.RemoveHs(Chem.AddHs(substructure_mol))
        #substructure_smiles.append(Chem.MolToSmiles(substructure_mol_validHs))





        # -------------
        node_indices_substructure[substructure_idx] = [i for i, x in enumerate(class_predictions_list) if x == substructure_idx]
        substructure_graph = protac_graph.subgraph(node_indices_substructure[substructure_idx])

        # Create a mapping from old indices (from the original graph) to new indices (for the RDKit molecule)
        index_mapping = {old_index: new_index for new_index, old_index in enumerate(substructure_graph.nodes())}

        substructure_mol = graph_to_mol(substructure_graph, index_mapping)  #OBS! Need to fix graph_to_mol()
        #add hydrogens if necessary! Sometimes kekule error, due to poor electron something
        substructure_mol_validHs = Chem.RemoveHs(Chem.AddHs(substructure_mol))
        substructure_smiles.append(Chem.MolToSmiles(substructure_mol_validHs))

        if boundary_bonds is not None:
            #get poi boundary atom indices (old indices)
            for boundary_bond in boundary_bonds:
                #get bond order of poi boundary bond from mol
                bond = protac_mol.GetBondBetweenAtoms(boundary_bond[0], boundary_bond[1])
                bondtype = bond.GetBondType()
                
                #if an old index exist in index_mapping
                if boundary_bond[0] in index_mapping:
                    #convert old index into new index
                    new_boundary_atom_index = index_mapping[boundary_bond[0]]
                elif boundary_bond[1] in index_mapping:
                    new_boundary_atom_index = index_mapping[boundary_bond[1]]
                else:
                    raise ValueError("boundary bond atoms not connected to boundary bond? Probably due to index error")


                    #add dummy atom [*:1] to graph
                    #get index of this dummy atom
                    #bind the boundary atom in the POI, using its new index, with the dummy atom. 
                        #use the same bond order as the original protac
                

        
        
    return substructure_smiles

def get_substructure_smiles_function_v2(protac_smiles, class_predictions, boundary_bonds = None):
    #OLD:
    #SMILES -> Graph
    #Graph + Class predictions -> Substructure graphs
    #Graph -> SMILES

    #NEW:
    #SMILES -> Mol
    #Mol -> editable mol + add dummyatoms -> Mol
    #Mol + classpredictions -> Fragments (Mol)
    #Mol + classpredictions + dummyatoms -> Fragments (Mol)
    #Mol -> SMILES
    

    protac_mol = Chem.MolFromSmiles(protac_smiles)

    #Get substructures without attachmentpoint
    substructures_smi = {}
    class_predictions_list = class_predictions.tolist()
    for substructure_idx, substructure_name in zip([0, 1, 2], ["POI SMILES", "LINKER SMILES", "E3 SMILES"]):
        node_indices_substructure = [i for i, x in enumerate(class_predictions_list) if x == substructure_idx]
        substructure_smi = Chem.MolFragmentToSmiles(protac_mol, node_indices_substructure, kekuleSmiles=True)
        substructures_smi[substructure_name] = substructure_smi
        

    #Get substructures with attachmentpoint
    
    #mol -> editable mol
    #add dummy atoms, store their atom idx
    #add bonds between dummy atoms and boundary bonds
    #remove boundary bonds
    #editable mol -> mol => mol with 3 unconnected parts
    #convert into smiles
        #split smiles by "." to get their substructures
    #identify substructures by dummy atoms

    editable_protac_mol = Chem.EditableMol(protac_mol)
    
    #add dummy atoms
    poi_dummyatom = Chem.Atom(0)
    poi_dummyatom.SetAtomMapNum(mapno=1)
    dummy_atom_idx1 = editable_protac_mol.AddAtom(poi_dummyatom) #for POI and linker
    dummy_atom_idx2 = editable_protac_mol.AddAtom(poi_dummyatom)

    e3_dummyatom = Chem.Atom(0)
    e3_dummyatom.SetAtomMapNum(mapno=2)
    dummy_atom_idx3 = editable_protac_mol.AddAtom(e3_dummyatom) #for E3 and linker
    dummy_atom_idx4 = editable_protac_mol.AddAtom(e3_dummyatom)
    
    dummy_atoms_indices_tuples = ((dummy_atom_idx1, dummy_atom_idx2), (dummy_atom_idx3, dummy_atom_idx4)) #structured in the same way as boundary_bonds 
    for boundary_bond, boundary_dummy_atoms_indices in zip(boundary_bonds, dummy_atoms_indices_tuples):
        bond = protac_mol.GetBondBetweenAtoms(boundary_bond[0], boundary_bond[1])
        bondtype = bond.GetBondType()
        for boundary_atom_idx, dummy_atom_idx in zip(boundary_bond, boundary_dummy_atoms_indices):
            if boundary_atom_idx == dummy_atom_idx:
                print(boundary_atom_idx)
                raise ValueError("Same atom idex of boundary and dummy atom. WHy?")
            else:
                editable_protac_mol.AddBond(beginAtomIdx = boundary_atom_idx, endAtomIdx = dummy_atom_idx, order =  bondtype)
        editable_protac_mol.RemoveBond(boundary_bond[0], boundary_bond[1])
        
    protac_mol_with_dummyatoms = editable_protac_mol.GetMol()
    split_protac_smi_with_dummyatoms = Chem.MolToSmiles(protac_mol_with_dummyatoms)
    substructure_smiles_with_dummyatoms = split_protac_smi_with_dummyatoms.split(".")
    
    substructure_smiles_with_dummyatoms_dict = {}
    for smi in substructure_smiles_with_dummyatoms:
        if "[*:2]" in smi:
            if "[*:1]" in smi:
                substructure_smiles_with_dummyatoms_dict["LINKER SMILES"] = smi
            else:
                substructure_smiles_with_dummyatoms_dict["E3 SMILES"] = smi
        else:
            substructure_smiles_with_dummyatoms_dict["POI SMILES"] = smi


    
    return substructures_smi, substructure_smiles_with_dummyatoms_dict

def one_hot_encode_boundary_nodes(protac_smiles, substructure_smiles):
    if isinstance(protac_smiles, list):
        protac_smiles = protac_smiles[0]
    if isinstance(substructure_smiles[0], list):
        substructure_smiles = substructure_smiles[0]
    boundary_POI_node_index, boundary_E3_node_index = boundary_ligand_nodes(protac_smiles=protac_smiles, substructure_smiles=substructure_smiles)

    #print(f'boundary_POI_node_index: {boundary_POI_node_index}')
    #print(f'boundary_E3_node_index: {boundary_E3_node_index}')
    mol = Chem.MolFromSmiles(protac_smiles)
    num_atoms = mol.GetNumAtoms()
    one_hot = torch.zeros((num_atoms, 3))
    #print(f'one_hot_zeros size: {one_hot.size()}')

    for i in range(num_atoms):
        if i in boundary_POI_node_index:
            one_hot[i][0] = 1
        elif i in boundary_E3_node_index:
            one_hot[i][2] = 1
        else: 
            one_hot[i][1] = 1 

    return one_hot


def normalize(vector, vector_max=None, vector_min=None):
    if vector_max is None:
        vector_max = vector.max()
    if vector_min is None:
        vector_min = vector.min()
    vector_norm = (vector-vector_min)/(vector_max-vector_min)
    return vector_norm

# Function to create the line graph G_e from graph G
def line_graph_custom(G):                                             #use nx.line_graph instead?
    # Initialize the line graph G_e
    G_e = nx.Graph()

    # Create a mapping of edges to nodes for G_e
    edge_to_node = {edge: idx for idx, edge in enumerate(G.edges())}

    # Add nodes to G_e, each node represents an edge in G
    for edge, node in edge_to_node.items():
        G_e.add_node(node, edge=edge)

    # Iterate through each edge in G and connect the corresponding nodes in G_e
    for edge in G.edges():
        # Find the node in G_e that represents this edge
        node_for_edge = edge_to_node[edge]
        # Find adjacent edges in G, and create edges between nodes in G_e
        for adj_edge in G.edges(edge[0]):
            if adj_edge != edge:
                # Ensure that we have the edge in the right order for lookup
                adj_edge = tuple(sorted(adj_edge))
                adj_node = edge_to_node[adj_edge]
                G_e.add_edge(node_for_edge, adj_node)
        for adj_edge in G.edges(edge[1]):
            if adj_edge != edge:
                # Ensure that we have the edge in the right order for lookup
                adj_edge = tuple(sorted(adj_edge))
                adj_node = edge_to_node[adj_edge]
                G_e.add_edge(node_for_edge, adj_node)

    G_e.remove_edges_from(nx.selfloop_edges(G_e))
    return G_e

def assign_pos_to_graph(smile, G):
    mol = Chem.MolFromSmiles(smile)
    AllChem.Compute2DCoords(mol)
    # Assign 2D coordinates to nodes
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        G.nodes[atom.GetIdx()]['pos'] = (pos.x, pos.y)
    pos = nx.get_node_attributes(G, 'pos')
    return G, pos


#def identify_correct_substructure_match()


### Gobal Descriptors
#Can be transformed to node descriptors via SubGraphLoop

#Established descriptors

def bonacich_custom(G, beta=0.1):
    adj_matrix = nx.adjacency_matrix(G).toarray()
    n = adj_matrix.shape[0]
    x = np.ones((n, 1))
    bonacich_values = np.linalg.inv(np.eye(n) - beta * adj_matrix.T) @ (x) # C = (I−βA)^−1 * I
    bonacich_values = bonacich_values.flatten()
    return bonacich_values

def betweenness_custom(G):
    betweenness = nx.betweenness_centrality(G)
    return betweenness

def eigenvector_custom(G):
    eigenvector = nx.eigenvector_centrality(G, max_iter=100000)
    return eigenvector

def katz_custom(G):
    eigenvector = nx.eigenvector_centrality(G, max_iter=100000)
    eigenvector_values = np.array([eigenvector[node] for node in G.nodes()])
    max_eigenvector = eigenvector_values.max()
    alpha = 1/(9*max_eigenvector) # α is a constant that determines the attenuation factor for distant nodes. It should be set so that it's smaller than the reciprocal of the largest eigenvalue of the adjacency matrix for the centrality scores to be meaningful.
    beta  = 0.1
    adj_matrix = nx.adjacency_matrix(G)
    I = np.identity(adj_matrix.shape[0])
    katz = np.dot(inv(I - alpha * adj_matrix), beta * np.ones(adj_matrix.shape[0]))   #Works better than nx.katz_centrality(G, alpha=0.9, beta=1), np.array([katz[node] for node in G.nodes()]), as this may fail to converge
    return katz



# ----------------------------------------------


# Global -> node pair descriptors

#wiener_matrix gives a value for each pair of nodes
def general_wiener_matrix(G):
    nodes = list(G.nodes())
    wiener_matrix = np.zeros((len(nodes), len(nodes)))
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i != j:
                wiener_matrix[i, j] = nx.shortest_path_length(G, node_i, node_j)
    return wiener_matrix

def general_modified_wiener_matrix(G):      
    wiener_matrix = general_wiener_matrix(G)
    modified_wiener_matrix = np.where(wiener_matrix != 0, 1 / wiener_matrix, 0) # Construct modified Wiener matrix by taking the reciprocal of the non-diagonal elements
    np.fill_diagonal(modified_wiener_matrix, 0)  # Ensure the diagonal is zero
    return modified_wiener_matrix


# ----------------------------------------------



# Personal descriptors

def ovality(G):
    num_nodes = G.number_of_nodes()  # Count the number of nodes
    diameter = nx.diameter(G)       # Get the diameter of the graph
    minor_axis_length = num_nodes/diameter
    return minor_axis_length

def eccentricity(G):
    num_nodes = G.number_of_nodes()  # Count the number of nodes
    diameter = nx.diameter(G)       # Get the diameter of the graph
    eccent = (1-num_nodes/(diameter**2))**0.5
    return eccent

def major_minor_axis_ratio(G):
    num_nodes = G.number_of_nodes()  # Count the number of nodes
    diameter = nx.diameter(G)       # Get the diameter of the graph
    major_minor_axis_ratio = num_nodes/(diameter**2)
    return major_minor_axis_ratio

def general_wiener_vector(G):                                                #Doesn seem informative
    return general_wiener_matrix(G).sum(axis=0)



def wiener_index(G):
    path_length_dict = dict(nx.all_pairs_shortest_path_length(G))
    wiener_index = sum(sum(d for d in length.values()) for length in path_length_dict.values()) / 2
    return wiener_index

def hyper_wiener_index(G):
    path_length_dict = dict(nx.all_pairs_shortest_path_length(G))
    hyper_wiener_index = sum(sum(d + d**2 for d in length.values()) for length in path_length_dict.values()) / 2
    return hyper_wiener_index

def general_modified_wiener_index(G): # Function to calculate modified Wiener index for general graphs
    modified_wiener_matrix = general_modified_wiener_matrix(G)
    modified_wiener_index = np.sum(modified_wiener_matrix) / 2 # Modified Wiener index is the sum of all elements since matrix is already reciprocal for non-diagonal elements
    return modified_wiener_index

def average_shortest_path_custom(G):
    avg = nx.average_shortest_path_length(G)
    num_nodes_subgraph = G.number_of_nodes()
    normalized_avg = avg/num_nodes_subgraph
    return normalized_avg



def subgraphLoop(G, radius_scan=None, rad_diam_proportion=2.5, custom_function=None, powerPreNorm=1, powerPostNorm=1, transformation=None):
    diameter = nx.diameter(G)   #Automatic and custom radius of the subgraphs that will be generated
    if radius_scan is None:
        radius_scan = diameter // rad_diam_proportion   #Automatically calc radius of subgraph based on diameter
            
    node_dict = {}              #Initialize a dictionary
    for node_index in G.nodes():
        node_dict[node_index] = 0
            
    for node_index in G.nodes():                                 #Loop over all nodes & create subgraphs
        subgraph = nx.ego_graph(G, node_index, radius=radius_scan) #Create subgraph at node, with radius r
                
        #Calculations
        output_custom_function = custom_function(subgraph) #All custom functions must only take 1 nx.graph as input
                
                #Assign values corresponding to the node(s) in the complete graph from the calculations
        if isinstance(output_custom_function, (int, float)):                    # Global metric will return 1 value
            node_dict[node_index]=custom_function(subgraph)                  #Apply custom function to subgraph, and store with the node index which the graph is centered at
        else:                                                                #Local metric. Transfer it from all nodes in subgraph to complete graph
            for subgraph_node_index in subgraph.nodes():
                node_dict[subgraph_node_index] += output_custom_function[subgraph_node_index]#output_custom_function.get(subgraph_node_index, 0) #The output is from a networkx function.  # Default to 0 if not found

    #Data processing
    Array = np.array([value for _, value in node_dict.items()])
    if transformation is not None:
        Array = transformation(Array)
    Array = np.power(Array, powerPreNorm)
    Array_norm = normalize(Array)
    Array_norm = np.power(Array_norm, powerPostNorm)
    return Array_norm


def graph_descriptor(G, descriptor=None):
    if descriptor == 'degree':
        degrees = np.array([d for n, d in G.degree()])
        calc_values = normalize(vector=degrees, vector_max=4, vector_min=1)
    elif descriptor == 'betweenness':
        betweenness = nx.betweenness_centrality(G)
        calc_values = np.array([betweenness[node] for node in G.nodes()])
    elif descriptor == 'eigenvector':
        eigenvector = nx.eigenvector_centrality(G, max_iter=100000)
        calc_values = np.array([eigenvector[node] for node in G.nodes()])
    elif descriptor == 'closeness':
        closeness = nx.closeness_centrality(G)
        calc_values = np.array([closeness[node] for node in G.nodes()])
    elif descriptor == 'bonacich':
        calc_values = bonacich_custom(G)
    elif descriptor == 'katz':
        calc_values = katz_custom(G)
    elif descriptor == 'local_eigenvectors_x':
        calc_values = subgraphLoop(G, custom_function=eigenvector_custom, powerPreNorm=0.1)
    elif descriptor == 'local_betweenness_5':
        calc_values = subgraphLoop(G, custom_function=betweenness_custom, radius_scan=5) 
    elif descriptor == 'local_normAvgShortestPath_5':
        calc_values = subgraphLoop(G, custom_function=average_shortest_path_custom, radius_scan=5)    
    elif descriptor == 'local_prod_betweenness_avgShortestPath_4':
        betweenness = nx.betweenness_centrality(G)
        betweenness_values = np.array([betweenness[node] for node in G.nodes()])
        avg_values = subgraphLoop(G, custom_function=average_shortest_path_custom, radius_scan=4, powerPostNorm=1/3)
        calc_values = normalize(betweenness_values*avg_values)
    else:
        raise ValueError(f'Give a valid descriptor:"{descriptor}"')
    calc_values = normalize(calc_values)
    return calc_values


def mol_to_simple_graph(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())    
    return G

from networkx import is_connected



def identify_bridge_nodes(G):
    bool_list = []
    bridge_nodes_idx = []

    #G = mol_to_simple_graph(mol)
    
    for idx in range(G.number_of_nodes()):
        G_temp = G.copy()
        G_temp.remove_node(idx)
        bridge_node = 1 - int(is_connected(G_temp))
        bool_list.append(bridge_node)
        if bridge_node:
            bridge_nodes_idx.append(idx)

    return bool_list, bridge_nodes_idx


def identify_bridge_nodes_and_reject_small_splits(G, smallest_allowed_subgraph_size=9):
    bool_list = []
    bridge_nodes_idx = []

    #G = mol_to_simple_graph(mol)
    
    for idx in range(G.number_of_nodes()):
        G_temp = G.copy()
        G_temp.remove_node(idx)
        bridge_node = 1 - int(is_connected(G_temp))

        allowed_subgraphs = 0
        if bridge_node:

            subgraphsizes = [len(c) for c in sorted(nx.connected_components(G_temp), key=len, reverse=True)]
            allowed_subgraphs = int(all(i >= smallest_allowed_subgraph_size for i in subgraphsizes))
        bridge_and_allowed = bridge_node and allowed_subgraphs
           
        
        bool_list.append(bridge_and_allowed)
        if bridge_and_allowed:
            bridge_nodes_idx.append(idx)
            

    return bool_list, bridge_nodes_idx

def identify_murcko_scaffold_atoms(mol):
    #only input protacs.
    ms_bool_list = []
    mol_ms = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
    match = mol.GetSubstructMatch(mol_ms)
    at_idx_list = list(match)
    for idx in range(mol.GetNumAtoms()):
        if idx in at_idx_list:
            ms_bool_list.append(1)
        else:
            ms_bool_list.append(0)
    return ms_bool_list, at_idx_list

def make_graph_with_pos(smile):
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

def identify_legal_nodes(mol, smallest_allowed_subgraph_size=9):
    smi = Chem.MolToSmiles(mol, canonical=True)
    G, pos = make_graph_with_pos(smi)
    bridge_bool_list, highlighted_nodes = identify_bridge_nodes(G) #identify_bridge_nodes_and_reject_small_splits(G, smallest_allowed_subgraph_size)
    ms_bool_list, at_idx_list = identify_murcko_scaffold_atoms(mol)
    bool_combined_list = []
    colors = []
    for b1, b2 in zip(bridge_bool_list, ms_bool_list):
        if b1*b2:
            bool_combined_list.append(1)
            colors.append('red')
        else:
            bool_combined_list.append(0)
            colors.append('blue')

    at_idx_combined_list = [idx for idx, bool in enumerate(bool_combined_list) if bool != 0]

    return bool_combined_list, at_idx_combined_list#, colors, pos


def remove_dummy_atoms(mol):
    atoms_to_remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    editable_mol = Chem.EditableMol(mol)
    for idx in sorted(atoms_to_remove, reverse=True):
        editable_mol.RemoveAtom(idx)
    return editable_mol.GetMol()

def identify_substructure_indices(protac_mol, poi_mol, e3_mol, protac_smile): #:return: List of atom indices that match the substructure .
    matches_poi = protac_mol.GetSubstructMatches(poi_mol)
    matches_e3 = protac_mol.GetSubstructMatches(e3_mol)

    if len(matches_poi) == 1 and len(matches_e3) == 1:
        poi_indices = [atom_idx for atom_idx in matches_poi[0]] 
        e3_indices = [atom_idx for atom_idx in matches_e3[0]]
    elif len(matches_poi) == 2 and len(matches_e3) == 2:                                                                                                        ##########################OBS Work in progress
        #assume POI = E3 

        #case 1: equal attatchment points to linker

        #case 2: unequal attachment points to linker

        e3_mol_no_attatchment_point = remove_dummy_atoms(e3_mol)
        poi_mol_no_attatchment_point = remove_dummy_atoms(poi_mol)
        poi_linker = Chem.DeleteSubstructs(protac_mol, e3_mol_no_attatchment_point)
        e3_linker = Chem.DeleteSubstructs(protac_mol, poi_mol_no_attatchment_point)

        matches_poi_linker = protac_mol.GetSubstructMatches(poi_linker)
        matches_e3_linker = protac_mol.GetSubstructMatches(e3_linker)

        #linker_indicies_from_intersection = list(set(matches_poi_linker).intersection(matches_e3_linker))

        
        poi_indices = matches_poi_linker
        #poi_indices.remove(linker_indicies_from_intersection)
        e3_indices = matches_e3_linker
        #e3_indices.remove(linker_indicies_from_intersection)
        #print(protac_smile)
        raise ValueError(f'2 substructure matches for protac: {protac_smile}')

    else:
        #print(protac_smile)
        raise ValueError(f'Complex substructure match for protac: {protac_smile}')


    # Calculate Linker indices by excluding POI and E3 indices
    all_indices = set(range(protac_mol.GetNumAtoms()))
    linker_indices = all_indices - set(poi_indices) - set(e3_indices)

    return list(set(poi_indices)), list(set(linker_indices)), list(set(e3_indices))   # Flatten the tuple of matches and return unique indices 

def create_protac_substructure_mapping(protac_smile, poi_smile, e3_smile):
    """
    Creates a mapping of atom indices from the original PROTAC molecule to their
    corresponding substructures (POI, E3, and Linker).
    :return: Dictionary with keys 'POI', 'E3', and 'Linker', each containing a list of atom indices.
    """
    protac_mol = Chem.MolFromSmiles(protac_smile)
    poi_mol = remove_dummy_atoms(Chem.MolFromSmiles(poi_smile))
    e3_mol = remove_dummy_atoms(Chem.MolFromSmiles(e3_smile))

    poi_indices, linker_indices, e3_indices = identify_substructure_indices(protac_mol, poi_mol, e3_mol, protac_smile)

    # Create the mapping
    substructure_mapping = {
        'POI': list(poi_indices),
        'Linker': list(linker_indices),
        'E3': list(e3_indices)}
    
    return substructure_mapping

def one_hot_encode_substructure_labels(protac_smile, poi_smile, e3_smile):
    substructure_mapping = create_protac_substructure_mapping(protac_smile, poi_smile, e3_smile) # Get dictionary of which atom ids belong to which substructure
    num_nodes = len(substructure_mapping['POI']) + len(substructure_mapping['Linker']) + len(substructure_mapping['E3'])
    one_hot_encoded = np.zeros((num_nodes, 3), dtype=int) # Initialize an array of zeros
    
    # Update the array based on the substructure mapping
    for col_idx, key in enumerate(['POI', 'Linker', 'E3']):
        indices = substructure_mapping[key]
        one_hot_encoded[indices, col_idx] = 1

    return one_hot_encoded

# Function to convert a molecule to a graph
def mol_to_graph(mol, graph_descriptor_list, transform=False):
    AllChem.ComputeGasteigerCharges(mol)  # Compute partial charges

    # Electronegativity lookup table (Pauling scale)
    electronegativity = {
                        'H': 2.20, 
                        'C': 2.55, 
                        'N': 3.04, 
                        'O': 3.44, 
                        'F': 3.98, 
                        'P': 2.19, 
                        'S': 2.58, 
                        'Cl': 3.16,
                        'Br': 2.96,  
                        'I': 2.66    
                        }

    # Assuming hydrophobic atoms (simple heuristic)
    hydrophobic_atoms = ['C', 'H']

    G = nx.Graph()
    for atom in mol.GetAtoms():
        # Partial Charge
        partial_charge = atom.GetProp('_GasteigerCharge')

        # Electronegativity
        atom_en = electronegativity.get(atom.GetSymbol(), None)

        # Hydrophobicity/Hydrophilicity
        hydrophobicity = atom.GetSymbol() in hydrophobic_atoms

        # Hydrogen Bond Donor/Acceptor (simple heuristic)
        # This is a simplistic approach and may not be accurate for all cases
        h_bond_donor = atom.GetSymbol() in ['N', 'O'] and atom.GetTotalNumHs() > 0
        h_bond_acceptor = atom.GetSymbol() in ['N', 'O']


        """ OBS! May Want to minimize the number of features to prevent model to memorize them"""
#[[6, 1, 2, 0, False, 2, False, rdkit.Chem.rdchem.HybridizationType.SP2, '-0.093589918573147773', 2.55, True, False, False]
        G.add_node(atom.GetIdx(),    
               atomic_num=atom.GetAtomicNum(),
               degree=atom.GetDegree(),
               explicit_bonds=atom.GetExplicitValence(),
               formal_charge=atom.GetFormalCharge(),
               aromatic=atom.GetIsAromatic()*1,
               hydrogens=atom.GetNumImplicitHs(),
               ring=atom.IsInRing()*1,
               #hybridization=atom.GetHybridization(),
               partial_charge=float(partial_charge),
               electronegativity=atom_en,
               #hydrophobicity=hydrophobicity*1,
               h_bond_donor=h_bond_donor*1,
               h_bond_acceptor=h_bond_acceptor*1
              )
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType(),
                   is_conjugated=bond.GetIsConjugated(),
                   in_ring=bond.IsInRing())                         ##If adding edge features here, also, expand smiles_to_data to unpack these from the graph.

    #All nodes and edges must be defined before graph descriptors can be calculated. Start calculating them here:
    graph_descriptors = graph_descriptor_list
    for descriptor_str in (graph_descriptors):
        graph_descriptor_dict = {}
        descriptor_value = graph_descriptor(G, descriptor=descriptor_str)
        for node in G.nodes():
            graph_descriptor_dict[node] = descriptor_value[node]
        nx.set_node_attributes(G, graph_descriptor_dict, descriptor_str)

    if transform is True:
        #Implement the line_graph function here. Think about downstream
        pass
    
    Graph = G
    return Graph

def mol_to_simple_graph(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),    
               atomic_num=atom.GetAtomicNum(),
               formal_charge=atom.GetFormalCharge(),
                aromatic=atom.GetIsAromatic()
              )
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType()
                   )                       
    return G

def graph_to_mol(G, index_mapping={}):
    # Create an empty editable molecule
    new_mol = Chem.RWMol()

    # Add atoms to the molecule
    for node, attr in G.nodes(data=True):
        atom = Chem.Atom(attr['atomic_num'])
        atom.SetFormalCharge(attr['formal_charge'])
        atom.SetIsAromatic(attr['aromatic'])
        new_mol.AddAtom(atom)

    # Add bonds to the molecule
    try:
        for start, end, attr in G.edges(data=True):
            bond_type = attr['bond_type']
            if index_mapping != {}:
                start = index_mapping[start]  #If
                end = index_mapping[end]
            new_mol.AddBond(start, end, bond_type)
    except:
        print(bond_type)
        print(start)
        print(end)
        pass

    # Convert to a standard RDKit molecule and return
    mol = new_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def smiles_to_data(protac_smile, substructure_smiles, graph_descriptor_list):
    mol = Chem.MolFromSmiles(protac_smile)
    G = mol_to_graph(mol, graph_descriptor_list)

    # Extract node features
    #num_nodes = len(G)
    #node_labels_initial_value = 0
    
    node_feature_raw = [list(G.nodes[node].values()) for node in G.nodes]
    node_features = torch.tensor(node_feature_raw, dtype=torch.float)

    boundary_edges = EdgeLabler(protac_smile, substructure_smiles)
    #boundary_edges_set = set(boundary_edges)

    # Extract edge indices and features
    edge_indices = []
    edge_labels = []
    edge_features = []
    for edge in G.edges(data=True):
        start, end = edge[0], edge[1]

        if (start, end) in boundary_edges:
            edge_labels.append(1)
        elif (end, start) in boundary_edges:
            edge_labels.append(1)
        else: 
            edge_labels.append(0)
        
        edge_indices.append((start, end))
        edge_indices.append((end, start))  # since it's an undirected graph
                        
        # Bond type one-hot encoding
        edge_feature_dict = edge[2]
        
        edge_feature = [
            int(edge_feature_dict['bond_type'] == Chem.rdchem.BondType.SINGLE),
            int(edge_feature_dict['bond_type'] == Chem.rdchem.BondType.DOUBLE),
            int(edge_feature_dict['bond_type'] == Chem.rdchem.BondType.TRIPLE),
            int(edge_feature_dict['bond_type'] == Chem.rdchem.BondType.AROMATIC),
            int(edge_feature_dict['is_conjugated']),
            int(edge_feature_dict['in_ring'])
            ]
        edge_features.extend([edge_feature, edge_feature])  # add twice for both directions

    edge_indices = torch.tensor(edge_indices).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)  

    poi_smile, _, e3_smile = substructure_split_sort(substructure_smiles)
    node_substructure_label_np = one_hot_encode_substructure_labels(protac_smile, poi_smile, e3_smile) #Direct node classification of all atoms of which substructure it belongs to    
    node_substructure_label = torch.from_numpy(node_substructure_label_np)

    assign_pos_to_graph(protac_smile, G)


    bool_combined_list, at_idx_combined_list = identify_legal_nodes(mol,smallest_allowed_subgraph_size=10)

    bool_combined_tensor = torch.tensor(bool_combined_list).reshape(-1, 1)
    

    data = Data(x=node_features, 
                edge_attr=edge_features,
                edge_labels=edge_labels,
                edge_index=edge_indices,
                node_substructure_label=node_substructure_label, 
                smiles=protac_smile,
                substructure_smiles=substructure_smiles,      
                G=G,
                legal_nodes=bool_combined_tensor
                ) 
    return data #, mol, colors, G, pos



def smiles_to_data_for_prediction(protac_smile, graph_descriptor_list, annotation=''):
    protac_smile = remove_stereo(protac_smile)
    mol = Chem.MolFromSmiles(protac_smile)
    G = mol_to_graph(mol, graph_descriptor_list)

    # Extract node features
    #num_nodes = len(G)
    #node_labels_initial_value = 0
    
    node_feature_raw = [list(G.nodes[node].values()) for node in G.nodes]
    node_features = torch.tensor(node_feature_raw, dtype=torch.float)

    #boundary_edges_set = set(boundary_edges)

    # Extract edge indices and features
    edge_indices = []

    edge_features = []
    for edge in G.edges(data=True):
        start, end = edge[0], edge[1]

        edge_indices.append((start, end))
        edge_indices.append((end, start))  # since it's an undirected graph
                        
        # Bond type one-hot encoding
        edge_feature_dict = edge[2]
        
        edge_feature = [
            int(edge_feature_dict['bond_type'] == Chem.rdchem.BondType.SINGLE),
            int(edge_feature_dict['bond_type'] == Chem.rdchem.BondType.DOUBLE),
            int(edge_feature_dict['bond_type'] == Chem.rdchem.BondType.TRIPLE),
            int(edge_feature_dict['bond_type'] == Chem.rdchem.BondType.AROMATIC),
            int(edge_feature_dict['is_conjugated']),
            int(edge_feature_dict['in_ring'])
            ]
        edge_features.extend([edge_feature, edge_feature])  # add twice for both directions

    edge_indices = torch.tensor(edge_indices).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)


    assign_pos_to_graph(protac_smile, G)


    bool_combined_list, at_idx_combined_list = identify_legal_nodes(mol,smallest_allowed_subgraph_size=10)

    bool_combined_tensor = torch.tensor(bool_combined_list).reshape(-1, 1)
    

    data = Data(x=node_features, 
                edge_attr=edge_features,
                edge_index=edge_indices,
                smiles=protac_smile,     
                G=G,
                legal_nodes=bool_combined_tensor,
                annotation=annotation
                ) 
    return data 



#Further criteria for legal nodes: The the POI-L node and Linker nodes can't be a part of the same ring => Calculate the resulting nodes for the linker, POI and E3. ...
# ... See if any ring has nodes from two classes. If a ring has nodes from Linker and E3, then the E3-L choice was poor. If the ring has nodes from Linker and POI, the POI-L choice was poor.
#def will_boundary_nodes_split_rings(mol, poi_L_idx, e3_l_idx):
#    ring_info = mol.GetRingInfo()
#    ring_idx = []
#    for atom in mol.GetAtoms():
#        if ring_info.IsAtomInRing(atom.GetIdx()) and (atom.GetIdx() != poi_L_idx) and (atom.GetIdx() != e3_l_idx):
#            ring_idx.append(atom.GetIdx())


def identify_bad_substructure_match(protac_smile, poi_smile, e3_smile): #:return: List of atom indices that match the substructure .
    #poi_smile, _, e3_smile = substructure_split_sort(substructure_smiles)
    protac_mol = Chem.MolFromSmiles(protac_smile)
    poi_mol = Chem.MolFromSmiles(poi_smile)
    e3_mol = Chem.MolFromSmiles(e3_smile)

    poi_mol = remove_dummy_atoms(poi_mol)
    e3_mol = remove_dummy_atoms(e3_mol)

    matches_poi = protac_mol.GetSubstructMatches(poi_mol)
    matches_e3 = protac_mol.GetSubstructMatches(e3_mol)

    if len(matches_poi) == 1 and len(matches_e3) == 1:
        return False
    elif len(matches_poi) == 2 and len(matches_e3) == 2:                                                                                                        ##########################OBS Work in progress
        return True
    else:
        return True
    
def remove_multiple_substrucmathes(test_set, p_column, poi_column, e3_column):
    test_set = test_set.copy()
    
    bad_substructure_match_idx = []
    for idx, row in test_set.iterrows():
        protac_smile = row[p_column]
        poi_smile = row[poi_column]
        e3_smile = row[e3_column]

        bad_match = identify_bad_substructure_match(protac_smile, poi_smile, e3_smile)

        if bad_match:
            bad_substructure_match_idx.append(idx)

    test_set.drop(bad_substructure_match_idx, inplace=True)
    test_set.reset_index(drop=True, inplace=True)
    return test_set

def prepare_data_set(test_set, p_column, poi_column, linker_column, e3_column):

    test_set['substructures'] = test_set.apply(lambda row: '.'.join([str(row[poi_column]), str(row[linker_column]), str(row[e3_column])]), axis=1)

    test_set = remove_multiple_substrucmathes(test_set, p_column, poi_column, e3_column) 

    test_set_substructures = test_set[['substructures']].copy()

    test_set_protacs = test_set[[p_column]].copy()
    test_set_protacs.rename(columns={p_column: 'Smiles'}, inplace=True)

    return test_set_protacs, test_set_substructures




#def find_atom_index_of_mapped_atoms(mol):
#    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 1 or atom.GetAtomMapNum() == 2]

def find_atom_index_of_mapped_atoms_detailed(mol):
    poi_l_attachment_point = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 1]
    e3_l_attachment_point = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 2]

    if len(poi_l_attachment_point) > 1 or len(e3_l_attachment_point) > 1:
        raise ValueError("Too many attachement points")

    return poi_l_attachment_point, e3_l_attachment_point

def remove_non_ring_atoms(mol):
    # Create an editable copy of the molecule
    emol = Chem.EditableMol(mol)

    atoms_to_remove = [-1]

    while len(atoms_to_remove) != 0:

        # Get indices of mapped atoms
        #atom_idx_list_of_mapped_atoms = find_atom_index_of_mapped_atoms(mol)
        poi_l_attachment_point, e3_l_attachment_point = find_atom_index_of_mapped_atoms_detailed(mol)
        atom_idx_list_of_mapped_atoms = poi_l_attachment_point + e3_l_attachment_point
        
        atoms_to_remove = []
        for atom in mol.GetAtoms():
            # Check if the atom is not one of the mapped atoms
            if atom.GetIdx() not in atom_idx_list_of_mapped_atoms:
                neighbors = atom.GetNeighbors()
                
                # Check if the atom has only one neighbor
                if len(neighbors) == 1:
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
                    
                    # Check if the bond is a single bond and the neighbor is a hydrogen atom
                    if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                        atoms_to_remove.append(atom.GetIdx())

        # Remove the identified atoms
        for idx in sorted(atoms_to_remove, reverse=True):
            emol.RemoveAtom(idx)

        # Get the modified molecule and sanitize it
        mol = emol.GetMol()
        Chem.SanitizeMol(mol)


    return mol

# Example usage
#mol = Chem.MolFromSmiles('[*:1]C#CCOCC(CCC1CCC1CC(CCCC)CC)OCC(c1cccc(c1C)C)OCC(C)C(=O)[*:2]')
#display(mol)
#modified_mol = remove_linker_non_ring_atoms(mol)
#display(modified_mol)

"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ReplaceCore


def attach_rings_to_linker(mol):

    # Convert the SMILES string to an RDKit molecule
    #mol = Chem.MolFromSmiles(linker_smiles)

    # Define the ring to attach (example: benzene ring with a dummy atom)
    ring_smiles_1 = '[C:1][C:1]1=[C:1][C:1]=[C:1][C:1]=[C:1]1'#'[U:1]1=[U:1][U:1]=[U:1][U:1]=[U:1]1'  # Benzene with a dummy atom at one position
    ring_smiles_2 = '[C:2][C:2]1=[C:2][C:2]=[C:2][C:2]=[C:2]1'#'[Au:2]1=[Au:2][Au:2]=[Au:2][Au:2]=[Au:2]1'
    ring_mol_1 = Chem.MolFromSmiles(ring_smiles_1)
    ring_mol_2 = Chem.MolFromSmiles(ring_smiles_2)

    # Iterate over the atoms and find dummy atoms
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == 1:
            mol = AllChem.ReplaceSubstructs(mol, Chem.MolFromSmiles("[*:1]"), ring_mol_1, replacementConnectionPoint=0)[0]
        elif atom.GetAtomMapNum() == 2:
            mol = AllChem.ReplaceSubstructs(mol, Chem.MolFromSmiles("[*:2]"), ring_mol_2, replacementConnectionPoint=0)[0]


    # Convert the modified molecule back to a SMILES string
    #modified_smiles = Chem.MolToSmiles(mol, canonical=True)
    Chem.GetSymmSSSR(mol)  # Finding rings and re-perceiving aromaticity


    return mol



def remove_rings_from_linker(mol):
    
    "Identifies specific rings (benzene rings with a dummy atom) in the molecule and replaces them with single dummy atoms."

    "Args: linker_smiles (str): SMILES string of the linker with attached rings."

    "Returns: str: SMILES string of the linker with rings replaced by dummy atoms."

    # Convert the SMILES string to an RDKit molecule
    #mol = Chem.MolFromSmiles(linker_smiles)

    # Define the substructures to be replaced (benzene rings with a dummy atom)
    ring_substructure_1 = Chem.MolFromSmiles('[C:1][C:1]1=[C:1][C:1]=[C:1][C:1]=[C:1]1')
    ring_substructure_2 = Chem.MolFromSmiles('[C:2][C:2]1=[C:2][C:2]=[C:2][C:2]=[C:2]1')

    # Define the dummy atoms to replace the rings
    dummy_1 = Chem.MolFromSmiles('[*:1]')
    dummy_2 = Chem.MolFromSmiles('[*:2]')

    # Replace the rings with dummy atoms

    #mol = Chem.ReplaceCore(mol, ring_substructure_1, labelByIndex=True)
    mol = AllChem.ReplaceSubstructs(mol, ring_substructure_1, dummy_1, replacementConnectionPoint=0)[0]
    mol = AllChem.ReplaceSubstructs(mol, ring_substructure_2, dummy_2, replacementConnectionPoint=0)[0]

    #mol = AllChem.ReplaceSubstructs(mol, ring_substructure_2, dummy_2, replacementConnectionPoint=0)[0]
    #display(mol)


    # Convert the modified molecule back to a SMILES string
    #modified_smiles = Chem.MolToSmiles(mol, canonical=True)
    Chem.GetSymmSSSR(mol)  # Finding rings and re-perceiving aromaticity
    Chem.SanitizeMol(mol)

    return mol



# Example usage
#linker_with_dummies = "[*:1]C#CCOCC(C)OCC(c1cccc(c1C)C)OCC(C)C(=O)[*:2]"# "[*:1]C#CCOCCOCC(C1=CC=CC=C1)OCCC(=O)[*:2]"# "[*:2]C(=O)CCOCCOCCOCC#C[*:1]"  # Example linker with dummy atoms
#mol = Chem.MolFromSmiles(linker_with_dummies)
#display(mol)
#modified_linker = attach_rings_to_linker(mol)
#display(modified_linker)
#modified_linker = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(modified_linker)
#modified_linker_inversed = remove_rings_from_linker(modified_linker)
#display(modified_linker_inversed)
#display(modified_linker_inversed)
"""



def linker_mol_to_ms(mol):

    if mol.GetNumAtoms() == 2: #only [*:1] and [*:2]
        return mol

    #if "[*:1]" and "[*:2]" in smiles


    poi_l_attachment_point, e3_l_attachment_point = find_atom_index_of_mapped_atoms_detailed(mol)

    emol = Chem.EditableMol(mol)
    
    #add one single bond between the attachment points
    try:
        emol.AddBond(poi_l_attachment_point[0], e3_l_attachment_point[0], Chem.rdchem.BondType.SINGLE)
    except:
        display(mol)
        print(f'poi_l_attachment_point:{poi_l_attachment_point}')
        print(f'e3_l_attachment_point:{e3_l_attachment_point}')
        print(Chem.MolToSmiles(mol, canonical=True))
        raise ValueError("Fail add bond")


    mol_circulized = emol.GetMol()
    try:
        # Sanitize the molecule
        Chem.GetSymmSSSR(mol_circulized)  # Finding rings and re-perceiving aromaticity
        Chem.SanitizeMol(mol_circulized)
    except: 
        raise ValueError("Fail GetSymmSSSR or SanitizeMol")
    
    #apply MS
    mol_circulized_ms = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol_circulized)
    ms_poi_l_attachment_point, ms_e3_l_attachment_point = find_atom_index_of_mapped_atoms_detailed(mol_circulized_ms)
    #mol_circulized_ms.GetBondBetweenAtoms(ms_poi_l_attachment_point, ms_e3_l_attachment_point).SetBondType(Chem.rdchem.BondType.UNSPECIFIED)

    emol_circulized_ms = Chem.EditableMol(mol_circulized_ms)

    #remove the bond between the attachment points
    emol_circulized_ms.RemoveBond(ms_poi_l_attachment_point[0], ms_e3_l_attachment_point[0])

    mol_ms = emol_circulized_ms.GetMol()

    try:
        # Sanitize the molecule
        Chem.GetSymmSSSR(mol_ms)  # Finding rings and re-perceiving aromaticity
        Chem.SanitizeMol(mol_ms)
    except: 
        raise ValueError("Fail GetSymmSSSR or SanitizeMol")

    return mol_ms

"""
linker_with_dummies = "[*:1]C#CCOCC(C)OCC(c1cccc(c1C)C)OCC(C)C(=O)[*:2]"# "[*:1]C#CCOCCOCC(C1=CC=CC=C1)OCCC(=O)[*:2]"# "[*:2]C(=O)CCOCCOCCOCC#C[*:1]"  # Example linker with dummy atoms
mol = Chem.MolFromSmiles(linker_with_dummies)
display(mol)
mol_ms = linker_mol_to_ms(mol)
display(mol_ms)"""



def get_anonymous_mol(mol):
    try:
        return rdMolHash.MolHash(mol, rdMolHash.HashFunction.AnonymousGraph)
    except:
        raise ValueError(f"Error processing molecule with rdMolHash.HashFunction.AnonymousGraph")

def get_anonymous_murcko(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Handle invalid SMILES strings
        raise ValueError("mol is None")

    smi_anon = get_anonymous_mol(mol)
    mol_anon = Chem.MolFromSmiles(smi_anon)

    if "[*:1]" in smiles and "[*:2]" in smiles: #is_linker = True
        #mol_ms = linker_mol_to_ms(mol)
        mol_anon_ms = linker_mol_to_ms(mol_anon)    
    else:
        #mol_ms = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
        mol_anon_ms = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol_anon)
    
    #smi_anon_ms = get_anonymous_mol(mol_ms)
    smi_anon_ms = Chem.MolToSmiles(mol_anon_ms, canonical=True)

    return smi_anon_ms
    
def generate_anonymous_murcko_in_df(dataframe, smiles_column):
    dataframe[smiles_column + '_AnonMS'] = dataframe[smiles_column].apply(get_anonymous_murcko)
    return dataframe


def get_murcko(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # Handle invalid SMILES strings
        raise ValueError("mol is None")

    if "[*:1]" in smiles and "[*:2]" in smiles: #is_linker = True
        #mol_ms = linker_mol_to_ms(mol)
        mol_ms = linker_mol_to_ms(mol)    
    else:
        #mol_ms = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
        mol_ms = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
    
    #smi_anon_ms = get_anonymous_mol(mol_ms)
    smi_ms = Chem.MolToSmiles(mol_ms, canonical=True)

    return smi_ms


def generate_murcko_in_df(dataframe, smiles_column):
    dataframe[smiles_column + '_MS'] = dataframe[smiles_column].apply(get_murcko)
    return dataframe




def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        else:
            print(f'Smile returned error: {smiles}')
            return None
    except:
        print(f'Smile returned error: {smiles}')
        return None
        #raise ValueError(f'Failed to process smile: {smiles}')



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
        unique_substructures[group] = list(group_df[substructure_col].dropna().unique())
    return unique_substructures



def select_random_substructures(unique_substructures_dict):
    """
    Selects a random substructure from a random group in the given dictionary.

    Args:
    unique_substructures_dict (dict): A dictionary with groups as keys and lists of substructures as values.

    Returns:
    str: A random substructure.
    """
    # Select a random group

    random_group = random.choice(list(unique_substructures_dict.keys()))
    # Select a random substructure from the group

    random_substructure = random.choice(unique_substructures_dict[random_group])

    return random_substructure


def select_n_random_substructures(unique_poi_substructures, unique_linker_substructures, unique_e3_substructures, num_protacs_to_generate):

    dics_list = [unique_poi_substructures, unique_linker_substructures, unique_e3_substructures]
    poi_list = []
    linker_list = []
    e3_list = []
    """
    Selects a random substructure from a random group in the given dictionary.

    Args:
    unique_substructures_dict (dict): A dictionary with groups as keys and lists of substructures as values.

    Returns:
    str: A random substructure.
    """

    for idx, dict_substruc in enumerate(dics_list):
        for i in range(num_protacs_to_generate):
            # Select a random group

            random_group = random.choice(list(dict_substruc.keys()))
            # Select a random substructure from the group

            random_substructure = random.choice(dict_substruc[random_group])

            if idx == 0:
                poi_list.append(random_substructure)
            elif idx == 1:
                linker_list.append(random_substructure)
            elif idx == 2:
                e3_list.append(random_substructure)

    return poi_list, linker_list, e3_list


def merge_molecules(mol1, mol2, atom_idx1, atom_idx2, bond_type = 'single'):
    # Combine the two molecules into a single editable molecule
    combined_mol = Chem.CombineMols(mol1, mol2)
    editable_mol = Chem.EditableMol(combined_mol)

    # Find neighbors of the attachment points
    neighbor_atom_idx1 = [nbr.GetIdx() for nbr in mol1.GetAtomWithIdx(atom_idx1).GetNeighbors() if nbr.GetAtomicNum() > 1][0]
    neighbor_atom_idx2 = [nbr.GetIdx() + mol1.GetNumAtoms() for nbr in mol2.GetAtomWithIdx(atom_idx2).GetNeighbors() if nbr.GetAtomicNum() > 1]
    
    if neighbor_atom_idx2 == []: #if linker has no length
        smi_e3_linker_with_e3_attachment = Chem.MolToSmiles(mol1, canonical=True)
        smi_e3_linker_with_poi_attachment = smi_e3_linker_with_e3_attachment.replace("[*:2]","[*:1]")
        mol_e3_linker_with_poi_attachment = Chem.MolFromSmiles(smi_e3_linker_with_poi_attachment)
        return mol_e3_linker_with_poi_attachment
    else:
        neighbor_atom_idx2 = neighbor_atom_idx2[0]


        #raise ValueError("Index out of range?")

    # Add a bond between the neighboring atoms (ignoring the dummy atoms)

    if bond_type == 'single':
        editable_mol.AddBond(neighbor_atom_idx1, neighbor_atom_idx2, order=rdchem.BondType.SINGLE)
    elif bond_type == 'rand_uniform':
        neighbor_atom1 = mol1.GetAtomWithIdx(neighbor_atom_idx1)
        neighbor_atom2 = mol2.GetAtomWithIdx(neighbor_atom_idx2-mol1.GetNumAtoms())
        highest_allowed_bondorder_atom_idx1 = neighbor_atom1.GetTotalNumHs() + 1 # +1 for the attatchment point
        highest_allowed_bondorder_atom_idx2 = neighbor_atom2.GetTotalNumHs() + 1
        highest_allowed_bondorder = min([highest_allowed_bondorder_atom_idx1, highest_allowed_bondorder_atom_idx2])
        possible_bonds = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE]
        allowed_bonds = possible_bonds[0:highest_allowed_bondorder]
        sampled_bond = random.sample(allowed_bonds, 1)[0]
        editable_mol.AddBond(neighbor_atom_idx1, neighbor_atom_idx2, order=sampled_bond)


    # Calculate the adjusted index for the attachment point in mol2
    adjusted_atom_idx2 = atom_idx2 + mol1.GetNumAtoms()

    # Remove the dummy atoms - IMPORTANT: remove the atom with the higher index first!
    max_idx = max(atom_idx1, adjusted_atom_idx2)
    min_idx = min(atom_idx1, adjusted_atom_idx2)

    editable_mol.RemoveAtom(max_idx)
    editable_mol.RemoveAtom(min_idx)

    # Get the modified molecule
    modified_mol = editable_mol.GetMol()

    # Sanitize the molecule to ensure its chemical validity
    Chem.SanitizeMol(modified_mol)

    return modified_mol

def reassemble_protac(poi_smiles, linker_smiles, e3_smiles, bond_type):

    if "[*:1]" in e3_smiles:
        raise ValueError(f"[*:1] found among E3-SMILES: {e3_smiles}]")
    elif "[*:2]" in poi_smiles:
        raise ValueError(f"[*:2] found among POI-SMILES: {poi_smiles}]")
    elif "[*:1]" not in linker_smiles or "[*:2]" not in linker_smiles:
        raise ValueError(f"[*:1] or [*:2] missing among Linker-SMILES: {linker_smiles}]")


    # Convert SMILES to RDKit Molecule objects
    poi_mol = Chem.MolFromSmiles(poi_smiles)
    linker_mol = Chem.MolFromSmiles(linker_smiles)
    e3_mol = Chem.MolFromSmiles(e3_smiles)

    # Find the indices of the attachment points
    poi_l_attachment_points, _ = find_atom_index_of_mapped_atoms_detailed(poi_mol)
    linker_poi_attachment_points, linker_e3_attachment_points = find_atom_index_of_mapped_atoms_detailed(linker_mol)
    _, e3_l_attachment_points = find_atom_index_of_mapped_atoms_detailed(e3_mol)

    # Ensure that each molecule has the correct number of attachment points
    if not poi_l_attachment_points or not linker_poi_attachment_points or not linker_e3_attachment_points or not e3_l_attachment_points:
        raise ValueError("Missing attachment points in one or more substructures")

    # Select the first (and only) attachment point for POI and E3, and the appropriate ones for the linker
    poi_idx = poi_l_attachment_points[0]
    linker_e3_idx = linker_e3_attachment_points[0]
    e3_idx = e3_l_attachment_points[0]

    # Merge E3 with Linker
    e3_linker_mol = merge_molecules(e3_mol, linker_mol, e3_idx, linker_e3_idx, bond_type = bond_type)
    linker_e3_mol_attachment_point, _ = find_atom_index_of_mapped_atoms_detailed(e3_linker_mol)
    linker_e3_mol_idx = linker_e3_mol_attachment_point[0]

    protac_mol = merge_molecules(e3_linker_mol, poi_mol, linker_e3_mol_idx, poi_idx, bond_type=bond_type)
    Chem.SanitizeMol(protac_mol)
    protac_smiles = Chem.MolToSmiles(protac_mol, canonical=True)

    return protac_smiles, protac_mol

def generate_random_protac_dataset(unique_poi_substructures, unique_linker_substructures, unique_e3_substructures, num_protacs_to_generate):
    
    poi_list, linker_list, e3_list = select_n_random_substructures(unique_poi_substructures, unique_linker_substructures, unique_e3_substructures, num_protacs_to_generate)
    protac_smiles_list = []
    #protac_mol_list = []
    for i in tqdm(range(num_protacs_to_generate)):
        poi_smiles = poi_list[i]
        linker_smiles = linker_list[i]
        e3_smiles = e3_list[i]
        try: 
            protac_smiles, protac_mol = reassemble_protac(poi_smiles, linker_smiles, e3_smiles)
        except:
            display(Chem.MolFromSmiles(poi_smiles))
            display(Chem.MolFromSmiles(linker_smiles))
            display(Chem.MolFromSmiles(e3_smiles))
            raise ValueError("eerrrr")
        protac_smiles_list.append(protac_smiles)
        #protac_mol_list.append(protac_mol)

    d = {'protac_smiles': protac_smiles_list, 'poi': poi_list, 'linker': linker_list, 'e3': e3_list}
    df = pd.DataFrame(data=d)
    
    return df
    

def show(name):
    if not isinstance(name, str):
        raise ValueError("Input a string")
    variable = [ (i,j) for i, j in globals().items() if i == name]
    if variable != []:
        try:
            len(variable[0][1])
            print(f'Name: {variable[0][0]},   Type: {type(variable[0][0])},   len: {len(variable[0][1])},  \n{variable[0][1]}')
        except:
            print(f'Name: {variable[0][0]},   Type: {type(variable[0][0])},  \n{variable[0][1]}')
        






def load_PROTACdataset(name, raw_data_path):
    pt_name = f'data_{name}'
    num_name_pt = len([f for f in os.listdir(raw_data_path+'/') if f.startswith(pt_name)])
    dummy_list = list(range(num_name_pt))
    dummy_df = pd.DataFrame({'DummyColumn': dummy_list}) #A hack to load the datasets. It requires the length of their datasets and their name.
    dataset = PROTACDataset(data=dummy_df, substructures=dummy_df, name=name)
    return dataset



class PROTACDataset(Dataset):
    # Change: Added `data` parameter to accept a DataFrame directly
    # TODO: Include auto labeling of graph for a PROTAC by its substructures. Then in an indexed list of all nodes, [num_nodes, 3], do a 1-hot encoding as a y value.
    # Evaluate using maximum common substructure divided by the number of atoms in the substructure (or total unique atoms in known and predicted substructure). If the predicted fragment consists of isolated nodes, solve by selecting the largest set of connected nodes as the ONLY part to evaluate (big punishment for unconnected or missing nodes)

    def __init__(self, graph_descriptor_list=[], filename=None, data=None, substructures=None, name='default_name', root=None, transform=None, pre_transform=None):
        """
        Initialize the dataset with either a filename or a DataFrame.
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        # Change: Initialize filename to None
        self.substructures = substructures
        self.data = data
        self.filename = filename
        self.graph_descriptor_list = graph_descriptor_list
        if isinstance(filename, str) and filename.endswith(".csv") and name == 'default_name':
            self.name = filename[:-4]
        elif name != 'default_name':
            self.name = name

        if root is None:  # Root is data directory by default
            current_directory = os.getcwd()
            root = os.path.join(current_directory, "..", "..", "data")

        # Change: Initialize the parent class at the end of the constructor
        super(PROTACDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered. """
        # Change: Check if self.filename is not None before returning
        if self.filename:
            return self.filename
        else:
            # Change: Return an empty list if no filename is provided
            return []

    @property
    def processed_file_names(self):
        """ If these files are found in processed_dir, processing is skipped """
        # Change: Check if data is already loaded before trying to read CSV
        if self.data is not None:
            self.data = self.data.reset_index(drop=True)
        else:
            # Change: Only read CSV if filename is provided
            if self.filename:
                self.data = pd.read_csv(self.raw_paths[0]).reset_index(drop=True)
        # Change: Generate file names based on the length of self.data
        return [f'data_{self.name}_{i}.pt' for i in range(len(self.data))]

    def download(self):
        # Change: Download method can remain empty or handle downloading if necessary
        pass

    def process(self):
        # Change: Check if self.data is a DataFrame and process it
        if isinstance(self.data, pd.DataFrame):
            if self.substructures is None:
                raise ValueError(f'If you give a dataframe of SMILES for the PROTACs, you must also give the SMILES for their substructures to get boundary edges!')
            
            for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
                data = smiles_to_data(protac_smile=row["Smiles"], substructure_smiles=self.substructures['substructures'][index].split("."), graph_descriptor_list=self.graph_descriptor_list)
                torch.save(data, os.path.join(self.processed_dir, f'data_{self.name}_{index}.pt'))
        else:
            # Handle case where self.data is not a DataFrame, e.g., a string of the name for a csv file!
           raise ValueError(f'Not implemented yet that you can use a CSV file')

    def len(self):
        # Change: Return the length of self.data
        return len(self.data)

    def get(self, idx):
        # Change: Simplified get method without graph and mol parameters
        data = torch.load(os.path.join(self.processed_dir, f'data_{self.name}_{idx}.pt'))
        return data
    

def get_predicted_boundary_nodes(model, data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        out = model(data.x, data.edge_index) #, edge_attr=data.edge_attr)
        min_vals = out.min(dim=0, keepdim=True).values
        max_vals = out.max(dim=0, keepdim=True).values
        out_normalized = 2 * ((out - min_vals) / (max_vals - min_vals)) - 1
        
        pred_boundary_nodes = out_normalized.argmax(dim=0)
        #print(f'pred_boundary_nodes: {pred_boundary_nodes}')
        #print(f' out_normalized: {out_normalized}')
        #print(f' pred_boundary_nodes: {pred_boundary_nodes}')
        POI_boundary_node = pred_boundary_nodes[0].item()
        E3_boundary_node = pred_boundary_nodes[2].item()
    return POI_boundary_node, E3_boundary_node

def process_boundaries_to_substructures(data, out, return_intermediate_path_nodes=False):
    """
    Converts the prediction (out) based on the boundary nodes of the POI and E3 into a definite node level prediction for all nodes.
    Useful for evaluation and using the model to predict new values
    
    """
    
    
    min_vals = out.min(dim=0, keepdim=True).values
    max_vals = out.max(dim=0, keepdim=True).values
    out_normalized = 2 * ((out - min_vals) / (max_vals - min_vals)) - 1
    #If the POI (or E3) is smaller than 9 atoms, then subtract the max_val from that node in out, and recalculate pred_boundary_nodes => Ensure that ligands are not too small.
    #If the removal of the POI (or E3) boundary node does not increase the number of disconected subgraphs by 1, then the node is not a boundary node, then subtract the max_val from that node in out, and recalculate pred_boundary_nodes
    pred_boundary_nodes = out_normalized.argmax(dim=0)
    #print(f'pred_boundary_nodes: {pred_boundary_nodes}')
    #print(f' out_normalized: {out_normalized}')
    #print(f' pred_boundary_nodes: {pred_boundary_nodes}')
    POI_boundary_node = pred_boundary_nodes[0].item()
    E3_boundary_node = pred_boundary_nodes[2].item()
    if POI_boundary_node == E3_boundary_node:

        #print(f'pre out_normalized[POI_boundary_node,0]: {out_normalized[POI_boundary_node,0]}')
        #print(f'pre out_normalized[POI_boundary_node,2]: {out_normalized[POI_boundary_node,2]}')

        if out_normalized[POI_boundary_node,0] > out_normalized[POI_boundary_node,2]:
            #print(f'min_vals: {min_vals}')
            #print(f'min_vals[0].item(): {min_vals[0].item()}')
            print(f'min_vals[2].item(): {min_vals[2].item()}')
           # print(f'out_normalized[POI_boundary_node,2]: {out_normalized[POI_boundary_node,2]}')
           # print(f'type out_normalized[POI_boundary_node,2]: {type(out_normalized[POI_boundary_node,2])}')
            out_normalized[POI_boundary_node,2] = min_vals[0, 2].item()
            pred_boundary_nodes = out_normalized.argmax(dim=0)
            E3_boundary_node = pred_boundary_nodes[2].item()
        else: # out_normalized[POI_boundary_node,0] < out_normalized[POI_boundary_node,2]:
           # print(f'min_vals: {min_vals}')
            #print(f'min_vals[0].item(): {min_vals[0].item()}')
            #print(f'min_vals[2].item(): {min_vals[2].item()}')
            #print(f'out_normalized[POI_boundary_node,2]: {out_normalized[POI_boundary_node,0]}')
            #print(f'type out_normalized[POI_boundary_node,2]: {type(out_normalized[POI_boundary_node,0])}')
            
            
            out_normalized[POI_boundary_node,0] = min_vals[0, 0].item()
            pred_boundary_nodes = out_normalized.argmax(dim=0)
            POI_boundary_node = pred_boundary_nodes[0].item()


        #print(f'post out_normalized[POI_boundary_node,0]: {out_normalized[POI_boundary_node,0]}')
        #print(f'post out_normalized[POI_boundary_node,2]: {out_normalized[POI_boundary_node,2]}')

        #print(f'E3_boundary_node: {E3_boundary_node}')
        #print(f'POI_boundary_node: {POI_boundary_node}')


    if isinstance(data.G, list):
        Graph_original = data.G[0].copy()
    else:
        Graph_original = data.G.copy()

    #Forgot what dictionary_shuffeled_indicies came from. I guess it is for IF it is present, then use it. If not present, it would give an error and do nothing. Was to check permutation invariance, and it was.
    try:
        shuffle_indicies_dictionary = data.dictionary_shuffeled_indicies
        POI_boundary_node = shuffle_indicies_dictionary[POI_boundary_node] #Update original node index of POI to the new corresponding shuffled index
        E3_boundary_node = shuffle_indicies_dictionary[E3_boundary_node]
    except AttributeError:
        dictionary_shuffeled_indicies = None 
    

    Graph = Graph_original.copy()
    Graph_ligands = Graph_original.copy()
    Graph_POI = Graph_original.copy()
    Graph_E3 = Graph_original.copy()
    
    #print(f'initial len(mol) : {Chem.MolFromSmiles(data.smiles[0]).GetNumAtoms()}')

    try:
        #print(f' POI_boundary_node: {POI_boundary_node}')
        #print(f'E3_boundary_node: {E3_boundary_node}')
        #print(type(Graph))
        #print(Graph)
        #print("Full graph")
        #vizualize_protac_From_Graph(Graph)
        #graph_test1 = Graph.copy()
        #graph_test2 = Graph.copy()
        #graph_test1.remove_node(POI_boundary_node)
        #graph_test2.remove_node(E3_boundary_node)
        #print("POI node:")
        #vizualize_protac_From_Graph(graph_test1)
        #print("E3 node:")
        #vizualize_protac_From_Graph(graph_test2)
        #print(f'POI_boundary_node: {POI_boundary_node}')
        #print(f'E3_boundary_node: {E3_boundary_node}')
        #print(f'Graph: {Graph}')
        path_nodes = nx.shortest_path(Graph, source=POI_boundary_node, target=E3_boundary_node)
        intermediate_path_nodes = path_nodes[1:-1]
    except nx.NetworkXNoPath:
        intermediate_path_nodes = []
        raise ValueError(f'Poor path between boundary nodes, or linker has no length. If no length, then procedure to better extract POI (and E3) is needed, possibly via deleting the other node (temporarily) and seeing which are connected to the other node')          #

    if POI_boundary_node == E3_boundary_node:
        Graph.remove_node(POI_boundary_node) 
    else:
        Graph.remove_node(POI_boundary_node) #assuming only one boundary node <=> One Attatchment point
        Graph.remove_node(E3_boundary_node)
        
    linker_nodes_set = set()
    if len(intermediate_path_nodes)>0:
        for linker_node in nx.descendants(Graph, intermediate_path_nodes[0]):
            linker_nodes_set.add(linker_node)
        linker_nodes_set.add(intermediate_path_nodes[0])
    linker_nodes = list(linker_nodes_set)

    Graph_ligands.remove_nodes_from(linker_nodes)
    Graph_POI.remove_nodes_from(linker_nodes)
    Graph_POI.remove_node(E3_boundary_node)
    Graph_E3.remove_nodes_from(linker_nodes)
    Graph_E3.remove_node(POI_boundary_node)
    #vizualize_protac_From_Graph(Graph_ligands)

    E3_nodes_set = set()
    for E3_node in nx.descendants(Graph_E3, E3_boundary_node):
        E3_nodes_set.add(E3_node)
    E3_nodes_set.add(E3_boundary_node)
    E3_nodes_set = E3_nodes_set - linker_nodes_set
    E3_nodes = list(E3_nodes_set)

    POI_nodes_set = set()
    for POI_node in nx.descendants(Graph_POI, POI_boundary_node):
        POI_nodes_set.add(POI_node)
    POI_nodes_set.add(POI_boundary_node)
    POI_nodes_set = POI_nodes_set - linker_nodes_set - E3_nodes_set
    POI_nodes = list(POI_nodes_set)

    total_node_list = POI_nodes + linker_nodes + E3_nodes
    

    pred_class_label_list = []
    for i in range(len(Graph_original)):
        matches = 0
        for j in range(len(total_node_list)):
            matches += int(i == total_node_list[j])
        if matches != 1:
            raise ValueError(f'There too many or no matches of node i in the following lists in process_boundaries_to_substructures(). Matches: {matches} for node {i}')
        
        if (i in POI_nodes) and (i in linker_nodes):
            raise ValueError('AARRRHHG')
        if (i in POI_nodes) and (i in E3_nodes):
            raise ValueError('AARRRHHG')
        if (i in E3_nodes) and (i in linker_nodes):
            raise ValueError('AARRRHHG')

        if i in POI_nodes:
            pred_class_label_list.append(0)
        elif i in linker_nodes:
            pred_class_label_list.append(1)
        elif i in E3_nodes:
            pred_class_label_list.append(2)
        else:
            print(f'i: {i}')
            #print(f'len(data.G[0]): {len(data.G[0])}')
            print(f'len(POI_nodes): {len(POI_nodes)}')
            print(f'len(linker_nodes): {len(linker_nodes)}')
            print(f'len(E3_nodes): {len(E3_nodes)}')
            print(f'(POI_nodes): {POI_nodes}')
            print(f'(linker_nodes): {linker_nodes}')
            print(f'(E3_nodes): {E3_nodes}')
            print(f'intermediate nodes: {intermediate_path_nodes}')
            raise ValueError('The number of nodes in G does not match the total count of nodes among the 3 lists (too many), or a node has been lost on the way of data processing.')

    #print(f'i: {i}')
    #print(f'len(data.G[0]): {len(data.G[0])}')
    #print(f'len(POI_nodes): {len(POI_nodes)}')
    #print(f'len(linker_nodes): {len(linker_nodes)}')
    #print(f'len(E3_nodes): {len(E3_nodes)}')
    #print(f'(POI_nodes): {POI_nodes}')
    #print(f'(linker_nodes): {linker_nodes}')
    #print(f'(E3_nodes): {E3_nodes}')
    #print(f'intermediate nodes: {intermediate_path_nodes}')
    #xxxxx=1
    pred_class_label_tensor = torch.tensor(pred_class_label_list)

    if return_intermediate_path_nodes is True:
        return intermediate_path_nodes, path_nodes, POI_nodes, linker_nodes,  E3_nodes
    else:
        return pred_class_label_tensor

def process_predicted_nodes_to_substructure_labels(raw_node_prediction):
    min_vals = raw_node_prediction.min(dim=1, keepdim=True).values #min & max value of each class individually
    max_vals = raw_node_prediction.max(dim=1, keepdim=True).values
    normalized_boundary_prediction =  ((raw_node_prediction - min_vals) / (max_vals - min_vals))  # normalized predictions spanning [0, 1]
    pred_node_classes = normalized_boundary_prediction.argmax(dim=1) # get idx of the max values for each class
    return pred_node_classes

def process_predicted_boundaries_to_substructure_labels(protac_smiles, raw_boundary_prediction):
    """
    Converts the prediction (raw_boundary_prediction) based on the boundary nodes of the POI and E3 into a definite node level prediction for all nodes.
    Useful for evaluation and using the model to predict new values
    
    """
   

    #-------------------------------------------------- get POI_boundary_node & E3_boundary_node --------------------------------------------------

    #Apply to each molecule in batch (for now)
    min_vals = raw_boundary_prediction.min(dim=0, keepdim=True).values #min & max value of each class individually
    max_vals = raw_boundary_prediction.max(dim=0, keepdim=True).values
    normalized_boundary_prediction =  ((raw_boundary_prediction - min_vals) / (max_vals - min_vals))  # normalized predictions spanning [0, 1]
     
    pred_boundary_nodes = normalized_boundary_prediction.argmax(dim=0) # get idx of the max values for each class

    POI_boundary_node = pred_boundary_nodes[0].item()
    E3_boundary_node = pred_boundary_nodes[2].item()
    n_highest_pred_node = 0
    while POI_boundary_node == E3_boundary_node: #if model is stupid and says the same node for both boundaries, prevent an error selecting the "most likely" as this node, and the other node after its "second most likely" node prediction.
        n_highest_pred_node +=1
        POI_boundary_node_probability = normalized_boundary_prediction[POI_boundary_node,0]
        E3_boundary_node_probability = normalized_boundary_prediction[E3_boundary_node,2]
        
        if POI_boundary_node_probability > E3_boundary_node_probability: #if model "belives" more in POI node than E3, choose POI as this node
            E3_boundary_node = torch.topk(normalized_boundary_prediction[:,2], k=n_highest_pred_node, dim=0).indices.tolist()[-1]
            normalized_boundary_prediction[E3_boundary_node,2] = 0 #min_vals[0, 2].item() #redefine the highest E3 to the lowest value 
            pred_boundary_nodes = normalized_boundary_prediction.argmax(dim=0) #Get new highest max values (for E3)
            E3_boundary_node = pred_boundary_nodes[2].item() #redefine E3 node idx as the second highest predicted
        else: 
            POI_boundary_node = torch.topk(normalized_boundary_prediction[:,2], k=n_highest_pred_node, dim=0).indices.tolist()[-1]
            normalized_boundary_prediction[POI_boundary_node,0] = 0 #min_vals[0, 0].item()
            pred_boundary_nodes = normalized_boundary_prediction.argmax(dim=0)
            POI_boundary_node = pred_boundary_nodes[0].item()
    
    if POI_boundary_node == E3_boundary_node:
        raise ValueError("POI_boundary_node is the same as E3_boundary_node")

    pred_class_label_tensor = process_boundaries_to_substructure_labels(protac_smiles, POI_boundary_node, E3_boundary_node)
    return pred_class_label_tensor



def process_boundaries_to_substructure_labels(protac_smiles, POI_boundary_node, E3_boundary_node):
        #-------------------------------------------------- graph preparations  --------------------------------------------------

    #prepare graphs
    mol = Chem.MolFromSmiles(protac_smiles)
    Graph_original = mol_to_simple_graph(mol)
    Graph = Graph_original.copy()
    Graph_ligands = Graph_original.copy()
    Graph_POI = Graph_original.copy()
    Graph_E3 = Graph_original.copy()
    
    #Get path between boundaries => Some of the linker nodes
    try:
        path_nodes = nx.shortest_path(Graph, source=POI_boundary_node, target=E3_boundary_node)
        intermediate_path_nodes = path_nodes[1:-1]
    except:# nx.NetworkXNoPath:
        intermediate_path_nodes = []
        #raise ValueError(f'Poor path between boundary nodes, or linker has no length. If no length, then procedure to better extract POI (and E3) is needed, possibly via deleting the other node (temporarily) and seeing which are connected to the other node')          #


    #-------------------------------------------------- get linker nodes  --------------------------------------------------


    #Better scheme: 
        # 1. delete POI boundary and get all descendants of the E3 boundary
        # 2. delete E3 boundary and get all descendants of the POI boundary
        # 3. define common descendants as the linker 
        # 4. get new graph, delete all descendants of the E3 boundary => define as POI
        # 5. get new graph, delete all descendants of the POI boundary => define as E3

    #Its better since shortest path algorithm may be slow & I am already using nx.descendants twice to get the exact same nodes.



    
    #if POI_boundary_node == E3_boundary_node: #Should never happen due to code above
    #    Graph.remove_node(POI_boundary_node) 
    #else:


    #Remove boundary nodes - ideally, the linker should be free from POI and E3 now
    try:
        Graph.remove_node(POI_boundary_node)
        Graph.remove_node(E3_boundary_node)
    except:
        raise ValueError("POI_boundary_node is the same as E3_boundary_node")
        
    #Get all nodes which are connected to the path 
    linker_nodes_set = set()
    if len(intermediate_path_nodes)>0:
        for linker_node in nx.descendants(Graph, intermediate_path_nodes[0]):
            linker_nodes_set.add(linker_node)
        linker_nodes_set.add(intermediate_path_nodes[0])
    linker_nodes = list(linker_nodes_set)

    #-------------------------------------------------- get E3 nodes  --------------------------------------------------


    Graph_ligands.remove_nodes_from(linker_nodes)
    Graph_POI.remove_nodes_from(linker_nodes)
    Graph_POI.remove_node(E3_boundary_node)
    Graph_E3.remove_nodes_from(linker_nodes)
    Graph_E3.remove_node(POI_boundary_node)
    #vizualize_protac_From_Graph(Graph_ligands)

    E3_nodes_set = set()
    for E3_node in nx.descendants(Graph_E3, E3_boundary_node):
        E3_nodes_set.add(E3_node)
    E3_nodes_set.add(E3_boundary_node)
    E3_nodes_set = E3_nodes_set - linker_nodes_set
    E3_nodes = list(E3_nodes_set)

    #-------------------------------------------------- get POI nodes  --------------------------------------------------


    POI_nodes_set = set()
    for POI_node in nx.descendants(Graph_POI, POI_boundary_node):
        POI_nodes_set.add(POI_node)
    POI_nodes_set.add(POI_boundary_node)
    POI_nodes_set = POI_nodes_set - linker_nodes_set - E3_nodes_set
    POI_nodes = list(POI_nodes_set)

    #-------------------------------------------------- get assign lables  --------------------------------------------------


    total_node_list  = POI_nodes + linker_nodes + E3_nodes
    pred_class_label_list = []
    for i in range(len(Graph_original)):
        matches = 0
        for j in range(len(total_node_list)):         #Seems inefficient
            matches += int(i == total_node_list[j])
        if matches != 1:
            raise ValueError(f'There too many or no matches of node i in the following lists in process_boundaries_to_substructures(). Matches: {matches} for node {i}')

        if i in POI_nodes:
            pred_class_label_list.append(0)
        elif i in linker_nodes:
            pred_class_label_list.append(1)
        elif i in E3_nodes:
            pred_class_label_list.append(2)
        else:
            raise ValueError('The number of nodes in G does not match the total count of nodes among the 3 lists (too many), or a node has been lost on the way of data processing.')


    pred_class_label_tensor = torch.tensor(pred_class_label_list)
    return pred_class_label_tensor


def remove_stereo(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.rdmolops.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)
    except:
        return np.nan



def vizualize_graph_descriptor(smile=None, print_line_graph=False, chosen_function=None, chosen_descriptor=None):        #Functional, not pretty. Work in progress
    """mol = Chem.MolFromSmiles(smile)
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
    pos = nx.get_node_attributes(Graph, 'pos')"""

    Graph, pos = make_graph_with_pos(smile)

    if print_line_graph is True:
        #Graph_L = nx.line_graph(Graph)
        Graph_L = line_graph_custom(Graph)
        #edge_centroids = compute_edge_centroids(Graph, pos)
        edge_centroids = {}
        for edge in Graph.edges():
            p1 = pos[edge[0]]
            p2 = pos[edge[1]]
            centroid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            edge_centroids[edge] = centroid
        pos_Graph_L = {node: edge_centroids[Graph_L.nodes[node]['edge']] for node in Graph_L.nodes()}
        Graph = Graph_L
        pos = pos_Graph_L

    if chosen_descriptor is None:
        if chosen_function is not None:
            chosen_descriptor = chosen_function(Graph)
        else:
            raise ValueError('Input either chosen_descriptor or chosen_function!')

    #Calculator node colors
    if isinstance(chosen_descriptor, dict):
        chosen_descriptor_list=[]
        for key, value in chosen_descriptor.items():
            chosen_descriptor_list.append(value) 
        chosen_descriptor = np.array(chosen_descriptor_list)
    norm_values = normalize(chosen_descriptor) 
    colors = [(0, 0, 0, 1), (1, 0, 0, 1)]  # RGBA tuples
    colormap = LinearSegmentedColormap.from_list('black_to_red', colors, N=100)
    colours = [colormap(x) for x in norm_values]
    
    plt.figure(figsize=(6, 4))
    nx.draw(Graph, pos, node_color=colours, with_labels=False, node_size=100)
    plt.axis('equal')

def vizualize_protac_From_Graph(graph, node_size=100, highlighted_nodes=[]):  # By atom color
    atom_color_mapping = {
        6: 'gray',   # Carbon
        7: 'blue',   # Nitrogen
        8: 'red',    # Oxygen
        16: 'yellow', # Sulfur
        53: 'purple', # Iodine
        9: 'pink',   # Fluoride (Fluorine)
        17: 'green',  # Chloride (Chlorine)
        35: 'orange',  # Bromine
    }
    highlight_color = 'gold'  # Color for highlighted nodes
    #colors = [atom_color_mapping[[d for n, d in graph.nodes.items()][node]['atomic_num']] for node in range(len(graph))]
    colors = [highlight_color if node in highlighted_nodes else atom_color_mapping[[d for n, d in graph.nodes.items()][node]['atomic_num']] for node in range(len(graph))]


    try: 
        plt.figure(figsize=(6, 4))
        pos = nx.get_node_attributes(graph, 'pos')
        nx.draw(graph, pos=pos, node_color=colors, with_labels=False, node_size=node_size)       #NO POS!!!!
    except:
        plt.close()
        plt.figure(figsize=(6, 4))
        nx.draw(graph, node_color=colors, with_labels=False, node_size=node_size)       #NO POS!!!!

    # Plotting
    
    plt.axis('equal')
    plt.show()

def make_groundtruth_graph(protac_smile, substructure_smiles):#(protac_mol, substructure_mapping):                                               ###########  WIP
    #protac_mol = Chem.MolFromSmiles(protac_smile)
    #protac_smile, poi_smile, e3_smile = substructure_split_sort(substructure_smiles)
    #substructure_mapping = create_protac_substructure_mapping(protac_smile, poi_smile, e3_smile)
    protac_mol = Chem.MolFromSmiles(protac_smile)
    AllChem.Compute2DCoords(protac_mol)
    poi_smile, linker_smile, e3_smile = substructure_split_sort(substructure_smiles)
    substructure_mapping = create_protac_substructure_mapping(protac_smile, poi_smile, e3_smile)

    G = nx.Graph()

    # Add nodes with coordinates and color coding
    for atom in protac_mol.GetAtoms():
        atom_idx = atom.GetIdx()
        pos = protac_mol.GetConformer().GetAtomPosition(atom_idx)
        #color = determine_color(substructure_mapping, atom_idx)
        if atom_idx in substructure_mapping['E3']:
            color = 'blue'
        elif atom_idx in substructure_mapping['Linker']:
            color = 'gray'
        elif atom_idx in substructure_mapping['POI']:
            color = 'red'
        else:
            color = 'black'  # Default color
        G.add_node(atom_idx, pos=(pos.x, pos.y), color=color)

    # Add edges
    for bond in protac_mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    colors = [G.nodes[node]['color'] for node in G.nodes]

    return G, colors


                                                       ###########  WIP
def vizualize_protac_From_Smile(smile, node_size=100):  # By atom color
    atom_color_mapping = {
    'C': 'gray',     # Carbon
    'N': 'blue',     # Nitrogen
    'O': 'red',      # Oxygen
    'S': 'yellow',   # Sulfur
    'I': 'purple',   # Iodine
    'F': 'teal',    # Fluoride
    'Cl': 'green',    # Chloride
    }
    # Convert SMILES to RDKit Molecule
    mol = Chem.MolFromSmiles(smile)

    # Create a networkx graph from the RDKit molecule
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), color=atom.GetSymbol())  # Store atom symbol

    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # Assign positions to the graph
    graph, pos = assign_pos_to_graph(smile, graph)

    # Map atom symbols to colors
    colors = [atom_color_mapping.get(graph.nodes[node]['color'], 'black') for node in graph.nodes]

    # Plotting
    plt.figure(figsize=(6, 4))
    nx.draw(graph, pos, node_color=colors, with_labels=False, node_size=node_size)
    plt.axis('equal')
    plt.show()




def vizualize_predicted_nodes(G):                                                ###########  WIP
    return []




def draw_smiles_with_titles(protac_smile, substructure_smiles):
    """
    Draws molecule structures for the given SMILES strings with titles.

    Args:
    protac_smile (str): SMILES string for the PROTAC molecule.
    poi_smile (str): SMILES string for the POI molecule.
    linker_smile (str): SMILES string for the Linker molecule.
    e3_smile (str): SMILES string for the E3 molecule.

    Returns:
    A matplotlib figure containing the rendered molecule structures with titles.
    """
    # Convert SMILES to RDKit Mol objects
    poi_smile, linker_smile, e3_smile = substructure_split_sort(substructure_smiles)
    smiles_list = [protac_smile, poi_smile, linker_smile, e3_smile]
    titles = ["PROTAC", "POI", "Linker", "E3"]
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]

    # Create a subplot grid
    num_mols = len(mols)
    fig, axs = plt.subplots(1, num_mols, figsize=(15, 5))

    # Render each molecule in its respective subplot
    for i, (mol, title) in enumerate(zip(mols, titles)):
        if mol:  # Check if the molecule is valid
            img = Draw.MolToImage(mol)
            axs[i].imshow(img)
            axs[i].axis('off')
            axs[i].set_title(title)
        else:
            axs[i].set_visible(False)

    plt.tight_layout()
    #return fig


def convert_fig_to_image(fig):
    """ Convert a Matplotlib figure to a PIL Image. """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    return img

def combine_two_figures(fig1, fig2):
    # Convert figures to images
    img1 = convert_fig_to_image(fig1)
    img2 = convert_fig_to_image(fig2)

    # Create a new figure for combining the images
    fig, axs = plt.subplots(1, 2, figsize=(12*1.3, 6*1.3))

    # Display images
    axs[0].imshow(np.asarray(img1))
    axs[0].axis('off')  # Turn off axis
    axs[1].imshow(np.asarray(img2))
    axs[1].axis('off')  # Turn off axis

    plt.show()


def highlight_multiple_substructures_and_graph(mol_smiles, substructure_smiles, print_smiles=False):
    """
    Draws a molecule with highlighted bonds where the substructures attach to the rest of the molecule.
    Additionally, it outputs a NetworkX graph and the indices of the corresponding bonds.
    
    Parameters:
    mol_smiles (str): The SMILES string of the molecule.
    substruct_smiles_list (list): A list of SMILES strings of the substructures with dummy atoms.
    
    Returns:
    G (NetworkX graph): The graph representation of the molecule.
    edge_indices (list): The indices of the atoms forming the corresponding bonds in the graph.
    """
    mol = Chem.MolFromSmiles(mol_smiles)
    G = nx.Graph()
    edge_indices = []
    circles_svg = ""

    # Add edges from the molecule to the graph
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # Create a drawer with an SVG renderer
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
    drawer.DrawMolecule(mol)

    poi_smile, linker_smile, e3_smile = substructure_split_sort(substructure_smiles)
    # Process each substructure
    for substruct_smiles in [poi_smile, e3_smile]:
        substruct_mol = Chem.MolFromSmiles(substruct_smiles)
        matches = mol.GetSubstructMatches(Chem.DeleteSubstructs(substruct_mol, Chem.MolFromSmiles('*')))

        if not matches:
            continue  # If no match is found, skip to the next substructure

        match = matches[0]  # Take the first match                                                                    #OBS! WIP! I need a function that can return "the" match which is correct

        # Find the corresponding bond
        for bond in mol.GetBonds():
            begin_atom_label = int(bond.GetBeginAtomIdx() in match)
            end_atom_label = int(bond.GetEndAtomIdx() in match)
            if begin_atom_label != end_atom_label:
                edge_indices.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

                # Calculate the midpoint of the bond for the SVG circle
                draw_coords = (drawer.GetDrawCoords(bond.GetBeginAtomIdx()),
                               drawer.GetDrawCoords(bond.GetEndAtomIdx()))
                mid_x = (draw_coords[0].x + draw_coords[1].x) / 2
                mid_y = (draw_coords[0].y + draw_coords[1].y) / 2
                radius = 4
                circles_svg += f'<circle cx="{mid_x}" cy="{mid_y}" r="{radius}" stroke="red" stroke-width="1" fill="none"/>'

    #Create an unique set of edges
    sorted_edge_indices = [tuple(sorted(t)) for t in edge_indices]

    # Step 2: Convert the list of sorted tuples to a set to remove duplicates
    unique_edge_set = set(sorted_edge_indices)

    # Step 3: Convert the set back to a list
    unique_edge_indices = list(unique_edge_set)


    if not edge_indices:
        print("No corresponding bonds were found.")
        return G, None

    # Finish the drawing and get the SVG text
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')

    # Add the circles to the SVG
    svg_with_circles = re.sub('</svg>', circles_svg + '</svg>', svg)

    # Display the modified SVG
    display(SVG(svg_with_circles))

    if print_smiles is True:
        print(f'PROTAC: {mol_smiles}')
        print(f'POI: {poi_smile}')
        print(f'Linker: {linker_smile}')
        print(f'E3: {e3_smile}')
    return G, unique_edge_indices



def MaxCommonSubstructure_maxatoms(true_class_labels, predicted_class_labels): #input: list, torch tensor?
    MCS_maxatoms_substructures = [-1, -1, -1] #initialize variable
    for label in [0, 1, 2]: #[POI, Linker, E3]
        indicies_substructure = [ true_label == label for true_label in true_class_labels]
        indicies_substructure_pred = [ pred_label == label for pred_label in predicted_class_labels]
        indicies_pred_is_substructure = [ bool_substructure and bool_substructure_pred for bool_substructure, bool_substructure_pred in zip(indicies_substructure,indicies_substructure_pred)]

        num_atoms_substructure = sum(indicies_substructure)
        num_atoms_substructure_pred = sum(indicies_substructure_pred)
        num_atoms_intersection_pred_true = sum(indicies_pred_is_substructure)
        union_num_atoms_pred_true = num_atoms_substructure + num_atoms_substructure_pred - num_atoms_intersection_pred_true
        num_atoms_maximum_common_substructure_pred_true = num_atoms_intersection_pred_true

        MCS_maxatoms_substructures[label] = num_atoms_maximum_common_substructure_pred_true / union_num_atoms_pred_true

    return MCS_maxatoms_substructures

def mislabelled_nodes_in_substructures(true_class_labels, predicted_class_labels):
    mislabelled_nodes_list = [0, 0, 0] #initialize variable
    for label in [0, 1, 2]: #[POI, Linker, E3]
        indicies_substructure = [ true_label == label for true_label in true_class_labels] # returns bool of which atoms are truely the substructure
        indicies_substructure_pred = [ pred_label == label for pred_label in predicted_class_labels] # returns bool of predicted substructure 
        indicies_correct_prediction = [ bool_substructure and bool_substructure_pred for bool_substructure, bool_substructure_pred in zip(indicies_substructure,indicies_substructure_pred)] #Correct prediction

        num_atoms_substructure = sum(indicies_substructure)
        num_atoms_correct_prediction = sum(indicies_correct_prediction)
        num_atoms_incorrect_prediction = num_atoms_substructure - num_atoms_correct_prediction

        mislabelled_nodes_list[label] = num_atoms_incorrect_prediction

    return mislabelled_nodes_list

def generate_3x3confusion_matrix(true_class_labels, class_predictions):
    # Confusion matrix initialization (3x3 matrix filled with 0s)
    confusion_matrix = [[0, 0, 0] for _ in range(3)]

    # Populating the confusion matrix
    for true_label, predicted_label in zip(true_class_labels, class_predictions):
        confusion_matrix[true_label][predicted_label] += 1

    return confusion_matrix

def add_3x3confusion_matrices(matrix1, matrix2):
    # Assuming both matrices are of the same size
    num_rows = len(matrix1)
    num_cols = len(matrix1[0])
    
    # Initialize a new matrix with the same size
    result_matrix = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

    # Add the corresponding elements of matrix1 and matrix2
    for i in range(num_rows):
        for j in range(num_cols):
            result_matrix[i][j] = matrix1[i][j] + matrix2[i][j]

    return result_matrix

def check_permutation_invariance(molecule_data, model, class_predictions, probabilities):
    num_nodes = molecule_data.num_nodes
    shuffle_indicies_dictionary = {} 
    int_list = list(range(0, num_nodes))
    np.random.shuffle(int_list)
    x_shuffeled = molecule_data.x.clone()
    
    G_pred = molecule_data.G.copy()

    for i in range(num_nodes):
        shuffle_indicies_dictionary[i] = int_list[i]   #dict[old] = shuffled
    for i in range(num_nodes):
        shuffled_index = shuffle_indicies_dictionary[i]
        x_shuffeled[shuffled_index] = molecule_data.x[i] 
        shuffled_edge_index = molecule_data.edge_index.clone()
    for i in range(len(molecule_data.edge_index[0])):
        start_node = molecule_data.edge_index[0][i].item()
        end_node = molecule_data.edge_index[1][i].item()
        shuffled_edge_index[0][i] = shuffle_indicies_dictionary[start_node]
        shuffled_edge_index[1][i] = shuffle_indicies_dictionary[end_node]


    molecule_data_shuffeled = Data(x=x_shuffeled, edge_index=shuffled_edge_index, G=G_pred, dictionary_shuffeled_indicies=shuffle_indicies_dictionary)
    class_predictions_shuffeled, probabilities_shuffeled = predict(model, molecule_data_shuffeled) 

    text = "Permutation invariant"
    for i in range(num_nodes):
        if class_predictions[i] == class_predictions_shuffeled[shuffle_indicies_dictionary[i]] is False:
            text = "Not permutation invariant!"
            raise IndexError("Not permutation invariant!")
        #If I can shuffle the nodes and edges, run the prediction, then unshuffle the nodes and edges, and still have the same result as if I didn't shuffle -> Then shuffeling does not affect result => Permutation invariant
    return text


def unpack_datapoint(chosen_dataset, idx):
    chosen_datapoint=chosen_dataset[idx]
    protac_smile = chosen_datapoint.smiles
    substructure_smiles = chosen_datapoint.substructure_smiles
    poi_smile, linker_smile, e3_smile = substructure_split_sort(substructure_smiles)
    #protac_mol = Chem.MolFromSmiles(protac_smile)
    G_pred=chosen_datapoint.G

    # Create a data object
    molecule_data = Data(x=chosen_datapoint.x, edge_index=chosen_datapoint.edge_index, G=G_pred)
    return molecule_data, G_pred, protac_smile, substructure_smiles, poi_smile, linker_smile, e3_smile



def predict(model, data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        out = model(data.x, data.edge_index) #, edge_attr=data.edge_attr)

        class_predictions = process_boundaries_to_substructures(data, out)
        #print(f'pred_class_label_list: {pred_class_label_list}')
        #class_predictions = pred_class_label_tensor
        probabilities = F.softmax(out, dim=1)                        #This is wrong, it means nothing. Yet it isnt used anywhere (so far...)
        #class_predictions = probabilities.argmax(dim=1)
    return class_predictions, probabilities


def predict_v2(model, data=None, protac_smiles=None):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        out = model(data.x, data.edge_index) #, edge_attr=data.edge_attr)
        class_predictions = process_boundaries_to_substructures_v2(protac_smiles, out)
        probabilities = F.softmax(out, dim=1)                        
    return class_predictions, probabilities


def val_test_split(validation_dataset, relative_test_size=0.3):

    val_size = len(validation_dataset)
    test_size = int(relative_test_size * val_size)
    val_size_new = val_size - test_size
    val_dataset_subset, test_dataset_subset  = random_split(validation_dataset, [val_size_new, test_size], generator=torch.Generator().manual_seed(42))
    
    val_dataset = [val_dataset_subset[i] for i in range(val_size_new)]
    test_dataset = [test_dataset_subset[i] for i in range(test_size)]
    return val_dataset, test_dataset



from torch_geometric.nn import GraphConv

#GCNConv worked well before, with a test accuracy of around 0.93
#GraphConv: maybe converges faster?

class NodeClassifierGNN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim):
        super(NodeClassifierGNN, self).__init__()
        self.conv1 = GraphConv(node_feature_dim, 16)# edge_dim=edge_feature_dim)
        self.conv2 = GraphConv(node_feature_dim, 16)# edge_dim=edge_feature_dim)
        self.conv3 = GraphConv(node_feature_dim, 16)# edge_dim=edge_feature_dim)
        self.conv4 = GraphConv(node_feature_dim, 16)# edge_dim=edge_feature_dim)
        self.out = torch.nn.Linear(16, 3)  # Output layer for 3 classes
        
    def forward(self, node_attr, edge_index): #, edge_attr):
        # First Graph Convolution Layer
        z = self.conv1(node_attr, edge_index)#, edge_attr)
        z = F.relu(z)

        z = self.conv2(node_attr, edge_index)#, edge_attr)
        z = F.relu(z)

        z = self.conv3(node_attr, edge_index)#, edge_attr)
        z = F.relu(z)

        z = self.conv4(node_attr, edge_index)#, edge_attr)
        z = F.relu(z)

        y = self.out(z)

        return y

# Training function
def train(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for data in loader:
        
        optimizer.zero_grad()

        #print(data.edge_index)
        print(data.x.dtype)
        print(data.edge_index.dtype)
        out = model(node_attr=data.x, edge_index=data.edge_index)#, edge_attr=data.edge_attr)
        
        
        #num_smiles = len(data.smiles)
        target_boundary_list = []
        for smile_idx, smile in enumerate(data.smiles):
                
            substructure_smiles = data.substructure_smiles[smile_idx]
            protac_smile = data.smiles[smile_idx]
            one_hot_boundary_nodes = one_hot_encode_boundary_nodes(protac_smiles=protac_smile, substructure_smiles=substructure_smiles)
            #print(f'one_hot_boundary_nodes: {one_hot_boundary_nodes}')
            #print(one_hot_boundary_nodes)
            # Convert one-hot encoded vectors to class indices
            target_boundary_for_idx = one_hot_boundary_nodes.argmax(dim=1) #0: POI boundary node, 1: non-boundary node (Linker and ligands), 2: E3 boundary node
            target_boundary_list.append(target_boundary_for_idx)
        
        target = target_boundary_list[0]
        print(target)
        print(type(target))
        print(target.size())
        if len(target_boundary_list) > 1:
            for i in range(1, len(target_boundary_list)):
                target = torch.cat((target, target_boundary_list[i]))
            #print(target)
            #print(f'data.node_substructure_label.argmax(dim=1): {data.node_substructure_label.argmax(dim=1)}, type: {type(data.node_substructure_label.argmax(dim=1))}')
            #print(f'target_for_idx: {target_boundary_for_idx}, type: {type(target_boundary_for_idx)}')
            #target = torch.cat((target,))
        #print(target)
        #show("target")
        
        
            #print(f'target: {target}')


        #if epoch > -1:
        #    print(f'target: {target}')
        #    print(f'one_hot_boundary_nodes: {one_hot_boundary_nodes}')
        #    print(f'one_hot_boundary_nodes.argmax(dim=1): {one_hot_boundary_nodes.argmax(dim=1)}')
        #    print(data)
        #    print(out)
        #    print(data.edge_index)
        #target = data.node_substructure_label.argmax(dim=1)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


    

# Evaluation function
def evaluate(model, loader, epoch):
    model.eval()
    correct = 0
    total = 0
    #correct_ori = 0
    #total_ori = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index)# , edge_attr=data.edge_attr)


            #mol = Chem.MolFromSmiles(data.smiles[0])
            #bool_combined_list, at_idx_combined_list, colors, pos = identify_legal_nodes(mol,smallest_allowed_subgraph_size=10)
            #bool_combined_tensor = torch.tensor(bool_combined_list).reshape(-1, 1)
            #out_processed = out * bool_combined_tensor                                          #Takes VERY long time. Precompute this, store in "dataset" to later retrive it
            #print(bool_combined_list)

            #legal_nodes_tensor = data.legal_nodes
            #out_processed = out * legal_nodes_tensor
            
            #print(f'data: {data}')
            #print(f'data[0]: {data[0]}')
            #print(f'out: {out}')
            #print(f'out[0]: {out[0:len(data[0])]}')
            #print(len(data))
            #print(len(data[0]))



            batch_size_graphs = [len(data[idx].x) for idx in range(len(data))]
            batch_size_graphs[:0] = [0]
            batch_cumulative_size_graphs = np.cumsum(batch_size_graphs)
            #print(batch_size_graphs)
            #print(batch_cumulative_size_graphs)

            pred_class_label_tensor_point_list = []
            for idx in range(len(data)):
                data_point = data[idx]
                out_point = out[batch_cumulative_size_graphs[idx]:batch_cumulative_size_graphs[idx+1]]
                pred_class_label_tensor_point = process_boundaries_to_substructures(data_point, out_point)
                pred_class_label_tensor_point_list.append(pred_class_label_tensor_point)

            pred_class_label_tensor =  pred_class_label_tensor_point_list[0]

            if len(pred_class_label_tensor_point_list) > 1:
                for i in range(1, len(pred_class_label_tensor_point_list)):
                    pred_class_label_tensor = torch.cat((pred_class_label_tensor, pred_class_label_tensor_point_list[i]))
            


            #print(np.cumsum([len(data_point.x) for data_point in data]))

            #np.cumsum([len(data_point) for data_point in data])
            
            
            
            #print(pred_class_label_tensor)
            #print(f'pred_class_label_tensor: {pred_class_label_tensor}')

            # Convert one-hot encoded vectors to class indices for comparison
            target_nodes = data.node_substructure_label.argmax(dim=1)
            #print(f'data.node_substructure_label: {data.node_substructure_label}')
            #print(f'data.data.node_substructure_label.argmax(dim=1): {data.node_substructure_label.argmax(dim=1)}')
            #print(f'target: {target}')
            #print(f'pred_class_label_list: {pred_class_label_list}')
            
            
            #if epoch > 2:
            #    print(f'target: {target}')
            #    print(f'pred_class_label_tensor: {pred_class_label_tensor}')
            #    print(f'pred_class_label_tensor: {process_boundaries_to_substructures(data, out)}')
            #    pass

            correct += pred_class_label_tensor.eq(target_nodes).sum().item()
            total += data.node_substructure_label.size(0)

            #correct_ori += process_boundaries_to_substructures(data, out).eq(target).sum().item()
            #total_ori += data.node_substructure_label.size(0)

    #print(f'Epoch {epoch+1}: Accuracy original: {correct_ori / total_ori}')
    return correct / total




def adjust_bond_types(mol):
    # Iterate over all bonds
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
            # Check if both atoms connected by the bond are in a ring
            if not (mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).IsInRing() and
                    mol.GetAtomWithIdx(bond.GetEndAtomIdx()).IsInRing()):
                # Adjust the bond type (e.g., to SINGLE or DOUBLE)
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)  # or DOUBLE, depending on context


def adjust_bond_types_for_kekulization(mol):
    # Iterate over all bonds
    for bond in mol.GetBonds():
        # Check for aromatic bonds
        if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
            # Get atoms connected by the bond
            atom1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())

            # Check if both atoms are in a ring
            if not (atom1.IsInRing() and atom2.IsInRing()):
                # Adjust the bond type (e.g., to SINGLE or DOUBLE)
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)  # or DOUBLE






from rdkit.Chem.rdmolops import GetAdjacencyMatrix
def get_girvan_newman_encoding(smiles:str):
    mol = Chem.MolFromSmiles(smiles)
    A = np.array(GetAdjacencyMatrix(mol))
    G = nx.convert_matrix.from_numpy_array(A)
    communities_generator = nx.community.girvan_newman(G)

    k = 3   # number of communities
    for _ in range(k-1):
        comms = list(next(communities_generator))
    out_array = np.zeros([A.shape[0], 1])
    for com_idx, com in enumerate(comms):
        for atom in com:
            out_array[atom] = com_idx
    return out_array




def find_connected_ring_systems(mol):
    """
    Identifies each connected ring system in the molecule and returns them as a list of lists,
    where each sublist contains the indices of the atoms in one connected ring system.
   
    Parameters:
    - mol (rdkit.Chem.Mol): The molecule to analyze.
   
    Returns:
    - List[List[int]]: A list of lists, with each sublist containing the atom indices of a connected ring system.
    """
    # Find the ring systems
    ring_info = mol.GetRingInfo()
    ring_atoms = ring_info.AtomRings()
   
    # Initialize a list to keep track of which atoms belong to which ring system
    ring_systems = []

    # Check each ring against existing ring systems to find connections
    for ring in ring_atoms:
        found = False
        for system in ring_systems:
            # If the ring shares atoms with an existing system, it's connected
            if not set(ring).isdisjoint(system):
                system.update(ring)
                found = True
                break
        if not found:
            # If the ring isn't connected to existing systems, start a new one
            ring_systems.append(set(ring))
   
    # Convert sets back to lists for easier use later
    ring_systems = [list(system) for system in ring_systems]
   
    return ring_systems


def find_non_ring_bonds(mol, ring_systems, exclude_bonds_connected_to_atoms_with_1_bond, datatype):
    non_ring_bonds_dict = {}

    for bond_idx, bond in enumerate(mol.GetBonds()):
        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()

        for system in ring_systems:
            ring_bond = False
            if start_atom_idx in system and end_atom_idx in system:
                ring_bond = True
                break
        
        if ring_bond is False:
            exclude_bond = False

            if exclude_bonds_connected_to_atoms_with_1_bond:
                start_atom = mol.GetAtomWithIdx(start_atom_idx)
                end_atom = mol.GetAtomWithIdx(end_atom_idx)

                start_atom_num_bonds = start_atom.GetTotalDegree() - start_atom.GetTotalNumHs()
                end_atom_num_bonds = end_atom.GetTotalDegree() - end_atom.GetTotalNumHs()
                
                if start_atom_num_bonds == 1 or end_atom_num_bonds == 1:
                    exclude_bond = True

            if not exclude_bond:
                non_ring_bonds_dict[bond_idx] = (start_atom_idx, end_atom_idx)

    if datatype == "dict":
        return non_ring_bonds_dict
    if datatype == "list":
        return list(non_ring_bonds_dict.values())

def get_all_splits_from_all_splittable_bonds(mol, splittable_bonds_list):
    #the following should be run outside and before this function:
        #mol = Chem.MolFromSmiles(smiles)
        #for atom in mol.GetAtoms():
        #    atom.SetProp('originalIdx', str(atom.GetIdx()))         
        #ring_systems = find_connected_ring_systems(mol)
        #splittable_bonds_list = find_non_ring_bonds(mol, ring_systems, exclude_bonds_connected_to_atoms_with_1_bond=True, datatype="list")



    # get all splittable bonds in the format of a list of tuples containing start and end atoms of these bonds
    # split each of these bonds and find the resulting atoms indices
    # add hydrogens to make a chemically valid molecules, get the original node idx, remove hydrogens and restore the editable molecule
    # store all indices of the all splits

    hydrogen_atom = rdchem.Atom(1)
    hydrogen_atom.SetProp('originalIdx', str(-1))    
    bond_to_num_Hs_to_add = {rdchem.BondType.SINGLE: 1,
                             rdchem.BondType.DOUBLE: 2,
                             rdchem.BondType.TRIPLE: 3}

    emol = Chem.EditableMol(mol)
    atom_indices_all_smallest_frags = {}
    for chronological_bond_idx, (start_atom_idx, end_atom_idx) in enumerate(splittable_bonds_list):

        bond = mol.GetBondBetweenAtoms(start_atom_idx, end_atom_idx)
        #bond_idx = bond.GetIdx()

        bondtype = bond.GetBondType()
        emol.RemoveBond(start_atom_idx, end_atom_idx)

        H_atoms_idx=[]
        for bond_atom_idx in (start_atom_idx, end_atom_idx):
            num_Hs_to_add = bond_to_num_Hs_to_add[bondtype]
            for _ in range(num_Hs_to_add):
                H_idx = emol.AddAtom(hydrogen_atom)
                H_atoms_idx.append(H_idx)
                emol.AddBond(bond_atom_idx, H_idx, order = rdchem.BondType.SINGLE)
        split_mol = emol.GetMol() 
        split_mol_frags = Chem.GetMolFrags(split_mol,asMols=True)
            
        if split_mol_frags[0].GetNumAtoms() < split_mol_frags[1].GetNumAtoms() :
            smallest_frag = split_mol_frags[0]
        else:
            smallest_frag = split_mol_frags[1]

        original_atom_idx_smallest_frag = []
        for atom in smallest_frag.GetAtoms():
            original_atom_idx = int(atom.GetProp('originalIdx'))
            if original_atom_idx > -1:
                original_atom_idx_smallest_frag.append(original_atom_idx)
        atom_indices_all_smallest_frags[chronological_bond_idx] = original_atom_idx_smallest_frag
            
        for H_atom_idx in sorted(H_atoms_idx, reverse=True):
            emol.RemoveAtom(H_atom_idx)
        emol.AddBond(start_atom_idx, end_atom_idx, order = bondtype)
    
    return atom_indices_all_smallest_frags
         

def merge_rings_if_sharing_at_least_2_atoms(sets):
    while True:
        did_merge = False # Flag to track if any merge occurred in this iteration.
        for i in range(len(sets)):
            if did_merge:
                break # If a merge occurred, restart from the beginning.
            for j in range(i + 1, len(sets)):
                if len(sets[i].intersection(sets[j])) > 1: # Check if two sets share 2 or more elements.
                    # Merge sets and update the list of sets.
                    sets[i] = sets[i].union(sets[j])
                    del sets[j] # Remove the merged set.
                    did_merge = True # Set the flag as true because a merge occurred.
                    break # Exit the loop to restart the process due to the modification of the list.
        if not did_merge:
            break # If no merges occurred in this iteration, exit the while loop.

    return sets

def get_components_from_ring_systems(mol, ring_systems):
    """
    Expands each ring system in the molecule based on bond connectivity, ensuring atoms connected
    to multiple ring systems are not included.
    
    Parameters:
    - mol (Chem.Mol): RDKit molecule object.
    - ring_systems (List[List[int]]): Initial ring systems identified by find_connected_ring_systems.
    
    Returns:
    - List[Set[int]]: Expanded ring systems with additional connected atoms.
    """
    # Convert list of lists to list of sets for easier manipulation
    ring_systems = [set(system) for system in ring_systems]

    ring_systems = merge_rings_if_sharing_at_least_2_atoms(ring_systems)

    # Create a mapping from atoms to their respective ring systems
    atom_to_systems = defaultdict(set)
    for i, system in enumerate(ring_systems):
        for atom_idx in system:
            atom_to_systems[atom_idx].add(i)

    # Work on a copy of the molecule to avoid altering the original
    editable_mol = Chem.RWMol(mol)

    for system_idx, original_system in enumerate(ring_systems):
        if system_idx == 5:
            pass
        system_atoms_to_add = set()

        # Make a temporary copy of the original system for safe iteration
        temp_system = set(original_system)

        for atom_idx in temp_system:
            atom = editable_mol.GetAtomWithIdx(atom_idx)
            neighbors = atom.GetNeighbors()

            # Check each neighbor to see if it forms a bridge to another ring system
            for neighbor in neighbors:
                neighbor_idx = neighbor.GetIdx()
                # Skip if neighbor is in the same ring system
                if neighbor_idx in temp_system:
                    continue

                # Cut the bond and check connectivity to other systems
                bond = editable_mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                if bond:  # If there's a bond connecting to a potential new atom
                    editable_mol.RemoveBond(atom_idx, neighbor_idx)
                    new_fragment = Chem.GetMolFrags(editable_mol, asMols=True, sanitizeFrags=False)
                    
                    # Determine if the neighbor belongs exclusively to this system
                    #frag_exclusive_to_this_system = None
                    for frag in new_fragment:
                        exclusive_to_this_system = True
                        frag_atom_indices = set(int(atom.GetProp('originalIdx')) for atom in frag.GetAtoms())
                        # If the fragment contains atoms from other ring systems, it's not exclusive
                        if not frag_atom_indices.isdisjoint(atom_to_systems.keys()) and not frag_atom_indices <= temp_system:
                            exclusive_to_this_system = False
                            continue
                        if exclusive_to_this_system:
                            system_atoms_to_add.update(frag_atom_indices)
                            #system_atoms_to_add.update(frag_atom_indices)

                    # Rebuild the molecule to restore the bond
                    editable_mol = Chem.RWMol(mol)

        # Update the original system with new atoms after checking all bonds
        original_system.update(system_atoms_to_add)

    return [list(system) for system in ring_systems]



def find_intercomponent_bonds(mol, components, allowed_bonds='all'):
    """
    Identifies bonds that are not contained within the same component.
    
    Parameters:
    - mol (Chem.Mol): The RDKit molecule object.
    - components (List[List[int]]): A list of components, where each component is a list of atom indices.
    
    Returns:
    - List[Tuple[int, int]]: A list of tuples, where each tuple contains the indices of atoms forming an intercomponent bond.
    """
    # Convert the list of components into a dictionary for faster lookup
    atom_to_component = {}
    for component_idx, component in enumerate(components):
        for atom_idx in component:
            atom_to_component[atom_idx] = component_idx

    # List to store bonds that connect different components or atoms not in any component
    intercomponent_bonds = []
    bond_idx_list = []

    # Iterate through all bonds in the molecule
    for bond in mol.GetBonds():
        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()

        # Determine the component of each atom in the bond
        start_component = atom_to_component.get(start_atom_idx, None)
        end_component = atom_to_component.get(end_atom_idx, None)

        # Check if bond connects different components or outside any component
        if start_component is None or end_component is None or start_component != end_component:
            start_atom_originalidx = mol.GetAtomWithIdx(start_atom_idx).GetProp('originalIdx')
            end_atom_originalidx = mol.GetAtomWithIdx(end_atom_idx).GetProp('originalIdx')
            if allowed_bonds != 'all':
                if (start_atom_originalidx, end_atom_originalidx) in allowed_bonds or (end_atom_originalidx, start_atom_originalidx) in allowed_bonds:
                    intercomponent_bonds.append((int(start_atom_originalidx), int(end_atom_originalidx)))
                    bond_idx_list.append(bond.GetIdx())
            else:
                intercomponent_bonds.append((start_atom_originalidx, end_atom_originalidx))
                bond_idx_list.append(bond.GetIdx())

    return intercomponent_bonds, bond_idx_list

def format_tuple_bonds_to_COO(bonds, to_tensor = True):
    start_atoms = []
    end_atoms = []
    for bond in bonds:
        start_atoms.append(bond[0])
        end_atoms.append(bond[1])
    coo_format = [start_atoms, end_atoms]
    if to_tensor:
        return torch.tensor(coo_format, dtype=torch.int64)
    else:
        return coo_format


def get_murcko_bonds(scaffold):
    murcko_bond_list = []
    for bond in scaffold.GetBonds():
        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        start_atom_originalidx = scaffold.GetAtomWithIdx(start_atom_idx).GetProp('originalIdx')
        end_atom_originalidx = scaffold.GetAtomWithIdx(end_atom_idx).GetProp('originalIdx')
        murcko_bond_list.append((start_atom_originalidx, end_atom_originalidx))
    return murcko_bond_list


from IPython.display import SVG
def draw_molecule_with_highlighted_atoms(mol, atoms_to_highlight):
    """
    Draws a molecule with specified atoms and bonds highlighted.
   
    Parameters:
    - smiles (str): SMILES string for the molecule.
    - atoms_to_highlight (set): Set of atom indices to highlight.
    - bonds_to_highlight (list): List of bond indices to highlight.
    - highlight_bond_colors (dict): Dictionary mapping bond indices to colors.
    """
    # Create molecule from SMILES
   
    # Initialize drawer
    d2d = Draw.rdMolDraw2D.MolDraw2DSVG(350*2, 300*2)
   
    # Set drawing options
    d2d.drawOptions().useBWAtomPalette()
    d2d.drawOptions().continuousHighlight = False
    d2d.drawOptions().highlightBondWidthMultiplier = 24
    d2d.drawOptions().setHighlightColour((0, 0, 1))
    d2d.drawOptions().fillHighlights = False
   
    # Draw the molecule with highlights
    d2d.DrawMolecule(mol,
                    highlightAtoms=atoms_to_highlight)
    d2d.FinishDrawing()
   
    # Convert drawing to image and display
    svg = d2d.GetDrawingText()
    display(SVG(svg.replace('svg:','')))


def draw_molecule_with_highlighted_bonds(mol, bonds_to_highlight):
    """
    Draws a molecule with specified atoms and bonds highlighted.
   
    Parameters:
    - smiles (str): SMILES string for the molecule.
    - atoms_to_highlight (set): Set of atom indices to highlight.
    - bonds_to_highlight (list): List of bond indices to highlight.
    - highlight_bond_colors (dict): Dictionary mapping bond indices to colors.
    """
    # Create molecule from SMILES
   
    # Initialize drawer
    d2d = Draw.rdMolDraw2D.MolDraw2DSVG(350*2, 300*2)
   
    # Set drawing options
    d2d.drawOptions().useBWAtomPalette()
    d2d.drawOptions().continuousHighlight = False
    d2d.drawOptions().highlightBondWidthMultiplier = 24
    d2d.drawOptions().setHighlightColour((0, 0, 1))
    d2d.drawOptions().fillHighlights = False
   
    # Draw the molecule with highlights
    d2d.DrawMolecule(mol,
                    highlightAtoms=[],
                     highlightBonds=bonds_to_highlight)
    d2d.FinishDrawing()
   
    # Convert drawing to image and display
    svg = d2d.GetDrawingText()
    display(SVG(svg.replace('svg:','')))


import itertools
import random

def calculate_occurrences(unique_substructure_list, N):
    """
    Calculates the number of times each unique element should appear in the final list,
    ensuring that the difference is at most 1.
    
    :param unique_count: The number of unique elements in the substructure list.
    :param N: The total number of PROTACs to generate.
    :return: A dictionary with the element index as the key and its required occurrences as the value.
    """

    unique_count = len(unique_substructure_list)
    base_occurrence = N // unique_count
    extra = N % unique_count # This many elements need to have one extra occurrence to reach N
    
    occurrences = {i: {'Occurance': base_occurrence, 'SMILES': unique_substructure_list[i]} for i in range(unique_count)}
    for i in range(extra):
        occurrences[i]['Occurance'] += 1 # Distribute the extra occurrences
    
    return occurrences

def expand_substructures(occurrences):
    """
    Expand the substructure list based on the calculated occurrences.
    
    :param occurrences: A dictionary with the element index as the key and a dictionary containing 'Occurance' and 'SMILES' as the value.
    :return: An expanded list of substructures.
    """
    expanded_list = []
    for i in occurrences:
        expanded_list.extend([occurrences[i]['SMILES']] * occurrences[i]['Occurance'])
    return expanded_list

def generate_protacs(POIs, Linkers, E3s, set_sizes = [], max_trial_count=5):
    """
    Generate a list of PROTACs, each consisting of one POI, one Linker, and one E3,
    ensuring each unique substructure appears as calculated by calculate_occurrences.
    
    :param POIs: List of unique POIs.
    :param Linkers: List of unique Linkers.
    :param E3s: List of unique E3s.
    :param N: Total number of PROTACs to generate.
    :return: List of PROTACs.
    """

    augmented_protac_substructure_sets_in_list = []
    if isinstance(set_sizes, (int, float)):
        set_sizes = [int(set_sizes)]
    for N in set_sizes:
    

        # Calculate occurrences for each substructure. Store in dictionary
        POI_occurrences = calculate_occurrences(POIs, N)
        Linker_occurrences = calculate_occurrences(Linkers, N)
        E3_occurrences = calculate_occurrences(E3s, N)
        
        # Expand the substructure lists to match the calculated occurrences
        expanded_POIs = expand_substructures(POI_occurrences)
        expanded_Linkers = expand_substructures(Linker_occurrences)
        expanded_E3s = expand_substructures(E3_occurrences)

        expanded_POIs_backup = expanded_POIs.copy()
        expanded_Linkers_backup = expanded_Linkers.copy()
        expanded_E3s_backup = expanded_E3s.copy()

        resulting_poi_list = []
        resulting_linker_list = []
        resulting_e3_list = []

        trials = 0
        while True and trials < max_trial_count:  
            # Combine the substructures to form PROTACs
            protacs = []
            for i in tqdm(range(N)):
                unique_protac_bool = False
                while unique_protac_bool == False:
                    sampeled_poi = random.sample(expanded_POIs, 1)[0]
                    sampeled_linker = random.sample(expanded_Linkers, 1)[0]
                    sampeled_e3 = random.sample(expanded_E3s, 1)[0]

                    sampeled_poi_first_index = expanded_POIs.index(sampeled_poi)
                    sampeled_linker_first_index = expanded_Linkers.index(sampeled_linker)
                    sampeled_e3_first_index = expanded_E3s.index(sampeled_e3)

                    protac_smiles, protac_mol = reassemble_protac(sampeled_poi, sampeled_linker, sampeled_e3, bond_type = 'rand_uniform')
                    if protac_smiles not in protacs:
                        add_protac = True
                        if len(augmented_protac_substructure_sets_in_list)>0:
                            for augmented_protac_substructure_set in augmented_protac_substructure_sets_in_list:
                                if protac_smiles in augmented_protac_substructure_set['PROTAC SMILES'].unique():
                                    add_protac = False
                        if add_protac:
                            unique_protac_bool = True
                            del expanded_POIs[sampeled_poi_first_index] 
                            del expanded_Linkers[sampeled_linker_first_index] 
                            del expanded_E3s[sampeled_e3_first_index] 
                            resulting_poi_list.append(sampeled_poi)
                            resulting_linker_list.append(sampeled_linker)
                            resulting_e3_list.append(sampeled_e3)
                            protacs.append(protac_smiles)


            if len(protacs) == len(set(protacs)): #Om dubletter genereras, försök igen!
                #if the set of protacs is not the same length as the set of unique protacs, then some protacs are duplicates, hence the selection must have been forced to occur by poor luck or too big test size.
                #so, solution is to redo everything. May take long time if test size is close to max allowed size.....
                break
            else:
                expanded_POIs = expanded_POIs_backup.copy()
                expanded_Linkers = expanded_Linkers_backup.copy()
                expanded_E3s = expanded_E3s_backup.copy()
                trials +=1
                if trials == max_trial_count:
                    raise ValueError(f"Failed to generated random dataset without duplicates {max_trial_count} times. Please try a smaller test_size in relation to the max_allowed size, or increase max_trial_count")

    
        data = {'PROTAC SMILES': protacs , 'POI SMILES': resulting_poi_list, 'LINKER SMILES': resulting_linker_list, 'E3 SMILES': resulting_e3_list}
        augmented_protac_substructure_set = pd.DataFrame(data)

        augmented_protac_substructure_sets_in_list.append(augmented_protac_substructure_set)

    return augmented_protac_substructure_sets_in_list