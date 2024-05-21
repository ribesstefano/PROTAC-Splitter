
from rdkit import Chem
import copy
import torch
import statistics
import pandas as pd
from statistics import median
import networkx as nx
from IPython.display import display
from rdkit.Chem import Draw
from IPython.display import SVG
import numpy as np
from scipy.linalg import inv

from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import rdmolops, rdchem







def avg(values):
    """
    Returns the average of a list of values, or None if the list is empty.
    """
    num_values = len(values)
    return sum(values) / num_values if num_values > 0 else None





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
        try:
            substructure_smi = Chem.MolFragmentToSmiles(protac_mol, node_indices_substructure, kekuleSmiles=True)
        except:
            substructure_smi = None
        substructures_smi[substructure_name] = substructure_smi


    if boundary_bonds == (None, None) or boundary_bonds is None:
        boundary_bonds = []
        for atom_idx, substructure_class in enumerate(class_predictions_list):
            atom = protac_mol.GetAtomWithIdx(atom_idx)
            neighbors = atom.GetNeighbors()
            for neighbor_atom in neighbors:
                neighbor_idx = neighbor_atom.GetIdx()
                if class_predictions_list[neighbor_idx] != substructure_class:
                    if [atom_idx, neighbor_idx] not in boundary_bonds:
                        boundary_bonds.append([neighbor_idx, atom_idx])


    #Get substructures with attachmentpoint
    
    #mol -> editable mol
    #add dummy atoms, store their atom idx
    #add bonds between dummy atoms and boundary bonds
    #remove boundary bonds
    #editable mol -> mol => mol with 3 unconnected parts
    #convert into smiles
        #split smiles by "." to get their substructures
    #identify substructures by dummy atoms

    if len(boundary_bonds) == 2:
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
    else: #len(boundary_bonds) != 2:
        substructure_smiles_with_dummyatoms_dict = {}
        substructure_smiles_with_dummyatoms_dict["LINKER SMILES"] = None
        substructure_smiles_with_dummyatoms_dict["E3 SMILES"] = None
        substructure_smiles_with_dummyatoms_dict["POI SMILES"] = None

    
    return substructures_smi, substructure_smiles_with_dummyatoms_dict






def aggregate_metrics_at_epoch(output, epoch, return_agg_output = False):
    aggregated_metrics = {}

    if return_agg_output:
        aggregated_output = copy.deepcopy(output)
        
    for dataset_name in output['metrics'].keys():
        for accuracy_origin in output['metrics'][dataset_name].keys():
            for structure_type in output['metrics'][dataset_name][accuracy_origin].keys():
                for metric_type in output['metrics'][dataset_name][accuracy_origin][structure_type].keys():
                    column_name = f'{accuracy_origin}_{structure_type}_{metric_type}'
                        
                    if metric_type in ["Atoms_wrong"]:

                        all_atoms_wrong = []
                        for atoms_wrong, occurance in output["metrics"][dataset_name][accuracy_origin][structure_type]["Atoms_wrong"][epoch].items():
                            all_atoms_wrong.extend([atoms_wrong]*occurance)


                        column_name_avg_atoms_wrong = f'Avg. {column_name}'
                        if column_name_avg_atoms_wrong not in aggregated_metrics:
                            aggregated_metrics[column_name_avg_atoms_wrong] = []
                        avg_atoms_wrong = avg(all_atoms_wrong)
                        aggregated_metrics[column_name_avg_atoms_wrong].append(avg_atoms_wrong)
                        
                        if return_agg_output:
                            aggregated_output["metrics"][dataset_name][accuracy_origin][structure_type]["Atoms_wrong"][epoch] = avg_atoms_wrong
                            
                            
                        column_name_median_atoms_wrong = f'Median {column_name}'
                        if column_name_median_atoms_wrong not in aggregated_metrics:
                            aggregated_metrics[column_name_median_atoms_wrong] = []
                        median_atoms_wrong = median(all_atoms_wrong)
                        aggregated_metrics[column_name_median_atoms_wrong].append(median_atoms_wrong)

                    else: 
                        macro_avg_metric = avg(output['metrics'][dataset_name][accuracy_origin][structure_type][metric_type][epoch])
                        if macro_avg_metric is not None:
                            if column_name not in aggregated_metrics:
                                aggregated_metrics[column_name] = []
                            aggregated_metrics[column_name].append(macro_avg_metric*100)
                        
                        if return_agg_output:
                            aggregated_output["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][epoch] = macro_avg_metric

            
            
        column_name = f'% Flipped PROTACs'
        if column_name not in aggregated_metrics:
            aggregated_metrics[column_name] = []
        aggregated_metrics[column_name].append(output["flip_fraction"][dataset_name][epoch]*100)

            
        for validity_type in output["validity_fraction"][dataset_name].keys():
            column_name = f'Validity {validity_type}'
            validity_frac = output["validity_fraction"][dataset_name][validity_type][epoch]
            if column_name not in aggregated_metrics:
                aggregated_metrics[column_name] = []
            aggregated_metrics[column_name].append(validity_frac*100)
    
    if return_agg_output:
        return aggregated_metrics, aggregated_output
    else:
        return aggregated_metrics





def process_predicted_boundaries_to_substructure_labels(protac_smiles, raw_boundary_prediction):
    """
    Converts the prediction (raw_boundary_prediction) based on the boundary nodes of the POI and E3 into a definite node level prediction for all nodes.
    Useful for evaluation and using the model to predict new values
    
    """
    pred_poi_boundary_idx, _, pred_e3_boundary_idx = raw_boundary_prediction.argmax(dim=0)
    poi_label = 0
    e3_label = 2
                
                
    n_highest_pred_node = 0
    if pred_poi_boundary_idx == pred_e3_boundary_idx:
        raw_prediction_protac_temporary = torch.clone(raw_boundary_prediction)    #SLOW?   #TopK is slow
    if True:
        while pred_poi_boundary_idx == pred_e3_boundary_idx:          
            n_highest_pred_node +=1
            
                

            poi_pred_val = raw_prediction_protac_temporary[pred_poi_boundary_idx, poi_label]
            e3_pred_val = raw_prediction_protac_temporary[pred_poi_boundary_idx, e3_label]
            if e3_pred_val>poi_pred_val:
                    #overwrite the value of the POI
                raw_prediction_protac_temporary[pred_poi_boundary_idx, poi_label] = float('-inf')  #min_vals[0, 0].item()
                    #find the index of second most likely POI 
                pred_poi_boundary_idx = torch.topk(raw_prediction_protac_temporary[:, poi_label], k=1, dim=0).indices

            else:
                raw_prediction_protac_temporary[pred_e3_boundary_idx, e3_label] = float('-inf') #min_vals[0, 2].item() #redefine the highest E3 to the lowest value 
                pred_e3_boundary_idx = torch.topk(raw_prediction_protac_temporary[:, e3_label], k=1, dim=0).indices

            if n_highest_pred_node > 2:
                print(f'pred_poi_boundary_idx: {pred_poi_boundary_idx}')
                print(f'pred_e3_boundary_idx: {pred_e3_boundary_idx}')
                print(f'e3_pred_val: {e3_pred_val}')
                print(f'poi_pred_val: {poi_pred_val}')
                print(f'raw_prediction_protac_temporary: {raw_prediction_protac_temporary}')
                print(f'raw_boundary_prediction: {raw_boundary_prediction}')

    pred_poi_boundary_idx = pred_poi_boundary_idx.item()
    pred_e3_boundary_idx = pred_e3_boundary_idx.item()

    if pred_poi_boundary_idx > raw_boundary_prediction.size(0) or pred_e3_boundary_idx > raw_boundary_prediction.size(0):
        raise ValueError(f"pred_poi_boundary_idx ({pred_poi_boundary_idx}) pred_e3_boundary_idx ({pred_e3_boundary_idx}) or larger than raw_boundary_prediction.size(0) {raw_boundary_prediction.size(0)}")
    #except:
    #    pred_poi_boundary_idx = 0
   #     pred_e3_boundary_idx = 1
    #    print("Fatal error avoided. Boundary node indices forced to be 0 and 1.")
    
    
    if pred_poi_boundary_idx == pred_e3_boundary_idx:
        raise ValueError("POI_boundary_node is the same as E3_boundary_node")

    pred_class_label_tensor = process_boundaries_to_substructure_labels(protac_smiles, pred_poi_boundary_idx, pred_e3_boundary_idx)
    return pred_class_label_tensor



def mol_to_simple_graph(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())    
    return G



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
        print(protac_smiles)
        display(Chem.MolFromSmiles(protac_smiles))
        print(f"POI_boundary_node: {POI_boundary_node}, {type(POI_boundary_node)}")
        print(f"E3_boundary_node: {E3_boundary_node}, {type(E3_boundary_node)}")
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






def aggregate_output_all_epochs(output):

    aggregated_output = copy.deepcopy(output)
        
    for dataset_name in output['metrics'].keys():
        for accuracy_origin in output['metrics'][dataset_name].keys():
            for structure_type in output['metrics'][dataset_name][accuracy_origin].keys():
                for metric_type in output['metrics'][dataset_name][accuracy_origin][structure_type].keys():
                    for epoch in output['metrics'][dataset_name][accuracy_origin][structure_type][metric_type].keys():

                        if metric_type in ["Atoms_wrong"]:
                            all_atoms_wrong = []
                            for atoms_wrong, occurance in output["metrics"][dataset_name][accuracy_origin][structure_type]["Atoms_wrong"][epoch].items():
                                all_atoms_wrong.extend([atoms_wrong]*occurance)
                            avg_atoms_wrong = avg(all_atoms_wrong)
                            aggregated_output["metrics"][dataset_name][accuracy_origin][structure_type]["Atoms_wrong"][epoch] = avg_atoms_wrong

                        else: 
                            macro_avg_metric = avg(output['metrics'][dataset_name][accuracy_origin][structure_type][metric_type][epoch])                       
                            aggregated_output["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][epoch] = macro_avg_metric
    
    return aggregated_output



def get_best_epoch(output, dataset, accuracy_origin, structure, metric_type, model_crossfold):
    if model_crossfold and False:
        avg_val_acc = {}
        min_num_epochs = min(len(list(fold_output['metrics'][dataset][accuracy_origin][structure][metric_type].keys())) for fold_output in output)
        starting_epoch = min(output[0]['metrics'][dataset][accuracy_origin][structure][metric_type].keys())
        for epoch in range(starting_epoch, min_num_epochs+starting_epoch):
            epoch_vals = []
            for fold_output in output:
                avg_val = avg(fold_output['metrics'][dataset][accuracy_origin][structure][metric_type][epoch])
                epoch_vals.append(avg_val)
            avg_epoch_val = avg(epoch_vals)
            avg_val_acc[epoch] = avg_epoch_val
        epoch_with_highest_max_val = max(avg_val_acc, key=avg_val_acc.get)
        

    else:
        protac_val_accs_over_epochs = output['metrics'][dataset][accuracy_origin][structure][metric_type]
        epoch_with_highest_max_val = max(protac_val_accs_over_epochs, key=protac_val_accs_over_epochs.get)
    
    return epoch_with_highest_max_val





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

        ring_bond = False
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
         



def get_boundary_bonds_v2(protac_mol, poi_smile, e3_smile):
        # ------ new strategy -------
    # theory: 
    #   connectionpoint is only at one bond/boundary. A perfect substructure match means all atoms of substructure
    #   is on one side of this boundary. The substructure has a fixed set of atoms it has to assign.
    #   If the substructure matches imperfectly, it means some atoms are assigned past the boundary.
    #   Since the boundary is a ONE bond (a so called bridge in graph theory), and there is a fixed set of atoms,
    #   it means some "allocations" are moved from the true substructure to the other side, meaning
    #   not all atoms in the true substructures are identified as the substructure.
    #   As the only connection to the substructure and the rest of the PROTAC is this one bridge,
    #   It means that there is now an unassigned fragment within the true substructure not connected to the protac
    #   => Removing the matched substructure will lead to 2 fragments, one small from the true substructure which was failed to be identified, and the rest of the protac
    #   IF there is a perfect match, there will only be 1 fragment.
    #       Use this as the criteria for good/bad matches.


    #not adapted for symmetric proacs with the same ligand on each side



    #make test such that removing the boundary bond will split the protac into 2 fragments (So it won't just linearize rings)
    #match the largest substructure first
    #for the atoms that a valid substructure was found, dissallow these atoms to be a valid match for next matches.


    poi_mol = Chem.MolFromSmiles(poi_smile)
    e3_mol = Chem.MolFromSmiles(e3_smile)
    poi_mol_wo_attachment = Chem.DeleteSubstructs(poi_mol, Chem.MolFromSmiles('*'))
    e3_mol_wo_attachment = Chem.DeleteSubstructs(e3_mol, Chem.MolFromSmiles('*'))
    poi_num_atoms = poi_mol_wo_attachment.GetNumAtoms()
    e3_num_atoms = poi_mol_wo_attachment.GetNumAtoms()

    if poi_num_atoms > e3_num_atoms:
        substructure_mols = [poi_mol_wo_attachment, e3_mol_wo_attachment]
        substructure_index = [0, 1]
    else:
        substructure_mols = [e3_mol_wo_attachment, poi_mol_wo_attachment]
        substructure_index = [1, 0]  

    boundary_POI_bond = -1
    boundary_E3_bond = -1

    valid_boundary_POI_bond = False
    valid_boundary_E3_bond = False

    matched_atoms = []
    for i, substructure in zip(substructure_index, substructure_mols):

        matches = protac_mol.GetSubstructMatches(substructure)

        for match in matches:
            if len(set(matched_atoms) & set(match)) > 0:
                #if this substructure match overlaps with an already identified substructure match
                #then skip this match
                continue
            editable_mol = Chem.RWMol(protac_mol)
            # Sort indices in descending order to avoid altering the indices of atoms to be deleted later
            for idx in sorted(match, reverse=True):
                editable_mol.RemoveAtom(idx)
            mol_without_substructure = Chem.Mol(editable_mol)
            fragments = rdmolops.GetMolFrags(mol_without_substructure, asMols=False)
            
            if len(fragments) == 0 :
                raise ValueError(f"No substructure match for PROTAC {Chem.MolToSmiles(protac_mol)} for {Chem.MolToSmiles(substructure)}!")
            elif len(fragments) > 1:  
                continue
            elif len(fragments) == 1: #if only one fragment remaining after removal => perfect match
                
                
                
                
                # Find boundary nodes for the POI and E3
                for bond in protac_mol.GetBonds():
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
                
                #validate that splitting this bond will split the molecule
                
                if valid_boundary_POI_bond is False and boundary_POI_bond != -1:
                    #the bond have been assigned, now validate it
                    editable_mol_for_validation = Chem.RWMol(protac_mol)
                    editable_mol_for_validation.RemoveBond(boundary_POI_bond[0], boundary_POI_bond[1])
                    protac_mol_for_validation = Chem.Mol(editable_mol_for_validation)
                    validation_fragments = rdmolops.GetMolFrags(protac_mol_for_validation, asMols=False)
                    if len(validation_fragments) == 2:
                        valid_boundary_POI_bond = True
                        matched_atoms.append(match)
                if valid_boundary_E3_bond is False and boundary_E3_bond != -1:
                    #the bond have been assigned, now validate it
                    editable_mol_for_validation = Chem.RWMol(protac_mol)
                    editable_mol_for_validation.RemoveBond(boundary_E3_bond[0], boundary_E3_bond[1])
                    protac_mol_for_validation = Chem.Mol(editable_mol_for_validation)
                    validation_fragments = rdmolops.GetMolFrags(protac_mol_for_validation, asMols=False)
                    if len(validation_fragments) == 2:
                        valid_boundary_E3_bond = True
                        matched_atoms.append(match)
            else:
                raise ValueError(f"Number of substructure matches: {len(fragments)}")
            
            
            
    if boundary_POI_bond == -1 or boundary_E3_bond == -1 or valid_boundary_E3_bond is False or valid_boundary_POI_bond is False:
        display(Chem.MolToSmiles(protac_mol))
        display(Chem.MolFromSmiles(poi_smile))
        display(Chem.MolFromSmiles(e3_smile))
        print(f'boundary_POI_bond: {boundary_POI_bond}. boundary_E3_bond: {boundary_E3_bond}')
        raise ValueError("Failed to assign boundary index")
    
    return boundary_POI_bond, boundary_E3_bond



def get_bond_labels(splittable_bonds, boundary_bonds, poi_label=0, linker_label = 1, e3_label = 2):
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

import copy

def get_boundarybond_labels(bond_labels):
    boundarybond_labels = copy.deepcopy(bond_labels)
    for protac_idx, bond_labels_protac in enumerate(boundarybond_labels):
        while 2 in bond_labels_protac:
            bond_idx = bond_labels_protac.index(2)
            boundarybond_labels[protac_idx][bond_idx] = 0
    
    return boundarybond_labels




def boundary_ligand_nodes_v2(protac_smiles, poi_smile, e3_smile, print_mols=False):

    mol = Chem.MolFromSmiles(protac_smiles)
    mol_to_match_into = Chem.MolFromSmiles(protac_smiles)
    if print_mols:
        display(mol)
    boundary_POI_node_index = -1
    boundary_E3_node_index = -1
    ligand_smiles_list = [poi_smile, e3_smile]   
    ligand_smi_dict = {'poi': poi_smile, 'e3': e3_smile}

    all_matched_atom_indices = {}

    #for each pair of matches, for poi and e3, select the first pair of matches that are exclusive
    for ligand_key, ligand_smile in ligand_smi_dict.items():
        
        substruct_mol_with_attachment = Chem.MolFromSmiles(ligand_smile)
        substruct_mol = Chem.DeleteSubstructs(substruct_mol_with_attachment, Chem.MolFromSmiles('*'))
        matches = mol.GetSubstructMatches(substruct_mol)
        all_matched_atom_indices[ligand_key] = matches #if multiple matches, store all of them.


        
        

    #select pair of matches:
    poi_matches = all_matched_atom_indices["poi"]
    e3_matches = all_matched_atom_indices["e3"]
    non_overlapping_match_of_poi_and_e3 = False
    for poi_match_idx, poi_match in enumerate(poi_matches):
        for e3_match_idx, e3_match in enumerate(e3_matches):
            poi_match_set = set(poi_match)
            e3_match_set = set(e3_match)
            #check that these sets are disjoint
            
            if poi_match_set.isdisjoint(e3_match_set): # if they dont share indices, the matches dont overlap
                
                #verify that the resulting linker is in contact with both the poi and e3
                #linker_indices = set(range(0, mol.GetNumAtoms())) - poi_match_set - e3_match_set

                matched_atom_indices = {}
                matched_atom_indices["poi"] = poi_matches[poi_match_idx]
                matched_atom_indices["e3"] = e3_matches[e3_match_idx]
                
                try_again = False
                for i, (ligand_smile, match) in enumerate(zip(ligand_smi_dict.values(), matched_atom_indices.values())): 

 
                #for i, ligand_smile in enumerate(ligand_smiles_list):
                    
                    #substruct_mol_with_attachment = Chem.MolFromSmiles(ligand_smile)
                    #substruct_mol = Chem.DeleteSubstructs(substruct_mol_with_attachment, Chem.MolFromSmiles('*'))
                    #matches = mol.GetSubstructMatches(substruct_mol)


                    
                    #if not matches:
                    #    continue  # If no match is found, skip to the next substructure
                    #match = matches[0]  # Take the first match                                         ##########################OBS!

                    if print_mols:
                        display(mol)
                        #display(substruct_mol)
                        draw_molecule_with_highlighted_atoms(mol=mol, atoms_to_highlight=match)

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
                
                substructure_label = process_boundaries_to_substructure_labels(protac_smiles, boundary_POI_node_index, boundary_E3_node_index)
                for i in range(2):
                    if sum(substructure_label==i) == 0:
                        try_again = True
                
                if try_again:
                    break
                        

                return boundary_POI_node_index, boundary_E3_node_index

                
                
    raise ValueError(f"Failed to identify non-overlapping matches for POI {poi_smile} and E3 {e3_smile} in PROTAC: {protac_smiles}")



def get_boundary_labels(protac_smiles, poi_smile, e3_smile):        #Returns np.array
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











# ----------------------- GRAPH DESCRIPTORS


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



def normalize(vector, vector_max=None, vector_min=None):
    if vector_max is None:
        vector_max = vector.max()
    if vector_min is None:
        vector_min = vector.min()
    vector_norm = (vector-vector_min)/(vector_max-vector_min)
    return vector_norm



def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def aggregate_metrics_for_crossfold(outputs, model):
    #want to take the average over each fold, for each epoch, dataset, structure type, metric type,

    epochs_all_splits = [set(output['loss'][model.train_set_name].keys()) for output in outputs]
    common_epochs = epochs_all_splits[0]

    for epoch_for_split in epochs_all_splits[1:]:
        common_epochs = common_epochs.intersection(epoch_for_split)

    max_epoch = max(common_epochs)
    min_epoch = min(common_epochs)

    measures_avg = copy.deepcopy(model.measures_template)
    measures_std = copy.deepcopy(model.measures_template)
    measures_concat = copy.deepcopy(model.measures_template)


    #loss
        #dataset names
            #epoch
    for dataset_name in outputs[0]['loss'].keys():
        for epoch in range(min_epoch, max_epoch+1):
            loss_list = [output['loss'][dataset_name][epoch] for output in outputs]
            measures_avg['loss'][dataset_name][epoch] = geo_mean(loss_list)
            measures_std['loss'][dataset_name][epoch] = statistics.stdev(loss_list)
        
        
        #boundary_bond_probs - not implemented

        #metrics
            #dataset names
                #accuracy origin
                    #structure
                        #metric type
                            #epoch
        for accuracy_origin in outputs[0]["metrics"][dataset_name].keys():
            for structure_type in outputs[0]["metrics"][dataset_name][accuracy_origin].keys():
                for metric_type in outputs[0]["metrics"][dataset_name][accuracy_origin][structure_type].keys():
                    for epoch in range(min_epoch, max_epoch+1):

                        metrics_list = []
                        for output in outputs:
                            metrics_raw = output["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][epoch] 
                            if isinstance(metrics_raw, list):
                                metrics_avg = avg(metrics_raw)
                            elif isinstance(metrics_raw, dict):
                                all_atoms_wrong = []
                                for atoms_wrong, occurance in metrics_raw.items():
                                    all_atoms_wrong.extend([atoms_wrong]*occurance)
                                metrics_avg = avg(all_atoms_wrong)
                            metrics_list.append(metrics_avg)
                  
                        measures_avg["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][epoch] = avg(metrics_list)
                        measures_std["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][epoch] = statistics.stdev(metrics_list)
                        
                        #measures_concat - deal with dict and list
                        if epoch not in measures_concat["metrics"][dataset_name][accuracy_origin][structure_type][metric_type]:
                            if isinstance(metrics_raw, list):
                                measures_concat["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][epoch] = []
                            elif isinstance(metrics_raw, dict):
                                measures_concat["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][epoch] = {}
                        if isinstance(metrics_raw, list):
                            measures_concat["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][epoch].append(metrics_raw)
                        elif isinstance(metrics_raw, dict):
                            for key, value in metrics_raw.items():
                                if key not in measures_concat["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][epoch]:
                                    measures_concat["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][epoch][key] = 0
                                measures_concat["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][epoch][key] += value

                            

        #validity_fraction
            #dataset names
                #'VALID SPLIT', 'POI SMILES', 'LINKER SMILES', 'E3 SMILES'
                    #epoch
        for validity_type in outputs[0]["validity_fraction"][dataset_name].keys():
            for epoch in range(min_epoch, max_epoch+1):
                validity_list = [output["validity_fraction"][dataset_name][validity_type][epoch] for output in outputs]
                measures_avg["validity_fraction"][dataset_name][validity_type][epoch] = avg(validity_list)
                measures_std["validity_fraction"][dataset_name][validity_type][epoch] = statistics.stdev(validity_list)

        #flip_fraction
            #dataset names
                #epoch
        for epoch in range(min_epoch, max_epoch+1):
            flip_list = [output["flip_fraction"][dataset_name][epoch] for output in outputs]
            measures_avg["flip_fraction"][dataset_name][epoch] = avg(flip_list)
            measures_std["flip_fraction"][dataset_name][epoch] = statistics.stdev(flip_list)

    return measures_avg, measures_std, measures_concat


def crossfolds_avg_std_at_epoch(outputs, epoch, dataset_names, print_df = False): 
    #first average within each fold, then average across folds - Because I want each fold to have equal weight (and not be determined by the number of PROTACs in each fold)

    aggregated_metrics_for_a_fold = {}
    for fold_id, output in enumerate(outputs):
        aggregated_metrics_for_a_fold[fold_id] = aggregate_metrics_at_epoch(output=output, epoch=epoch)
    
    avg_metrics = {column: [] for column in aggregated_metrics_for_a_fold[0].keys()}
    std_metrics = {column: [] for column in aggregated_metrics_for_a_fold[0].keys()}

    if print_df:
        printable_metrics = {column: [] for column in aggregated_metrics_for_a_fold[0].keys()}
    
    for dataset_id, dataset_name in enumerate(dataset_names):
        for column in aggregated_metrics_for_a_fold[0].keys():
            fold_values = []
            for fold_id in range(0, len(outputs)):
                fold_values.append(aggregated_metrics_for_a_fold[fold_id][column][dataset_id])
            
            average_fold_value = avg(fold_values)
            std_fold_value = statistics.stdev(fold_values)

            avg_metrics[column].append(average_fold_value)
            std_metrics[column].append(std_fold_value)
            if print_df:
                printable_metrics[column].append(f'{round(average_fold_value,1)} \u00B1 {round(std_fold_value, 1)}')

    
    if print_df:
        printable_df = pd.DataFrame(data=printable_metrics)
        printable_df.index = dataset_names
        print(printable_df)
    
    return {'avg_metrics': avg_metrics, 'std_metrics': std_metrics, 'rows': dataset_names}        