
import torch
from rdkit import Chem
from torch_geometric.utils import from_smiles
from torch_geometric.data import InMemoryDataset


import os
import pickle
import pandas as pd
from tqdm import tqdm



from data_curation_augmentation_splitting_functions import (remove_stereo)

from pipeline_functions import (boundary_ligand_nodes_v2,
                                get_boundary_labels,
                                process_boundaries_to_substructure_labels,
                                mol_to_simple_graph,
                                graph_descriptor,
                                find_connected_ring_systems, 
                                find_non_ring_bonds, 
                                get_all_splits_from_all_splittable_bonds,
                                get_boundary_bonds_v2, 
                                get_bond_labels, 
                                get_boundarybond_labels,
                                get_girvan_newman_encoding
)






class ProtacDataset(InMemoryDataset):
    def __init__(self, protac_input, mode, model_type, node_descriptors, transform=None, graph_descriptor_list=[], use_girvan_newman_encodings=False, precompute_splits = False):
        super(ProtacDataset, self).__init__()
        self.mode = mode
        self.model_type = model_type
        self.graph_descriptor_list = graph_descriptor_list
        self.use_girvan_newman_encodings = use_girvan_newman_encodings
        self.node_descriptors = node_descriptors
        self.precompute_splits = precompute_splits


        self.validate_mode()             
        self.process_input(protac_input)
        self.initialize_mols_and_graphs()

        if self.graph_descriptor_list:
            self.compute_graph_descriptors()

        if model_type == "link_pred":
            self.initialize_link_prediction_attributes()
        elif model_type == "node_pred":
            self.initialize_node_prediction_attributes()
        elif model_type == "boundary_pred":
            pass

        if self.use_girvan_newman_encodings:
            self.compute_girvan_newman_encodings()

        self.dataset = self._generate_dataset()

    def validate_mode(self):
        valid_modes = ["predict", "train", "eval"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose from {valid_modes}.")

    def process_input(self, protac_input):
        if self.mode == "predict":
            self.process_predict_mode(protac_input)
        elif self.mode in ["train", "eval"]:
            self.process_train_or_eval_mode(protac_input)
        else:
            raise ValueError("Unsupported mode provided.")

    def process_predict_mode(self, protac_input):
        if isinstance(protac_input, list):
            self.protac_smiles = [remove_stereo(smi) for smi in protac_input]
        elif isinstance(protac_input, str):
            self.protac_smiles = [remove_stereo(protac_input)]
        else:
            raise TypeError("For prediction, input must be a string or list of strings.")

    def process_train_or_eval_mode(self, protac_input):
        self.protac_smiles = protac_input['PROTAC SMILES'].apply(remove_stereo).tolist()
        self.poi_smiles = protac_input['POI SMILES'].apply(remove_stereo).tolist()
        self.e3_smiles = protac_input['E3 SMILES'].apply(remove_stereo).tolist()
        self.boundary_indices, self.boundary_labels, self.substructure_labels = self.compute_boundary_and_substructure_labels()

    def compute_boundary_and_substructure_labels(self):
        boundary_indices = []
        boundary_labels = []
        substructure_labels = []

        for protac_idx, (smiles, poi_smile, e3_smile) in enumerate(zip(self.protac_smiles, self.poi_smiles, self.e3_smiles)):
            boundary_index = boundary_ligand_nodes_v2(smiles, poi_smile, e3_smile)
            boundary_indices.append(boundary_index)

            boundary_label = get_boundary_labels(smiles, poi_smile, e3_smile) #for boundary node pred
            boundary_labels.append(boundary_label)

            substructure_label = process_boundaries_to_substructure_labels(smiles, boundary_index[0], boundary_index[1])
            substructure_labels.append(substructure_label)
       
        return boundary_indices, boundary_labels, substructure_labels

    def initialize_mols_and_graphs(self):
        self.mols = [Chem.MolFromSmiles(smile) for smile in self.protac_smiles]
        self.graphs = [mol_to_simple_graph(mol) for mol in self.mols]

    def compute_graph_descriptors(self):
        self.graph_descriptors = [self.compute_descriptors_for_graph(graph) for graph in self.graphs]

    def compute_descriptors_for_graph(self, graph):
        descriptors = {descriptor: torch.from_numpy(graph_descriptor(graph, descriptor=descriptor)).type(torch.float32).reshape(-1, 1) for descriptor in self.graph_descriptor_list}
        return descriptors

    def initialize_link_prediction_attributes(self):
        for mol in self.mols:
            for atom in mol.GetAtoms():
                atom.SetProp('originalIdx', str(atom.GetIdx()))

        self.ring_systems = [find_connected_ring_systems(mol) for mol in self.mols]
        self.splittable_bonds = [find_non_ring_bonds(mol, system, exclude_bonds_connected_to_atoms_with_1_bond=True, datatype="list")
                                 for mol, system in zip(self.mols, self.ring_systems)]            
        
        if self.mode != "predict":
            self.initialize_boundary_and_splittable_bonds()
        
    def initialize_boundary_and_splittable_bonds(self):
        if self.precompute_splits:
            self.resulting_node_splits_from_all_splittable_bonds = [get_all_splits_from_all_splittable_bonds(mol, bonds) for mol, bonds in zip(self.mols, self.splittable_bonds)]
        self.boundary_bonds = [get_boundary_bonds_v2(mol, poi, e3) for mol, poi, e3 in zip(self.mols, self.poi_smiles, self.e3_smiles)]
        self.bond_labels = get_bond_labels(self.splittable_bonds, self.boundary_bonds, poi_label = 0, linker_label = 1, e3_label = 2)
        self.boundarybond_labels = get_boundarybond_labels(self.bond_labels)

    def initialize_node_prediction_attributes(self):
        if self.mode != "predict":
            self.substructure_labels = [process_boundaries_to_substructure_labels(smiles, boundary_nodes_indices[0], boundary_nodes_indices[1]) 
                                        for smiles, boundary_nodes_indices in zip(self.protac_smiles, self.boundary_indices)] #POI:0, Linker:1, E3:2
        for mol in self.mols:
            for atom in mol.GetAtoms():
                atom.SetProp('originalIdx', str(atom.GetIdx()))
        self.ring_systems = [find_connected_ring_systems(mol) for mol in self.mols]
        self.splittable_bonds = [find_non_ring_bonds(mol, system, exclude_bonds_connected_to_atoms_with_1_bond=True, datatype="list")
                                 for mol, system in zip(self.mols, self.ring_systems)]   
        if self.precompute_splits:
            self.resulting_node_splits_from_all_splittable_bonds = [get_all_splits_from_all_splittable_bonds(mol, bonds) for mol, bonds in zip(self.mols, self.splittable_bonds)]
            
            self.all_possible_linker_atom_idx_for_all_protacs = []
            self.all_possible_ligands_atom_idx_for_all_protacs = []
            for protac_idx, node_splits_from_all_splittable_bonds in enumerate(self.resulting_node_splits_from_all_splittable_bonds):
                splittable_bonds_idx = list(node_splits_from_all_splittable_bonds.keys())
                all_possible_linker_atom_idx = []
                all_possible_ligands_atom_idx = []
                num_nodes = self.mols[protac_idx].GetNumAtoms()
                protac_node_indices = set(list(range(num_nodes)))
                for i in range(len(splittable_bonds_idx)-1):
                    splittable_bonds_idx_slice = splittable_bonds_idx[i+1:]
                    for j in splittable_bonds_idx_slice:

                        atom_idx_split_0 = set(node_splits_from_all_splittable_bonds[i])
                        atom_idx_split_1 = set(node_splits_from_all_splittable_bonds[j])

                        if atom_idx_split_0.isdisjoint(atom_idx_split_1):
                            pass
                        elif len(atom_idx_split_0)>len(atom_idx_split_1):
                            atom_idx_split_0 = protac_node_indices - atom_idx_split_0
                        elif len(atom_idx_split_0)<len(atom_idx_split_1):
                            atom_idx_split_1 = protac_node_indices - atom_idx_split_1
                        else:
                            raise ValueError("One set should contain the other, but are the same size => Same predicted boundary bonds => Linker has no atoms")
                                    
                        linker_atom_idx_plus_split_1 = protac_node_indices - atom_idx_split_0
                        linker_atom_idx = linker_atom_idx_plus_split_1 - atom_idx_split_1

                        all_possible_linker_atom_idx.append(list(linker_atom_idx))
                        all_possible_ligands_atom_idx.append([list(atom_idx_split_0), list(atom_idx_split_1)])

                self.all_possible_linker_atom_idx_for_all_protacs.append(all_possible_linker_atom_idx)
                self.all_possible_ligands_atom_idx_for_all_protacs.append(all_possible_ligands_atom_idx)


    def compute_girvan_newman_encodings(self):
        self.girvan_newman_encodings = [torch.tensor(get_girvan_newman_encoding(smile), dtype=torch.float32) for smile in self.protac_smiles]

    def _generate_dataset(self, allowed_descriptors=None):
        # Convert SMILES to graph data objects, possibly simplifying features based on needs.
        self.dataset = [from_smiles(smile) for smile in self.protac_smiles]
        if self.node_descriptors == "rdkit":
            pass
        elif self.node_descriptors == "ones":
            for datapoint in self.dataset:
                datapoint.x = torch.ones((datapoint.x.size(0),1), dtype=torch.float32)
        elif self.node_descriptors == "empty":
            for datapoint in self.dataset:
                datapoint.x = torch.ones((datapoint.x.size(0),0), dtype=torch.float32)
        

        for idx, data in enumerate(self.dataset):
            # Initial node features based on 'rdkit' descriptors are assumed to be included by from_smiles.
            # Extend or replace node features based on node_descriptors.
            node_features = self.get_node_features(data, idx, allowed_descriptors)

            # Update node features for the datapoint.
            data.x = node_features

            # Process edges: ensuring the correct type for edge indices and attributes.
            data.edge_index = data.edge_index.type(torch.int64)
            data.edge_attr = data.edge_attr.type(torch.float32)
            data.x = data.x.type(torch.float32)

            # Update dataset with additional information based on mode and model_type.
            self.update_data_based_on_mode(data, idx)
            self.update_data_for_link_pred(data, idx)
            self.update_data_for_node_pred(data, idx)

        return self.dataset

    def get_node_features(self, data, idx, allowed_descriptors):
        # Base node features could be modified here based on descriptors and the model type.
        node_features = data.x

        if "girvan_newman" in self.node_descriptors and (allowed_descriptors is None or "girvan_newman" in allowed_descriptors):
            node_features = torch.cat((node_features, self.girvan_newman_encodings[idx].unsqueeze(1)), dim=1)

        if self.graph_descriptor_list:
            for descriptor_name, descriptor_tensor in self.graph_descriptors[idx].items():
                if allowed_descriptors is None or descriptor_name in allowed_descriptors:
                    node_features = torch.cat((node_features, descriptor_tensor), dim=1)

        return node_features

    def update_data_based_on_mode(self, data, idx):
        if self.mode != "predict":
            data.substructure_labels = torch.tensor(self.substructure_labels[idx], dtype=torch.long)
            if self.model_type == "boundary_pred":
                data.boundary_labels = torch.tensor(self.boundary_labels[idx], dtype=torch.long)

    def update_data_for_link_pred(self, data, idx):
        if self.model_type == "link_pred":
            data.splittable_bonds = torch.tensor(self.splittable_bonds[idx], dtype=torch.int64)
            data.graphs = self.graphs[idx]

            if self.precompute_splits:
                data.splittable_bond_idx_to_node_incides = self.resulting_node_splits_from_all_splittable_bonds[idx]

            if self.mode != "predict":
                data.boundary_bonds = torch.tensor(self.boundary_bonds[idx], dtype=torch.int64)
                data.bond_labels = torch.tensor(self.bond_labels[idx], dtype=torch.int64)
                data.boundarybond_labels = torch.tensor(self.boundarybond_labels[idx], dtype=torch.int64)

    def update_data_for_node_pred(self, data, idx):
        if self.model_type == "node_pred":
            
            data.graphs = self.graphs[idx]
            if self.precompute_splits:
                data.splittable_bond_idx_to_node_incides = self.resulting_node_splits_from_all_splittable_bonds[idx]
                
                data.all_possible_linker_atom_idx_for_all_protacs = self.all_possible_linker_atom_idx_for_all_protacs[idx]
                data.all_possible_ligands_atom_idx_for_all_protacs = self.all_possible_ligands_atom_idx_for_all_protacs[idx]

            if self.mode != "predict":
                data.substructure_labels = torch.tensor(self.substructure_labels[idx], dtype=torch.int64)

    def __len__(self):
        return len(self.protac_smiles)
   
    def __getitem__(self, idx):
        return self.dataset[idx]

    def set_mode(self, mode):
        self.mode = mode #to allow the retrival of graphs when doing postprocessing on the predictions or evaluations

    def set_model_type(self, model_type):
        self.model_type = model_type

    def set_use_graph_descriptors(self, graph_descriptor_list):
        self.graph_descriptor_list = graph_descriptor_list
    
    def set_node_descriptors(self, node_descriptors):
        self.node_descriptors = node_descriptors







class ProtacLoader:
    def __init__(self, dataset_paths={}, node_descriptors=[], graph_descriptors=[], model_type="link_pred", model_crossfold=False, precompute_splits=False):
        self.dataset_paths = dataset_paths
        self.datasets = {}
        self.model_type = model_type
        self.dataset_names = dataset_paths.keys()
        self.model_crossfold = model_crossfold
        self.node_descriptors = node_descriptors
        self.graph_descriptors = graph_descriptors 

    #loads a saved dict
    def load_datasets_from_file(self, dataset_path):
        if not os.path.exists(dataset_path):
            print(f"Dataset file not found: {dataset_path}")
            return
        with open(dataset_path, 'rb') as file:
            self.datasets = pickle.load(file)
        return self.datasets
    
    def save_datasets_to_file(self, dataset_path):
        if os.path.exists(dataset_path):
            print(f"Dataset file already exists: {dataset_path}")
            return
        with open(dataset_path, 'wb') as file:
            pickle.dump(self.datasets, file)


    
    #generate new data via PROTACDataset
    def load_dataframes(self):
        dataframes = {}
        for name, path in self.dataset_paths.items():
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
            dataframes[name] = pd.read_csv(path)
        return dataframes

    def initialize_datasets(self, dataframes, num_crossfolds=5, precompute_splits=False):
        self.precompute_splits = precompute_splits
        if self.model_crossfold:
            self.handle_cross_validation_sets(dataframes=dataframes,crossfolds=num_crossfolds)
        for name in tqdm(self.dataset_names, desc=f'Loading datasets'):
            if self.model_crossfold and name in ["Train", "Validation"]:
                continue
            if name in dataframes:
                self.datasets[name] = ProtacDataset(protac_input=dataframes[name], mode="train", 
                                                    model_type=self.model_type, node_descriptors=self.node_descriptors, 
                                                    graph_descriptor_list=self.graph_descriptors, precompute_splits=self.precompute_splits)
        return self.datasets

    def handle_cross_validation_sets(self, dataframes, crossfolds=5):
        """
        Handles the initialization of cross-validation datasets.
        """
        if 'Train' not in dataframes or 'Validation' not in dataframes:
            print("Training or Validation dataframes are missing for cross-validation.")
            return

        # Merge and shuffle training and validation dataframes, then assign cross-fold IDs
        trainval_df = self.merge_for_crossvalidation(dataframes)
        if trainval_df is None:
            print("Failed to merge training and validation datasets.")
            return

        trainval_df = self.assign_crossfold_ids(trainval_df, crossfolds=crossfolds)
        
        # Initialize datasets for each cross-validation fold
        self.datasets['train_CV'] = []
        self.datasets['val_CV'] = []

        for fold in tqdm(range(crossfolds), desc=f"Loading crossfolds"):
            # Filter the merged dataframe for the current fold's training and validation sets
            val_df = trainval_df[trainval_df['Crossfold_ID'] == fold]
            train_df = trainval_df[trainval_df['Crossfold_ID'] != fold]

            # Initialize ProtacDataset instances for the current fold
            train_dataset = ProtacDataset(protac_input=train_df, mode="train", 
                                        model_type=self.model_type, node_descriptors=self.node_descriptors, 
                                        graph_descriptor_list=self.graph_descriptors, precompute_splits=self.precompute_splits)
            val_dataset = ProtacDataset(protac_input=val_df, mode="train", 
                                        model_type=self.model_type, node_descriptors=self.node_descriptors, 
                                        graph_descriptor_list=self.graph_descriptors, precompute_splits=self.precompute_splits)

            # Store the initialized datasets
            self.datasets['train_CV'].append(train_dataset)
            self.datasets['val_CV'].append(val_dataset)

    #generate crossvalidation
    def merge_for_crossvalidation(self, dataframes):
        trainval_df = pd.concat([dataframes["Train"], dataframes["Validation"]]).sample(frac=1).reset_index(drop=True)
        print(f'All PROTACs are unique in trainvalset: {len(trainval_df["PROTAC SMILES"]) == len(trainval_df["PROTAC SMILES"].unique())}')
        return trainval_df

    def assign_crossfold_ids(self, dataframe, crossfolds=5):
        crossfold_ids = list(range(crossfolds)) * (len(dataframe) // crossfolds + 1)
        while len(crossfold_ids) > len(dataframe):
            crossfold_ids.pop()
        dataframe["Crossfold_ID"] = crossfold_ids
        return dataframe