
from protacdataset import ProtacDataset


from pipeline_functions import (avg,
                                get_substructure_smiles_function_v2,
                                aggregate_metrics_at_epoch,
                                process_predicted_boundaries_to_substructure_labels)

from data_curation_augmentation_splitting_functions import (compute_countMorgFP)

from rdkit import Chem, DataStructs
import networkx as nx

import pandas as pd
import copy
from contextlib import nullcontext
from statistics import median

import torch
import torch.nn as nn
import torch.nn.functional as F



from torch_geometric.loader import DataLoader
from torch_geometric.nn import (GraphConv, GCNConv, GATConv, SAGEConv, 
                                TransformerConv, BatchNorm, GraphNorm, 
                                MeanSubtractionNorm)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


compute_countMorgFP








class PROTACSplitter(torch.nn.Module):
    def __init__(self, init_params):
        super(PROTACSplitter, self).__init__()

        # Assign basic parameters from init_params for existing configurations
        self.assign_basic_parameters(init_params)
        
        # Activation function setup
        self.set_activation_function(init_params['activationfunction'])
        self.last_activation_function = init_params.get("last_activation_function", None)
        
        # Layer mapping for GNN layers
        self.layer_map = self.get_layer_map()

        # Initialize GNN layers based on configurations
        self.define_gnn_layers(init_params)

        if self.model_type == "link_pred":
            # Initialize output layers based on configurations
            self.output_layers = self.initialize_output_layers(init_params.get('output_layer_config', []))
            
            # NEW: Initialize boundary layers based on configurations
            self.boundary_layers = self.initialize_output_layers(init_params.get('boundary_layer_config', []))
        
        # Define normalization layers if applicable
        self.define_normalization_layers()
        
        # Additional model configurations
        self.configure_additional_model_parameters(init_params)
        
        # Softmax for boundary bond probabilities
        self.softmax_for_boundary_bond_probs = nn.Softmax(dim=0)


    def define_gnn_layers(self, init_params):
        self.layers = torch.nn.ModuleList()

        if self.gnn_layer_type == "GINConv":              
            raise ValueError("OBS! GINConv not implmented yet. It seems to use an MLP as input")
        else:
            self.GNNLayer = self.layer_map[self.gnn_layer_type]


            # First layer
        if self.use_edge_information:
            if self.gnn_layer_type == 'TransformerConv':
                self.layers.append(self.GNNLayer(init_params['node_feature_dim'], self.layer_dims[0], edge_dim=3, beta=self.TransformerConvBeta, heads=self.TransformerConvHeads, dropout=self.TransformerConvDropout))
            else:
                self.layers.append(self.GNNLayer(init_params['node_feature_dim'], self.layer_dims[0], edge_dim=3))
        else:
            self.layers.append(self.GNNLayer(init_params['node_feature_dim'], self.layer_dims[0]))


            # Hidden layers
        for i in range(0, len(self.layer_dims)-1):
            if self.gnn_layer_type == 'TransformerConv':
                self.layers.append(self.GNNLayer(self.layer_dims[i]*self.TransformerConvHeads, self.layer_dims[i+1], beta=self.TransformerConvBeta, heads=self.TransformerConvHeads, dropout=self.TransformerConvDropout))
            else:
                self.layers.append(self.GNNLayer(self.layer_dims[i], self.layer_dims[i+1]))

        

    def assign_basic_parameters(self, init_params):
        self.use_edge_information = init_params["use_edge_information"]
        self.use_jumping_knowledge = init_params['use_jumping_knowledge']
        self.use_skip_connections = init_params["use_skip_connections"]
        self.use_batch_normalization = init_params["use_batch_normalization"]
        self.use_graph_normalization = init_params["use_graph_normalization"]
        self.use_MeanSubtractionNorm = init_params["use_MeanSubtractionNorm"]
        self.dropout = init_params["dropout_rate"]
        self.regularization_types = init_params['regularization_types']
        self.gnn_layer_type = init_params['gnn_layer_type']
        self.model_type = init_params['model_type']
        self.layer_dims = init_params['layer_dims']
        self.final_layer_dim = init_params['final_layer_dim']
        self.layer_dims[-1] = self.final_layer_dim
        self.TransformerConvBeta = init_params['TransformerConvBeta']
        self.TransformerConvHeads = init_params['TransformerConvHeads']
        self.TransformerConvDropout = init_params['TransformerConvDropout']
        self.num_predicted_classes = init_params['num_predicted_classes']
        self.device = init_params['device']
        self.poi_label = 0
        self.e3_label = 2

    def set_activation_function(self, activation_type):
        if activation_type == "relu":
            self.ActFun = F.relu
        elif activation_type == "lrelu":
            self.ActFun = nn.LeakyReLU()
        # Extendable for other activation functions

    def get_layer_map(self):
        layer_map = {
            'GraphConv': GraphConv,
            'GCNConv': GCNConv,
            'GATConv': GATConv,
            'SAGEConv': SAGEConv,
            'TransformerConv': TransformerConv,
            # Placeholder for AttentiveFP if available and others
        }
        if 'GINConv' in layer_map:
            raise ValueError("OBS! GINConv not implemented yet. It seems to use an MLP as input")
        return layer_map

    def define_normalization_layers(self):
        if self.use_batch_normalization:
            self.batch_norm_layers = torch.nn.ModuleList([BatchNorm(dim) for dim in self.layer_dims[:-1]])
        if self.use_graph_normalization:
            self.graph_norm_layers = torch.nn.ModuleList([GraphNorm(dim) for dim in self.layer_dims[:-1]])
        if self.use_MeanSubtractionNorm:
            self.MeanSubtractionNorm_layers = torch.nn.ModuleList([MeanSubtractionNorm() for dim in self.layer_dims[:-1]])
   

    def initialize_output_layers(self, layer_configs):
        output_layers = torch.nn.ModuleDict()  
        for idx, config in enumerate(layer_configs):
            layer_type = config.get('type', 'linear')  
            layer_key_prefix = f"{layer_type}_{idx}"  

            depth = config.get('depth', 1)  # Default depth is 1
            base_in_features = self.layer_dims[-1] if config['in_features'] == -1 else config['in_features']

            in_features = base_in_features
            if 'global' in layer_type:
                in_features += base_in_features
            if self.gnn_layer_type == 'TransformerConv' and idx == 0:  # For first layer with TransformerConv
                in_features += base_in_features * (self.TransformerConvHeads - 1)
            if 'symmetric' in layer_type:
                in_features += base_in_features
            
            for d in range(depth):
                layer_key = f"{layer_key_prefix}_{d}"  # Unique key for each layer in depth
                if d == depth - 1:  # Last layer
                    out_features = config['out_features']
                else:
                    out_features = config.get('intermediate_features', base_in_features)
                layer = torch.nn.Linear(in_features, out_features)
                in_features = out_features
                
                output_layers[layer_key] = layer

        return output_layers
    
    def configure_additional_model_parameters(self, init_params):
        self.pooling_layers = torch.nn.ModuleList()
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.states = []
        

    


    def process_gnnlayers(self, z, edge_index, edge_attr):
        skip_z = [] if self.use_skip_connections else None
        
        # Apply GNN layers
        for layer_idx, layer in enumerate(self.layers):
            
            if self.use_edge_information and layer_idx == 0:
                z = layer(z, edge_index, edge_attr=edge_attr)
            else: 
                z = layer(z, edge_index)
                
            if self.use_batch_normalization and layer_idx < len(self.batch_norm_layers):
                z = self.batch_norm_layers[layer_idx](z)
            if self.use_graph_normalization and layer_idx < len(self.graph_norm_layers):
                z = self.graph_norm_layers[layer_idx](z)
            if self.use_MeanSubtractionNorm and layer_idx < len(self.MeanSubtractionNorm_layers):
                z = self.MeanSubtractionNorm_layers[layer_idx](z)

            z = self.ActFun(z)
            
            if self.use_skip_connections:
                skip_z += [z]

            z = F.dropout(z, p=self.dropout, training=self.training)
           
            if self.use_skip_connections:
                if len(skip_z)>1:
                    if z.size() == skip_z[-2].size():
                        z = z + skip_z[-2]

        return z

    def process_outlayers(self, input, layers):
        "Process input through a sequence of layers."
        for idx, (layer_key, layer) in enumerate(layers.items()):  # Apply all but the last layer with activation
            if idx == len(layers)-1:
                #last layer has been defined. Skip ActFun and dropout
                break
            if idx == 0 and 'symmetric' in layer_key:
                input = layer(input[0]) + layer(input[1])
            else:
                input = layer(input)
            input = self.ActFun(input)
            # Apply dropout between intermediate layers if enabled
            #input = F.dropout(input, p=self.dropout, training=self.training)
        # No activation after the last layer
        return layer(input)
    

    def forward(self, batch, edges_to_predict = None):
        z, edge_index = batch["x"], batch["edge_index"]

        if self.use_edge_information:
            edge_attr = batch["edge_attr"]
        else:
            edge_attr = None

        z = self.process_gnnlayers(z, edge_index, edge_attr)

        if self.model_type == "link_pred":
            z_edges_0 = torch.cat((z[edges_to_predict[:,0]], z[edges_to_predict[:,1]]), dim=1) if edges_to_predict is not None else None
            z_edges_1 = torch.cat((z[edges_to_predict[:,1]], z[edges_to_predict[:,0]]), dim=1) if edges_to_predict is not None else None
            z_edges = (z_edges_0, z_edges_1)
        
            if self.model_type == "link_pred" and edges_to_predict is None:
                raise ValueError("edges_to_predict was is None")

            # Boundary location pred
            y_location = self.process_outlayers(z_edges, self.boundary_layers)

            # Boundary type pred
            y_type = self.process_outlayers(z if edges_to_predict is None else z_edges, self.output_layers)
        
        elif self.model_type == "node_pred" or self.model_type == "boundary_pred":
            y_type = z
            y_location = None

        return y_type, y_location







    def get_model_type(self):
        return self.model_type



    def get_protac_data(self, protac_smiles, precompute_splits=False):
        return ProtacDataset(protac_input=protac_smiles, 
                             mode="predict", 
                             model_type=self.model_type,
                             node_descriptors=self.node_descriptors, 
                             graph_descriptor_list=self.graph_descriptor_list,
                             precompute_splits = precompute_splits) #as smiles or list of smiles is fed, and not Dataframe with POI and E3 smiles, only mode="predict" can be used
        


    def train_model(self, train_params):
        
        datasets_dict = train_params['datasets_dict']
        criterion = train_params['criterion']
        lr = train_params['lr']
        optimizer = train_params['optimizer'](self.parameters(), lr=lr)
        epochs = train_params['epochs']
        max_epochs = train_params['max_epochs']

        val_early_stopping = train_params['val_early_stopping']
        median_over_n_val_losses = train_params['median_over_n_val_losses']
        min_over_n_val_losses = train_params['min_over_n_val_losses']
        stop_if_val_acc_below_x_at_n_list = train_params['stop_if_val_acc_below_x_at_n_list']

        print_every_n_epochs = train_params['print_every_n_epochs']
        self.compute_pretrained_values = train_params['compute_pretrained_values']
        self.compute_rand_accuracy = train_params['compute_rand_accuracy']

        fp_function = train_params['fp_function']
        e3_database_fps = fp_function(train_params['e3_library'])
        poi_database_fps = fp_function(train_params['poi_library'])
        self.use_new_postprocessing_algorithm = train_params['use_new_postprocessing_algorithm']
        self.param_to_opt = train_params["param_to_opt"]

        self.weight_class_loss = train_params["weight_class_loss"]
        
        if "Train" in datasets_dict:
            self.train_set_name = "Train"
        elif "train_CV" in datasets_dict:
            self.train_set_name = "train_CV"
        else:
            raise ValueError("No dataset named 'Train' or 'train_CV'!")
        
        if "Validation" in datasets_dict:
            self.val_set_name = "Validation"
        elif "val_CV" in datasets_dict:
            self.val_set_name = "val_CV"
        else:
            raise ValueError("No dataset named 'Train' or 'train_CV'!")
    
        if self.compute_rand_accuracy is True:
            datasets_dict["Dummy"] = datasets_dict[self.train_set_name] #do random predictions on training protacs
        
        
        self.training = False


        if not hasattr(self, 'measures'):
            self.measures = {'loss': {}, 
                            'boundary_bond_probs': {},
                            'metrics': {},
                            'validity_fraction': {},
                            'flip_fraction': {}
                            }

        
        

        self.node_descriptors = None #training_data.node_descriptors
        self.graph_descriptor_list = None #training_data.graph_descriptor_list

        

        self.total_nodes = {}

        
           
        loaders = {}
        for i, (dataset_name, dataset) in enumerate(datasets_dict.items()):
            if dataset_name == "Dummy" and self.compute_rand_accuracy is False:
                continue

            #set params for dataloaders
            if dataset_name == "Dummy":
                shuffle = True
                batch_size = 1
            elif dataset_name == self.train_set_name:
                shuffle = True
                batch_size = train_params['batch_size']
            else:
                shuffle = False
                batch_size = train_params['batch_size']

            loaders[dataset_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

            #metrics for dataset
            for measure_key in self.measures.keys():
                self.measures[measure_key][dataset_name] = {}
 
            #make sure same node_descriptors and graph_descriptor_list for all datasets
            if self.node_descriptors is None:
                self.node_descriptors = dataset.node_descriptors
            elif self.node_descriptors != dataset.node_descriptors:
                raise ValueError(f"self.node_descriptors: {self.node_descriptors} != dataset.node_descriptors ({dataset_name}): {dataset.node_descriptors}")
            if self.graph_descriptor_list is None:
                self.graph_descriptor_list = dataset.graph_descriptor_list
            elif self.graph_descriptor_list != dataset.graph_descriptor_list:
                raise ValueError(f"self.graph_descriptor_list: {self.graph_descriptor_list} != dataset.graph_descriptor_list ({dataset_name}): {dataset.graph_descriptor_list}")

            self.total_nodes[dataset_name] = None 

        
        #Store this in model to simplify code and skip recalculating on every epoch (slow) 
        self.total_train_nodes = None
        self.total_val_nodes = None
        num_testsets = max([0, len(datasets_dict)-2])
        self.total_tests_nodes = [None for _ in range(num_testsets)] 




        #self.structure_types_template = {"PROTAC": [], "POI": [], "LINKER": [], "E3": [], "LIGANDS": []}
        metrics_template_protac = {'Atoms_wrong': {}, 'Accuracy': {}}
        metrics_template_ligands = {'Atoms_wrong': {}, 'Accuracy': {}, 'Precision': {}, 'Recall': {}, "F1": {}}
        metrics_template_substructures = {'Atoms_wrong': {}, 'Precision': {}, 'Recall': {}, "F1": {}}

        self.large_metrics_template = {
                        "model": {"PROTAC": copy.deepcopy(metrics_template_protac), 
                                   "LIGANDS": copy.deepcopy(metrics_template_ligands), 
                                   "LINKER": copy.deepcopy(metrics_template_substructures), 
                                   "POI": copy.deepcopy(metrics_template_substructures), 
                                   "E3": copy.deepcopy(metrics_template_substructures) }, 
                        
                        "model+check_library": {"PROTAC": copy.deepcopy(metrics_template_protac), 
                                                "LIGANDS": copy.deepcopy(metrics_template_ligands), 
                                                "LINKER": copy.deepcopy(metrics_template_substructures), 
                                                "POI": copy.deepcopy(metrics_template_substructures), 
                                                "E3": copy.deepcopy(metrics_template_substructures) }
                        }
        
        for dataset_name in self.measures["metrics"].keys():
            self.measures["metrics"][dataset_name] = copy.deepcopy(self.large_metrics_template)

            
        for dataset_name in self.measures["validity_fraction"].keys():
            self.measures["validity_fraction"][dataset_name] = {'VALID SPLIT': {}, 'POI SMILES': {}, 'LINKER SMILES': {}, 'E3 SMILES': {}}


        self.measures_template = copy.deepcopy(self.measures)


        # Training loop
        self.epoch = 0
        delta_val_criteria = 1
        self.have_computed_pretrained_values = False
        while self.epoch < max_epochs and ((self.epoch < epochs and val_early_stopping is False) or (val_early_stopping and delta_val_criteria >= 0)):
            self.epoch += 1
            measures_epoch = copy.deepcopy(self.measures_template)

            for dataset_name, dataset in datasets_dict.items():
                
                
                if dataset_name == self.train_set_name:
                    train_or_eval = "train"
                    get_loss = True
                elif dataset_name == self.val_set_name:
                    train_or_eval = "eval"
                    get_loss = True
                else: #test sets
                    train_or_eval = "eval"
                    get_loss = False
                
                if self.compute_pretrained_values and self.have_computed_pretrained_values is False:
                    train_or_eval = "eval"
                    self.have_computed_pretrained_values = True
                    self.epoch =- 1  

                measures_epoch = self._train_or_evaluate_epoch(dataset_name=dataset_name, 
                                                                 loader=loaders[dataset_name], 
                                                                 criterion=criterion, 
                                                                 optimizer=optimizer,
                                                                 train_or_eval=train_or_eval, 
                                                                 e3_database_fps=e3_database_fps, 
                                                                 poi_database_fps=poi_database_fps, 
                                                                 fp_function=fp_function,
                                                                 measures_epoch=measures_epoch,
                                                                 get_loss = get_loss)
            
            self._store_measures(measures_epoch=measures_epoch) #not yet implemented

            #save current parameters
            self.states.append(self.state_dict())

            # Print metrics every n epochs
            if self.epoch % print_every_n_epochs == 0:
                self._print_epoch_metrics()


            #Early stopping
            if val_early_stopping:
                delta_val_criteria = self._early_stopping(max_epochs, 
                                                          median_over_n_val_losses, 
                                                          stop_if_val_acc_below_x_at_n_list, 
                                                          min_over_n_val_losses)

        return self.measures



    def _train_or_evaluate_epoch(self, dataset_name, loader, criterion, train_or_eval, optimizer=None, e3_database_fps=None, poi_database_fps=None, fp_function = None, max_num_rand_pred_protacs=200, get_loss = True, measures_epoch = None):        
        total_loss = 0
        count_rand_pred = 0

        
        if self.total_nodes[dataset_name] is None:
            total_nodes = 0
            compute_total_nodes = True
        else:
            total_nodes = self.total_nodes[dataset_name]
            compute_total_nodes = False

        if train_or_eval == "train":
            self.training = True #for dropout
            self.train()
        elif train_or_eval == "eval":
            self.eval()


        
        if measures_epoch is None:
            measures_epoch = copy.deepcopy(self.measures_template) #may be the case when predicting



        #initialize this epochs variables
        #CHECK: validity_fraction -> dataset_name -> validity dict -> epoch -> fractions
        for validity_key in measures_epoch["validity_fraction"][dataset_name].keys():
            measures_epoch["validity_fraction"][dataset_name][validity_key] = 0 #start as a counter, divide by total in the end to get fraction

        #CHECK: flip_fraction -> dataset_name -> epoch -> fraction
        measures_epoch["flip_fraction"][dataset_name] = 0 #start as a counter, divide by total in the end to get fraction

        #metrics -> dataset_name -> accuracy_origin -> structure_type -> metric -> epoch -> data
        for accuracy_origin in measures_epoch["metrics"][dataset_name].keys():
            for structure_type in measures_epoch["metrics"][dataset_name][accuracy_origin].keys():
                for metric_key in measures_epoch["metrics"][dataset_name][accuracy_origin][structure_type].keys():
                    if metric_key == "Atoms_wrong":
                        data_type = dict
                    else:
                        data_type = list
                    measures_epoch["metrics"][dataset_name][accuracy_origin][structure_type][metric_key] = data_type()

        

        protac_counter = 0 
        with nullcontext() if train_or_eval == "train" else torch.no_grad(): 
            for batch in loader:
                
                batch = batch.to(self.device)


                if dataset_name == "Dummy":
                    if count_rand_pred>max_num_rand_pred_protacs:
                        break
                    else:
                        count_rand_pred += len(batch)
                    randomize_y = True
                else:
                    randomize_y = False

                
                if train_or_eval == "train":
                    optimizer.zero_grad()  # Reset gradients for next batch

                y_type, y_location, predictions_and_prob, boundary_bond_probs_dataset = self.batch_to_output_and_classpredictions(batch=batch, e3_database_fps=e3_database_fps, poi_database_fps=poi_database_fps, fp_function=fp_function, randomize_y=randomize_y)
                measures_epoch = self._count_correct_and_total_nodes(batch=batch, 
                                                                      predictions_and_prob=predictions_and_prob, 
                                                                      dataset_name=dataset_name,
                                                                      measures_epoch = measures_epoch)

                if get_loss:
                    loss = self._compute_loss(criterion, batch, y_type, y_location)
                    total_loss += loss.item() 

                if train_or_eval == "train":
                    loss.backward() 
                    optimizer.step()  

                if compute_total_nodes:
                    total_nodes += batch.num_nodes

                protac_counter += len(batch)

        # Store total nodes for future epochs
        if compute_total_nodes: 
            self.total_nodes[dataset_name] = total_nodes

        #Maksure to set to false for dropout
        if self.training:
            self.training = False

                
        measures_epoch['loss'][dataset_name] = (total_loss) / total_nodes  
        measures_epoch["flip_fraction"][dataset_name] = measures_epoch["flip_fraction"][dataset_name] / protac_counter

        for validity_key in measures_epoch["validity_fraction"][dataset_name].keys():
            measures_epoch["validity_fraction"][dataset_name][validity_key] = measures_epoch["validity_fraction"][dataset_name][validity_key] / protac_counter
            
        
        return measures_epoch
            

    def _store_measures(self, measures_epoch):
        for dataset_name in self.measures["validity_fraction"].keys():
            #loss
            self.measures['loss'][dataset_name][self.epoch] = measures_epoch['loss'][dataset_name]

            #validity fraction
            for validity_key in measures_epoch["validity_fraction"][dataset_name].keys():
                self.measures["validity_fraction"][dataset_name][validity_key][self.epoch] = measures_epoch["validity_fraction"][dataset_name][validity_key]

            #flips
            self.measures["flip_fraction"][dataset_name][self.epoch] = measures_epoch["flip_fraction"][dataset_name]

            #metrics
            for accuracy_origin in self.measures["metrics"][dataset_name].keys():
                for structure_type in self.measures["metrics"][dataset_name][accuracy_origin].keys():
                    for metric_type in self.measures["metrics"][dataset_name][accuracy_origin][structure_type].keys():
                        self.measures["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][self.epoch] = measures_epoch["metrics"][dataset_name][accuracy_origin][structure_type][metric_type]



    def _count_correct_and_total_nodes(self, batch, predictions_and_prob, dataset_name, measures_epoch):
        

        true_protac_nodes_dataset = batch["substructure_labels"].cpu()


        pred_protac_nodes_dataset_dict = {}
        for accuracy_origin in predictions_and_prob.keys():
            pred_protac_nodes_dataset_dict[accuracy_origin] = predictions_and_prob[accuracy_origin][0].cpu()
            

        for protac_idx in range(len(batch)):
            start_idx = batch.ptr[protac_idx]
            end_idx = batch.ptr[protac_idx+1]      

            protac_smiles = batch[protac_idx].smiles
            true_protac_nodes = true_protac_nodes_dataset[start_idx: end_idx]
            
            true_poi_nodes_bool = 0 == true_protac_nodes
            true_linker_nodes_bool = 1 == true_protac_nodes
            true_e3_nodes_bool = 2 == true_protac_nodes
            true_ligand_nodes_bool = (true_poi_nodes_bool) | (true_e3_nodes_bool)

            for accuracy_origin in self.measures["metrics"][dataset_name].keys():
                
                pred_protac_nodes = pred_protac_nodes_dataset_dict[accuracy_origin][start_idx: end_idx]
                if pred_protac_nodes.size(0) == 0:
                    if accuracy_origin == "model+check_library":
                        pred_protac_nodes = pred_protac_nodes_dataset_dict["model"][start_idx: end_idx]
                    else:
                        print(start_idx)
                        print(end_idx)
                        print(len(batch))
                        print(batch)
                        print(pred_protac_nodes_dataset_dict[accuracy_origin])
                        print(predictions_and_prob[accuracy_origin][0])
                        raise ValueError("No prediction was made!")

                pred_poi_nodes_bool = 0 == pred_protac_nodes
                pred_linker_nodes_bool = 1 == pred_protac_nodes
                pred_e3_nodes_bool = 2 == pred_protac_nodes
                pred_ligand_nodes_bool = pred_poi_nodes_bool | pred_e3_nodes_bool

                for structure_type in self.measures["metrics"][dataset_name][accuracy_origin].keys():

                    if structure_type == "PROTAC":
                        true_structure_nodes = true_protac_nodes
                        pred_structure_nodes = pred_protac_nodes
                    elif structure_type == "POI":
                        true_structure_nodes = true_poi_nodes_bool
                        pred_structure_nodes = pred_poi_nodes_bool
                    elif structure_type == "LINKER":
                        true_structure_nodes = true_linker_nodes_bool
                        pred_structure_nodes = pred_linker_nodes_bool
                    elif structure_type == "E3":
                        true_structure_nodes = true_e3_nodes_bool
                        pred_structure_nodes = pred_e3_nodes_bool
                    elif structure_type == "LIGANDS":
                        true_structure_nodes = true_ligand_nodes_bool
                        pred_structure_nodes = pred_ligand_nodes_bool

                    if sum(true_structure_nodes) == 0:
                        print(true_structure_nodes)
                        print(pred_structure_nodes)
                        print(structure_type)
                        print(protac_smiles)
                        print(true_protac_nodes_dataset[start_idx: end_idx])

                    metrics_structure = self._calc_eval_metrics(true_structure_nodes=true_structure_nodes,
                                                                pred_structure_nodes=pred_structure_nodes,
                                                                structure_type = structure_type)
                    
                    
                    
                    
                    
                    for metric_type in metrics_structure[structure_type].keys():
                        if isinstance(measures_epoch["metrics"][dataset_name][accuracy_origin][structure_type][metric_type], dict): #eg. for Atoms_wrong
                            value_to_dict = metrics_structure[structure_type][metric_type]
                            if value_to_dict not in measures_epoch["metrics"][dataset_name][accuracy_origin][structure_type][metric_type]: 
                                measures_epoch["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][value_to_dict] = 0
                            measures_epoch["metrics"][dataset_name][accuracy_origin][structure_type][metric_type][value_to_dict] += 1
                        elif isinstance(measures_epoch["metrics"][dataset_name][accuracy_origin][structure_type][metric_type], list):
                            measures_epoch["metrics"][dataset_name][accuracy_origin][structure_type][metric_type].append(metrics_structure[structure_type][metric_type])
                        else:
                            raise TypeError("data structure is neither list or dict!")


                

                substructures_smi, substructure_smi_with_attachmentpoints = get_substructure_smiles_function_v2(protac_smiles=protac_smiles, 
                                                                                                            class_predictions=pred_protac_nodes, #pred_protac_nodes, 
                                                                                                            boundary_bonds =  (None, None)) #if none, will calculate it, which will work for any model_type. If not a perfect split, then invalid smiles with attachmentpoints. Smiles withoutattachmentpoints may or may not be valid still.
                
                
                if accuracy_origin == "model":
                    ligand_linker_acc = measures_epoch["metrics"][dataset_name][accuracy_origin]["LIGANDS"]["Accuracy"][-1]
                    protac_acc = measures_epoch["metrics"][dataset_name][accuracy_origin]["PROTAC"]["Accuracy"][-1]

                    if ligand_linker_acc > protac_acc:
                        measures_epoch["flip_fraction"][dataset_name] += 1


                    for substructure_key, substructure_smiles in substructures_smi.items():
                        if substructure_smiles != None and '.' not in substructure_smiles:
                            measures_epoch["validity_fraction"][dataset_name][substructure_key] += 1



                    if None not in list(substructure_smi_with_attachmentpoints.values()):
                        measures_epoch["validity_fraction"][dataset_name]["VALID SPLIT"] += 1

        return measures_epoch
       

    def _calc_eval_metrics(self, true_structure_nodes, pred_structure_nodes, structure_type):
        metrics_structure = {structure_type: {}}

        

        if structure_type != "PROTAC":
            average = "binary"
            
            precision = precision_score(y_true=true_structure_nodes, y_pred=pred_structure_nodes, average=average) if "precision" in self.param_to_opt or self.param_to_opt == [None] else 0
            recall = recall_score(y_true=true_structure_nodes, y_pred=pred_structure_nodes, average=average) if "recall" in self.param_to_opt or self.param_to_opt == [None] else 0
            f1 = f1_score(y_true=true_structure_nodes, y_pred=pred_structure_nodes, average=average) if "f1" in self.param_to_opt or self.param_to_opt == [None] else 0

            metrics_structure[structure_type]["Precision"] = precision
            metrics_structure[structure_type]["Recall"] = recall
            metrics_structure[structure_type]["F1"] = f1

        atoms_wrong_in_protac = (true_structure_nodes != pred_structure_nodes).sum().item() #Total number of wrong predictions. Not using "FP" or "FP" as it is multiclass
        metrics_structure[structure_type]["Atoms_wrong"] = atoms_wrong_in_protac

        

        if structure_type in ["PROTAC", "LIGANDS"]:
            accuracy = accuracy_score(y_true=true_structure_nodes, y_pred=pred_structure_nodes) if "accuracy" in self.param_to_opt or self.param_to_opt == [None] else 0
            metrics_structure[structure_type]["Accuracy"] = accuracy
        
        return metrics_structure
            



    def _print_epoch_metrics(self, epoch = None):
        if epoch is None or epoch == -1:
            epoch = self.epoch
        
        print(f"Epoch {epoch} Metrics:")

        printout_dict = aggregate_metrics_at_epoch(output=self.measures, epoch=epoch)

        printout_df = pd.DataFrame(printout_dict)
        row_names = [dataset_name for dataset_name in self.measures['metrics'].keys()]
        printout_df.index = row_names
        print(printout_df.round(1))
        


    def _early_stopping(self, max_epochs, median_over_n_val_losses, stop_if_val_acc_below_x_at_n_list, min_over_n_val_losses):
        delta_val_criteria = 1
        stopping_cause = "Mistaken stopping - Look for error in _early_stopping()"
        if max_epochs == self.epoch:
            delta_val_criteria = -1
            stopping_cause = "Stopping: Max epoch reached"
        elif median_over_n_val_losses is not None and len(self.metrics['avg_val_loss']) >= 2*median_over_n_val_losses:
            median_last_n_to_2n_epochs = median(self.metrics['avg_val_loss'][-2*median_over_n_val_losses:-median_over_n_val_losses])
            median_last_n_epochs = median(self.metrics['avg_val_loss'][-median_over_n_val_losses:]) 
            delta_val_criteria = median_last_n_to_2n_epochs - median_last_n_epochs # continue if delta_val_criteria is positive -> validation loss is decreasing
            stopping_cause = "Early stopping: Increaing median validation loss"
        elif stop_if_val_acc_below_x_at_n_list is not None: #a list of criteria on validation accuracy - val acc must be above  val_frac_criteria   at  n_epochs
            for stop_if_val_acc_below_x_at_n_dict in stop_if_val_acc_below_x_at_n_list:
                n_epochs = stop_if_val_acc_below_x_at_n_dict['n_epochs']
                val_frac_criteria = stop_if_val_acc_below_x_at_n_dict['val_frac_criteria']
                if self.epoch >= n_epochs:
                    avg_val_acc = avg(self.measures['metrics'][self.val_set_name]["model"]["PROTAC"]["Accuracy"][self.epoch])
                    delta_val_criteria = avg_val_acc - val_frac_criteria
                    stopping_cause = f"Early stopping: Validation accuracy below {val_frac_criteria} at or after {n_epochs} epochs"
        elif min_over_n_val_losses is not None and self.epoch >= 2*min_over_n_val_losses:
            val_losses_list = list(self.measures['loss'][self.val_set_name].values())
            min_last_n_to_2n_epochs = min(val_losses_list[-2*min_over_n_val_losses:-min_over_n_val_losses])
            min_last_n_epochs = min(val_losses_list[-min_over_n_val_losses:]) 
            delta_val_criteria = min_last_n_to_2n_epochs - min_last_n_epochs # continue if delta_val_criteria is positive -> validation loss is decreasing
            stopping_cause = f"Early stopping: Increasing lowest validation loss (The lowest validation loss between epochs {self.epoch-2*min_over_n_val_losses} and {self.epoch-1*min_over_n_val_losses} is lower than the lowest validation loss between epochs {self.epoch-1*min_over_n_val_losses+1} and {self.epoch})"

        if delta_val_criteria < 0:
            print(stopping_cause)
        
        return delta_val_criteria


    def _predict_bond_idx(self, raw_prediction_protac):
        pred_poi_boundary_bond_idx, _, pred_e3_boundary_bond_idx = raw_prediction_protac.argmax(dim=0)

                
                
        n_highest_pred_node = 1
        if pred_poi_boundary_bond_idx == pred_e3_boundary_bond_idx:
            raw_prediction_protac_temporary = torch.clone(raw_prediction_protac)    #SLOW?   #TopK is slow
        while pred_poi_boundary_bond_idx == pred_e3_boundary_bond_idx:          
                           
            poi_pred_val = raw_prediction_protac_temporary[pred_poi_boundary_bond_idx, self.poi_label]
            e3_pred_val = raw_prediction_protac_temporary[pred_poi_boundary_bond_idx, self.e3_label]
            if e3_pred_val>poi_pred_val:
                        #overwrite the value of the POI
                raw_prediction_protac_temporary[pred_poi_boundary_bond_idx,self.poi_label] = float('-inf')  #min_vals[0, 0].item()
                        #find the index of second most likely POI 
                pred_poi_boundary_bond_idx = torch.topk(raw_prediction_protac_temporary[:,self.poi_label], k=n_highest_pred_node, dim=0).indices.tolist()[-1]
                        #get the index
                pred_poi_boundary_bond_idx, _, pred_e3_boundary_bond_idx = raw_prediction_protac_temporary.argmax(dim=0)
            else:
                raw_prediction_protac_temporary[pred_e3_boundary_bond_idx,self.e3_label] = float('-inf') #min_vals[0, 2].item() #redefine the highest E3 to the lowest value 
                pred_e3_boundary_bond_idx = torch.topk(raw_prediction_protac_temporary[:,self.e3_label], k=n_highest_pred_node, dim=0).indices.tolist()[-1]
                pred_poi_boundary_bond_idx, _, pred_e3_boundary_bond_idx = raw_prediction_protac_temporary.argmax(dim=0) #Get new highest max values (for E3)
                    
                    
            n_highest_pred_node +=1
        
        return pred_poi_boundary_bond_idx, pred_e3_boundary_bond_idx


    def _predict_node_labels_and_probabilities_for_batch(self, batch, raw_prediction, splittable_bonds_ptr = None, e3_database_fps=None, poi_database_fps=None, fp_function=None, predicted_boundary_bonds_dict={}):
        boundary_bond_probs = []
        if self.model_type == "boundary_pred":
            class_predictions = torch.empty(size=(0,), dtype=torch.float32)
            probabilites = torch.empty(size=(0,self.num_predicted_classes), dtype=torch.float32)
            probabilites = probabilites.to(self.device)

            for protac_idx in range(len(batch)):
                
                protac_smiles = batch[protac_idx]["smiles"]
                start_row_idx = batch["ptr"][protac_idx].item()
                end_row_idx = batch["ptr"][protac_idx+1].item()
                raw_prediction_protac =  raw_prediction[start_row_idx:end_row_idx]  #chunck up raw_prediction into each protac
                class_prediction_protac = process_predicted_boundaries_to_substructure_labels(protac_smiles=protac_smiles, raw_boundary_prediction=raw_prediction_protac)
                probabilites_protac = F.softmax(raw_prediction_protac, dim=1)

                """if 0 not in class_prediction_protac or 1 not in class_prediction_protac or 2 not in class_prediction_protac:
                    mol = Chem.MolFromSmiles(protac_smiles)
                    G, pos = make_graph_with_pos(protac_smiles)
                    colors = []
                    if colors == []:
                        for label in class_prediction_protac:
                            if label.item() == 0: #POI
                                colors.append('red')
                            elif label.item() == 1: #linker
                                colors.append('gray')
                            elif label.item() == 2: #E3
                                colors.append('blue')
                    if len(pos) != len(colors):
                        raise ValueError("Different lengths between 'pos' and 'colors'")                              #This error occurs only sometimes......
                    nx.draw_networkx(G, pos=pos,  node_color=colors, with_labels=False, node_size = 50)
                    plt.axis('equal')
                    plt.show()
                    print(class_prediction_protac)
                    print(raw_prediction_protac)
                    raise ValueError("missing predicted class") """

                class_predictions = torch.cat((class_predictions, class_prediction_protac), dim=0)
                probabilites = torch.cat((probabilites, probabilites_protac), dim=0)

            class_predictions_checked_against_library = class_predictions.clone()

        elif self.model_type == "node_pred": 
            class_predictions = torch.empty(size=(0,), dtype=torch.float32) #On CPU when created
            class_predictions_checked_against_library = torch.empty(size=(0,), dtype=torch.float32) #On CPU when created
            
            probabilites = torch.empty(size=(0,), dtype=torch.float32)  #not yet implemented

            class_predictions = raw_prediction.argmax(dim=1)
            probabilites = F.softmax(raw_prediction, dim=1)
            class_predictions_checked_against_library = class_predictions.clone()
        elif self.model_type == "spit_pred":
            if True: 
                poi_diff_prob = raw_prediction[:,0] - raw_prediction[:,1] - raw_prediction[:,2]
                linker_diff_prob = raw_prediction[:,1] - raw_prediction[:,0] - raw_prediction[:,2]
                e3_diff_prob = raw_prediction[:,2] - raw_prediction[:,0] - raw_prediction[:,1]
                #ligand_diff_prob = -linker_diff_prob
                for protac_idx in range(len(batch)):
                    protac_data = batch[protac_idx]
                    #print(protac_data)

                    start_row_idx = batch.ptr[protac_idx]
                    end_row_idx = batch.ptr[protac_idx+1]
                    raw_prediction_protac =  raw_prediction[start_row_idx:end_row_idx]

                    protac_poi_diff_prob =  poi_diff_prob[start_row_idx:end_row_idx]
                    protac_linker_diff_prob =  linker_diff_prob[start_row_idx:end_row_idx]
                    protac_e3_diff_prob =  e3_diff_prob[start_row_idx:end_row_idx]
                    #protac_ligand_diff_prob =  ligand_diff_prob[start_row_idx:end_row_idx]

                    best_diff_prob = 0
                    best_split_idx = None
                    #print(protac_data[0])
                    for split_idx, (linker_atoms, (ligand_0_atoms, ligand_1_atoms)) in enumerate(zip(protac_data.all_possible_linker_atom_idx_for_all_protacs, protac_data.all_possible_ligands_atom_idx_for_all_protacs)):
                        #print(protac_linker_diff_prob.size())
                        #print(ligand_0_atoms)
                        #print(-protac_linker_diff_prob[ligand_0_atoms] )
                        ligand_0_diff_prob = -protac_linker_diff_prob[ligand_0_atoms]
                        ligand_1_diff_prob = -protac_linker_diff_prob[ligand_1_atoms]
                        ligands_diff_prob = ligand_0_diff_prob.sum().item() + ligand_1_diff_prob.sum().item()
                        #print(ligands_diff_prob)
                        linker_diff_prob = protac_linker_diff_prob[linker_atoms].sum().item()

                        

                        if best_diff_prob < (ligands_diff_prob + linker_diff_prob):
                            best_diff_prob = ligands_diff_prob + linker_diff_prob
                            best_split_idx = split_idx
                    
                    
                    linker_atoms = protac_data.all_possible_linker_atom_idx_for_all_protacs[best_split_idx]
                    ligand_0_atoms, ligand_1_atoms = protac_data.all_possible_ligands_atom_idx_for_all_protacs[best_split_idx]
                    score_poie3 = protac_poi_diff_prob[ligand_0_atoms].sum().item() + protac_e3_diff_prob[ligand_1_atoms].sum().item()
                    score_e3poi = protac_poi_diff_prob[ligand_1_atoms].sum().item() + protac_e3_diff_prob[ligand_0_atoms].sum().item()
                    if score_poie3 > score_e3poi:
                        poi_atoms = ligand_0_atoms
                        e3_atoms = ligand_1_atoms
                    else:
                        poi_atoms = ligand_1_atoms
                        e3_atoms = ligand_0_atoms


                    num_nodes = protac_data.x.size(0)
                    class_prediction_protac = torch.empty(size=(num_nodes,), dtype=torch.float32)
                    for class_idx, substructure_nodes_list in zip([0, 1, 2], [poi_atoms, linker_atoms, e3_atoms]):
                        for node_idx in substructure_nodes_list:
                            class_prediction_protac[node_idx] = class_idx
                    
                    
                    
                    if e3_database_fps is not None or poi_database_fps is not None:
                        #fp_function = compute_countMorgFP
                        protac_smiles = batch[protac_idx]["smiles"]
                        tanimoto_similarity_cutoff = 0.75
                        class_prediction_protac_checked_against_library = self._flip_ligands_for_protac(class_prediction_protac=class_prediction_protac, 
                                                                                tanimoto_similarity_cutoff=tanimoto_similarity_cutoff, 
                                                                                protac_smiles=protac_smiles, 
                                                                                fp_function = fp_function,
                                                                                e3_database_fps=e3_database_fps,
                                                                                poi_database_fps = poi_database_fps)
                        class_prediction_protac_checked_against_library = class_prediction_protac_checked_against_library.to(self.device)
                        class_predictions_checked_against_library = class_predictions_checked_against_library.to(self.device)
                        class_predictions_checked_against_library = torch.cat((class_predictions_checked_against_library, class_prediction_protac_checked_against_library), dim=0)

                    class_predictions = class_predictions.to(self.device)
                    class_prediction_protac = class_prediction_protac.to(self.device)
                    class_predictions = torch.cat((class_predictions, class_prediction_protac), dim=0)
                    

                    probabilites_protac = torch.empty(size=(num_nodes,), dtype=torch.float32)  #YET TO IMPLEMENT - probability for each class for every bond (dim=1), or probability for which bond is the boundary (for each class) (dim=0)
                    probabilites_protac = probabilites_protac.to(self.device)
                    probabilites = probabilites.to(self.device)
                    probabilites = torch.cat((probabilites, probabilites_protac), dim=0)
                    











        elif self.model_type == "link_pred":
            class_predictions = torch.empty(size=(0,), dtype=torch.float32) #On CPU when created
            class_predictions_checked_against_library = torch.empty(size=(0,), dtype=torch.float32) #On CPU when created
            
            probabilites = torch.empty(size=(0,), dtype=torch.float32)  #not yet implemented

            for protac_idx in range(len(batch)):
                if splittable_bonds_ptr is not None:
                    start_row_idx = splittable_bonds_ptr[protac_idx]
                    end_row_idx = splittable_bonds_ptr[protac_idx+1]
                    raw_prediction_protac =  raw_prediction[start_row_idx:end_row_idx]  #chunck up raw_prediction into each protac
                else:
                    raw_prediction_protac = raw_prediction

                predicted_boundary_bonds = predicted_boundary_bonds_dict[protac_idx]

                if predicted_boundary_bonds is None:
                    pred_poi_boundary_bond_idx, pred_e3_boundary_bond_idx = self._predict_bond_idx(raw_prediction_protac)
                else:
                    pred_poi_boundary_bond_idx, pred_e3_boundary_bond_idx = predicted_boundary_bonds

                poi_boundary_classprob = self.softmax_for_boundary_bond_probs(raw_prediction_protac[pred_poi_boundary_bond_idx])
                e3_boundary_classprob = self.softmax_for_boundary_bond_probs(raw_prediction_protac[pred_e3_boundary_bond_idx])
                #print(f'poi_boundary_classprob: {poi_boundary_classprob}')
                #print(f'e3_boundary_classprob: {e3_boundary_classprob}')
                #confidence_poi_boundary = poi_boundary_classprob[0] / (poi_boundary_classprob[0]+poi_boundary_classprob[2])
                #confidence_e3_boundary = e3_boundary_classprob[2] / (e3_boundary_classprob[0]+e3_boundary_classprob[2])
                #confidence_lowest_boundary = min([confidence_poi_boundary.item(), confidence_e3_boundary.item()])
                boundary_bond_probs.append({'POI BOUNDARY': poi_boundary_classprob.tolist(), 'E3 BOUNDARY': e3_boundary_classprob.tolist()})
                
                
                
                if self.use_new_postprocessing_algorithm:
                    #----------------new algorithm----------------
                    num_nodes = batch[protac_idx]["x"].size(0)
                    protac_node_indices = set(list(range(num_nodes)))
                    poi_nodes = set(batch[protac_idx]["splittable_bond_idx_to_node_incides"][pred_poi_boundary_bond_idx.item()]) #node_indices_smallest_frag_from_poi_boundary
                    e3_nodes = set(batch[protac_idx]["splittable_bond_idx_to_node_incides"][pred_e3_boundary_bond_idx.item()]) #node_indices_smallest_frag_from_e3_boundary

                    #If the sets share any same elements, that means that precomputation wasnt perfect. Take instead the compliment set = all other node indices = "flip" which fragment to take (this one was the smallest) = Take the other side of the split protac
                    if poi_nodes.isdisjoint(e3_nodes):
                        #original splits are not overlapping => good split
                        pass
                    else:
                        #if they overlap, one of the splits must contain the other split => find the largest set of these two, as that shall contain the other set, and flip it
                        if len(poi_nodes)>len(e3_nodes):
                            #poi set contains e3 set, flip poi set
                            poi_nodes = protac_node_indices - poi_nodes
                        elif len(poi_nodes)<len(e3_nodes):
                            #as the 
                            e3_nodes = protac_node_indices - e3_nodes
                        else:
                            raise ValueError("One set should contain the other, but are the same size => Same predicted boundary bonds => Linker has no atoms")
                    #splits are now good

                    #When the poi and e3 splits are good, get the linker by removing these elements from the whole protac
                    linker_e3_nodes = protac_node_indices - poi_nodes
                    linker_nodes = linker_e3_nodes - e3_nodes
                    #print(f'linker_nodes: {linker_nodes}')

                    #----------------new algorithm----------------
                
                else:
                    splittable_bonds = batch[protac_idx]["splittable_bonds"]
                    poi_node_0 = splittable_bonds[pred_poi_boundary_bond_idx][0].item() 
                    poi_node_1 = splittable_bonds[pred_poi_boundary_bond_idx][1].item() 
                    e3_node_0 = splittable_bonds[pred_e3_boundary_bond_idx][0].item() 
                    e3_node_1 = splittable_bonds[pred_e3_boundary_bond_idx][1].item()
                    
                    
                    G = batch[protac_idx]["graphs"].copy()
                    graph = G.copy()
                    if graph.has_edge(poi_node_0, poi_node_1):
                        graph.remove_edge(poi_node_0, poi_node_1)
                    elif graph.has_edge(poi_node_1, poi_node_0):
                        graph.remove_edge(poi_node_1, poi_node_0)
                    if graph.has_edge(e3_node_0, e3_node_1):
                        graph.remove_edge(e3_node_0, e3_node_1)
                    elif graph.has_edge(e3_node_1, e3_node_0):
                        graph.remove_edge(e3_node_1, e3_node_0)
                    

                    substructures_from_boundaries = {}
                    for boundary_node in [poi_node_0, poi_node_1, e3_node_0, e3_node_1]:
                        skip_this_itteration = False
                        found_linker_again = False
                        if not found_linker_again:
                            for found_substructure in substructures_from_boundaries.values():
                                if boundary_node in found_substructure:
                                    skip_this_itteration = True #it found the linker again
                                    found_linker_again = True
                                    break
                        if skip_this_itteration:
                            continue
                        
                        substructure_from_boundary = nx.descendants(graph, boundary_node)
                        substructure_from_boundary.add(boundary_node)
                        
                        has_poi_boundary = poi_node_0 in substructure_from_boundary or poi_node_1 in substructure_from_boundary
                        has_e3_boundary = e3_node_0 in substructure_from_boundary or e3_node_1 in substructure_from_boundary
                        if has_poi_boundary and has_e3_boundary:
                            substructure_type = 'linker'
                        elif has_poi_boundary:
                            substructure_type = 'poi'
                        elif has_e3_boundary:
                            substructure_type = 'e3'
                        substructures_from_boundaries[substructure_type] = substructure_from_boundary

                    poi_nodes = substructures_from_boundaries["poi"]
                    linker_nodes = substructures_from_boundaries["linker"]
                    e3_nodes = substructures_from_boundaries["e3"]
                


                if poi_nodes.isdisjoint(linker_nodes) and poi_nodes.isdisjoint(e3_nodes) and linker_nodes.isdisjoint(e3_nodes):
                    pass #substructures shares no nodes in common -> good
                else:
                    raise ValueError("The substructures shares nodes")
                num_nodes = batch[protac_idx]["x"].size(0)
                if num_nodes != len(poi_nodes) + len(linker_nodes) +len(e3_nodes):
                    
                    print(f'num_nodes: {num_nodes}, (len(poi_nodes) + len(linker_nodes) +len(e3_nodes)): {len(poi_nodes) + len(linker_nodes) +len(e3_nodes)}')
                    print(f'len(poi_nodes): {len(poi_nodes)},  len(linker_nodes): {len(linker_nodes)},  len(e3_nodes): {len(e3_nodes)}')
                    raise ValueError("The substructures dont have as many nodes as the full protac")
                
         
                class_prediction_protac = torch.empty(size=(num_nodes,), dtype=torch.float32)
                for class_idx, substructure_nodes_list in zip([0, 1, 2], [poi_nodes, linker_nodes, e3_nodes]):
                    for node_idx in substructure_nodes_list:
                        class_prediction_protac[node_idx] = class_idx

                
                if e3_database_fps is not None or poi_database_fps is not None:
                    #fp_function = compute_countMorgFP
                    protac_smiles = batch[protac_idx]["smiles"]
                    tanimoto_similarity_cutoff = 0.75
                    class_prediction_protac_checked_against_library = self._flip_ligands_for_protac(class_prediction_protac=class_prediction_protac, 
                                                                            tanimoto_similarity_cutoff=tanimoto_similarity_cutoff, 
                                                                            protac_smiles=protac_smiles, 
                                                                            fp_function = fp_function,
                                                                            e3_database_fps=e3_database_fps,
                                                                            poi_database_fps = poi_database_fps)
                    class_prediction_protac_checked_against_library = class_prediction_protac_checked_against_library.to(self.device)
                    class_predictions_checked_against_library = class_predictions_checked_against_library.to(self.device)
                    class_predictions_checked_against_library = torch.cat((class_predictions_checked_against_library, class_prediction_protac_checked_against_library), dim=0)

                class_predictions = class_predictions.to(self.device)
                class_prediction_protac = class_prediction_protac.to(self.device)
                class_predictions = torch.cat((class_predictions, class_prediction_protac), dim=0)
                

                probabilites_protac = torch.empty(size=(num_nodes,), dtype=torch.float32)  #YET TO IMPLEMENT - probability for each class for every bond (dim=1), or probability for which bond is the boundary (for each class) (dim=0)
                probabilites_protac = probabilites_protac.to(self.device)
                probabilites = probabilites.to(self.device)
                probabilites = torch.cat((probabilites, probabilites_protac), dim=0)
                
        
        else:
            raise ValueError(f"Please set a valid model_type. Current modeltype: {self.model_type}")
        
        predictions_and_prob = {}
        predictions_and_prob["model"] = (class_predictions, probabilites)
        predictions_and_prob["model+check_library"] = (class_predictions_checked_against_library, 0)


        return predictions_and_prob, boundary_bond_probs
    
    
    def _compute_loss(self, criterion, batch, output, yb):
        target = self._get_target(batch=batch)

        if self.model_type == "link_pred" and len(batch["ptr"])==2: #if batchsize > 1, then multiple separete loss calc must be done, one for each graph
            class_loss = criterion(output, target)  

            output_poi_e3 = output[:,[0,2]]
            output_poi_e3_T = torch.transpose(output_poi_e3, 0, 1)
            poi_bond_index = (batch["bond_labels"] == 0).nonzero(as_tuple=False)[0]
            e3_bond_index = (batch["bond_labels"] == 2).nonzero(as_tuple=False)[0]
            bond_targets = torch.cat((poi_bond_index, e3_bond_index), dim=0)
            bond_loss = criterion(output_poi_e3_T, bond_targets)
            weight_class_loss = self.weight_class_loss
            weight_bond_loss = 1 - weight_class_loss
            loss = class_loss*weight_class_loss + bond_loss* weight_bond_loss
        else:
            loss = criterion(output, target)

        #boundarybond prediction loss
        if self.model_type == "link_pred":
            target_boundarybonds = batch["boundarybond_labels"]
            boundary_loss = criterion(yb, target_boundarybonds)
            loss += boundary_loss    

        regularization_loss = 0
        if self.regularization_types is not None:
            if 'L1_model_params' in self.regularization_types:
                l1_param_norm = sum(p.abs().sum() for p in self.parameters())
                l1_param_norm_avg = l1_param_norm /self.num_params
                regularization_loss += l1_param_norm_avg
            if 'L2_model_params' in self.regularization_types:
                l2_param_norm = sum(p.pow(2).sum() for p in self.parameters())
                l2_param_norm_rms = (l2_param_norm /self.num_params)**(0.5)
                regularization_loss += l2_param_norm_rms
            loss += regularization_loss

        return loss


    
    def _flip_ligands_for_protac(self, class_prediction_protac, tanimoto_similarity_cutoff, protac_smiles, fp_function, e3_database_fps, poi_database_fps):
        #Get predicted poi without attachmentpoint
        protac_mol = Chem.MolFromSmiles(protac_smiles)
        class_prediction_protac_library_checked = class_prediction_protac.clone()

        max_similarity = 0
        if e3_database_fps is not None:
            poi_node_label= 0
            node_indices_poi = [i for i, x in enumerate(class_prediction_protac_library_checked.tolist()) if x == poi_node_label]
            poi_mol = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(protac_mol, node_indices_poi, kekuleSmiles=True))
            poi_fp = fp_function([poi_mol])
            max_similarity_poi = max(DataStructs.BulkTanimotoSimilarity(poi_fp[0], e3_database_fps))
            if max_similarity<max_similarity_poi:
                max_similarity = max_similarity_poi

        if poi_database_fps is not None:
            e3_node_label=2
            node_indices_e3 = [i for i, x in enumerate(class_prediction_protac_library_checked.tolist()) if x == e3_node_label]
            e3_mol = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(protac_mol, node_indices_e3, kekuleSmiles=True))
            e3_fp = fp_function([e3_mol])
            max_similarity_e3 = max(DataStructs.BulkTanimotoSimilarity(e3_fp[0], poi_database_fps))
            if max_similarity<max_similarity_e3:
                max_similarity = max_similarity_e3

        if max_similarity > tanimoto_similarity_cutoff:  
            class_prediction_protac_library_checked[class_prediction_protac_library_checked==2] = 3
            class_prediction_protac_library_checked[class_prediction_protac_library_checked==0] = 2
            class_prediction_protac_library_checked[class_prediction_protac_library_checked==3] = 0


        return class_prediction_protac_library_checked

        
    def predict(self, protac_smiles, e3_database_fps = None, poi_database_fps = None, fp_function = compute_countMorgFP, use_library=True, return_probs = False, precompute_splits = False):
        if self.model_type == "link_pred":
            get_boundary_bonds = True
        else:
            get_boundary_bonds = False
        class_predictions, _,  (pred_poi_boundary_bond, pred_e3_boundary_bond), boundary_bond_probs_protac = self.predict_node_labels_and_probabilities_for_protac(protac_smiles=protac_smiles, 
                                                                                                                                       get_boundary_bonds=get_boundary_bonds, 
                                                                                                                                       e3_database_fps = e3_database_fps, 
                                                                                                                                       poi_database_fps = poi_database_fps, 
                                                                                                                                       fp_function = fp_function, 
                                                                                                                                       use_library=use_library,
                                                                                                                                       precompute_splits=precompute_splits)
        substructures_smi, substructure_smi_with_attachmentpoints = get_substructure_smiles_function_v2(protac_smiles, 
                                                                                                        class_predictions, 
                                                                                                        boundary_bonds =  (pred_poi_boundary_bond.tolist(), pred_e3_boundary_bond.tolist()))
            
        substructures_smi["PROTAC SMILES"] = protac_smiles
        substructure_smi_with_attachmentpoints["PROTAC SMILES"] = protac_smiles

        if return_probs:
            return substructures_smi, substructure_smi_with_attachmentpoints
        else:
            return substructures_smi, substructure_smi_with_attachmentpoints
    
    def _get_target(self, batch):
        if self.model_type == "node_pred":
            return batch["substructure_labels"]
        elif self.model_type == "boundary_pred":
            return batch["boundary_labels"]
        elif self.model_type == "link_pred":
            return batch["bond_labels"]
        else:
            raise ValueError("Undefined modeltype")
    

    def batch_to_output_and_classpredictions(self, batch, e3_database_fps=None, poi_database_fps=None, fp_function = compute_countMorgFP, boundary_bond_probs_dataset=[], randomize_y=False):
        #forward
        splittable_bonds_ptr = None
        if randomize_y is False:
            if self.model_type == "link_pred":   # rename target to "y" maybe datalodaer can concatenate properly then and skip this for loop
                splittable_bonds = batch["splittable_bonds"]
                if len(batch)>1:     
                    splittable_bonds_batch = torch.empty(batch["splittable_bonds"].size(), dtype=torch.int64)  
                    splittable_bonds_batch = splittable_bonds_batch.to(self.device)        
                    splittable_bonds_ptr = [0]*(len(batch)+1)
                    for protac_idx in range(len(batch)):
                        splittable_bonds_ptr[protac_idx+1] = splittable_bonds_ptr[protac_idx] + batch[protac_idx]["splittable_bonds"].size()[0]
                        bond_range = range(splittable_bonds_ptr[protac_idx], splittable_bonds_ptr[protac_idx+1])
                        splittable_bonds_batch[bond_range, :] = splittable_bonds[bond_range, :] + batch["ptr"][protac_idx].item()
                else:
                    splittable_bonds_batch = batch["splittable_bonds"]
                y_type_raw, y_location_raw = self.forward(batch, edges_to_predict=splittable_bonds_batch)  # Forward pass
            else:
                y_type_raw, y_location_raw = self.forward(batch)
        
        else: #randomize_y is True
            target = self._get_target(batch=batch)
            
            y_type_raw = torch.rand(target.size()[0], self.num_predicted_classes)
            y_type_raw = y_type_raw.to(self.device)

            y_location_raw = torch.rand(target.size()[0], 2)
            y_location_raw = y_location_raw.to(self.device)

        y_type = F.softmax(y_type_raw, dim=1)

     

        #predict boundary bonds from y_location_raw
        predicted_boundary_bonds_dict = {}
        if self.model_type == "link_pred":
            for protac_idx in range(len(batch)):
                if splittable_bonds_ptr is not None:
                    start_row_idx = splittable_bonds_ptr[protac_idx]
                    end_row_idx = splittable_bonds_ptr[protac_idx+1]
                    y_location_raw_protac =  y_location_raw[start_row_idx:end_row_idx]  #chunck up raw_prediction into each protac
                else:
                    y_location_raw_protac = y_location_raw

                #print(y_location_raw_protac)
                #print(y_location_raw_protac[:,0])
                predicted_boundary_bonds = torch.topk(y_location_raw_protac[:,0], k=2, dim=0).indices.tolist()
                if (y_type[predicted_boundary_bonds[0], 0] + y_type[predicted_boundary_bonds[1], 2])  > (y_type[predicted_boundary_bonds[0], 2] + y_type[predicted_boundary_bonds[1], 0]):
                    #first bond POI-B like and second bond is more E3-b like than the other way around
                    pass
                else:
                    #first bond is assumed to be poi-boundary, second being E3-boundary in _predict_node_labels_and_probabilities_for_batch
                    predicted_boundary_bonds = [predicted_boundary_bonds[1], predicted_boundary_bonds[0]]
                
                predicted_boundary_bonds_dict[protac_idx] = predicted_boundary_bonds
        else:
            predicted_boundary_bonds = None


        #Get node predictions / classes of nodes
        predictions_and_prob, boundary_bond_probs_batch = self._predict_node_labels_and_probabilities_for_batch(batch=batch, 
                                                                                                                raw_prediction=y_type, 
                                                                                                                splittable_bonds_ptr=splittable_bonds_ptr, 
                                                                                                                e3_database_fps=e3_database_fps, 
                                                                                                                poi_database_fps=poi_database_fps, 
                                                                                                                fp_function=fp_function,
                                                                                                                predicted_boundary_bonds_dict=predicted_boundary_bonds_dict)
        boundary_bond_probs_dataset.extend(boundary_bond_probs_batch)

        return y_type, y_location_raw, predictions_and_prob, boundary_bond_probs_dataset

    def predict_node_labels_and_probabilities_for_protac(self, protac_smiles='', get_boundary_bonds = False, e3_database_fps = None, poi_database_fps=None, fp_function = None, use_library=True, precompute_splits=False):
        #prepare data
        if isinstance(protac_smiles, list):
            protac_smiles = protac_smiles
        protac_data = self.get_protac_data(protac_smiles, precompute_splits)
        loader = DataLoader(dataset=protac_data, batch_size=1, shuffle=False)

        self.eval()  
        with torch.no_grad():
            for protac in loader: 
                protac = protac.to(self.device)
                
                output, yb, predictions_and_prob, boundary_bond_probs_dataset = self.batch_to_output_and_classpredictions(batch=protac, e3_database_fps = e3_database_fps, poi_database_fps = poi_database_fps, fp_function = fp_function)


                if use_library:
                    class_predictions = predictions_and_prob["model"][0]
                else:
                    class_predictions = predictions_and_prob["model+check_library"][0]
                probabilites = predictions_and_prob["model"][1]
                
            if get_boundary_bonds:
                pred_poi_boundary_bond_idx, pred_e3_boundary_bond_idx = self._predict_bond_idx(raw_prediction_protac=output)
                #Get splittable bonds from protac_data
                splittable_bonds_protac = protac["splittable_bonds"]
                #get boundary bond atoms!
                pred_poi_boundary_bond = splittable_bonds_protac[pred_poi_boundary_bond_idx]
                pred_e3_boundary_bond = splittable_bonds_protac[pred_e3_boundary_bond_idx]
            
                return class_predictions, probabilites, (pred_poi_boundary_bond, pred_e3_boundary_bond), boundary_bond_probs_dataset
            else:
                return class_predictions, probabilites, (None, None), boundary_bond_probs_dataset

