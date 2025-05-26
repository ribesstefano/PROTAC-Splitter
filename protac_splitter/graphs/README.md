# Graph-Based PROTAC-Splitter

## Heuristic Betweenness Centrality

```python
idx = 3765
for i in range(10):
    # sample = held_out_df.sample(n=1, random_state=42 + i).iloc[0]
    sample = held_out_df.iloc[i]
    # sample = held_out_df.iloc[i]
    protac_smiles = sample['PROTAC SMILES']
    wh_smiles = sample['POI Ligand SMILES with direction']
    lk_smiles = sample['Linker SMILES with direction']
    e3_smiles = sample['E3 Binder SMILES with direction']

    protac = Chem.MolFromSmiles(protac_smiles)
    wh = Chem.MolFromSmiles(wh_smiles)
    lk = Chem.MolFromSmiles(lk_smiles)
    e3 = Chem.MolFromSmiles(e3_smiles)

    # display_mol(Chem.MolFromSmiles(protac_smiles), w=1500, h=600)
    get_mapped_protac_img(protac_smiles, wh_smiles, lk_smiles, e3_smiles, w=1500, h=600, display_image=True, useSVG=False)
    # wh_edge = get_atom_idx_at_attachment(protac, wh, lk)
    # e3_edge = get_atom_idx_at_attachment(protac, e3, lk)
    
    ret = nx_split(protac_smiles, representative_e3s_fp, morgan_fp_generator, use_capacity_weight=False, betweenness_threshold=0.4)
    e3_smiles = ret['e3']
    wh_smiles = ret['poi']
    linker_smiles = ret['linker']
    top_nodes = ret['top_nodes']
    centrality = ret['centrality']

    # display_mol(Chem.MolFromSmiles(e3_smiles), w=800, h=400, legend="E3")
    # display_mol(Chem.MolFromSmiles(linker_smiles), w=800, h=400, legend="Linker")
    # display_mol(Chem.MolFromSmiles(wh_smiles), w=800, h=400, legend="WH")
    
    display_mol(Chem.MolFromSmiles('.'.join([wh_smiles, linker_smiles, e3_smiles])), w=800, h=400, legend="Graph-based split")
    

    display(Draw.MolToImage(
        protac,
        size=(1500, 400),
        highlightColor=(1, 0, 1, 0.3), # Light purple
        highlightAtoms=top_nodes, # Highlight the top nodes
        legend=f"Graph nodes: {top_nodes} (Betweenness centrality: {centrality[top_nodes[0]]:.3f})",
    ))
```


## Graph Edge Classifier Example

Example of how to use the GraphEdgeClassifier to train a model on a dataset of PROTACs and their ligands, and then predict edges in new PROTACs.

```python
label_cols = [c for c in train_set.columns if c.startswith("label_")]
train_set = sets["train"].dropna(subset=label_cols)
train_set = train_set[(train_set["label_e3_split"] + train_set["label_wh_split"]) <= 1]
X_train = train_set.drop(columns=label_cols)

graph_features = [c for c in X_train.columns if c.startswith("graph_")]
# graph_features = [
#     "graph_betweenness",
#     "graph_degree",
#     "graph_degree_r2",
#     "graph_degree_r3",
# ]
categorical_features = ["chem_bond_type", "chem_atom_u", "chem_atom_v"]
fingerprint_features = [c for c in X_train.columns if c.startswith("chem_mol_fp_")]

# Instantiate and train
clf = GraphEdgeClassifier(
    graph_features=graph_features,
    categorical_features=categorical_features,
    fingerprint_features=fingerprint_features,
    use_descriptors=False,
    use_fingerprints=False,
    binary=True,
)
y_train = train_set["label_is_split"].astype("int32") if clf.binary else GraphEdgeClassifier.build_multiclass_target(train_set)

clf.fit(X_train, y_train)
clf.save("../models/edge_classifier_bin.joblib")
print(f"Model saved to ../models/edge_classifier_bin.joblib")

label_cols = [c for c in train_set.columns if c.startswith("label_")]
train_set = sets["train"].dropna(subset=label_cols)
train_set = train_set[(train_set["label_e3_split"] + train_set["label_wh_split"]) <= 1]
X_train = train_set.drop(columns=label_cols)

graph_features = [c for c in X_train.columns if c.startswith("graph_")]
# graph_features = [
#     "graph_betweenness",
#     "graph_degree",
#     "graph_degree_r2",
#     "graph_degree_r3",
# ]
categorical_features = ["chem_bond_type", "chem_atom_u", "chem_atom_v"]
fingerprint_features = [c for c in X_train.columns if c.startswith("chem_mol_fp_")]

# Instantiate and train
clf = GraphEdgeClassifier(
    graph_features=graph_features,
    categorical_features=categorical_features,
    fingerprint_features=fingerprint_features,
    use_descriptors=False,
    use_fingerprints=False,
    binary=False,
)
y_train = train_set["label_is_split"].astype("int32") if clf.binary else GraphEdgeClassifier.build_multiclass_target(train_set)

clf.fit(X_train, y_train)
clf.save("../models/edge_classifier.joblib")
print(f"Model saved to ../models/edge_classifier.joblib")
```