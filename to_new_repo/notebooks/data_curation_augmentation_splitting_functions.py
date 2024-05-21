


# Search control+f: "# ----" to find all sections

import re
import random

import numpy as np
import pandas as pd

from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolHash, rdFMCS, rdMolAlign, rdchem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

import networkx as nx

from matplotlib.ticker import MaxNLocator
from collections import defaultdict
import itertools

from IPython.display import display, SVG
import matplotlib.pyplot as plt





# ------------------------------------------ SVG AND DRAWING FUNCTIONS



def save_as_svg(svg_content, filename, num_mols):
    """Save SVG content to a file."""
    with open(filename, 'w') as file:
        data = str(svg_content.data)
        data = data.replace('1500', str(500*num_mols))
        file.write(data)


    

def align_molecules_2D(ref_mol, to_align_mol):
    AllChem.Compute2DCoords(ref_mol)
    AllChem.Compute2DCoords(to_align_mol)
    # Find the maximum common substructure and use it to align molecules
    mcs = rdFMCS.FindMCS([ref_mol, to_align_mol])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    ref_match = ref_mol.GetSubstructMatch(mcs_mol)
    align_match = to_align_mol.GetSubstructMatch(mcs_mol)
    atom_map = list(zip(align_match, ref_match))
    rdMolAlign.AlignMol(to_align_mol, ref_mol, atomMap=atom_map)
    return to_align_mol

def align_molecules_by_coordinates(ref_mol, to_align_mol):
    # Find the maximum common substructure
    AllChem.Compute2DCoords(to_align_mol)
    mcs = rdFMCS.FindMCS([ref_mol, to_align_mol])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    ref_match = ref_mol.GetSubstructMatch(mcs_mol)
    align_match = to_align_mol.GetSubstructMatch(mcs_mol)

    # Copy the coordinates from the reference molecule to the molecule to be aligned
    ref_conf = ref_mol.GetConformer()
    align_conf = to_align_mol.GetConformer()
    for ref_idx, align_idx in zip(ref_match, align_match):
        ref_pos = ref_conf.GetAtomPosition(ref_idx)
        align_conf.SetAtomPosition(align_idx, ref_pos)

    return to_align_mol



def draw_molecule_to_svg(mol, size=(500, 500), scale=1.0):
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().fixedBondLength = scale
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = re.sub(r'\<\?xml.*?\?\>', '', svg)  # Remove XML declaration
    svg = svg.replace('<svg', '<g').replace('</svg>', '</g>')  # Replace svg tags with g tags
    return svg

def combine_svgs(svgs, output_filename, dimensions=None, size=(500, 500), xy_shifts=None):
    if dimensions is None:
        dimensions = (len(svgs), 1)
    if xy_shifts is None:
        xy_shifts = [(0,0) for i in range(dimensions[0]*dimensions[1])]
    

    width, height = size
    grid_width, grid_height = dimensions
    # Include only one XML declaration and the opening <svg> tag
    combined_svg = f'<?xml version="1.0" standalone="no"?>\n'
    combined_svg += f'<svg width="{grid_width * width}px" height="{grid_height * height}px" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">\n'

    # Arrange SVGs in a grid
    for i, (svg, xy_shift) in enumerate(zip(svgs, xy_shifts)):
        x = (i % grid_width) * width
        y = (i // grid_width) * height
        combined_svg += f'<g transform="translate({x+xy_shift[0]},{y-xy_shift[1]})">{svg}</g>\n'

    combined_svg += '</svg>'
    with open(output_filename, 'w') as file:
        file.write(combined_svg)



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
    svg = svg.replace('svg:','')
    display(SVG(svg))

    return svg



def align_mol_2D_ver2(template, query):


    mcs = rdFMCS.FindMCS([template, query])
    patt = Chem.MolFromSmarts(mcs.smartsString)

    query_match = query.GetSubstructMatch(patt)
    template_match = template.GetSubstructMatch(patt)

    rms = AllChem.AlignMol(query, template, atomMap=list(zip(query_match,template_match)))
    return template, query






def transform_molecule(mol, degrees, translate_x=0, translate_y=0, flip_x_axis=False):
    """Apply rotation, translation, and optionally flip the molecule."""
    radians = np.deg2rad(degrees)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians),  np.cos(radians), 0],
        [0,               0,               1]
    ])
    AllChem.Compute2DCoords(mol)

    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        new_pos = np.dot(rotation_matrix, pos)
        new_pos[0] += translate_x  # Translate along the x-axis
        new_pos[1] += translate_y  # Translate along the y-axis
        if flip_x_axis:
            new_pos[1] = -new_pos[1]  # Flip along the x-axis
        conf.SetAtomPosition(i, new_pos)





def tailored_framework_example(mol_ms):
    #remove lone atoms
    #define all atoms to be atom number 1
    #define all bonds to be single bonds

    mol_ms_w = Chem.RWMol(mol_ms)
    atom_idx_to_remove = []
    for atom in mol_ms_w.GetAtoms():
        if atom.GetDegree() == 1: # lone atom. Need to remove it to create the generic framework.
            atom_idx_to_remove.append(atom.GetIdx())
            continue
        atom.SetAtomicNum(0)
    
    for bond in mol_ms_w.GetBonds():
        bond.SetBondType(Chem.rdchem.BondType.SINGLE)

    atom_idx_to_remove.sort(reverse=True)
    for atom_idx in atom_idx_to_remove:
        mol_ms_w.RemoveAtom(atom_idx)
    
    mol_ms_new = mol_ms_w.GetMol()
    return mol_ms_new


# ---------------------------------------------- CHEMOINFORMATICS










def remove_stereo(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.rdmolops.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)
    except:
        return np.nan


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.rdmolops.RemoveStereochemistry(mol)
    return mol


def compute_RDKitFP(smiles, maxPath=7, fpSize=2048):
    if isinstance(smiles[0], str):
        mols = [get_mol(smi) for smi in smiles]
    else:
        mols = smiles #assume mols were fed instead
    rdgen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=maxPath, fpSize=fpSize)
    fps = [rdgen.GetCountFingerprint(mol) for mol in mols]
    return fps

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



def tanimoto_similarity_matrix(fps):
    """
    Calculate a symmetric Tanimoto similarity matrix for a list of fingerprints using bulk operations.
    
    Parameters:
    - fps: list, RDKit fingerprint objects for which to calculate similarity.
    
    Returns:
    - np.array, Symmetric square matrix of Tanimoto similarity.
    """
    num_fps = len(fps)
    sim_matrix = np.zeros((num_fps, num_fps)) # Initialize a square matrix of zeros

    for i in tqdm(range(num_fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim = np.array(similarities)
        sim_matrix[i, :] = sim
        sim_matrix[i, i] = 1 # Set diagonal to 1 as the similarity to self is 1

    return sim_matrix



# ----------------------------------------------- DATA SPLITTING


def tanimoto_distance_matrix(sim_matrix=None, fps = None):
    if sim_matrix is None:
        sim_matrix = tanimoto_similarity_matrix(fps=fps) 
    return 1-sim_matrix

from sklearn.cluster import HDBSCAN
def perform_hdbscan(distance_matrix, tanimoto_distance_cutoff, min_cluster_size=2, min_samples=None):
    """
    Cluster data using the HDBSCAN algorithm on a precomputed distance matrix.

    Parameters:
    - distance_matrix: np.array, Precomputed distance matrix where distances[i, j] is the distance between i and j.
    - tanimoto_distance_cutoff: float, The distance threshold. Clusters below this value will be merged. 
    - min_cluster_size: int, The minimum size of clusters; defaults to 5.
    - min_samples: int, The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - clusterer: HDBSCAN object, Fitted HDBSCAN model.
    """
    clusterer = HDBSCAN(metric='precomputed', 
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon = tanimoto_distance_cutoff,
                        allow_single_cluster = True)
    clusterer.fit(distance_matrix)
    return clusterer


def get_hdbscan_clusters(smi_list, fp_function, max_allowed_tanimoto_similarity = 0.4):
    fps = fp_function(smi_list)
    sim_matrix = tanimoto_similarity_matrix(fps = fps)
    distance_matrix = tanimoto_distance_matrix(sim_matrix = sim_matrix)

    tanimoto_distance_cutoff = 1 - max_allowed_tanimoto_similarity
    clusterer = perform_hdbscan(distance_matrix=distance_matrix, tanimoto_distance_cutoff = tanimoto_distance_cutoff)
    
    return clusterer.labels_


def get_test_clusters(clusterer_labels, test_set_minfraction = 0.10, test_set_maxfraction = 0.20):

    #test_set_minfraction = 0.10 #stop inclding more clusters when this fraction is succeeded
    #test_set_maxfraction = 0.20 #if the addition of the final cluster goes from below minmum to above this max, undo final addition

    num_mols = len(clusterer_labels)

    cumulative_count_mols_to_test = 0
    cluster_ids_for_test = []
    cluster_ids_for_test_dict = {}

    #count the number of molecules in each cluster
    cluster_counts = Counter(clusterer_labels)

    #want a list of clusters sorted by size
    labels_sorted_mostcommon = sort_by_most_common_elements(clusterer_labels)
    ids_largest_clusters = get_unique_perserve_order(labels_sorted_mostcommon)  #each mol has a label, the label is a cluster. The number of labels is the number of mol in the cluster

    #itterativly add clusters to test set, until min set test fraction is reached.
    for cluster_id in reversed(ids_largest_clusters):
        num_mol_in_cluster = cluster_counts[cluster_id]
        cumulative_count_mols_to_test += num_mol_in_cluster
        cluster_ids_for_test.append(cluster_id)
        cluster_ids_for_test_dict[cluster_id] = num_mol_in_cluster
        test_set_fraction = cumulative_count_mols_to_test/num_mols
        if test_set_fraction>test_set_minfraction:
            break

    #if max test fraction is succeded, remove last cluster that made it tip over
    if test_set_fraction>test_set_maxfraction:
        cluster_ids_for_test.remove(cluster_id)
        del cluster_ids_for_test_dict[cluster_id]
        cumulative_count_mols_to_test -= num_mol_in_cluster

    print(f'test_set_fraction: {round(cumulative_count_mols_to_test/num_mols*100,2)} % ({cumulative_count_mols_to_test} molecules)')
    return cluster_ids_for_test, cluster_ids_for_test_dict


def plot_hists(values, binwidth_plural = [0.1, 0.05, 0.01], title=""):

    value_min = min(values)
    value_max = max(values)

    num_plots = len(binwidth_plural)
    fig, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(8*num_plots, 6), sharex=True, sharey=False)
    for ax_id, (binwidth, ax) in enumerate(zip(binwidth_plural, axs)):
        bins = list(np.arange(value_min, value_max + binwidth, binwidth))
        ax.hist(values, range=(value_min,value_max), bins=bins)
        hist_title = f'{title} (Binwidth = {binwidth})'
        ax.set_title(hist_title)
    fig.supylabel('Y value')
    fig.supxlabel('Count')
    plt.show


def validate_test_set(test_set, training_set, cutoff, fp_function=compute_countMorgFP, plot_highest_sim_match = False, save_fig=False, save_title=""):                                                            #OBS! Does not take into aaccount if cutoff and cutoff_MS are different!
    """
    Validates that no molecule in the test setf has a Tanimoto similarity above the cutoff
    with any molecule in the training set. Returns a list of maximum similarities for test set molecules.
    """
  
    test_fps = fp_function(test_set)
    train_fps = fp_function(training_set)
    max_similarities = []

    highest_similarity = 0

    below_cutoff = True
    for test_idx, test_fp in enumerate(test_fps):
        similarities = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        max_similarity = max(similarities)
        max_similarities.append(max_similarity)

        if plot_highest_sim_match and max_similarity > highest_similarity:
            highest_similarity = max_similarity
            train_idx = similarities.index(max_similarity)
            highest_similarity_indicies = [test_idx, train_idx]

        if max_similarity > cutoff:
            below_cutoff = False

    if plot_highest_sim_match:
        test_idx = highest_similarity_indicies[0]
        train_idx = highest_similarity_indicies[1]
        test_smi = test_set[test_idx]
        train_smi = training_set[train_idx]
        
        test_mol = Chem.MolFromSmiles(test_smi)
        train_mol = Chem.MolFromSmiles(train_smi)

        print(f"The most similar test molecule (right) {test_smi} to any other training molecule:")
        display(test_mol)
        print(f"It is similar to following train molecule (left) {train_smi}, with similarity of {highest_similarity}")
        display(train_mol)

        if save_fig:
            #AllChem.Compute2DCoords(train_mol)
            #test_mol = align_molecules_by_coordinates(train_mol, test_mol)
        
            train_svg = draw_molecule_to_svg(train_mol, scale=30)  # Adjust scale as needed
            test_svg = draw_molecule_to_svg(test_mol, scale=30)

            filename = f"fig_method/most_similar_train_test_{save_title}"
            combine_svgs(svgs=[train_svg, test_svg], output_filename=f'{filename}.svg')
            display(SVG(filename=f'{filename}.svg'))


    return below_cutoff, max_similarities


def distribute_clusters(clusters, num_sets_out=3):
    # Sort clusters by size in descending order
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1], reverse=True)
   
    # Initialize three sets with labels and total size
    sets = [{'cluster_ids': [], 'total_size': 0} for _ in range(num_sets_out)]
   
    # Assign each cluster to the set with the smallest total size
    for label, size in sorted_clusters:
        # Find the set with the minimum total size
        
        min_set = min(sets, key=lambda x: x['total_size'])
        # Add the cluster to this set
        min_set['cluster_ids'].append(label)
        min_set['total_size'] += size

    sets_dict = { idx: set for idx, set in enumerate(sets)}
   
    return sets_dict

def get_test_trainval_smi(smi_list, fp_function, max_allowed_tanimoto_similarity = 0.4, test_set_minfraction = 0.10, test_set_maxfraction = 0.20, binwidth_plural = [5, 1], substructure_type="", save_fig = False):

    #get test and trainval clusters
    substructure_wo_attachent_clusters = get_hdbscan_clusters(smi_list=smi_list, fp_function=fp_function, max_allowed_tanimoto_similarity=max_allowed_tanimoto_similarity)
    if type(substructure_wo_attachent_clusters) is np.ndarray:
        substructure_wo_attachent_clusters = substructure_wo_attachent_clusters.tolist()
    plot_hists(values = substructure_wo_attachent_clusters, binwidth_plural = binwidth_plural, title="HDBSCAN cluster sizes")
    plt.show()
    substructure_wo_attachment_test_clusters, substructure_wo_attachment_test_clusters_dict = get_test_clusters(substructure_wo_attachent_clusters, test_set_minfraction = test_set_minfraction, test_set_maxfraction = test_set_maxfraction)
    substructure_wo_attachment_trainval_clusters = list(  set(substructure_wo_attachent_clusters)  -  set(substructure_wo_attachment_test_clusters)   )
    
    # Create an empty dictionary to store the clusters
    map_cluster_label_to_unique_substructures_without_attachment = {}

    # Iterate through both lists simultaneously
    for smi, cluster_label in zip(smi_list, substructure_wo_attachent_clusters):
        if cluster_label not in map_cluster_label_to_unique_substructures_without_attachment:
            map_cluster_label_to_unique_substructures_without_attachment[cluster_label] = []
        map_cluster_label_to_unique_substructures_without_attachment[cluster_label].append(smi)


    substructure_wo_attachment_test_clusters_subsplit_withSize = distribute_clusters(substructure_wo_attachment_test_clusters_dict, num_sets_out=3)
    for testset_idx, testset_clusters_and_sizes in substructure_wo_attachment_test_clusters_subsplit_withSize.items():
        print(f"Testset {testset_idx} has {testset_clusters_and_sizes['total_size']} entries ({round(testset_clusters_and_sizes['total_size']/len(smi_list)*100,2)} %)")


    #check if any testsplit contains the -1 group (defined as "noise" by HDBSCAN)
    #if -1 exists, then get how many molecules is in the "noise" group
    #subtract that number from the split with -1 and delete the key -1 from that test split
    #Then get all smiles for the -1 group
    #itterativly designate one SMILES to the testsplit with the fewest SMILES
    #Store in the form of a dictionary and then add onto the test_unique_substructure_without_attachment for that split (after test_unique_substructure_without_attachment has been given its smiles)
    


    #get number of molecules in splits
    split_sizes = [substructure_wo_attachment_test_clusters_subsplit_withSize[i]["total_size"] for i in range(len(substructure_wo_attachment_test_clusters_subsplit_withSize))]
    smi_to_testsplits = {idx: [] for idx in range(len(split_sizes))}



    #check if any testsplit contains the -1 group (defined as "noise" by HDBSCAN)
    for split_idx, split_dict in substructure_wo_attachment_test_clusters_subsplit_withSize.items():
        if -1 in split_dict['cluster_ids']:
            #if -1 exists, then get how many molecules is in the "noise" group
            num_noisy_mols = substructure_wo_attachent_clusters.count(-1)
            #subtract that number from the split with -1 and delete the key -1 from that test split
            substructure_wo_attachment_test_clusters_subsplit_withSize[split_idx]["total_size"] -= num_noisy_mols
            split_sizes[split_idx] -= num_noisy_mols
            #remove the group from the split
            substructure_wo_attachment_test_clusters_subsplit_withSize[split_idx]["cluster_ids"].remove(-1)
            
            #Then get all smiles for the -1 group
            noise_substructure_without_attachment = map_cluster_label_to_unique_substructures_without_attachment[-1]
    
            
            
            for smi in noise_substructure_without_attachment:
                min_size = min(split_sizes)
                idx_min_size = split_sizes.index(min_size)
                smi_to_testsplits[idx_min_size].append(smi)
                split_sizes[idx_min_size] += 1

            

    #dict with split idx to cluster idx
    substructure_wo_attachment_test_clusters_subsplit = {}
    for split_idx, split_dict in substructure_wo_attachment_test_clusters_subsplit_withSize.items():
        substructure_wo_attachment_test_clusters_subsplit[split_idx] = split_dict["cluster_ids"]

    

    trainval_unique_substructure_without_attachment = [map_cluster_label_to_unique_substructures_without_attachment[trainval_cluster] for trainval_cluster in substructure_wo_attachment_trainval_clusters]
    trainval_unique_substructure_without_attachment = [smi for sub_list in trainval_unique_substructure_without_attachment for smi in sub_list] #unpack listlist


    #get smiles 
    test_unique_substructure_without_attachment_splits = {}
    for split_idx, substructure_wo_attachment_test_clusters in substructure_wo_attachment_test_clusters_subsplit.items():
        test_unique_substructure_without_attachment = [map_cluster_label_to_unique_substructures_without_attachment[test_cluster] for test_cluster in substructure_wo_attachment_test_clusters]
        test_unique_substructure_without_attachment = [smi for sub_list in test_unique_substructure_without_attachment for smi in sub_list] #unpack listlist
        
        test_unique_substructure_without_attachment += smi_to_testsplits[split_idx]

        #sanity check
        if set(test_unique_substructure_without_attachment) & set(trainval_unique_substructure_without_attachment):
            raise ValueError("A test SMILES is in the training set!")
        else:
            print(f"All good, split was successfull for split idx: {split_idx}")


        #plot max tanimoto similarity between the trainval and test sets
        is_valid, max_similarities = validate_test_set(test_set=test_unique_substructure_without_attachment, 
                                                    training_set=trainval_unique_substructure_without_attachment, 
                                                    cutoff=max_allowed_tanimoto_similarity, 
                                                    fp_function=fp_function, 
                                                    plot_highest_sim_match = True)
        print(f'count max_similarities: {len(max_similarities)}')
    
    
        fig, ax = plt.subplots()
        # Plotting the maximum similarity for all test molecules, against the trainset
        plt.hist(max_similarities, bins=30, alpha=0.75)
        print(f'All most similar pairs are below cutoff ({max_allowed_tanimoto_similarity}) is {is_valid}')
        plt.xlabel('Maximum Tanimoto Similarity')
        plt.ylabel('Frequency')
        print(f'Distribution of Maximum Tanimoto Similarity (Test vs. Train)')
        plt.show()
        if save_fig:
            fig.savefig(f'fig_similarity_test_trainval/simtesttrain{substructure_type}_{split_idx}.svg', format='svg', dpi=1200, bbox_inches='tight')
            fig.savefig(f'fig_similarity_test_trainval/simtesttrain{substructure_type}_{split_idx}.png', format='png', dpi=1200, bbox_inches='tight')
            

        if is_valid:
            print(f"SMILES successfully split into trainval and test set without any pair with similarity above cutoff ({max_allowed_tanimoto_similarity})")
            test_unique_substructure_without_attachment_splits[split_idx] = test_unique_substructure_without_attachment
            
            
    return {'test': test_unique_substructure_without_attachment_splits, 'trainval': trainval_unique_substructure_without_attachment}




def validate_no_general_ms_leakage(test_smi_set, training_smi_set):

    test_smi_anon_ms_list = [get_anonymous_murcko(smi) for smi in test_smi_set]
    training_smi_anon_ms_list = [get_anonymous_murcko(smi) for smi in training_smi_set]

    test_smi_anon_ms_set = set(test_smi_anon_ms_list)
    training_smi_anon_ms_set = set(training_smi_anon_ms_list)

    validated = True
    for test_smi_anon_ms in test_smi_anon_ms_set:
        if test_smi_anon_ms in training_smi_anon_ms_set:
            validated = False
            print(f'Test anonMS {test_smi_anon_ms} in training set!')

    if validated:
        print("No AnonMS in test set occurs in training set!")

def butina_clustering(smi_list, cutoff, fp_function=compute_countMorgFP):
    fps_list = fp_function(smi_list)
    dists = []
    nfps = len(fps_list)
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps_list[i],fps_list[:i])
        dists.extend([1-x for x in sims])
    clusters = Butina.ClusterData(dists,nfps,cutoff,isDistData=True)
    return clusters

def plot_butina_clustering(smi_list, clusters, cutoff, yscale='log', substructure_str=''):
    substructure_unique_smi = list(set(smi_list))
    clusters_sorted = sorted(clusters, key=lambda x: len(x))

    num_substructures_butina = 0
    for cluster in clusters:
        num_substructures_butina += len(cluster)
    
    if num_substructures_butina != len(substructure_unique_smi):
        raise ValueError(f"Number of substructures butina got does not match the number of unique smiles for {substructure_str}")

    clusters_smi = []
    for cluster in clusters_sorted:
        temp_smi_list = []
        for substructure_idx in cluster:
            smi = substructure_unique_smi[substructure_idx]
            temp_smi_list.append(smi)
        temp_smi_set = set(temp_smi_list)
        temp_smi_list = list(temp_smi_set)
        clusters_smi.append(temp_smi_list)

    print(f'Number of clusters: {len(clusters_smi)}')

    clusters_sorted_decending = sorted(clusters, key=lambda x: -len(x))
    clusters_sizes = [len(cluster) for cluster in clusters_sorted_decending]
    fig, ax = plt.subplots()
    x_ticks = list(range(len(clusters_sizes)))
    bar_colors = ['tab:blue']*(len(clusters_sizes))
    ax.bar(x_ticks, clusters_sizes, color=bar_colors)
    ax.set_ylabel('Substructure count')
    ax.set_xlabel('Cluster ID')
    #ax.set_title(f'Butina clustering of {substructure_str} with cutoff at {cutoff}')
    plt.yscale(yscale)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()
    return fig




def butina_clustering_substructures_with_fixed_cutoff(smi_sets_without_attachment, cutoff, plot_top_clusters = None, plot = False, yscale='log', test_or_trainval = '', save_fig=False):
    clusters_for_substructures = {}
    for substructure, substructure_str in zip(smi_sets_without_attachment, ["POI", "LINKER", "E3"]):
        if substructure_str == "LINKER":
            fp_function = compute_RDKitFP
        else:
            fp_function = compute_countMorgFP
        clusters = butina_clustering(substructure, cutoff=cutoff, fp_function=fp_function)
        clusters_for_substructures[substructure_str] = clusters

        print(f'{substructure_str}:')
        print(f'Number of unique substructures: {len(substructure)}')
        if plot:
            fig = plot_butina_clustering(substructure, clusters, cutoff, yscale)

            if plot_top_clusters is not None:
                clusters_sorted_decending = sorted(clusters, key=lambda x: -len(x))
                for cluster_idx in plot_top_clusters:
                    cluster = clusters_sorted_decending[cluster_idx]
                    for substructure_idx in cluster:
                        smi = substructure[substructure_idx]
                        display(Chem.MolFromSmiles(smi))
            
            if save_fig:
                fig.savefig(f'fig_butinaclustering/cutoff{cutoff}_{test_or_trainval}_{substructure_str}.svg', format='svg', dpi=1200, bbox_inches='tight')
                fig.savefig(f'fig_butinaclustering/cutoff{cutoff}_{test_or_trainval}_{substructure_str}.png', format='png', dpi=1200, bbox_inches='tight')
        

    
    return clusters_for_substructures




def get_clusters_substructures_attachments_dict(clusters_for_substructures, smi_set_without_attachment, unique_substructures_without_attachments_dict_to_with_attachments):

    #POI: {cluster_idx: {variant_idx: (attachment_idx, ...), ... }, ...}
    clusters_substructures_attachments_dict = {'POI': {}, 'LINKER': {}, 'E3': {}}

    for i, substruct_str in enumerate(["POI", "LINKER", "E3"]):
        clusters = clusters_for_substructures[substruct_str]

        for cluster_idx, cluster in enumerate(clusters):
            clusters_substructures_attachments_dict[substruct_str][cluster_idx] = {}
            for substructure_idx, substructure_idx_without_attachment in enumerate(cluster):
                clusters_substructures_attachments_dict[substruct_str][cluster_idx][substructure_idx] = {}
                substructure_without_attachment = smi_set_without_attachment[i][substructure_idx_without_attachment]
                substructure_plural_with_attachment = unique_substructures_without_attachments_dict_to_with_attachments[substruct_str][substructure_without_attachment]

                for attachment_idx, substructure_with_attachment in enumerate(substructure_plural_with_attachment):
                    clusters_substructures_attachments_dict[substruct_str][cluster_idx][substructure_idx][attachment_idx] = substructure_with_attachment

    return clusters_substructures_attachments_dict












# ---------------------------------------------- GENERAL DATAPROCESSING

from itertools import repeat, chain
from collections import Counter

def sort_by_most_common_elements(y):
    return list(chain.from_iterable(repeat(i, c) for i,c in Counter(y).most_common()))

def get_unique_perserve_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def count_unique_SMILES_and_MurckoScaffolds(smiles_list):
    count_dict = {}
    smiles_list_unique = get_unique_perserve_order(smiles_list)
    count_dict["SMILES Count"] = len(smiles_list_unique)
    murcko_list_unique = get_unique_perserve_order([get_murcko(smi) for smi in smiles_list_unique])
    count_dict["MurckoScaffold Count"] = len(murcko_list_unique)
    framework_list_unique = get_unique_perserve_order([get_anonymous_murcko(smi) for smi in smiles_list_unique])
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
        unique_substructures[group] = list(group_df[substructure_col].dropna().unique())
    return unique_substructures



# -----------------------------------------------  PROCESS RDKIT DATA

def add_attachments(list_without_attachments, dict_map_without_to_with):
    smiles_with_attachment = []
    for smi in list_without_attachments:
        smiles_with_attachment.extend(dict_map_without_to_with[smi])
    smiles_with_attachment = list(set(smiles_with_attachment))
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_with_attachment]
    return smiles_with_attachment, mols

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


def prepare_data_set(test_set, p_column, poi_column, linker_column, e3_column):

    test_set['substructures'] = test_set.apply(lambda row: '.'.join([str(row[poi_column]), str(row[linker_column]), str(row[e3_column])]), axis=1)

    test_set = remove_multiple_substrucmathes(test_set, p_column, poi_column, e3_column) 

    test_set_substructures = test_set[['substructures']].copy()

    test_set_protacs = test_set[[p_column]].copy()
    test_set_protacs.rename(columns={p_column: 'Smiles'}, inplace=True)

    return test_set_protacs, test_set_substructures

  
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

def remove_dummy_atoms(mol):
    atoms_to_remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    editable_mol = Chem.EditableMol(mol)
    for idx in sorted(atoms_to_remove, reverse=True):
        editable_mol.RemoveAtom(idx)
    return editable_mol.GetMol()


def get_boundary_bondtype(mol, bondtype_count = None):

    if bondtype_count is None:
        bondtype_count = {'SINGLE': 0, 'DOUBLE': 0, 'TRIPLE': 0}
    bondtype_to_str = {rdchem.BondType.SINGLE: 'SINGLE',
                             rdchem.BondType.DOUBLE: 'DOUBLE',
                             rdchem.BondType.TRIPLE: 'TRIPLE'}

    # Find dummy atoms by symbol or index
    dummy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    for dummy_atom_idx in reversed(dummy_atoms): # reverse to avoid index shifting issues
        #identify the order of the bond
        atom = mol.GetAtomWithIdx(dummy_atom_idx)
        neighbors = atom.GetNeighbors()

        for neighbour_atom in neighbors:
            neighbour_idx = neighbour_atom.GetIdx()
            dummy_atom_bond = mol.GetBondBetweenAtoms(dummy_atom_idx, neighbour_idx)
            dummy_atom_bondtype = dummy_atom_bond.GetBondType()
            bondtype_count[bondtype_to_str[dummy_atom_bondtype]] += 1


    return bondtype_count




def remove_dummy_atom_from_mol(mol, output = "smiles"):

    if Chem.MolToSmiles(mol) == "O=C(CCCCCCCCCC[*:1])[*:2]":
        pass


    # Find dummy atoms by symbol or index
    dummy_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']

    hydrogen_atom = rdchem.Atom(1)
    bond_to_num_Hs_to_add = {rdchem.BondType.SINGLE: 1,
                             rdchem.BondType.DOUBLE: 2,
                             rdchem.BondType.TRIPLE: 3}

    #identify the order of the bond
    #add that many hydrogens to the other atom the dummy atom is connected to
    #remove the dummy atom
    #RemoveHs
    
    #for each dummy atom
    emol = Chem.RWMol(mol)
    for dummy_atom_idx in reversed(dummy_atoms): # reverse to avoid index shifting issues
        #identify the order of the bond
        atom = mol.GetAtomWithIdx(dummy_atom_idx)
        neighbors = atom.GetNeighbors()
        

        for neighbour_atom in neighbors:
            neighbour_idx = neighbour_atom.GetIdx()
            dummy_atom_bond = emol.GetBondBetweenAtoms(dummy_atom_idx, neighbour_idx)
            dummy_atom_bondtype = dummy_atom_bond.GetBondType()
            num_Hs_to_add = bond_to_num_Hs_to_add[dummy_atom_bondtype]
            for _ in range(num_Hs_to_add):
                hydrogen_atom_idx = emol.AddAtom(hydrogen_atom)
                emol.AddBond(neighbour_idx, hydrogen_atom_idx, order = rdchem.BondType.SINGLE)
        
        emol.RemoveAtom(dummy_atom_idx)

    substruct_mol_without_attachment_extra_hydrogens = Chem.Mol(emol)
    substruct_mol_without_attachment = Chem.RemoveHs(substruct_mol_without_attachment_extra_hydrogens)
    
    Chem.GetSymmSSSR(substruct_mol_without_attachment)
    Chem.SanitizeMol(substruct_mol_without_attachment)

    substruct_mol_without_attachment = Chem.MolFromSmiles(Chem.MolToSmiles(substruct_mol_without_attachment)) #verify this can be done...

    dummy_atoms = [atom.GetIdx() for atom in substruct_mol_without_attachment.GetAtoms() if atom.GetSymbol() == '*']
    if dummy_atoms != []:
        print(f'SMILES: {Chem.MolToSmiles(mol)}')
        display(mol)
        print("new mol:")
        display(substruct_mol_without_attachment)
        raise ValueError("dummy atoms still present!")


    if output == "smiles":
        return Chem.MolToSmiles(substruct_mol_without_attachment, canonical= True)
    else:
        return substruct_mol_without_attachment




def remove_dummy_atom_from_smiles(smiles, output = "smiles"):
    mol = Chem.MolFromSmiles(smiles)
    return remove_dummy_atom_from_mol(mol, output)


def find_atom_index_of_mapped_atoms_detailed(mol):
    poi_l_attachment_point = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 1]
    e3_l_attachment_point = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == 2]

    if len(poi_l_attachment_point) > 1 or len(e3_l_attachment_point) > 1:
        raise ValueError("Too many attachement points")

    return poi_l_attachment_point, e3_l_attachment_point

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



def get_bond_idx(smi, bonds_start_end_atoms):
    mol = Chem.MolFromSmiles(smi)

    bond_indices = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()

        if [begin_idx, end_idx] in bonds_start_end_atoms or [end_idx, begin_idx] in bonds_start_end_atoms:
            bond_indices.append(bond.GetIdx())
        elif (begin_idx, end_idx) in bonds_start_end_atoms or (end_idx, begin_idx) in bonds_start_end_atoms:
            bond_indices.append(bond.GetIdx())
    
    return bond_indices





# ------------------------------------------ GRAPH FUNCTIONS





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





# ----------------------------------------- DATA AUGMENTAION




def shuffle_dict_keys(original_dict):
    # Extract keys and values
    keys = list(original_dict.keys())
    values = list(original_dict.values())
    
    # Shuffle the keys
    random.shuffle(keys)
    
    # Reconstruct the dictionary with shuffled keys
    shuffled_dict = dict(zip(keys, values))
    return shuffled_dict

import random

def select_combination(clusters_dict, unused_substructures):
    selected_combination = []
    for substructure_type, clusters in clusters_dict.items():
        if unused_substructures[substructure_type]:
            # Prioritize unused substructures if any
            cluster_idx, substructure_wo_attachment_idx, attachment_point_idx = unused_substructures[substructure_type].pop()
        else:
            # Random selection if all substructures have been used
            cluster_idx = random.choice(list(clusters.keys())) # Select a cluster
            cluster = clusters[cluster_idx]
            substructure_wo_attachment_idx = random.choice(list(cluster.keys())) # Select a variant within the cluster
            substructure_wo_attachment = cluster[substructure_wo_attachment_idx]
            attachment_point_idx = random.choice(list(substructure_wo_attachment.keys())) # Select an attachment point

        selected_combination.append((cluster_idx, substructure_wo_attachment_idx, attachment_point_idx))
    return tuple(selected_combination)

def generate_protacs_indices(clusters_dict, num_protacs=None, num_protacs_factor= 1, factor_list=[1, 1, 1], force_use_all_attachment_points = True):
    used_combinations = set()
    augmented_protac_substructureindices_list = []
    # Initialize tracking for unused substructures
    if force_use_all_attachment_points:

        clusters_dict_shuffled = clusters_dict.copy()

        for substructure_type, clusters in clusters_dict.items():
            for cluster_idx, cluster in clusters.items():
                for substructure_idx, substructure in cluster.items():
                    clusters_dict_shuffled[substructure_type][cluster_idx][substructure_idx] = shuffle_dict_keys(substructure)   #shuffle attachmentpoints
                clusters_dict_shuffled[substructure_type][cluster_idx] = shuffle_dict_keys(cluster)
            clusters_dict_shuffled[substructure_type] = shuffle_dict_keys(clusters)

        unused_substructures = {
            substructure_type: {
                (cluster_idx, substructure_idx, attachment_point_idx)
                for cluster_idx, cluster in clusters.items()
                for substructure_idx, substructure in cluster.items()
                for attachment_point_idx in substructure.keys()
            }
            for substructure_type, clusters in clusters_dict_shuffled.items()
            }
    
        num_substructures_with_attachmentpoints_for_each_type = [len(unused_substructures[substructure_type]) for substructure_type in clusters_dict_shuffled.keys()]
        num_substructures_with_attachmentpoints_for_each_type = [f*val for f, val in zip(factor_list, num_substructures_with_attachmentpoints_for_each_type)] # factor_list is to allow to ignore training substructures, when they are mixed with test substructures
        substructure_class_with_most_unique_substructures_with_attachmentpoints = num_substructures_with_attachmentpoints_for_each_type.index(max(num_substructures_with_attachmentpoints_for_each_type))
        max_num_substructures_with_attachmentpoints = num_substructures_with_attachmentpoints_for_each_type[substructure_class_with_most_unique_substructures_with_attachmentpoints]
        
        if num_protacs == None:
            num_protacs = max_num_substructures_with_attachmentpoints
        num_protacs = int(num_protacs*num_protacs_factor)

        if num_protacs < max_num_substructures_with_attachmentpoints:
            print(f"Number of PROTACs ({num_protacs}) is less than some substructures max-count: {num_substructures_with_attachmentpoints_for_each_type}")

        clusters_dict_out = clusters_dict_shuffled

    else:
        unused_substructures = {substructure_type: () for substructure_type, clusters in clusters_dict.items()}
        clusters_dict_out = clusters_dict
    
    while len(augmented_protac_substructureindices_list) < num_protacs:
        combination = select_combination(clusters_dict_out, unused_substructures)
        if combination not in used_combinations:
            augmented_protac_substructureindices_list.append(combination)
            used_combinations.add(combination)
    
    return augmented_protac_substructureindices_list, clusters_dict_out






def get_cluster_substructure_attachment_counts(augmented_protac_substructureindices_list):
    # protacs is a list of tuples, where each tuple contains 3 elements (for POI, linker, E3)
    # Each element is a tuple itself, containing indices for cluster, substructure, and attachment point

    # Initialize structures to count occurrences
    cluster_counts = defaultdict(lambda: defaultdict(int))
    substructure_counts = defaultdict(lambda: defaultdict(int))
    attachment_point_counts = defaultdict(lambda: defaultdict(int))

    # Process protacs to fill the structures
    for protac in augmented_protac_substructureindices_list:
        for idx, substructure_type in enumerate(["POI", "Linker", "E3"]):
            cluster_idx, substructure_idx, attachment_point_idx = protac[idx]
            
            # Count occurrences
            cluster_counts[substructure_type][(cluster_idx,)] += 1
            substructure_counts[substructure_type][(cluster_idx, substructure_idx)] += 1
            attachment_point_counts[substructure_type][(cluster_idx, substructure_idx, attachment_point_idx)] += 1
    
    return cluster_counts, substructure_counts, attachment_point_counts
            


def sort_counts(counts):
    """
    Sorts the counts dictionary by cluster ID, then substructure ID, then attachment point ID.
    Returns a sorted list of tuples from the counts dictionary.
    """
    # Sorting first by cluster ID, then by substructure ID, then by attachment point ID
    return sorted(counts.items(), key=lambda item: item[0])





def plot_counts_with_patterns_and_sorting(counts, title, colors=None, ylabel=''):
    """Plots the counts with hatching patterns to differentiate clusters, sorted and without a border, with slightly overlapping bars."""
    # Define colors and hatches
    if colors is None:
        colors = ['darkgreen', 'lightgreen'] # Alternating colors for clusters
    
    fig, ax = plt.subplots()
    
    # Sort the counts dictionary
    sorted_counts = sort_counts(counts)
    
    # Variables to manage colors and hatches based on cluster
    current_cluster = None
    color_index = -1

    # Increase the bar width by 5% to make them overlap
    bar_width = 1.05 # Slightly increase bar width to ensure overlap and eliminate gaps

    for i, ((cluster_idx, *rest), value) in enumerate(sorted_counts):
        # Change color at the start of a new cluster
        if current_cluster != cluster_idx and colors is not None:
            current_cluster = cluster_idx
            color_index = (color_index + 1) % len(colors) # Cycle through colors
        
        # Draw the bar with specific color, no edge color, and increased width for overlap
        ax.bar(i, value, width=bar_width, color=colors[color_index], edgecolor='none')
    
    #ax.set_title(title)
    plt.xlabel(ylabel) #, fontsize=18)
    plt.ylabel('Substructure count')


    #if len(sorted_counts)<=5:
    #    fig.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.show()

    return fig

def barplot_from_protac_substructureindices_list(augmented_protac_substructureindices_list, dataset_name = '', save_figs = False):
   
    
    cluster_counts, substructure_counts, attachment_point_counts = get_cluster_substructure_attachment_counts(augmented_protac_substructureindices_list=augmented_protac_substructureindices_list)
    for substructure_type in ["POI", "Linker", "E3"]:
        print(f"{substructure_type} Cluster distribution")
        fig = plot_counts_with_patterns_and_sorting(cluster_counts[substructure_type], f'{substructure_type} - Cluster Distribution', colors=["darkgreen"], ylabel="Cluster ID")
        
        if save_figs:
            fig.savefig(f'fig_substructuredist/{dataset_name}_clustercounts_{substructure_type}.svg', format='svg', dpi=1200, bbox_inches='tight')
            fig.savefig(f'fig_substructuredist/{dataset_name}_clustercounts_{substructure_type}.png', format='png', dpi=1200, bbox_inches='tight')
        
        #print(f"{substructure_type} Distribution of Substructures without attachmentpoints, within clusters")
        #plot_counts_with_patterns_and_sorting(substructure_counts[substructure_type], f'{substructure_type} - Substructure Distribution', colors=['darkgreen', 'lightgreen'], ylabel="Substructure without attachment ID")
        
        print(f"{substructure_type} Distribution of Substructures with attachmentpoints, within clusters")
        fig = plot_counts_with_patterns_and_sorting(attachment_point_counts[substructure_type], f'{substructure_type} - Attachment Point Distribution', colors=['darkgreen', 'lightgreen'], ylabel="Substructure ID")
        
        if save_figs:
            fig.savefig(f'fig_substructuredist/{dataset_name}_substructurecounts_{substructure_type}.svg', format='svg', dpi=1200, bbox_inches='tight')
            fig.savefig(f'fig_substructuredist/{dataset_name}_substructurecounts_{substructure_type}.png', format='png', dpi=1200, bbox_inches='tight')
        
    

def generate_protac_from_indices_list(clusters_substructures_attachments_dict, augmented_protac_substructureindices_list, bond_type = 'rand_uniform'):
    smiles_dict = {'PROTAC SMILES': [], 'POI SMILES': [], 'LINKER SMILES': [], 'E3 SMILES': [], 'POI SMILES WITHOUT ATTACHMENT': [], 'LINKER SMILES WITHOUT ATTACHMENT': [], 'E3 SMILES WITHOUT ATTACHMENT': []}
    for augmented_protac_tuple in tqdm(augmented_protac_substructureindices_list):
        poi_tuple = augmented_protac_tuple[0]
        linker_tuple = augmented_protac_tuple[1]
        e3_tuple = augmented_protac_tuple[2]

        poi_with_attachment = clusters_substructures_attachments_dict["POI"][poi_tuple[0]][poi_tuple[1]][poi_tuple[2]]
        linker_with_attachment = clusters_substructures_attachments_dict["LINKER"][linker_tuple[0]][linker_tuple[1]][linker_tuple[2]]
        e3_with_attachment = clusters_substructures_attachments_dict["E3"][e3_tuple[0]][e3_tuple[1]][e3_tuple[2]]

        protac_smiles, protac_mol = reassemble_protac(poi_with_attachment, linker_with_attachment, e3_with_attachment, bond_type = bond_type)

        smiles_dict['PROTAC SMILES'].append(protac_smiles)
        smiles_dict['POI SMILES'].append(poi_with_attachment)
        smiles_dict['LINKER SMILES'].append(linker_with_attachment)
        smiles_dict['E3 SMILES'].append(e3_with_attachment)

        poi_without_attachment = remove_dummy_atom_from_smiles(poi_with_attachment, output = "smiles")
        linker_without_attachment = remove_dummy_atom_from_smiles(linker_with_attachment, output = "smiles")
        e3_without_attachment = remove_dummy_atom_from_smiles(e3_with_attachment, output = "smiles")
        
        smiles_dict['POI SMILES WITHOUT ATTACHMENT'].append(poi_without_attachment)
        smiles_dict['LINKER SMILES WITHOUT ATTACHMENT'].append(linker_without_attachment)
        smiles_dict['E3 SMILES WITHOUT ATTACHMENT'].append(e3_without_attachment)
    
    return smiles_dict







# -------------------- GENERATE SPLITS



def generate_splits(poi_test_smi_set_without_attachment_splits, linker_test_smi_set_without_attachment_splits, e3_test_smi_set_without_attachment_splits,
                    poi_trainval_smi_set_without_attachment, linker_trainval_smi_set_without_attachment, e3_trainval_smi_set_without_attachment,
                    unique_substructures_without_attachments_dict_to_with_attachments, unique_substructures_with_attachments, fixed_cutoff):
    
    poi_trainval_smi_set = []
    for smi in poi_trainval_smi_set_without_attachment:
        poi_trainval_smi_set.extend(unique_substructures_without_attachments_dict_to_with_attachments['POI'][smi])
        poi_trainval_smi_set = list(set(poi_trainval_smi_set)) #duplicates are introduced via matching in the substructures without attachment points multiple times
        poi_trainval_set = [Chem.MolFromSmiles(smi) for smi in poi_trainval_smi_set]

    linker_trainval_smi_set = []
    for smi in linker_trainval_smi_set_without_attachment:
        linker_trainval_smi_set.extend(unique_substructures_without_attachments_dict_to_with_attachments['LINKER'][smi])
        linker_trainval_smi_set = list(set(linker_trainval_smi_set))
        linker_trainval_set = [Chem.MolFromSmiles(smi) for smi in linker_trainval_smi_set]

    e3_trainval_smi_set = []
    for smi in e3_trainval_smi_set_without_attachment:
        e3_trainval_smi_set.extend(unique_substructures_without_attachments_dict_to_with_attachments['E3'][smi])
        e3_trainval_smi_set = list(set(e3_trainval_smi_set))
        e3_trainval_set = [Chem.MolFromSmiles(smi) for smi in e3_trainval_smi_set]


    poi_trainval_substruct_smi_df = pd.DataFrame({"POI SMILES": poi_trainval_smi_set})
    linker_trainval_substruct_smi_df = pd.DataFrame({"LINKER SMILES": linker_trainval_smi_set})
    e3_trainval_substruct_smi_df = pd.DataFrame({"E3 SMILES": e3_trainval_smi_set})
    poi_trainval_substruct_smi_df.to_csv('../data/poi_trainval_substructures_with_attachment.csv', index=False)
    linker_trainval_substruct_smi_df.to_csv('../data/linker_trainval_substructures_with_attachment.csv', index=False)
    e3_trainval_substruct_smi_df.to_csv('../data/e3_trainval_substructures_with_attachment.csv', index=False)

    poi_trainval_smi_set_without_attachment_for_df = [remove_dummy_atom_from_mol(mol, output="smiles") for mol in poi_trainval_set]
    linker_trainval_smi_set_without_attachment_for_df = [remove_dummy_atom_from_mol(mol, output="smiles") for mol in linker_trainval_set]
    e3_trainval_smi_set_without_attachment_for_df = [remove_dummy_atom_from_mol(mol, output="smiles") for mol in e3_trainval_set]
    poi_trainval_substruct_smi_df_without_attachment = pd.DataFrame({"POI SMILES": poi_trainval_smi_set_without_attachment_for_df})
    linker_trainval_substruct_smi_df_without_attachment = pd.DataFrame({"LINKER SMILES": linker_trainval_smi_set_without_attachment_for_df})
    e3_trainval_substruct_smi_df_without_attachment = pd.DataFrame({"E3 SMILES": e3_trainval_smi_set_without_attachment_for_df})
    poi_trainval_substruct_smi_df_without_attachment.to_csv('../data/poi_trainval_substructures_without_attachment.csv', index=False)
    linker_trainval_substruct_smi_df_without_attachment.to_csv('../data/linker_trainval_substructures_without_attachment.csv', index=False)
    e3_trainval_substruct_smi_df_without_attachment.to_csv('../data/e3_trainval_substructures_without_attachment.csv', index=False)

        
    print(f' Trainval counts of poi, with attachment: {count_unique_SMILES_and_MurckoScaffolds(poi_trainval_smi_set)} \n')
    print(f' Trainval counts of linker, with attachment: {count_unique_SMILES_and_MurckoScaffolds(linker_trainval_smi_set)} \n')
    print(f' Trainval counts of e3, with attachment: {count_unique_SMILES_and_MurckoScaffolds(e3_trainval_smi_set)} \n')






    trainval_smi_set_without_attachment = [poi_trainval_smi_set_without_attachment, linker_trainval_smi_set_without_attachment, e3_trainval_smi_set_without_attachment]


    

    trainval_clusters_for_substructures = butina_clustering_substructures_with_fixed_cutoff(
                                                            smi_sets_without_attachment = trainval_smi_set_without_attachment, 
                                                            cutoff = fixed_cutoff,
                                                            plot_top_clusters = [], 
                                                            plot = True,
                                                            yscale='linear',
                                                            test_or_trainval = "trainval")

    trainval_clusters_substructures_attachments_dict = get_clusters_substructures_attachments_dict(trainval_clusters_for_substructures,
                                                                                                    trainval_smi_set_without_attachment,
                                                                                                    unique_substructures_without_attachments_dict_to_with_attachments)

    
    
    
    
    
    num_splits = 3
    split_dict = {}
    for split_idx in range(num_splits):
        poi_test_smi_set_without_attachment = poi_test_smi_set_without_attachment_splits[split_idx]
        linker_test_smi_set_without_attachment = linker_test_smi_set_without_attachment_splits[split_idx]
        e3_test_smi_set_without_attachment = e3_test_smi_set_without_attachment_splits[split_idx]

        


        poi_test_smi_set = []
        for smi in poi_test_smi_set_without_attachment:
            poi_test_smi_set.extend(unique_substructures_without_attachments_dict_to_with_attachments['POI'][smi])
        poi_test_smi_set = list(set(poi_test_smi_set))
        poi_test_set = [Chem.MolFromSmiles(smi) for smi in poi_test_smi_set]

        linker_test_smi_set = []
        for smi in linker_test_smi_set_without_attachment:
            linker_test_smi_set.extend(unique_substructures_without_attachments_dict_to_with_attachments['LINKER'][smi])
        linker_test_smi_set = list(set(linker_test_smi_set))
        linker_test_set = [Chem.MolFromSmiles(smi) for smi in linker_test_smi_set]

        e3_test_smi_set = []
        for smi in e3_test_smi_set_without_attachment:
            e3_test_smi_set.extend(unique_substructures_without_attachments_dict_to_with_attachments['E3'][smi])
        e3_test_smi_set = list(set(e3_test_smi_set))
        e3_test_set = [Chem.MolFromSmiles(smi) for smi in e3_test_smi_set]

        test_smi_set_without_attachment = [poi_test_smi_set_without_attachment, linker_test_smi_set_without_attachment, e3_test_smi_set_without_attachment]



        
        poi_test_smi_set_without_attachment_full = list(itertools.chain.from_iterable(poi_test_smi_set_without_attachment_splits.values()))
        linker_test_smi_set_without_attachment_full = list(itertools.chain.from_iterable(linker_test_smi_set_without_attachment_splits.values()))
        e3_test_smi_set_without_attachment_full = list(itertools.chain.from_iterable(e3_test_smi_set_without_attachment_splits.values()))


        poi_test_smi_set_full, poi_test_set_full = add_attachments(poi_test_smi_set_without_attachment_full, unique_substructures_without_attachments_dict_to_with_attachments['POI'])
        linker_test_smi_set_full, linker_test_set_full = add_attachments(linker_test_smi_set_without_attachment_full, unique_substructures_without_attachments_dict_to_with_attachments['LINKER'])
        e3_test_smi_set_full, e3_test_set_full = add_attachments(e3_test_smi_set_without_attachment_full, unique_substructures_without_attachments_dict_to_with_attachments['E3'])


        poi_test_substruct_smi_df = pd.DataFrame({"POI SMILES": poi_test_smi_set_full})
        linker_test_substruct_smi_df = pd.DataFrame({"LINKER SMILES": linker_test_smi_set_full})
        e3_test_substruct_smi_df = pd.DataFrame({"E3 SMILES": e3_test_smi_set_full})
        poi_test_substruct_smi_df.to_csv('../data/poi_test_substructures_with_attachment.csv', index=False)
        linker_test_substruct_smi_df.to_csv('../data/linker_test_substructures_with_attachment.csv', index=False)
        e3_test_substruct_smi_df.to_csv('../data/e3_test_substructures_with_attachment.csv', index=False)

        poi_test_smi_set_without_attachment_for_df = [remove_dummy_atom_from_mol(mol, output="smiles") for mol in poi_test_set_full]
        linker_test_smi_set_without_attachment_for_df = [remove_dummy_atom_from_mol(mol, output="smiles") for mol in linker_test_set_full]
        e3_test_smi_set_without_attachment_for_df = [remove_dummy_atom_from_mol(mol, output="smiles") for mol in e3_test_set_full]
        poi_test_substruct_smi_df_without_attachment = pd.DataFrame({"POI SMILES": poi_test_smi_set_without_attachment_for_df})
        linker_test_substruct_smi_df_without_attachment = pd.DataFrame({"LINKER SMILES": linker_test_smi_set_without_attachment_for_df})
        e3_test_substruct_smi_df_without_attachment = pd.DataFrame({"E3 SMILES": e3_test_smi_set_without_attachment_for_df})
        poi_test_substruct_smi_df_without_attachment.to_csv('../data/poi_test_substructures_without_attachment.csv', index=False)
        linker_test_substruct_smi_df_without_attachment.to_csv('../data/linker_test_substructures_without_attachment.csv', index=False)
        e3_test_substruct_smi_df_without_attachment.to_csv('../data/e3_test_substructures_without_attachment.csv', index=False)





        save_fig = False

        fp_function_dict = {"POI": compute_countMorgFP,
                            "LINKER": compute_RDKitFP,
                            "E3": compute_countMorgFP}

        cutoffs_dict = {"POI": 0.45,
                            "LINKER": 0.45,
                            "E3": 0.5}

        test_sets_all_splits_for_validation = {"POI": poi_test_smi_set_without_attachment_full,
                                "LINKER": linker_test_smi_set_without_attachment_full,
                                "E3": e3_test_smi_set_without_attachment_full}

        train_sets_for_validation = {"POI": poi_trainval_smi_set_without_attachment,
                    "LINKER": linker_trainval_smi_set_without_attachment,
                    "E3": e3_trainval_smi_set_without_attachment}

        for substructure_str in train_sets_for_validation.keys():
            #plot max tanimoto similarity between the trainval and test sets
            print(substructure_str)
            is_valid, max_similarities = validate_test_set(test_set=test_sets_all_splits_for_validation[substructure_str], 
                                                                training_set=train_sets_for_validation[substructure_str], 
                                                                cutoff=cutoffs_dict[substructure_str], 
                                                                fp_function=fp_function_dict[substructure_str], 
                                                                plot_highest_sim_match = True,
                                                                save_fig=False,
                                                                save_title=substructure_str)
            print(f'count max_similarities: {len(max_similarities)}')
            fig, ax = plt.subplots()
                # Plotting the maximum similarity for all test molecules, against the trainset
            plt.hist(max_similarities, bins=30, alpha=0.75)
            print(f'All most similar pairs are below cutoff ({cutoffs_dict[substructure_str]}) is {is_valid}')
            plt.xlabel('Maximum Tanimoto Similarity')
            plt.ylabel('Frequency')
            print(f'Distribution of Maximum Tanimoto Similarity (Test vs. Train)')
            plt.show()
            if save_fig:
                fig.savefig(f'fig_method/most_similar_train_test_histogram_{substructure_str}.png', format='png', dpi=1200, bbox_inches='tight')
                fig.savefig(f'fig_method/most_similar_train_test_histogram_{substructure_str}.svg', format='svg', dpi=1200, bbox_inches='tight')


        for substructure_type, smi_list in unique_substructures_with_attachments.items():
            print(f' All counts of {substructure_type}, with attachment: {count_unique_SMILES_and_MurckoScaffolds(smi_list)}')
        
        print(f' Test counts of poi, with attachment, split {split_idx}: {count_unique_SMILES_and_MurckoScaffolds(poi_test_smi_set)}')


        print(f' Test counts of linker, with attachment, split {split_idx}: {count_unique_SMILES_and_MurckoScaffolds(linker_test_smi_set)}')


        print(f' Test counts of e3, with attachment, split {split_idx}: {count_unique_SMILES_and_MurckoScaffolds(e3_test_smi_set)}')


        print(f' Test counts of poi, with attachment, ALL SPLITS: {count_unique_SMILES_and_MurckoScaffolds(poi_test_smi_set_full)}')
        print(f' Test counts of linker, with attachment, ALL SPLITS: {count_unique_SMILES_and_MurckoScaffolds(linker_test_smi_set_full)}')
        print(f' Test counts of e3, with attachment, ALL SPLITS: {count_unique_SMILES_and_MurckoScaffolds(e3_test_smi_set_full)}')




        #Validate that no general MS in testset is in trainset

        print("POI:")
        validate_no_general_ms_leakage(poi_test_smi_set, poi_trainval_smi_set)
        print("Linker:")
        validate_no_general_ms_leakage(linker_test_smi_set, linker_trainval_smi_set)
        print("E3:")
        validate_no_general_ms_leakage(e3_test_smi_set, e3_trainval_smi_set)





        test_smi_set = [poi_test_smi_set, linker_test_smi_set, e3_test_smi_set] #test_AnonMS_to_smi #
        test_smi_MS_set_with_duplicates = [[get_murcko(smi) for smi in testset] for testset in test_smi_set ]
        test_smi_MS_set = [list(set(testset)) for testset in test_smi_MS_set_with_duplicates]
        test_smi_AnonMS_set_with_duplicates = [[get_anonymous_murcko(smi) for smi in testset] for testset in test_smi_set ]
        test_smi_AnonMS_set = [list(set(testset)) for testset in test_smi_AnonMS_set_with_duplicates]

        train_smi_set = [poi_trainval_smi_set, linker_trainval_smi_set, e3_trainval_smi_set] #train_AnonMS_to_smi

        for train_smi_set_substructure, substructure in zip(train_smi_set, ["POI", "Linker", "E3"]):
            print(f'Number of trainval {substructure}: {len(train_smi_set_substructure)}')
        print(f'Number of unique combinations: {len(train_smi_set[0])*len(train_smi_set[1])*len(train_smi_set[2]) / 1000**2} million')
        print("\n")
        for test_smi_set_substructure, substructure in zip(test_smi_set, ["POI", "Linker", "E3"]):
            print(f'Number of test {substructure}: {len(test_smi_set_substructure)}')
        print(f'Number of unique combinations: {len(test_smi_set[0])*len(test_smi_set[1])*len(test_smi_set[2])}')
        print("\n")
        for test_smi_MS_set_substructure, substructure in zip(test_smi_MS_set, ["POI", "Linker", "E3"]):
            print(f'Number of test MS for {substructure}: {len(test_smi_MS_set_substructure)}')
        print(f'Number of unique combinations: {len(test_smi_MS_set[0])*len(test_smi_MS_set[1])*len(test_smi_MS_set[2])}')
        print("\n")
        for test_smi_AnonMS_set_substructure, substructure in zip(test_smi_AnonMS_set, ["POI", "Linker", "E3"]):
            print(f'Number of test AnonMS for {substructure}: {len(test_smi_AnonMS_set_substructure)}')
        print(f'Number of unique combinations: {len(test_smi_AnonMS_set[0])*len(test_smi_AnonMS_set[1])*len(test_smi_AnonMS_set[2])}')




        
        

        test_clusters_for_substructures = butina_clustering_substructures_with_fixed_cutoff(
                                                            smi_sets_without_attachment = test_smi_set_without_attachment, 
                                                            cutoff = fixed_cutoff,
                                                            plot_top_clusters = [], 
                                                            plot = True,
                                                            yscale='linear',
                                                            test_or_trainval = "test")


        


        test_clusters_substructures_attachments_dict = get_clusters_substructures_attachments_dict(test_clusters_for_substructures,
                                                                                                    test_smi_set_without_attachment,
                                                                                                    unique_substructures_without_attachments_dict_to_with_attachments)



        #generate indices of substructures

        #1 unknown substructure
        test_poi_clusters_substructures_attachments_dict = trainval_clusters_substructures_attachments_dict.copy()
        test_poi_clusters_substructures_attachments_dict["POI"] = test_clusters_substructures_attachments_dict["POI"]

        test_linker_clusters_substructures_attachments_dict = trainval_clusters_substructures_attachments_dict.copy()
        test_linker_clusters_substructures_attachments_dict["LINKER"] = test_clusters_substructures_attachments_dict["LINKER"]

        test_e3_clusters_substructures_attachments_dict = trainval_clusters_substructures_attachments_dict.copy()
        test_e3_clusters_substructures_attachments_dict["E3"] = test_clusters_substructures_attachments_dict["E3"]


        #2 unknown substructures
        test_poilinker_clusters_substructures_attachments_dict = test_clusters_substructures_attachments_dict.copy()
        test_poilinker_clusters_substructures_attachments_dict["E3"] = trainval_clusters_substructures_attachments_dict["E3"]

        test_poie3_clusters_substructures_attachments_dict = test_clusters_substructures_attachments_dict.copy()
        test_poie3_clusters_substructures_attachments_dict["LINKER"] = trainval_clusters_substructures_attachments_dict["LINKER"]

        test_e3linker_clusters_substructures_attachments_dict = test_clusters_substructures_attachments_dict.copy()
        test_e3linker_clusters_substructures_attachments_dict["POI"] = trainval_clusters_substructures_attachments_dict["POI"]



        
        #3 unknown
        num_protacs_factor = 4 # the number new augmented unique protacs equal to num_protacs_factor multiplied by the number of substructures which there are most
        test_protac_augmented_protac_substructureindices_list, test_clusters_substructures_attachments_dict_out = generate_protacs_indices(clusters_dict=test_clusters_substructures_attachments_dict, 
                                                                                        num_protacs_factor=num_protacs_factor,
                                                                                        force_use_all_attachment_points=True)




    
        #2 unkown

        #Try to make sure that each unknown substructure are at least found 4 times in the each test set. Unknown E3 are few of, so the PROTAC count in Test E3Linker is set to that of the Test POILinker

        test_poilinker_augmented_protac_substructureindices_list, test_poilinker_clusters_substructures_attachments_dict_out = generate_protacs_indices(clusters_dict=test_poilinker_clusters_substructures_attachments_dict, 
                                                                                        num_protacs_factor=5,
                                                                                        factor_list=[1, 1, 0],
                                                                                        force_use_all_attachment_points=True)
        test_poie3_augmented_protac_substructureindices_list, test_poie3_clusters_substructures_attachments_dict = generate_protacs_indices(clusters_dict=test_poie3_clusters_substructures_attachments_dict, 
                                                                                        num_protacs_factor=5,
                                                                                        factor_list=[1, 0, 1],
                                                                                        force_use_all_attachment_points=True)
        test_e3linker_augmented_protac_substructureindices_list, test_e3linker_clusters_substructures_attachments_dict_out = generate_protacs_indices(clusters_dict=test_e3linker_clusters_substructures_attachments_dict, 
                                                                                        num_protacs_factor=5,
                                                                                        factor_list=[0, 1, 1],
                                                                                        force_use_all_attachment_points=True)



        
        #1 unknown

        #Try to make sure that each unknown substructure are at least found 4 times in the each test set. Unknown E3 are few of, so the PROTAC count in Test E3 is set to that of the Test POI


        test_poi_augmented_protac_substructureindices_list, test_poi_clusters_substructures_attachments_dict_out = generate_protacs_indices(clusters_dict=test_poi_clusters_substructures_attachments_dict, 
                                                                                        num_protacs_factor=5,
                                                                                        factor_list=[1, 0, 0],
                                                                                        force_use_all_attachment_points=True)

        test_linker_augmented_protac_substructureindices_list, test_linker_clusters_substructures_attachments_dict = generate_protacs_indices(clusters_dict=test_linker_clusters_substructures_attachments_dict, 
                                                                                        num_protacs_factor=5,
                                                                                        factor_list=[0, 1, 0],
                                                                                        force_use_all_attachment_points=True)

        test_e3_augmented_protac_substructureindices_list, test_e3_clusters_substructures_attachments_dict_out = generate_protacs_indices(clusters_dict=test_e3_clusters_substructures_attachments_dict, 
                                                                                        num_protacs_factor=25,
                                                                                        factor_list=[0, 0, 1],
                                                                                        force_use_all_attachment_points=True)




        
        #barplot_from_protac_substructureindices_list(augmented_protac_substructureindices_list=trainval_augmented_protac_substructureindices_list, dataset_name='trainval', save_figs = True)
        #barplot_from_protac_substructureindices_list(augmented_protac_substructureindices_list=test_protac_augmented_protac_substructureindices_list, dataset_name='test_protac', save_figs = True)
        #barplot_from_protac_substructureindices_list(augmented_protac_substructureindices_list=test_poilinker_augmented_protac_substructureindices_list)
        #barplot_from_protac_substructureindices_list(augmented_protac_substructureindices_list=test_poie3_augmented_protac_substructureindices_list)
        #barplot_from_protac_substructureindices_list(augmented_protac_substructureindices_list=test_e3linker_augmented_protac_substructureindices_list)
        #barplot_from_protac_substructureindices_list(augmented_protac_substructureindices_list=test_poi_augmented_protac_substructureindices_list)
        #barplot_from_protac_substructureindices_list(augmented_protac_substructureindices_list=test_linker_augmented_protac_substructureindices_list)
        #barplot_from_protac_substructureindices_list(augmented_protac_substructureindices_list=test_e3_augmented_protac_substructureindices_list)




        bond_type = 'rand_uniform'
        test_protac_smiles_with_attachments = generate_protac_from_indices_list(test_clusters_substructures_attachments_dict_out,
                                                                            test_protac_augmented_protac_substructureindices_list,
                                                                            bond_type = bond_type)
        test_poilinker_smiles_with_attachments = generate_protac_from_indices_list(test_poilinker_clusters_substructures_attachments_dict_out,
                                                                            test_poilinker_augmented_protac_substructureindices_list,
                                                                            bond_type = bond_type)
        test_poie3_smiles_with_attachments = generate_protac_from_indices_list(test_poie3_clusters_substructures_attachments_dict,
                                                                            test_poie3_augmented_protac_substructureindices_list,
                                                                            bond_type = bond_type)
        test_e3linker_smiles_with_attachments = generate_protac_from_indices_list(test_e3linker_clusters_substructures_attachments_dict_out,
                                                                            test_e3linker_augmented_protac_substructureindices_list,
                                                                            bond_type = bond_type)
        test_poi_smiles_with_attachments = generate_protac_from_indices_list(test_poi_clusters_substructures_attachments_dict_out,
                                                                            test_poi_augmented_protac_substructureindices_list,
                                                                            bond_type = bond_type)
        test_linker_smiles_with_attachments = generate_protac_from_indices_list(test_linker_clusters_substructures_attachments_dict,
                                                                            test_linker_augmented_protac_substructureindices_list,
                                                                            bond_type = bond_type)
        test_e3_smiles_with_attachments = generate_protac_from_indices_list(test_e3_clusters_substructures_attachments_dict_out,
                                                                            test_e3_augmented_protac_substructureindices_list,
                                                                            bond_type = bond_type)


        #generate dataframes

        
        test_protac_df = pd.DataFrame(test_protac_smiles_with_attachments)

        test_poilinker_df = pd.DataFrame(test_poilinker_smiles_with_attachments)
        test_poie3_df = pd.DataFrame(test_poie3_smiles_with_attachments)
        test_e3linker_df = pd.DataFrame(test_e3linker_smiles_with_attachments)

        test_poi_df = pd.DataFrame(test_poi_smiles_with_attachments)
        test_linker_df = pd.DataFrame(test_linker_smiles_with_attachments)
        test_e3_df = pd.DataFrame(test_e3_smiles_with_attachments)


        split_dict[split_idx] = {"Test PROTAC": test_protac_df,
                                 "Test POI": test_poi_df,
                                 "Test Linker": test_linker_df,
                                 "Test E3": test_e3_df,
                                 "Test POILINKER": test_poilinker_df,
                                 "Test POIE3": test_poie3_df,
                                 "Test E3Linker": test_e3linker_df}

        if split_idx == 0:
            #trainval_train_to_val_fraction = 1/trainval_fraction - 1
            trainval_fraction = 0.2 
            num_train_protacs_factor = 1

            num_trainval_protacs = num_train_protacs_factor/(1-trainval_fraction)
            num_protacs_factor = num_trainval_protacs*(1-trainval_fraction) + num_trainval_protacs*trainval_fraction # the number new augmented unique protacs equal to num_protacs_factor multiplied by the number of substructures which there are most
            trainval_augmented_protac_substructureindices_list, trainval_clusters_substructures_attachments_dict_out = generate_protacs_indices(clusters_dict=trainval_clusters_substructures_attachments_dict, 
                                                                                                                        num_protacs_factor=num_protacs_factor,
                                                                                                                        force_use_all_attachment_points=True)

            
            
            trainval_smiles_with_attachments = generate_protac_from_indices_list(trainval_clusters_substructures_attachments_dict_out,
                                                                            trainval_augmented_protac_substructureindices_list,
                                                                            bond_type = bond_type)
            
            trainval_df = pd.DataFrame(trainval_smiles_with_attachments) #spit df into training and validation
            train_df = trainval_df.sample(frac=1-trainval_fraction)
            val_df = trainval_df.drop(train_df.index)

            split_dict[split_idx]["Train"] = train_df
            split_dict[split_idx]["Validation"] = val_df


    return split_dict