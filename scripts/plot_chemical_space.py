import os
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

from rdkit import Chem
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from datasets import load_dataset

from protac_splitter.chemoinformatics import canonize

matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
})

def main():
    parser = argparse.ArgumentParser(
        description="Plot chemical space of PROTACs based on their SMILES strings."
    )
    parser.add_argument(
        "--protac_db_path",
        type=str,
        required=True,
        help="Path to the PROTAC-DB v3.0 CSV file.",
    )
    parser.add_argument(
        "--protac_pedia_path",
        type=str,
        required=True,
        help="Path to the PROTAC-Pedia CSV file.",
    )
    parser.add_argument(
        "--internal_data_path",
        type=str,
        default=None,
        help="Path to the internal PROTAC-Splitter dataset (optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chemical_space_plots",
        help="Directory to save the output plots.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=2,
        help="Number of processes to use for parallel processing.",
    )
    parser.add_argument(
        "--num_proc_fp_gen",
        type=int,
        default=8,
        help="Number of processes to use for fingerprint generation.",
    )
    parser.add_argument(
        "--test_internal_data",
        action='store_true',
        help="If set, will test plotting for internal PROTAC-Splitter dataset.",
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    def get_substructs(row):
        text = row['text']
        labels = row['labels']
        return {
            'PROTAC SMILES': text,
            'POI Ligand SMILES with direction': labels.split('.')[2],
            'Linker SMILES with direction': labels.split('.')[1],
            'E3 Binder SMILES with direction': labels.split('.')[0],
        }

    ds = load_dataset(
        'ailab-bio/PROTAC-Splitter-Dataset',
        'clustered',
        hf_token=os.getenv('HF_TOKEN', None),
    )
    ds = ds.map(get_substructs, num_proc=args.num_proc, remove_columns=['text', 'labels'])
    train_df = ds['train'].to_pandas()
    val_df = ds['validation'].to_pandas()
    test_df = ds['test'].to_pandas()
    held_out_df = ds['held_out'].to_pandas()

    # Load PROTAC-DB v3.0 and PROTAC-Pedia datasets
    protacdb_df = pd.read_csv(args.protac_db_path, low_memory=False).dropna(subset=['Smiles'])
    protacpedia_df = pd.read_csv(args.protac_pedia_path, low_memory=False).dropna(subset=['PROTAC SMILES'])

    # Rename the 'Smiles' column in protacdb_df to 'PROTAC SMILES'
    protacdb_df = protacdb_df.rename(columns={'Smiles': 'PROTAC SMILES'})

    # Drop duplicate rows in protacdb_df and protacpedia_df based on 'PROTAC SMILES'
    protacdb_df = protacdb_df.drop_duplicates(subset=['PROTAC SMILES'])
    protacpedia_df = protacpedia_df.drop_duplicates(subset=['PROTAC SMILES'])

    # Canonize the SMILES in the protacdb and protacpedia dataframes
    tqdm.pandas(desc="Canonizing PROTAC SMILES")
    protacdb_df['PROTAC SMILES'] = protacdb_df['PROTAC SMILES'].progress_apply(lambda x: canonize(x))
    protacpedia_df['PROTAC SMILES'] = protacpedia_df['PROTAC SMILES'].progress_apply(lambda x: canonize(x))

    # Map the "PROTAC SMILES" of protacdb and protacpedia to the columns of held_out_df: 'POI Ligand SMILES with direction', 'Linker SMILES with direction', 'E3 Binder SMILES with direction'
    def map_protac_smiles(row):
        protac_smiles = row['PROTAC SMILES']
        if protac_smiles in held_out_df['PROTAC SMILES'].values:
            return held_out_df[held_out_df['PROTAC SMILES'] == protac_smiles].iloc[0].to_dict()
        
        # If not found, return the original row
        return {
            'PROTAC SMILES': protac_smiles,
            'POI Ligand SMILES with direction': None,
            'Linker SMILES with direction': None,
            'E3 Binder SMILES with direction': None,
        }

    tqdm.pandas(desc="Mapping PROTAC SMILES to substructures")
    protacdb_df = protacdb_df.progress_apply(map_protac_smiles, axis=1, result_type='expand')
    protacpedia_df = protacpedia_df.progress_apply(map_protac_smiles, axis=1, result_type='expand')

    # Drop rows with NaN values in the SMILES columns
    protacdb_df = protacdb_df.dropna(subset=['POI Ligand SMILES with direction', 'Linker SMILES with direction', 'E3 Binder SMILES with direction'])
    protacpedia_df = protacpedia_df.dropna(subset=['POI Ligand SMILES with direction', 'Linker SMILES with direction', 'E3 Binder SMILES with direction'])

    if args.internal_data_path is not None:
        internal_df = pd.read_csv(args.internal_data_path, low_memory=False)
        if 'labels' in internal_df.columns and 'text' in internal_df.columns:
            # If the internal dataset has 'text' and 'labels', we need to extract the substructures
            tqdm.pandas(desc="Extracting substructures from internal dataset")
            internal_df = internal_df.progress_apply(get_substructs, axis=1, result_type='expand')

    # Get PROTAC-DB v3.0 and PROTAC-Pedia SMILES data
    ligands = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list),
        'held_out': defaultdict(list),
        'protacdb': defaultdict(list),
        'protacpedia': defaultdict(list),
    }
    if args.internal_data_path is not None:
        ligands['internal'] = defaultdict(list)
    for ligand_name in ['PROTAC SMILES', 'E3 Binder SMILES with direction', 'POI Ligand SMILES with direction', 'Linker SMILES with direction']:
        for ligand in train_df[ligand_name].unique():
            ligands['train'][ligand_name].append(ligand)

        for ligand in val_df[ligand_name].unique():
            ligands['val'][ligand_name].append(ligand)

        for ligand in test_df[ligand_name].unique():
            ligands['test'][ligand_name].append(ligand)
        
        for ligand in held_out_df[ligand_name].unique():
            ligands['held_out'][ligand_name].append(ligand)
            
        for ligand in protacdb_df[ligand_name].unique():
            ligands['protacdb'][ligand_name].append(ligand)
        
        for ligand in protacpedia_df[ligand_name].unique():
            ligands['protacpedia'][ligand_name].append(ligand)
        
        if args.internal_data_path is not None:
            for ligand in internal_df[ligand_name].unique():
                ligands['internal'][ligand_name].append(ligand)

    morgan_fp_generator = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=2, # 16
        fpSize=512, # 1024
        useBondTypes=True,
        includeChirality=True,
    )

    def bitvect_to_numpy(bitvect):
        return np.frombuffer(bitvect.ToBitString().encode(), 'u1') - ord('0')

    ligands_fp = {}
    for split, ligand_dict in ligands.items():
        ligands_fp[split] = defaultdict(list)
        for ligand_name, ligand_list in ligand_dict.items():
            filename = os.path.join(args.output_dir, f'{split}_{ligand_name.split(" ")[0].lower()}_fp.npy')
            # Load the FP if it exists
            if os.path.exists(filename):
                print(f"Loading {split} {ligand_name} FP from file: {filename}")
                ligands_fp[split][ligand_name] = np.load(filename)
                continue

            print(f"Generating {split} {ligand_name} FPs...")
            fp_list = morgan_fp_generator.GetFingerprints([Chem.MolFromSmiles(smiles) for smiles in ligand_list], numThreads=args.num_proc_fp_gen)
            fp_list = [bitvect_to_numpy(fp) for fp in fp_list]
            ligands_fp[split][ligand_name] = fp_list

            print(f"Saving {split} {ligand_name} FP to file: {filename}")
            np.save(filename, np.array(fp_list))
        print()

    column_names = [
        'PROTAC SMILES',
        'E3 Binder SMILES with direction',
        'POI Ligand SMILES with direction',
        'Linker SMILES with direction',
    ]
    for column_name in column_names:
        ligand_name = column_name.split(" ")[0].lower()
        ligand_name_ext = ' '.join(column_name.split(" ")[:(1 if 'linker' in ligand_name else 2)])
        
        # Stack all the embeddings together for visualization
        if column_name == 'PROTAC SMILES':
            fp_list = [
                ligands_fp['train'][column_name],
                ligands_fp['val'][column_name],
                ligands_fp['test'][column_name],
                ligands_fp['protacdb'][column_name],
                ligands_fp['protacpedia'][column_name],
            ]
        else:
            fp_list = [
                ligands_fp['protacdb'][column_name],
                ligands_fp['protacpedia'][column_name],
            ]
        if args.internal_data_path is not None:
            if args.test_internal_data:
                # Add a small noise to the internal data for testing
                fps = np.array(ligands_fp['internal'][column_name])
                internal_fp = fps + np.random.normal(0, 0.5, size=fps.shape)
                fp_list.append(internal_fp.tolist())
            else:    
                fp_list.append(ligands_fp['internal'][column_name])
        all_embeddings = np.vstack(fp_list)

        # Save the PCA embeddings for visualization
        filename = os.path.join(args.output_dir, f'all_{ligand_name}_embeddings_pca.npy')
        if os.path.exists(filename):
            print(f"Loading all {ligand_name.title()} embeddings PCA from file: {filename}")
            all_embeddings_pca = np.load(filename)
        else:
            # Run PCA on all embeddings for visualization
            pca = PCA(n_components=2, random_state=42)
            all_embeddings_pca = pca.fit_transform(all_embeddings)
            print(f"Saving all {ligand_name.title()} embeddings PCA to file: {filename}")
            np.save(filename, all_embeddings_pca)

        # Create a DataFrame for visualization
        df_embeddings = pd.DataFrame(all_embeddings_pca, columns=['x', 'y'])
        split_labels = []
        if column_name == 'PROTAC SMILES':
            split_labels = ['Train (Synthetic)'] * len(ligands_fp['train'][column_name]) + \
                           ['Validation (Synthetic)'] * len(ligands_fp['val'][column_name]) + \
                           ['Test (Synthetic)'] * len(ligands_fp['test'][column_name]) + \
                           ['PROTAC-DB v3.0'] * len(ligands_fp['protacdb'][column_name]) + \
                           ['PROTAC-Pedia'] * len(ligands_fp['protacpedia'][column_name])
            if args.internal_data_path is not None:
                split_labels += ['Internal Data'] * len(ligands_fp['internal'][column_name])
        else:
            split_labels = [f'{ligand_name_ext}s - PROTAC-DB v3.0'] * len(ligands_fp['protacdb'][column_name]) + \
                           [f'{ligand_name_ext}s - PROTAC-Pedia'] * len(ligands_fp['protacpedia'][column_name])
            if args.internal_data_path is not None:
                split_labels += [f'{ligand_name_ext}s - Internal'] * len(ligands_fp['internal'][column_name])
        df_embeddings['split'] = split_labels

        # Plot the PCA embeddings using seaborn
        palette = ['#83B8FE', '#FFA54C', '#94ED67', '#FF7FFF']
        colors = {
            'Train (Synthetic)': '#FFD700',       # Gold (yellowish)
            'Validation (Synthetic)': '#EE82EE',      # Violet
            'Test (Synthetic)': '#94ED67',    # Green (from palette)
            'PROTAC-DB v3.0': '#83B8FE',              # Blue (from palette)
            'PROTAC-Pedia': '#FF7F50',       # Coral (closer to orange)
            'Internal Data': '#8A2BE2', # Blue Violet
            'Orange': '#FFA54C',
            'Gray': '#D3D3D3',
            'Medium Slate Blue (indigo)': '#7B68EE',
            'Orchid (violet)': '#DA70D6',
            'Dark Turquoise (blueish)': '#00CED1',
            'Pale Green': '#98FB98',
            'Hot Pink (closer to violet)': '#FF69B4',
        }
        if column_name == 'PROTAC SMILES':
            palette = list(colors.values())[:df_embeddings['split'].nunique()]
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df_embeddings, x='x', y='y', hue='split', alpha=0.6, palette=palette, s=12, edgecolor='black', linewidth=0.1, rasterized=True)
        else:
            palette = ['#83B8FE', '#FF7FFF']
            if args.internal_data_path is not None:
                palette.append('#94ED67')
            plt.figure(figsize=(6, 6))
            sns.scatterplot(data=df_embeddings, x='x', y='y', hue='split', alpha=0.6, palette=palette, edgecolor='black')
        # Add legend and labels
        plt.xlabel('PCA Component 1', fontdict={'fontsize': 12})
        plt.ylabel('PCA Component 2', fontdict={'fontsize': 12})
        print('-' * 80)
        print(f"Plotting PCA for {column_name} with {len(df_embeddings)} points")
        print('-' * 80)
        plt.title(f'') # PCA of PROTAC SMILES Fingerprints')
        if column_name == 'PROTAC SMILES':
            plt.legend(title='', loc='upper left', fontsize=12, markerscale=2.2)
        else:
            plt.legend(title='', loc='lower left', fontsize=12, markerscale=2.2)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        # Save the plot
        plot_filename = os.path.join(args.output_dir, f'pca_{ligand_name}.pdf')
        plt.savefig(plot_filename, bbox_inches='tight')


if __name__ == "__main__":
    main()