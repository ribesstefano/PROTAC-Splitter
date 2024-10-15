import os
import sys
import argparse
from typing import Literal, Tuple, Dict, Optional, Tuple, Generator
from collections import Counter, defaultdict
from itertools import product
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import numba as nb
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
from rdkit import rdBase
from datasets import Dataset, DatasetDict, concatenate_datasets
from functools import partial
from datasets.utils.logging import disable_progress_bar


if 'ipykernel' in sys.modules:
    from tqdm.auto import tqdm  # for notebooks
else:
    from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), 'protac_splitter'))

from protac_splitter.protac_cheminformatics import reassemble_protac
from protac_splitter.evaluation import check_substructs


# Disable the RDKit warnings that pop up when RDKit fails to create molecules
RDLogger.DisableLog("rdApp.*")
blocker = rdBase.BlockLogs()


def safe_display(*args):
    """Displays content only if running in a Jupyter notebook."""
    if 'ipykernel' in sys.modules:
        from IPython.display import display
        display(*args)
    else:
        print(*args)


def remove_stereo(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.rdmolops.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        # print(e)
        return np.nan


def randomize_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=False, doRandom=True)
    except Exception as e:
        # print(e)
        return np.nan


def canonize_smiles(smiles: str) -> str:
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    except:
        return np.nan


@nb.jit(nopython=True)
def levenshtein_distance(s1: str, s2: str) -> int:
    """ Returns the Levenshtein distance between two strings.
    
    Args:
        s1: The first string.
        s2: The second string.
    
    Returns:
        The Levenshtein distance between the two strings.
    """
    # Create a matrix of zeros with dimensions len(s1) + 1 x len(s2) + 1
    matrix = np.zeros((len(s1) + 1, len(s2) + 1))
    # Fill the first row with the index of each character in s1
    for i in range(len(s1) + 1):
        matrix[i, 0] = i
    # Fill the first column with the index of each character in s2
    for j in range(len(s2) + 1):
        matrix[0, j] = j
    # Iterate over the matrix and fill in the values
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            # If the characters are the same, the cost is 0
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            # Fill in the matrix with the minimum of the three possible values
            matrix[i, j] = min(
                matrix[i - 1, j] + 1,
                matrix[i, j - 1] + 1,
                matrix[i - 1, j - 1] + cost
            )
    # Return the bottom right value of the matrix
    return matrix[-1, -1]


def get_ordered_substruct(
        protac_smiles: str,
        substruct1: str,
        substruct2: str,
) -> Tuple[str, str]:
    """Returns the first substructure that is found in the PROTAC SMILES string.
    
    Args:
        protac_smiles: The PROTAC SMILES string.
        substruct1: The first substructure string.
        substruct2: The second substructure string.

    Returns:
        The first substructure that is found in the PROTAC SMILES string.
    """
    # Remove stereochemistry and attachment points from the SMILES strings
    protac_smiles = remove_stereo(protac_smiles)
    substruct1_nodir = remove_stereo(substruct1).replace('[*:1]', '').replace('[*:2]', '')
    substruct2_nodir = remove_stereo(substruct2).replace('[*:1]', '').replace('[*:2]', '')

    # Remove all digits from the PROTAC SMILES string
    protac_smiles = ''.join([i for i in protac_smiles if not i.isdigit()])
    substruct1_nodir = ''.join([i for i in substruct1_nodir if not i.isdigit()])
    substruct2_nodir = ''.join([i for i in substruct2_nodir if not i.isdigit()])

    # Get a proportion of the PROTAC SMILES string that is the same length as
    # the substructure strings
    protac_sub1 = protac_smiles[:len(substruct1_nodir)]
    protac_sub2 = protac_smiles[:len(substruct2_nodir)]

    # # Check how "similar" the protac_sub1 string is to substruct1_nodir string
    # sub1_sim = sum([1 for i, j in zip(protac_sub1, substruct1_nodir) if i == j]) / len(substruct1_nodir)
    # # Check how "close" the protac_sub2 string is to substruct2_nodir string
    # sub2_sim = sum([1 for i, j in zip(protac_sub2, substruct2_nodir) if i == j]) / len(substruct2_nodir)
    # # Return the substructure that is more similar to the PROTAC SMILES string
    # return substruct1 if sub1_sim > sub2_sim else substruct2

    # Check how "similar" the protac_sub1 string is to substruct1_nodir string
    sub1_dist = levenshtein_distance(protac_sub1, substruct1_nodir)
    sub2_dist = levenshtein_distance(protac_sub2, substruct2_nodir)

    # Return the substructure that is more similar to the PROTAC SMILES string
    if sub1_dist < sub2_dist:
        return substruct1, substruct2
    else:
        return substruct2, substruct1
    

@nb.jit(nopython=True)
def join_substructures(
        protac_smiles: str,
        e3_smiles: str,
        linker_smiles: str,
        poi_smiles: str,
        random_order_prob: float = 0.0,
        fixed_first_sub: Optional[Literal['e3', 'poi']] = 'e3',
) -> str:
    """Joins the substructures with the linker to create a PROTAC SMILES string. Default is to always have the E3 substructure first, then the linker, and finally the POI substructure.

    Args:
        protac_smiles: The PROTAC SMILES string.
        e3_smiles: The E3 ligand SMILES string.
        linker_smiles: The linker SMILES string.
        poi_smiles: The POI ligand SMILES string.
        random_order_prob: The probability of randomly ordering the substructures.
        fixed_first_sub: The first substructure to be fixed in the PROTAC SMILES string.
        
    Returns:
        The concatenated SMILES strings of the substructures.
    """
    if np.random.rand() < random_order_prob:
        # Randomly order all the three substructures
        rand_idx = np.random.permutation(np.arange(3))
        substructs = [e3_smiles, linker_smiles, poi_smiles]
        return f'{substructs[rand_idx[0]]}.{substructs[rand_idx[1]]}.{substructs[rand_idx[2]]}'
    elif fixed_first_sub == 'e3':
        return f'{e3_smiles}.{linker_smiles}.{poi_smiles}'
    elif fixed_first_sub == 'poi':
        return f'{poi_smiles}.{linker_smiles}.{e3_smiles}'
    # else:
    #     first_substruct, second_substruct = get_ordered_substruct(protac_smiles, e3_smiles, poi_smiles)
    #     return f'{first_substruct}.{linker_smiles}.{second_substruct}'


def get_unique_substructs(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        'e3': df['E3 Binder SMILES with direction'].unique(),
        'linker': df['Linker SMILES with direction'].unique(),
        'poi': df['POI Ligand SMILES with direction'].unique(),
    }


def get_substruct_prob(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    unique_substructs = get_unique_substructs(df)
    probs = {
        'e3': np.array([1 / df['E3 Binder SMILES with direction'].value_counts()[sub] for sub in unique_substructs['e3']]),
        'linker': np.array([1 / df['Linker SMILES with direction'].value_counts()[sub] for sub in unique_substructs['linker']]),
        'poi': np.array([1 / df['POI Ligand SMILES with direction'].value_counts()[sub] for sub in unique_substructs['poi']]),
    }
    # Normalize the probabilities so that they sum to 1
    return {k: v / v.sum() for k, v in probs.items()}


def get_fingerprint(smiles: str, morgan_fpgen = None, radius: int = 10, fpSize: int = 512) -> np.ndarray:
    """ Get the Morgan fingerprint of a molecule.
    
    Args:
        smiles (str): The SMILES string of the molecule.
        morgan_fpgen: The Morgan fingerprint generator.

    Returns:
        np.ndarray: The Morgan fingerprint.
    """
    if morgan_fpgen is None:
        morgan_fpgen = AllChem.GetMorganGenerator(
            radius=radius,
            fpSize=fpSize,
            includeChirality=True,
        )
    return morgan_fpgen.GetFingerprint(Chem.MolFromSmiles(smiles))


def push_ds_to_hub(
        config_name: str,
        split: Optional[str] = None,
        dataset: Dataset | DatasetDict = None,
        train_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
):
    """Push the dataset to the Hugging Face Hub.
    
    Args:
        config_name: The name of the configuration.
        dataset: The dataset to push to the Hub.
        train_df: The training DataFrame.
        test_df: The test DataFrame.
    """
    if train_df is not None and test_df is not None:
        dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df, preserve_index=False),
            'test': Dataset.from_pandas(test_df, preserve_index=False),
        })
    elif dataset is None:
        raise ValueError('Either dataset or train_df and test_df must be provided.')
    dataset.push_to_hub(
        'ailab-bio/PROTAC-Splitter-Dataset',
        config_name=config_name,
        split=split,
        private=True,
        token=os.getenv('HF_TOKEN'),
        max_shard_size="2GB",
    )

def shuffle_substructs(s: str, shuffle_prob: float = 0.0) -> str:
    if np.random.rand() < shuffle_prob:
        substructs = s.split('.')
        np.random.shuffle(substructs)
        return '.'.join(substructs)
    else:
        return s


def check_dataframes(dfs: pd.DataFrame) -> bool:
    tqdm.pandas(desc='Checking train dataset')
    train_check = all(dfs['train'].progress_apply(lambda x: check_substructs(
            x['PROTAC SMILES'],
            x['POI Ligand SMILES with direction'],
            x['Linker SMILES with direction'],
            x['E3 Binder SMILES with direction'],
        ), axis=1)
    )
    tqdm.pandas(desc='Checking test dataset')
    test_check = all(dfs['test'].progress_apply(lambda x: check_substructs(
            x['PROTAC SMILES'],
            x['POI Ligand SMILES with direction'],
            x['Linker SMILES with direction'],
            x['E3 Binder SMILES with direction'],
        ), axis=1)
    )
    return train_check and test_check


def check_dataset(
    ds: Dataset,
    num_proc: int = 1,
) -> bool:
    """
    Checks whether all samples in a dataset are correct.

    Args:
        ds (Dataset): The dataset to check.
        checking_function (callable): A function that takes 'text' and 'labels' and returns True if the sample is correct.
        num_proc (int): Number of processes to use.

    Returns:
        bool: True if all samples are correct, False otherwise.
    """
    # Filter out incorrect samples using the checking function
    incorrect_samples = ds.filter(
        lambda example: not check_substructs(protac_smiles=example['text'], pred=example['labels']),
        num_proc=num_proc,
    )
    # If there are any incorrect samples, return False
    if len(incorrect_samples) > 0:
        print(f'Found {len(incorrect_samples)} incorrect samples.')
        print(pd.DataFrame(incorrect_samples))
        pd.DataFrame(incorrect_samples).to_csv('incorrect_samples.csv', index=False)
        return False
    else:
        return True

def convert_df_to_text(df: pd.DataFrame) -> pd.DataFrame:
    """ Convert a DataFrame to a DataFrame with 'text' and 'labels' columns.
    
    Args:
        df (pd.DataFrame): The DataFrame to convert.

    Returns:
        pd.DataFrame: The converted DataFrame.
    """
    protac_col = 'PROTAC SMILES'
    e3_col = 'E3 Binder SMILES with direction'
    linker_col = 'Linker SMILES with direction'
    poi_col = 'POI Ligand SMILES with direction'
    df['labels'] = df.apply(lambda x: join_substructures(x[protac_col], x[e3_col], x[linker_col], x[poi_col]), axis=1)
    df = df.rename(columns={'PROTAC SMILES': 'text'})
    return df[['text', 'labels']]


def main(
        num_proc: int = 16,
        max_train_samples: int = 10_000,
        disable_progress_bars: bool = False,
):
    # Disable the progress bar for Hugging Face Datasets (like map, filter, etc.)
    if disable_progress_bars:
        disable_progress_bar()

    np.random.seed(42)

    data_dir = os.path.join(os.getcwd(), 'data')

    ds = {
        'standard': {
            'train': pd.read_csv(os.path.join(data_dir, 'datasets', 'standard', 'train.csv')),
            'test': pd.read_csv(os.path.join(data_dir, 'datasets', 'standard', 'test.csv'))
        },
        'hardest': {
            'train': pd.read_csv(os.path.join(data_dir, 'datasets', 'hardest', 'train.csv')),
            'test': pd.read_csv(os.path.join(data_dir, 'datasets', 'hardest', 'test.csv'))
        },
        'e3_unique': {
            'train': pd.read_csv(os.path.join(data_dir, 'datasets', 'e3_unique', 'train.csv')),
            'test': pd.read_csv(os.path.join(data_dir, 'datasets', 'e3_unique', 'test.csv'))
        },
        'linker_unique': {
            'train': pd.read_csv(os.path.join(data_dir, 'datasets', 'linker_unique', 'train.csv')),
            'test': pd.read_csv(os.path.join(data_dir, 'datasets', 'linker_unique', 'test.csv'))
        },
        'poi_unique': {
            'train': pd.read_csv(os.path.join(data_dir, 'datasets', 'poi_unique', 'train.csv')),
            'test': pd.read_csv(os.path.join(data_dir, 'datasets', 'poi_unique', 'test.csv'))
        }
    }
    text_dfs = {}

    print('Checking dataframes...')
    for config_name, dataframes in ds.items():
        
        # TODO: The hardest split is not done yet.
        if config_name != 'standard':
            continue

        if not check_dataframes(dataframes):
            print(f'Error in {config_name} dataset: incorrect samples.')
            sys.exit(1)

        print(f'Checking {config_name} dataset...')
        train_df = convert_df_to_text(dataframes['train'])
        test_df = convert_df_to_text(dataframes['test'])
        text_dfs[config_name] = {'train': train_df, 'test': test_df}

        train_check = check_dataset(Dataset.from_pandas(train_df, preserve_index=False), num_proc=max(1, num_proc // 2))
        test_check = check_dataset(Dataset.from_pandas(test_df, preserve_index=False), num_proc=max(1, num_proc // 2))

        if not train_check or not test_check:
            print(f'Error in {config_name} dataset: reassembly failed.')
            sys.exit(1)
    print('All dataframes are correct (i.e., reassemblying substructures works).')

    shuffle_prob = 0.3 # Augmentation not used, for now...

    for config_name, train_test_dfs in ds.items():
        # TODO: The hardest split is not done yet.
        if config_name != 'standard':
            continue

        print('-' * 80)
        print(f'Processing {config_name} dataset...')
        print('-' * 80)
        
        train_df = text_dfs[config_name]['train']
        test_df = text_dfs[config_name]['test']

        # Get the test dataset to use for all the augmented datasets
        train_ds = Dataset.from_pandas(train_df, preserve_index=False, split='train')
        test_ds = Dataset.from_pandas(test_df, preserve_index=False, split='test')
        train_ds = train_ds.shuffle(seed=42)
        test_ds = test_ds.shuffle(seed=42)

        push_ds_to_hub(dataset=train_ds, config_name=config_name, split='train')
        push_ds_to_hub(dataset=test_ds, config_name=config_name, split='test')

        num_aug_samples = max_train_samples - len(train_df)
        if num_aug_samples <= 0:
            raise ValueError('The number of augmented samples must be greater than 0.')
        # ----------------------------------------------------------------------
        # Getting Randomized Dataset
        # ----------------------------------------------------------------------
        print('-' * 80)
        print(f'Processing {config_name} dataset with randomized substructures...')
        print('-' * 80)
        print(f'Number of iterations: {num_aug_samples // len(train_df)}')
        
        # Concatenate the training dataset until we reach the desired number of
        # samples
        ds_list = [
            train_ds.shuffle(seed=42)
            for _ in range(num_aug_samples // len(train_df))
        ]
        ds_list.append(
            train_ds.select(range(num_aug_samples % len(train_df))).shuffle(seed=42)
        )
        randomized_ds = concatenate_datasets(ds_list)

        # Apply the randomization to the SMILES strings
        rand_smiles = lambda x: {'text': randomize_smiles(x['text']), 'labels': randomize_smiles(x['labels'])}
        randomized_ds = randomized_ds.map(rand_smiles, num_proc=num_proc)

        # Concatenate the original training dataset with the randomized dataset
        train_aug_ds = concatenate_datasets([train_ds, randomized_ds])

        push_ds_to_hub(dataset=train_aug_ds, config_name=f'{config_name}_randomized', split='train')
        push_ds_to_hub(dataset=test_ds, config_name=f'{config_name}_randomized', split='test')

        # ----------------------------------------------------------------------
        # Getting Recombined Dataset
        # ----------------------------------------------------------------------
        print('-' * 80)
        print(f'Processing {config_name} dataset with recombined substructures...')
        print('-' * 80)

        prng = np.random.RandomState(42)

        # Get unique substructures
        unique_e3s = train_test_dfs['train']['E3 Binder SMILES with direction'].unique()
        unique_linkers = train_test_dfs['train']['Linker SMILES with direction'].unique()
        unique_pois = train_test_dfs['train']['POI Ligand SMILES with direction'].unique()

        # Generate a dataset of randomly sampled indeces to the substructures
        num_samples = num_aug_samples * 10  # Generate more combinations to account for failures
        indices_df = pd.DataFrame({
            'e3_idx': prng.randint(0, len(unique_e3s), size=num_samples),
            'linker_idx': prng.randint(0, len(unique_linkers), size=num_samples),
            'poi_idx': prng.randint(0, len(unique_pois), size=num_samples),
        }).drop_duplicates().iloc[:num_aug_samples * 2, :]

        # Create a dataset from the DataFrame
        combinations_ds = Dataset.from_pandas(indices_df)

        def generate_recombined_protac(example, unique_e3s, unique_linkers, unique_pois, test_smiles_set, test_fps, similarity_threshold, prng):
            e3_smiles = unique_e3s[example['e3_idx']]
            linker_smiles = unique_linkers[example['linker_idx']]
            poi_smiles = unique_pois[example['poi_idx']]

            bond_types = ['single', 'double', 'triple']
            bonds_comb = list(product(bond_types, bond_types))
            prng.shuffle(bonds_comb)

            new_protac = None
            for (e3_bond_type, poi_bond_type) in bonds_comb:
                try:
                    new_protac, _ = reassemble_protac(
                        poi_smiles=poi_smiles,
                        linker_smiles=linker_smiles,
                        e3_smiles=e3_smiles,
                        e3_bond_type=e3_bond_type,
                        poi_bond_type=poi_bond_type,
                    )
                except Exception:
                    continue
                if new_protac:
                    break
            
            if new_protac is None:
                return {'text': None, 'labels': None}
            
            # Check if new_protac is in test set
            if new_protac in test_smiles_set:
                return {'text': None, 'labels': None}

            # Compute fingerprint
            fp = get_fingerprint(new_protac)
            similarities = DataStructs.BulkTanimotoSimilarity(fp, test_fps)
            avg_similarity = np.mean(similarities)

            if avg_similarity > similarity_threshold:
                return {'text': None, 'labels': None}

            labels = join_substructures(new_protac, e3_smiles, linker_smiles, poi_smiles)
            return {'text': new_protac, 'labels': labels}

        # Precompute fingerprints for the test PROTACs
        test_smiles = test_df['text'].tolist()
        test_smiles_set = set(test_smiles)
        test_fps = [get_fingerprint(s) for s in test_smiles]
        similarity_threshold = 0.4

        # For each combination of substructures, generate a recombined PROTAC
        recombined_ds = combinations_ds.map(
            generate_recombined_protac,
            fn_kwargs={
                'unique_e3s': unique_e3s,
                'unique_linkers': unique_linkers,
                'unique_pois': unique_pois,
                'test_smiles_set': test_smiles_set,
                'test_fps': test_fps,
                'similarity_threshold': similarity_threshold,
                'prng': prng,
            },
            num_proc=num_proc,
            remove_columns=combinations_ds.column_names,
        )
        # Filter out the failed reassemblies
        recombined_ds = recombined_ds.filter(lambda x: x['text'] is not None).select(range(num_aug_samples))

        # Concatenate the original training dataset with the recombined dataset
        train_aug_ds = concatenate_datasets([train_ds, recombined_ds])

        push_ds_to_hub(dataset=train_aug_ds, config_name=f'{config_name}_recombined', split='train')
        push_ds_to_hub(dataset=test_ds, config_name=f'{config_name}_recombined', split='test')

        # ----------------------------------------------------------------------
        # Getting Randomized + Recombined Dataset
        # ----------------------------------------------------------------------
        print('-' * 80)
        print(f'Processing {config_name} dataset with recombined and randomized substructures...')
        print('-' * 80)

        # Divide the dataset in two halves
        train_even_ds = recombined_ds.filter(lambda example, idx: idx % 2 == 0, with_indices=True)
        train_odd_ds = recombined_ds.filter(lambda example, idx: idx % 2 == 1, with_indices=True)

        # Randomize the SMILES strings in the even half
        train_even_ds = train_even_ds.map(lambda example: {'text': randomize_smiles(example['text']), 'labels': randomize_smiles(example['labels'])}, num_proc=num_proc)

        # Concatenate the two halves back together
        train_aug_ds = concatenate_datasets([train_ds, train_even_ds, train_odd_ds])

        push_ds_to_hub(dataset=train_aug_ds, config_name=f'{config_name}_randomized_recombined', split='train')
        push_ds_to_hub(dataset=test_ds, config_name=f'{config_name}_randomized_recombined', split='test')

    print('All done!')


if __name__ == '__main__':
    # Setup main argument parser
    parser = argparse.ArgumentParser(description='Generate augmented datasets for PROTACs-Splitter.')
    parser.add_argument('--num_proc', type=int, default=16, help='The number of processes to use for parallel processing.')
    parser.add_argument('--max_train_samples', type=int, default=10000, help='The maximum number of training samples to generate.')
    parser.add_argument('--disable_progress_bars', action='store_true', help='Disable progress bars.')
    args = parser.parse_args()

    main(num_proc=args.num_proc, max_train_samples=args.max_train_samples, disable_progress_bars=args.disable_progress_bars)