import os
import sys
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


def get_randomized_ds_generator(
        df: pd.DataFrame,
        num_samples: int,
        original_train_first: bool = False,
) -> Generator[Dict[str, str], None, None]:
    """Generate a dataset with randomized substructures.

    Args:
        df: The dataset containing the substructures.
        num_samples: The maximum number of combinations to generate.
        original_train_first: Whether to include the original training data first.

    Returns:
        A generator that yields the randomized substructures as 'text' and 'labels'.
    """
    total_samples = num_samples

    # Add the original training data first
    if original_train_first:
        total_samples -= len(df)
        for _, row in df.iterrows():
            yield {'text': row['text'], 'labels': row['labels']}

    # Generate the randomized SMILES N times to get the desired number of samples
    for i in range(max(num_samples // len(df), 1)):
        for j, row in df.iterrows():
            if i * len(df) + j >= total_samples:
                break
            yield {'text': randomize_smiles(row['text']), 'labels': randomize_smiles(row['labels'])}


def rand_combinations_generator(
        unique_substructs: Dict[str, np.ndarray],
        probs: Dict[str, Optional[np.ndarray]],
        num_samples: int,
) -> Generator[Tuple[str, str, str], None, None]:
    """Generate random combinations of substructures. Each sample is a dictionary of 'e3', 'linker', 'poi' substructures, each sampled from the supplied unique substructures.

    Args:
        unique_substructs: The unique substructures: a dictionary of lists of unique substructures for 'e3', 'linker', and 'poi'.
        probs: The probabilities of each substructure:  a dictionary of list of probabilities for each 'e3', 'linker', and 'poi'.
        num_samples: The number of combinations to generate.

    Returns:
        A generator that yields the random combinations of substructures.
    """
    
    # Setup random number generator
    prng = np.random.RandomState(42)

    # Create a mapping from a substructure to an integer ID (index)
    keys = ['e3', 'linker', 'poi']
    substruct2idx = {}
    for key in keys:
        substruct2idx[key] = {idx: item for idx, item in enumerate(unique_substructs[key])}
    
    # To avoid repetitions, create a set to store yielded combinations
    yielded_combs = set()

    # Generate combinations
    while len(yielded_combs) < num_samples:
        # Generate a combination, i.e., a tuple of randomly sampled indices
        comb_indices = tuple(prng.choice(len(unique_substructs[k]), p=probs[k]) for k in keys)

        # Encode the combination into single integer before checking set
        # membership. We assume that each unique substructure has less
        # than 2^32 items, i.e., ~4B items.
        encoded = (comb_indices[0] << 64) | (comb_indices[1] << 32) | comb_indices[2]

        # Check if the encoded combination is already in the set. If not,
        # add it to the set and yield the combination.
        if encoded not in yielded_combs:
            yielded_combs.add(encoded)
            comb = {k: substruct2idx[k][idx] for k, idx in zip(keys, comb_indices)}
            yield comb


def bond_combinations_generator(num_iter: int):
    """Generate random combinations of bond types for the E3 and POI substructures to reassemble PROTACs.

    Args:
        num_iter: The number of combinations to generate.

    Returns:
        A generator that yields the random combinations of bond types.
    """
    prng = np.random.RandomState(42)
    bond_types = ['single', 'double', 'triple']
    bonds_comb = list(product(bond_types, bond_types))
    for _ in range(num_iter):
        # Shuffle the bond types now for reproducibility
        prng.shuffle(bonds_comb)
        yield bonds_comb


def recombined_protac_generator(
        train_df: pd.DataFrame,
        num_samples: int,
        uniform_sampling: bool = True,
) -> Generator[Dict[str, str], None, None]:
    """Generate recombined PROTACs.

    Args:
        train_df: The training dataset.
        num_samples: The number of combinations to generate.
        uniform_sampling: Whether to sample the substructures uniformly.

    Returns:
        A generator that yields the recombined PROTACs as 'text' and 'labels'.
    """
    # Get unique substructures, then convert them to NumBa-friendly dict format
    unique_substructs = get_unique_substructs(train_df)

    # Get substructure probabilities, then convert them to NumBa-friendly dict format
    if uniform_sampling:
        probs = {k: None for k in ['e3', 'linker', 'poi']}
    else:
        probs = get_substruct_prob(train_df)

    # Get the generators for the substructures to combine and random bond types
    substruct_combs = rand_combinations_generator(unique_substructs, probs, num_samples * 2)
    bonds_combs = bond_combinations_generator(num_samples * 2)

    # Iterate over the combinations and yield the recombined PROTACs
    yielded_samples = 0
    for substruct_comb, bonds_comb in zip(substruct_combs, bonds_combs):
        # Get the first PROTAC that can be reassembled given the random bonds  
        new_protac = None
        for (e3_bond_type, poi_bond_type) in bonds_comb:
            try:
                new_protac, _ = reassemble_protac(
                    poi_smiles=substruct_comb['poi'],
                    linker_smiles=substruct_comb['linker'],
                    e3_smiles=substruct_comb['e3'],
                    e3_bond_type=e3_bond_type,
                    poi_bond_type=poi_bond_type,
                )
            except Exception as e:
                pass
            if new_protac:
                break
        if new_protac is None:
            continue

        yield {
            'text': new_protac,
            'labels': join_substructures(new_protac, substruct_comb['e3'], substruct_comb['linker'], substruct_comb['poi']),
        }

        yielded_samples += 1
        if yielded_samples >= num_samples:
            break


def get_recombined_text_generator(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        num_samples: int = 5000,
        uniform_sampling: bool = True,
        similarity_threshold: float = 0.4,
        original_train_first: bool = False,
) -> Generator[Dict[str, str], None, None]:
    """Generate recombined PROTACs making sure that they are not in the test set and have low similarity.

    Args:
        train_df: The training dataset.
        test_df: The test dataset.
        num_samples: The number of combinations to generate.
        uniform_sampling: Whether to sample the substructures uniformly.
        similarity_threshold: The Tanimoto similarity threshold.
        original_train_first: Whether to include the original training data first.

    Returns:
        A generator that yields the recombined PROTACs as 'text' and 'labels'.
    """
    total_samples = num_samples

    if original_train_first:
        total_samples -= len(train_df)
        for _, row in train_df.iterrows():
            yield {'text': row['PROTAC SMILES'], 'labels': join_substructures(row['PROTAC SMILES'], row['E3 Binder SMILES with direction'], row['Linker SMILES with direction'], row['POI Ligand SMILES with direction'])}

    # Precompute fingerprints for the test PROTACs
    test_fps = test_df['PROTAC SMILES'].apply(get_fingerprint).to_list()

    # Generate recombined PROTACs
    yielded_examples = 0
    for recombined in recombined_protac_generator(train_df, num_samples * 10, uniform_sampling):
        # Calculate bulk Tanimoto similarity and average similarity
        fp = get_fingerprint(recombined['text'])
        similarities = DataStructs.BulkTanimotoSimilarity(fp, test_fps)
        avg_similarity = np.mean(similarities)

        # Check that the new PROTAC is not in the test set and has low similarity
        if recombined['text'] in test_df['PROTAC SMILES'].values or avg_similarity > similarity_threshold:
            continue

        yield recombined

        yielded_examples += 1
        if yielded_examples >= total_samples:
            break


def push_ds_to_hub(
        config_name: str,
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


def check_datasets(d):
    tqdm.pandas(desc='Checking train dataset')
    train_check = all(d['train'].progress_apply(lambda x: check_substructs(
            x['PROTAC SMILES'],
            x['POI Ligand SMILES with direction'],
            x['Linker SMILES with direction'],
            x['E3 Binder SMILES with direction'],
        ), axis=1)
    )
    tqdm.pandas(desc='Checking test dataset')
    test_check = all(d['test'].progress_apply(lambda x: check_substructs(
            x['PROTAC SMILES'],
            x['POI Ligand SMILES with direction'],
            x['Linker SMILES with direction'],
            x['E3 Binder SMILES with direction'],
        ), axis=1)
    )
    return train_check and test_check


def main():
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

    # print('Checking datasets...')
    # for config_name, datasets in ds.items():
    #     print(f'Checking {config_name} dataset...')
    #     if not check_datasets(datasets):
    #         print(f'Error in {config_name} dataset: reassembly failed.')
    #         sys.exit(1)
    # print('All datasets are correct (i.e., reassemblying substructures works).')

    shuffle_prob = 0.3 # Augmentation not used, for now...
    max_train_samples = 1_000_000 # 1_000_000

    for config_name, train_test_dfs in ds.items():
        # TODO: The hardest split is not done yet.
        if config_name != 'standard':
            continue

        print('-' * 80)
        print(f'Processing {config_name} dataset...')
        print('-' * 80)

        def convert_df_to_text(df: pd.DataFrame) -> pd.DataFrame:
            protac_col = 'PROTAC SMILES'
            e3_col = 'E3 Binder SMILES with direction'
            linker_col = 'Linker SMILES with direction'
            poi_col = 'POI Ligand SMILES with direction'
            df['labels'] = df.apply(lambda x: join_substructures(x[protac_col], x[e3_col], x[linker_col], x[poi_col]), axis=1)
            df = df.rename(columns={'PROTAC SMILES': 'text'})
            return df[['text', 'labels']]
        
        train_df = convert_df_to_text(train_test_dfs['train'])
        test_df = convert_df_to_text(train_test_dfs['test'])

        # Get the test dataset to use for all the augmented datasets
        train_ds = Dataset.from_pandas(train_df, preserve_index=False, split='train')
        test_ds = Dataset.from_pandas(test_df, preserve_index=False, split='test')
        train_ds = train_ds.shuffle(seed=42)
        test_ds = test_ds.shuffle(seed=42)

        print(train_ds)
        print(test_ds)

        push_ds_to_hub(dataset=train_ds, config_name=config_name)
        push_ds_to_hub(dataset=test_ds, config_name=config_name)

        num_aug_samples = max_train_samples - len(train_df)
        if num_aug_samples <= 0:
            raise ValueError('The number of augmented samples must be greater than 0.')
        # ----------------------------------------------------------------------
        # Getting Randomized Dataset
        # ----------------------------------------------------------------------
        print('-' * 80)
        print(f'Processing {config_name} dataset with randomized substructures...')
        print('-' * 80)

        train_aug_ds = Dataset.from_generator(
            get_randomized_ds_generator,
            gen_kwargs={'df': train_df, 'num_samples': num_aug_samples},
            split='train',
        )
        train_aug_ds = concatenate_datasets([train_ds, train_aug_ds]).shuffle(seed=42)

        push_ds_to_hub(dataset=train_aug_ds, config_name=f'{config_name}_randomized')
        push_ds_to_hub(dataset=test_ds, config_name=f'{config_name}_randomized')

        # ----------------------------------------------------------------------
        # Getting Recombined Dataset
        # ----------------------------------------------------------------------
        print('-' * 80)
        print(f'Processing {config_name} dataset with recombined substructures...')
        print('-' * 80)

        train_aug_ds = Dataset.from_generator(
            get_recombined_text_generator,
            gen_kwargs={'train_df': train_test_dfs['train'], 'test_df': train_test_dfs['test'], 'num_samples': num_aug_samples},
            split='train',
        )
        print(train_aug_ds)

        train_aug_ds = concatenate_datasets([train_ds, train_aug_ds]).shuffle(seed=42)

        print(train_aug_ds)

        push_ds_to_hub(dataset=train_aug_ds, config_name=f'{config_name}_recombined')
        push_ds_to_hub(dataset=test_ds, config_name=f'{config_name}_recombined')

        # ----------------------------------------------------------------------
        # Getting Randomized + Recombined Dataset
        # ----------------------------------------------------------------------
        print('-' * 80)
        print(f'Processing {config_name} dataset with recombined and randomized substructures...')
        print('-' * 80)

        # Divide the dataset in two halves
        train_even_ds = train_aug_ds.filter(lambda example, idx: idx % 2 == 0, with_indices=True)
        train_odd_ds = train_aug_ds.filter(lambda example, idx: idx % 2 == 1, with_indices=True)

        # Randomize the SMILES strings in the even half
        train_even_ds = train_even_ds.map(lambda example: {'text': randomize_smiles(example['text']), 'labels': randomize_smiles(example['labels'])})

        # Concatenate the two halves back together
        train_aug_ds = concatenate_datasets([train_ds, train_even_ds, train_odd_ds]).shuffle(seed=42)

        push_ds_to_hub(dataset=train_aug_ds, config_name=f'{config_name}_randomized_recombined')
        push_ds_to_hub(dataset=test_ds, config_name=f'{config_name}_randomized_recombined')


if __name__ == '__main__':
    main()