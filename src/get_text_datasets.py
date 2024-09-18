import os
import sys
from typing import Literal, Tuple, Dict, Optional, Tuple, Generator
from collections import Counter, defaultdict
from itertools import product

import numpy as np
import numba as nb
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
from rdkit import rdBase
from datasets import Dataset, DatasetDict

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


# @nb.jit()
def rand_combinations(
        unique_substructs: Dict[str, np.ndarray],
        probs: Dict[str, Optional[np.ndarray]],
        # unique_e3s: np.ndarray,
        # unique_linkers: np.ndarray,
        # unique_pois: np.ndarray,
        # prob_e3s: Optional[np.ndarray],
        # prob_linkers: Optional[np.ndarray],
        # prob_pois: Optional[np.ndarray],
) -> Generator[Tuple[str, str, str], None, None]:

    keys = ['e3', 'linker', 'poi']
    substruct2idx = {}
    for key in keys:
        substruct2idx[key] = {idx: item for idx, item in enumerate(unique_substructs[key])}
    
    # Step 3: Initialize
    yielded_combs = set()
    max_combinations = np.prod([len(unique_substructs[k]) for k in keys])

    # Step 4: Generate combinations
    for _ in range(max_combinations):
        # Generate a combination tuple of indices
        comb_indices = tuple(np.random.choice(len(unique_substructs[s]), p=probs[s]) for s in keys)

        # Encode into single integer before checking set membership
        # Assuming each substructure has less than 2^32 items, i.e., ~4B items
        # NOTE: This option is more memory-efficient but requires encoding and
        # decoding.
        encoded = (comb_indices[0] << 64) | (comb_indices[1] << 32) | comb_indices[2]
        if encoded not in yielded_combs:
            yielded_combs.add(encoded)
            comb = {s: substruct2idx[s][idx] for s, idx in zip(keys, comb_indices)}
            yield comb


def get_recombined_df(
        df: Dict[str, pd.DataFrame],
        max_combinations: int = 5000,
        uniform_sampling: bool = True,
        similarity_threshold: float = 0.4,
        max_tries_per_combination: int = 15,
        verbose: int = 0,
) -> pd.DataFrame:
    """Recombine the substructures of PROTACs to create new PROTACs.

    Args:
        df: The dataset containing the substructures.
        max_combinations: The maximum number of combinations to generate.
        uniform_sampling: Whether to sample uniformly from the substructures.
        similarity_threshold: The Tanimoto similarity threshold.
        verbose: The verbosity level.

    Returns:
        pd.DataFrame: The recombined PROTACs.
    """
    # Get unique substructures, then convert them to NumBa-friendly dict format
    unique_substructs = get_unique_substructs(df['train'])

    # Get substructure probabilities, then convert them to NumBa-friendly dict format
    if uniform_sampling:
        probs = {k: None for k in ['e3', 'linker', 'poi']}
    else:
        probs = get_substruct_prob(df['train'])

    # Precompute fingerprints of test PROTACs
    test_fps = df['test']['PROTAC SMILES'].apply(get_fingerprint).to_list()

    # Generation loop
    recombined_df = []
    recombined_len = 0
    for comb in tqdm(rand_combinations(unique_substructs, probs), total=max_combinations * 2, desc='Recombining PROTACs'):
        if recombined_len >= max_combinations:
            break
        new_protac = None
        for _ in range(max_tries_per_combination):
            try:
                new_protac, _ = reassemble_protac(
                    poi_smiles=comb['poi'],
                    linker_smiles=comb['linker'],
                    e3_smiles=comb['e3'],
                    e3_bond_type='rand_uniform',
                    poi_bond_type='rand_uniform',
                )
            except Exception as e:
                pass
            if new_protac:
                break
        if new_protac is None:
            continue

        # Calculate bulk Tanimoto similarity
        fp = get_fingerprint(new_protac)
        similarities = DataStructs.BulkTanimotoSimilarity(fp, test_fps)
        avg_similarity = np.mean(similarities)

        if new_protac in df['test']['PROTAC SMILES'].values or avg_similarity > similarity_threshold:
            continue

        recombined_df.append({
            'PROTAC SMILES': new_protac,
            'E3 Binder SMILES with direction': comb['e3'],
            'Linker SMILES with direction': comb['linker'],
            'POI Ligand SMILES with direction': comb['poi'],
        })
        recombined_len += 1

        if recombined_len < 5 and verbose:
            print(f'{comb["e3"]}.{comb["linker"]}.{comb["poi"]}')
            print(new_protac)
            print(randomize_smiles(new_protac))
            safe_display(Chem.MolFromSmiles(new_protac))
            print('-' * 80)

    return pd.DataFrame(recombined_df)

    # # combinations = product(
    # #     np.random.choice(unique_substructs['e3'], min(max_samples, unique_substructs['e3'].size), replace=False, p=probs['e3']),
    # #     np.random.choice(unique_substructs['linker'], min(max_samples, unique_substructs['e3'].size), replace=False, p=probs['linker']),
    # #     np.random.choice(unique_substructs['poi'], min(max_samples, unique_substructs['e3'].size), replace=False, p=probs['poi']),
    # # )
    # # num_combinations = min(max_samples, unique_substructs['e3'].size) * min(max_samples, unique_substructs['e3'].size) * min(max_samples, unique_substructs['e3'].size)
    # # if verbose:
    # #     print(f'Maximum number of samples: {max_samples}')
    # #     print(f'Number of actual combinations: {num_combinations:,}')

    # np.random.shuffle(unique_substructs['e3']),
    # np.random.shuffle(unique_substructs['linker']),
    # np.random.shuffle(unique_substructs['poi']),
    # combinations = product(
    #     unique_substructs['e3'],
    #     unique_substructs['linker'],
    #     unique_substructs['poi'],
    # )

    # # Precompute fingerprints of test PROTACs
    # test_fps = df['test']['PROTAC SMILES'].apply(get_fingerprint).to_list()

    # recombined_df = []
    # recombined_len = 0
    # for i, (e3, linker, poi) in tqdm(enumerate(combinations), total=max_combinations * 2, desc='Recombining PROTACs'):
    #     if recombined_len >= max_combinations:
    #         break
    #     new_protac = None
    #     while not new_protac:
    #         try:
    #             new_protac, _ = reassemble_protac(
    #                 poi,
    #                 linker,
    #                 e3,
    #                 e3_bond_type='rand_uniform',
    #                 poi_bond_type='rand_uniform',
    #             )
    #         except:
    #             pass

    #     # Calculate bulk Tanimoto similarity
    #     fp = get_fingerprint(new_protac)
    #     similarities = DataStructs.BulkTanimotoSimilarity(fp, test_fps)
    #     avg_similarity = np.mean(similarities)

    #     if new_protac in df['test']['PROTAC SMILES'].values or avg_similarity > similarity_threshold:
    #         continue

    #     recombined_df.append({
    #         'PROTAC SMILES': new_protac,
    #         'E3 Binder SMILES with direction': e3,
    #         'Linker SMILES with direction': linker,
    #         'POI Ligand SMILES with direction': poi,
    #     })
    #     recombined_len += 1
    #     if i < 5 and verbose:
    #         print(f'{e3}.{linker}.{poi}')
    #         print(new_protac)
    #         print(randomize_smiles(new_protac))
    #         safe_display(Chem.MolFromSmiles(new_protac))
    #         print('-' * 80)

    # return pd.DataFrame(recombined_df)


def push_ds_to_hub(train_df, test_df, config_name):
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_df, preserve_index=False),
        'test': Dataset.from_pandas(test_df, preserve_index=False),
    })
    dataset_dict.push_to_hub(
        'ailab-bio/PROTAC-Splitter-Dataset',
        config_name=config_name,
        private=True,
        token=os.getenv('HF_TOKEN'),
    )


def shuffle_substructs(s: str, shuffle_prob: float = 0.0) -> str:
    if np.random.rand() < shuffle_prob:
        substructs = s.split('.')
        np.random.shuffle(substructs)
        return '.'.join(substructs)
    else:
        return s


def randomize_dataset(df):
    df['text'] = df['text'].apply(randomize_smiles)
    df['labels'] = df['labels'].apply(randomize_smiles)
    return df


def shuffle_dataset(df, shuffle_prob=0.3):
    df['labels'] = df['labels'].apply(shuffle_substructs, shuffle_prob=shuffle_prob)
    return df


def check_dataset(d):
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

print('Checking datasets...')
for config_name, datasets in ds.items():
    print(f'Checking {config_name} dataset...')
    if not check_dataset(datasets):
        print(f'Error in {config_name} dataset: reassembly failed.')
        sys.exit(1)
print('All datasets are correct (i.e., reassemblying substructures works).')

protac_col = 'PROTAC SMILES'
e3_col = 'E3 Binder SMILES with direction'
linker_col = 'Linker SMILES with direction'
poi_col = 'POI Ligand SMILES with direction'

max_rand_samples = 100_000
shuffle_prob = 0.3 # Augmentation not used, for now...

text_ds = {}
for config_name, datasets in ds.items():
    # TODO: The hardest split is not done yet.
    if config_name != 'standard':
        continue

    print('-' * 80)
    print(f'Processing {config_name} dataset...')
    print('-' * 80)

    text_ds[config_name] = {}
    for split, dataset in datasets.items():
        text_df = ds[config_name][split].copy()

        # Add 'labels' column
        tqdm.pandas(desc=f'Joining {config_name} {split} substructures')
        text_df['labels'] = text_df.progress_apply(lambda x: join_substructures(x[protac_col], x[e3_col], x[linker_col], x[poi_col]), axis=1)

        # Rename 'PROTAC SMILES' column to 'text'
        text_df = text_df.rename(columns={'PROTAC SMILES': 'text'})
        text_ds[config_name][split] = text_df[['text', 'labels']]

    train_df = text_ds[config_name]['train']
    test_df = text_ds[config_name]['test']
    push_ds_to_hub(train_df, test_df, config_name)

    print('-' * 80)
    print(f'Processing {config_name} dataset with randomized substructures...')
    print('-' * 80)
    # TODO: Add configuration with randomized data
    randomized_df = []
    num_samples = len(train_df)
    while num_samples < max_rand_samples:
        tmp = randomize_dataset(train_df.copy())
        randomized_df.append(tmp)
        num_samples += len(tmp)
    randomized_df = pd.concat(randomized_df)[['text', 'labels']].drop_duplicates()
    print(f'Number of randomized PROTACs: {len(randomized_df)}')
    push_ds_to_hub(pd.concat([train_df, randomized_df]), test_df, f'{config_name}_randomized')

    print('-' * 80)
    print(f'Processing {config_name} dataset with recombined substructures...')
    print('-' * 80)
    # TODO: Add configuration with recombined data
    recombined_df = get_recombined_df(datasets, max_combinations=len(randomized_df))
    tqdm.pandas(desc=f'Joining {config_name} recombined substructures', total=len(recombined_df))
    recombined_df['labels'] = recombined_df.progress_apply(lambda x: join_substructures(x[protac_col], x[e3_col], x[linker_col], x[poi_col]), axis=1)
    recombined_df = recombined_df.rename(columns={'PROTAC SMILES': 'text'})
    recombined_df = recombined_df[['text', 'labels']].drop_duplicates()
    print(f'Number of recombined PROTACs: {len(recombined_df)}')
    push_ds_to_hub(pd.concat([train_df, recombined_df]), test_df, f'{config_name}_recombined')

    print('-' * 80)
    print(f'Processing {config_name} dataset with recombined and randomized substructures...')
    print('-' * 80)
    # TODO: Add configuration with recombined and randomized data
    rec_rand_df = recombined_df.copy()
    # Shuffle the rows of the DataFrame
    rec_rand_df = rec_rand_df.sample(frac=1).reset_index(drop=True)
    # Get 50% of the DataFrame rows
    rand_df = rec_rand_df.iloc[:len(rec_rand_df) // 2, :].copy()
    rec_df = rec_rand_df.iloc[len(rec_rand_df) // 2:, :].copy()
    # Randomize one half of the DataFrame
    rand_df = randomize_dataset(rand_df)
    # Join them back together
    rec_rand_df = pd.concat([rec_df, rand_df]).reset_index(drop=True)
    rec_rand_df = rec_rand_df[['text', 'labels']].drop_duplicates()
    print(f'Number of recombined and randomized PROTACs: {len(rec_rand_df)}')
    push_ds_to_hub(pd.concat([train_df, rec_rand_df]), test_df, f'{config_name}_randomized_recombined')