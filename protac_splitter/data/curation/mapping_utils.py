from rdkit import Chem
import pandas as pd

from protac_splitter.chemoinformatics import (
    canonize_smiles,
    remove_stereo,
    get_mol_id,
)

def update_dictionary(
        dictionary: pd.DataFrame,
        substr_to_add: list,
        morgan_fp_generator = None,
        verbose: int = 0,
) -> pd.DataFrame:
    """ Updates a dictionary with a list of additional substructures.
    
    The dictionary is a dataframe with columns 'SMILES', 'Molecule', 'ID', and 'FP'.

    Args:
        dictionary: The input dictionary dataframe.
        substr_to_add: The list of additional substructures.

    Returns:
        The updated dictionary dataframe.
    """
    # Canonize the SMILES strings
    substr_to_add = [canonize_smiles(smiles) for smiles in substr_to_add if smiles is not None]
    substr_to_add = list(set(substr_to_add))

    # Remove entries already in the dictionary
    for smiles in substr_to_add:
        if not dictionary.empty and smiles in dictionary[f'SMILES'].unique().tolist():
            if verbose > 1:
                print(f'\tWARNING. SMILES already in the dictionary: {smiles}')
            # Remove it from the list
            substr_to_add.remove(smiles)

    new_entries = []
    for smiles in substr_to_add:
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as e:
            if verbose:
                print(e)
            mol = None
        # Remove entries that result in invalid molecules
        if mol is None:
            continue
        new_entries.append({
            'SMILES': smiles,
            'Molecule': mol,
            'ID': get_mol_id(smiles),
        })
        # Try adding its no-stereochemistry version as well
        smiles_nostereo = remove_stereo(smiles)
        if smiles_nostereo is not None and smiles_nostereo != smiles:
            mol_nostereo = Chem.MolFromSmiles(smiles_nostereo)
            if mol_nostereo is not None:
                new_entries.append({
                    'SMILES': canonize_smiles(smiles_nostereo),
                    'Molecule': mol_nostereo,
                    'ID': get_mol_id(smiles_nostereo),
                })
    new_entries = pd.DataFrame(new_entries).drop_duplicates()
    
    if len(new_entries) > 0:
        # Add fingerprints to the new entries
        if morgan_fp_generator is None:
            morgan_fp_generator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, useBondTypes=True, includeChirality=True)

        new_entries['FP'] = new_entries['Molecule'].apply(lambda x: morgan_fp_generator.GetFingerprint(x) if x is not None else None)
        if verbose:
            print(f'Number of substructures added to the dictionary: {len(new_entries)}')

    # Return the updated dictionary
    return pd.concat([dictionary, pd.DataFrame(new_entries)], axis=0).drop_duplicates(subset='SMILES').reset_index(drop=True)