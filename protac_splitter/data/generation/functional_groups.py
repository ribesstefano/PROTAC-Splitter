from typing import Dict, Optional, Union
from collections import defaultdict, Counter
import json

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

from protac_splitter.chemoinformatics import (
    get_atom_idx_at_attachment,
    canonize_smarts,
)
from protac_splitter.display_utils import (
    safe_display,
    display_mol,
)


def get_functional_group_at_attachment(
        protac: Chem.Mol,
        substruct: Chem.Mol,
        linker: Chem.Mol,
        n_hops: int = 1,
        timeout: Optional[Union[int, float]] = None,
        return_dict: bool = False,
        verbose: int = 0,
) -> Union[str, Dict[str, str]]:
    """ Get the functional group at the attachment point of a substructure in the PROTAC molecule.
    
    Args:
        protac: The PROTAC molecule.
        substruct: The substructure of the PROTAC that contains the attachment point, e.g., the POI or E3 ligase.
        linker: The linker molecule.
        n_hops: The number of hops to consider for the neighborhood.
        timeout: The timeout for the substructure search.
        return_dict: Whether to return the functional groups as a dictionary.
        verbose: Verbosity level.
    
    Returns:
        str | Dict[str, str]: The SMARTS of the functional group at the attachment point. If return_dict is True, a dictionary with the SMARTS of the functional groups at the attachment point and at the "two sides" of the attachment point (keys: 'attachment', 'substruct', 'linker').
    """
    protac = Chem.AddHs(protac)
    substruct = Chem.AddHs(substruct)
    linker = Chem.AddHs(linker)

    attachment_idxs = get_atom_idx_at_attachment(
            protac=protac,
            substruct=substruct,
            linker=linker,
            timeout=timeout,
            return_dict=True,
            verbose=0,
    )
    # Get all neighboring atoms that are n_hops away from the attachment point
    if attachment_idxs is None:
        return None
    if len(attachment_idxs) != 2:
        return None
    if verbose:
        print(f'Attachment points: {attachment_idxs}')
        img = Draw.MolToImage(protac, highlightAtoms=attachment_idxs.values(), size=(800, 500))
        safe_display(img)
        print('Neighbors:')

    # Recursively find neighbors at n_hops distance
    neighborhood = set([protac.GetAtomWithIdx(idx) for idx in attachment_idxs.values()])
    def find_neighbors(atom, hops, excluded_atom_idx=None):
        if hops <= 0:
            return
        for neighbor in atom.GetNeighbors():
            if excluded_atom_idx is not None and neighbor.GetIdx() == excluded_atom_idx:
                neighborhood.add(neighbor)
                continue
            neighborhood.add(neighbor)
            find_neighbors(neighbor, hops - 1)
    
    for idx in attachment_idxs.values():
        find_neighbors(protac.GetAtomWithIdx(idx), n_hops)

    # Display the neighborhood
    if verbose:
        print(f'Neighbors at {n_hops} hops:')
        # Get options to display all hydrogen atoms
        options = Draw.DrawingOptions()
        # Add a legend to the image
        options.legend = 'Neighbors at attachment points'
        img = Draw.MolToImage(protac, highlightAtoms=[a.GetIdx() for a in neighborhood], size=(800, 500), options=options)
        safe_display(img)
    
    # # NOTE: The following is an overkill, there is an RDKit function to extract a substructure
    # neighborhood_mol = extract_atoms_as_molecule(protac, [a.GetIdx() for a in neighborhood])
    # neighborhood_smarts = canonize_smarts(Chem.MolToSmarts(neighborhood_mol))

    # Extract the SMARTS given the atom indices of the neighborhood
    neighborhood_idxs = [a.GetIdx() for a in neighborhood]
    neighborhood_smarts = Chem.MolFragmentToSmarts(protac, neighborhood_idxs)
    neighborhood_smarts = canonize_smarts(neighborhood_smarts)

    if verbose:
        print(neighborhood_smarts)
        display_mol(Chem.MolFromSmarts(neighborhood_smarts), display_svg=False)

    if return_dict:
        smarts = {}
        smarts['attachment'] = neighborhood_smarts
        # Get the SMARTS at the attachment point and at its "two sides"
        for side, idx in attachment_idxs.items():
            # NOTE: We know that attachment_idxs is a dictionary with two keys,
            # 'susbtruct' and 'linker', so we can directly use the other key
            other_side = 'linker' if side == 'substruct' else 'substruct'
            excluded_atom_idx = attachment_idxs[other_side]
            neighborhood = {protac.GetAtomWithIdx(idx)}
            find_neighbors(protac.GetAtomWithIdx(idx), n_hops, excluded_atom_idx=excluded_atom_idx)

            # Get the atom indices of the neighborhood
            neighborhood_idxs = [a.GetIdx() for a in neighborhood]
            
            # Copy the PROTAC molecule and set the excluded_atom_idx to a dummy
            p = Chem.Mol(protac)
            p.GetAtomWithIdx(excluded_atom_idx).SetAtomicNum(0)

            # Extract the SMARTS from the copied PROTAC given the indeces
            s = Chem.MolFragmentToSmarts(p, neighborhood_idxs)
            smarts[other_side] = canonize_smarts(s)

        return smarts

    return neighborhood_smarts


def get_functional_group_at_attachment_side(
    substruct: Chem.Mol,
    attachment_id: Optional[int] = None,
    n_hops: int = 2,
    add_Hs: bool = True,
) -> Optional[str]:
    """ Get the functional group at the attachment point of a substructure in the PROTAC molecule.
    
    Args:
        substruct: The substructure of the PROTAC that contains the attachment point, e.g., the POI or E3 ligase.
        attachment_id: The attachment point ID in the substructure. E.g., 1 for the POI, as in "[*:1]".
        n_hops: The number of hops to consider for the neighborhood. Default is 2.
        add_Hs: Whether to add hydrogens to the substructure.

    Returns:
        str: The SMARTS of the functional group at the attachment point. None if failed.
    """
    if add_Hs:
        substruct = Chem.AddHs(substruct)

    # Get the atom index of the attachment point, i.e., a dummy atom
    attachment_idx2map = {}
    for atom in substruct.GetAtoms():
        if atom.GetAtomicNum() == 0:
            # Get the mapped atom index
            attachment_idx2map[atom.GetIdx()] = atom.GetAtomMapNum()
    
    if not attachment_idx2map:
        return None

    # If we are dealing with a linker, get the specific attachment point
    if attachment_id is not None:
        attachment_idx = [k for k, v in attachment_idx2map.items() if v == attachment_id]
        if not attachment_idx:
            return None
        attachment_idx = attachment_idx[0]
    else:
        attachment_idx = list(attachment_idx2map.keys())[0]

    neighborhood = {substruct.GetAtomWithIdx(attachment_idx)}
    def find_neighbors(atom, hops):
        if hops <= 0:
            return
        for neighbor in atom.GetNeighbors():
            neighborhood.add(neighbor)
            find_neighbors(neighbor, hops - 1)
    
    find_neighbors(substruct.GetAtomWithIdx(attachment_idx), n_hops)
    neighborhood_idxs = [a.GetIdx() for a in neighborhood]

    neighborhood_smarts = Chem.MolFragmentToSmarts(substruct, neighborhood_idxs)
    if neighborhood_smarts:
        return canonize_smarts(neighborhood_smarts)
    
    return None


def get_functional_groups_distributions(
        df: pd.DataFrame,
        get_side_chain_info: bool = False,
        timeout: Optional[Union[int, float]] = None,
        filename_distributions: Optional[str] = None,
        filename_mappings: Optional[str] = None,
        filename_df_with_functional_groups: Optional[str] = None,
        load_from_file: bool = True,
        verbose: int = 0,
) -> Dict[str, Dict[str, set]]:
    """ Get the distributions of functional groups at attachment points in a dataframe of PROTACs.
    
    The input dataframe should contain the following columns:
        - 'PROTAC SMILES': The SMILES of the PROTAC.
        - 'POI Ligand SMILES with direction': The SMILES of the POI ligand.
        - 'Linker SMILES with direction': The SMILES of the linker.
        - 'E3 Binder SMILES with direction': The SMILES of the E3 binder.

    Args:
        df: The DataFrame containing the PROTACs.
        get_side_chain_info: Whether to get the side chain information along with the functional groups at the attachment points.
        timeout: The timeout for the substructure search. Default is None.
        verbose: Verbosity level.

    Returns:
        Dict[str, Dict[str, set]]: The distributions of functional groups at attachment points in PROTACs.
    """
    smarts_counter = Counter()
    e3_smarts_counter = Counter()
    poi_smarts_counter = Counter()
    substr_smarts_counter = {
        'poi2linker': defaultdict(Counter),
        'linker2poi': defaultdict(Counter),
        'e32linker': defaultdict(Counter),
        'linker2e3': defaultdict(Counter),
    }
    # Assign to each functional group the list of substructures that appear in the df
    poi_substr2fg = defaultdict(set)
    e3_substr2fg = defaultdict(set)
    # Assign to each substructure the list of functional groups that appear in the df
    poi_fg_2_substr = defaultdict(set)
    e3_fg_2_substr = defaultdict(set)
    substr_fg_2_linker = defaultdict(set)

    linker2fg = defaultdict(dict)

    if load_from_file:
        if filename_distributions is not None and filename_mappings is not None:
            with open(filename_distributions, 'r') as f:
                fg_distr = json.load(f)
            with open(filename_mappings, 'r') as f:
                fg_mappings = json.load(f)
            ret = {}
            ret.update(fg_distr)
            ret.update(fg_mappings)
            return ret
        else:
            print(f'WARNING: No filename provided to load the mappings from. The functional groups will be recomputed.')
    
    df_with_functional_groups = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        protac_smiles = row['PROTAC SMILES']
        poi_smiles = row['POI Ligand SMILES with direction']
        linker_smiles = row['Linker SMILES with direction']
        e3_smiles = row['E3 Binder SMILES with direction']
            
        protac = Chem.MolFromSmiles(protac_smiles)
        poi = Chem.MolFromSmiles(poi_smiles)
        e3 = Chem.MolFromSmiles(e3_smiles)
        linker = Chem.MolFromSmiles(linker_smiles)

        if None in [protac, poi, e3, linker]:
            print(f'WARNING: Could not parse the following SMILES:')
            print(f'PROTAC: {protac_smiles}')
            print(f'POI: {poi_smiles}')
            print(f'Linker: {linker_smiles}')
            print(f'E3: {e3_smiles}')
            print('-' * 80)

        # We have a bit of care with the linker, as it can be empty
        try:
            _ = Chem.molzip(Chem.MolFromSmiles('.'.join([poi_smiles, linker_smiles, e3_smiles])))
        except:
            print(f'WARNING: The linker might be empty: {linker_smiles}')
            linker = None
        
        if linker is not None:
            fg_poi = get_functional_group_at_attachment(protac, poi, linker, timeout=timeout, return_dict=get_side_chain_info)
            fg_e3 = get_functional_group_at_attachment(protac, e3, linker, timeout=timeout, return_dict=get_side_chain_info)
        else:
            # If the linker is empty, then we use the other side as the linker
            fg_poi = get_functional_group_at_attachment(protac, poi, e3, return_dict=get_side_chain_info)
            fg_e3 = get_functional_group_at_attachment(protac, e3, poi, return_dict=get_side_chain_info)

        if get_side_chain_info:
            if fg_poi is not None:
                smarts_counter.update([fg_poi['attachment']])
                poi_smarts_counter.update([fg_poi['substruct']])
                substr_smarts_counter['poi2linker'][fg_poi['substruct']].update([fg_poi['linker']])
                substr_smarts_counter['linker2poi'][fg_poi['linker']].update([fg_poi['substruct']])
                linker2fg[linker_smiles]['poi'] = fg_poi['attachment']

                poi_substr2fg[poi_smiles].append(fg_poi['attachment'])
                poi_fg_2_substr[fg_poi['attachment']].update([poi_smiles])
            
            if fg_e3 is not None:
                smarts_counter.update([fg_e3['attachment']])
                e3_smarts_counter.update([fg_e3['substruct']])
                substr_smarts_counter['e32linker'][fg_e3['substruct']].update([fg_e3['linker']])
                substr_smarts_counter['linker2e3'][fg_e3['linker']].update([fg_e3['substruct']])
                linker2fg[linker_smiles]['e3'] = fg_e3['attachment']

                e3_substr2fg[e3_smiles].update(fg_e3['attachment'])
                e3_fg_2_substr[fg_e3['attachment']].update([e3_smiles])
        else:
            if fg_poi is not None:
                smarts_counter.update([fg_poi])
                poi_smarts_counter.update([fg_poi])
                poi_substr2fg[poi_smiles].update([fg_poi])
                poi_fg_2_substr[fg_poi].update([poi_smiles])
                substr_fg_2_linker[fg_poi].update([linker_smiles])
            if fg_e3 is not None:
                smarts_counter.update([fg_e3])
                e3_smarts_counter.update([fg_e3])
                e3_substr2fg[e3_smiles].update([fg_e3])
                e3_fg_2_substr[fg_e3].update([e3_smiles])
                substr_fg_2_linker[fg_e3].update([linker_smiles])
        
        # Update the DataFrame with the functional groups
        if fg_poi is not None:
            row['POI Ligand Functional Group'] = fg_poi
        if fg_e3 is not None:
            row['E3 Binder Functional Group'] = fg_e3
        df_with_functional_groups.append(row)

    # Normalize all the counts to probability distributions
    fg_distr = {k: v / smarts_counter.total() for k, v in smarts_counter.items()}
    e3_fg_distr = {k: v / e3_smarts_counter.total() for k, v in e3_smarts_counter.items()}
    poi_fg_distr = {k: v / poi_smarts_counter.total() for k, v in poi_smarts_counter.items()}

    # Sort the probability distributions
    fg_distr = dict(sorted(fg_distr.items(), key=lambda x: x[1], reverse=True))
    e3_fg_distr = dict(sorted(e3_fg_distr.items(), key=lambda x: x[1], reverse=True))
    poi_fg_distr = dict(sorted(poi_fg_distr.items(), key=lambda x: x[1], reverse=True))

    if not get_side_chain_info:
        ret = {
            'fg_distr': fg_distr,
            'e3_fg_distr': e3_fg_distr,
            'poi_fg_distr': poi_fg_distr,
            'poi_fg_2_substr': poi_fg_2_substr,
            'e3_fg_2_substr': e3_fg_2_substr,
            'substr_fg_2_linker': substr_fg_2_linker,
        }

    # Normalize the linker-to-substructure to probability distributions
    if get_side_chain_info:
        side_fg_distr = defaultdict(dict)
        for direction, smarts2counter in substr_smarts_counter.items():
            for smarts, counter in smarts2counter.items():
                side_fg_distr[direction][smarts] = {k: v / counter.total() for k, v in counter.items()}
                side_fg_distr[direction][smarts] = dict(sorted(side_fg_distr[direction][smarts].items(), key=lambda x: x[1], reverse=True))

            if verbose:
                # Display the top 5 functional groups
                print('-' * 80)
                print(f'{"-".join(direction.upper().split("2"))}:')
                print('-' * len(direction) + '-' * 2)
                for i, (smarts, probs) in enumerate(side_fg_distr[direction].items()):
                    if i >= 5:
                        break
                    print(f'{smarts}:')
                    for j, (sma, prob) in enumerate(probs.items()):
                        if j >= 5:
                            break
                        print(f'\t{prob:.2%} -> {sma}')
        ret = {
            'fg_distr': fg_distr,
            'e3_fg_distr': e3_fg_distr,
            'poi_fg_distr': poi_fg_distr,
            'poi_fg_2_substr': poi_fg_2_substr,
            'e3_fg_2_substr': e3_fg_2_substr,
            'substr_fg_2_linker': substr_fg_2_linker,
            'side_fg_distr': side_fg_distr,
        }

    if filename_distributions is not None:
        # Save to JSON file
        distributions = {k: v for k, v in ret.items() if 'distr' in k}
        with open(filename_distributions, 'w') as f:
            json.dump(distributions, f, indent=4)
        print(f'Functional group distributions saved to: {filename_distributions}')

    if filename_mappings is not None:
        # Convert sets to lists to make the data serializable
        fg_mappings = {k: {sk: list(s) for sk, s in v.items()} for k, v in ret.items() if 'distr' not in k}

        with open(filename_mappings, 'w') as f:
            json.dump(fg_mappings, f, indent=4)
        print(f'Functional group mappings saved to: {filename_mappings}')

    df_with_functional_groups = pd.DataFrame(df_with_functional_groups)
    ret['dataframe'] = df_with_functional_groups

    if filename_df_with_functional_groups is not None:
        df_with_functional_groups.to_csv(filename_df_with_functional_groups, index=False)
        print(f'DataFrame with functional groups saved to: {filename_df_with_functional_groups}')

    return ret