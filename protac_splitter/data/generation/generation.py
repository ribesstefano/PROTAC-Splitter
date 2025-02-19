import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem

from protac_splitter.evaluation import check_reassembly


def generate_protacs(
        poi_fg_distr: Dict[str, float],
        e3_fg_distr: Dict[str, float],
        substr_fg_2_linker: Dict[str, List[str]],
        poi_fg_2_substr: Dict[str, List[str]],
        e3_fg_2_substr: Dict[str, List[str]],
        num_samples: int,
        random_state: int = 42,
        batch_size: int = 1000,
        max_workers: int = 4,
        original_df: Optional[pd.DataFrame] = None,
        filename_generated_df: Optional[str] = None,
        base_data_dir: Optional[str] = None,
        cover_all_smiles: bool = False,
) -> pd.DataFrame:
    """ Generate PROTACs given the distributions of functional groups at attachment points.
    
    Args:
        poi_fg_distr: The distribution of functional groups at the POI attachment point.
        e3_fg_distr: The distribution of functional groups at the E3 attachment point.
        substr_fg_2_linker: The mapping of functional groups to linkers.
        poi_fg_2_substr: The mapping of functional groups to POI substrates.
        e3_fg_2_substr: The mapping of functional groups to E3 substrates.
        num_samples: The number of PROTACs to generate.
        random_state: The random state for reproducibility.
        batch_size: The batch size for generating PROTACs.
        max_workers: The maximum number of workers for the ThreadPoolExecutor.
        original_df: The original DataFrame containing the PROTACs. Must have a
                     column named 'PROTAC SMILES' containing the strings to
                     avoid generating. The check is done on strings, so make
                     sure to canonize/standardize the SMILES strings.
        filename_generated_df: The filename to save the generated PROTACs.

    Returns:
        pd.DataFrame: The DataFrame containing the generated PROTACs.
    """

    np.random.seed(random_state)
    final_df = pd.DataFrame()
    total_batches = int(np.ceil(num_samples / batch_size))

    def generate_protac_batch(batch_size: int, random_state: int) -> List[dict]:
        np.random.seed(random_state)

        # Sample functional groups for POI and E3
        poi_fgs = np.random.choice(list(poi_fg_distr.keys()), size=batch_size, p=list(poi_fg_distr.values()))
        e3_fgs = np.random.choice(list(e3_fg_distr.keys()), size=batch_size, p=list(e3_fg_distr.values()))

        # Map functional groups to corresponding substrates
        # NOTE: When size argument is specified, the output is a numpy array.
        # NOTE: If the functional group is not in the dictionary, the output is an empty numpy array.
        poi_samples = [
            np.random.choice(poi_fg_2_substr.get(fg, []), size=1 if fg in poi_fg_2_substr and poi_fg_2_substr[fg] else 0)
            for fg in poi_fgs
        ]
        e3_samples = [
            np.random.choice(e3_fg_2_substr.get(fg, []), size=1 if fg in e3_fg_2_substr and e3_fg_2_substr[fg] else 0)
            for fg in e3_fgs
        ]

        generated_protacs = []

        for poi_smiles, poi_fg, e3_smiles, e3_fg in zip(poi_samples, poi_fgs, e3_samples, e3_fgs):
            # Check if poi_smiles and e3_smiles are not an empty numpy array
            if poi_smiles.size == 0 or e3_smiles.size == 0:
                continue

            # Convert the numpy arrays to strings
            poi_smiles, e3_smiles = poi_smiles[0], e3_smiles[0]
            
            linkers = set(substr_fg_2_linker.get(poi_fg, [])) & set(substr_fg_2_linker.get(e3_fg, []))
            if not linkers:
                continue

            linker_smiles = np.random.choice(list(linkers))

            # Get the PROTAC by combining the POI, linker, and E3
            ligands_smiles = '.'.join([poi_smiles, linker_smiles, e3_smiles])
            protac = Chem.MolFromSmiles(ligands_smiles)

            if protac is None:
                continue
            try:
                protac = Chem.molzip(protac)
            except:
                continue

            # Sanitize molecule
            try:
                zero_on_success = Chem.SanitizeMol(protac, catchErrors=True)
                if zero_on_success != 0:
                    continue
                protac_smiles = Chem.MolToSmiles(protac, canonical=True)
            except:
                continue

            if original_df is not None and protac_smiles in original_df['PROTAC SMILES'].values:
                continue
            
            # Check if PROTAC can be reassembled
            if not check_reassembly(protac_smiles, ligands_smiles):
                continue

            generated_protacs.append({
                'PROTAC SMILES': protac_smiles,
                'POI Ligand SMILES with direction': poi_smiles,
                'Linker SMILES with direction': linker_smiles,
                'E3 Binder SMILES with direction': e3_smiles,
                'POI Ligand Functional Group': poi_fg,
                'E3 Binder Functional Group': e3_fg,
            })

        return generated_protacs

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in tqdm(range(total_batches), desc="Generating Batches"):
            futures.append(executor.submit(generate_protac_batch, batch_size, random_state + i))

        for i, future in tqdm(enumerate(futures), desc="Processing Results", total=total_batches):
            generated_batch = future.result()
            if generated_batch:
                batch_df = pd.DataFrame(generated_batch)
                final_df = pd.concat([final_df, batch_df]).drop_duplicates()
                if i % 100 == 0:
                    if base_data_dir:
                        batch_df.to_csv(os.path.join(base_data_dir, f'generated_protacs_batch={i}.csv'), index=False)
                    else:
                        batch_df.to_csv(f'generated_protacs_batch={i}.csv', index=False)
                    if filename_generated_df:
                        final_df.to_csv(filename_generated_df, index=False)
                if len(final_df) >= num_samples:
                    break

        if not final_df.empty:
            generated_pois = set(final_df['POI Ligand SMILES with direction'].unique())
            generated_e3s = set(final_df['E3 Binder SMILES with direction'].unique())
            generated_linkers = set(final_df['Linker SMILES with direction'].unique())
        else:
            generated_pois = set()
            generated_e3s = set()
            generated_linkers = set()

        # Check how we covered the available substructures
        avail_pois = set()
        avail_e3s = set()
        avail_linkers = set()
        for fg in poi_fg_2_substr:
            avail_pois.update(set(poi_fg_2_substr[fg]))
        for fg in e3_fg_2_substr:
            avail_e3s.update(set(e3_fg_2_substr[fg]))
        for fg in substr_fg_2_linker:
            avail_linkers.update(set(substr_fg_2_linker[fg]))
            
        e3_coverage = len(generated_e3s) / len(avail_e3s)
        poi_coverage = len(generated_pois) / len(avail_pois)
        linker_coverage = len(generated_linkers) / len(avail_linkers)

        print(f"POI coverage:    {poi_coverage:.3%}")
        print(f"E3 coverage:     {e3_coverage:.3%}")
        print(f"Linker coverage: {linker_coverage:.3%}")
        
        # Get the "leftover" ligands
        leftover_pois = avail_pois - generated_pois
        leftover_e3s = avail_e3s - generated_e3s
        leftover_linkers = avail_linkers - generated_linkers
        
        covering_df = []

        with tqdm(total=len(leftover_pois) + len(leftover_e3s) + len(leftover_linkers), desc="Covering Leftover Ligands") as pbar:
            while True:
                if not cover_all_smiles:
                    break

                # Randomly select a POI, E3, and linker
                if not leftover_pois:
                    pois_to_sample = avail_pois
                else:
                    pois_to_sample = leftover_pois
                if not leftover_e3s:
                    e3s_to_sample = avail_e3s
                else:
                    e3s_to_sample = leftover_e3s
                if not leftover_linkers:
                    linkers_to_sample = avail_linkers
                else:
                    linkers_to_sample = leftover_linkers

                poi_smiles = np.random.choice(list(pois_to_sample))
                e3_smiles = np.random.choice(list(e3s_to_sample))
                linker_smiles = np.random.choice(list(linkers_to_sample))
                
                # Get the PROTAC by combining the POI, linker, and E3
                ligands_smiles = '.'.join([poi_smiles, linker_smiles, e3_smiles])
                protac = Chem.MolFromSmiles(ligands_smiles)
                if protac is None:
                    continue
                try:
                    protac = Chem.molzip(protac)
                except:
                    continue
                
                # Sanitize molecule
                try:
                    zero_on_success = Chem.SanitizeMol(protac, catchErrors=True)
                    if zero_on_success != 0:
                        continue
                    protac_smiles = Chem.MolToSmiles(protac, canonical=True)
                except:
                    continue
                
                if original_df is not None and protac_smiles in original_df['PROTAC SMILES'].values:
                    continue
                
                # Check if PROTAC can be reassembled
                if not check_reassembly(protac_smiles, ligands_smiles):
                    continue
                
                covering_df.append({
                    'PROTAC SMILES': protac_smiles,
                    'POI Ligand SMILES with direction': poi_smiles,
                    'Linker SMILES with direction': linker_smiles,
                    'E3 Binder SMILES with direction': e3_smiles,
                    'POI Ligand Functional Group': None,
                    'E3 Binder Functional Group': None,
                })
                
                generated_pois.add(poi_smiles)
                generated_e3s.add(e3_smiles)
                generated_linkers.add(linker_smiles)
                
                ligands_added = 0
                if poi_smiles in leftover_pois:
                    leftover_pois.remove(poi_smiles)
                    ligands_added += 1
                if e3_smiles in leftover_e3s:
                    leftover_e3s.remove(e3_smiles)
                    ligands_added += 1
                if linker_smiles in leftover_linkers:
                    leftover_linkers.remove(linker_smiles)
                    ligands_added += 1
                
                e3_coverage = len(generated_e3s) / len(avail_e3s)
                poi_coverage = len(generated_pois) / len(avail_pois)
                linker_coverage = len(generated_linkers) / len(avail_linkers)

                # Update the pbar and write the coverage
                pbar.update(ligands_added)
                pbar.set_postfix({
                    'POI': f"{poi_coverage:.2%}",
                    'E3': f"{e3_coverage:.2%}",
                    'Linker': f"{linker_coverage:.2%}",
                })
                
                if not leftover_pois and not leftover_e3s and not leftover_linkers:
                    break

        final_df = pd.concat([final_df, pd.DataFrame(covering_df)]).drop_duplicates()

    # Save to file if specified
    if filename_generated_df:
        final_df.to_csv(filename_generated_df, index=False)
        print(f"Generated PROTACs saved to: {filename_generated_df}")

    return final_df