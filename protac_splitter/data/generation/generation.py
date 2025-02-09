import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem


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
        original_df: The original DataFrame containing the PROTACs.
        filename_generated_df: The filename to save the generated PROTACs.

    Returns:
        pd.DataFrame: The DataFrame containing the generated PROTACs.
    """

    np.random.seed(random_state)
    final_df = pd.DataFrame()
    total_batches = int(np.ceil(num_samples / batch_size))

    def generate_protac_batch(batch_size: int) -> List[dict]:
        # Sample functional groups for POI and E3
        poi_fgs = np.random.choice(list(poi_fg_distr.keys()), size=batch_size, p=list(poi_fg_distr.values()))
        e3_fgs = np.random.choice(list(e3_fg_distr.keys()), size=batch_size, p=list(e3_fg_distr.values()))

        # Map functional groups to corresponding substrates
        poi_samples = [np.random.choice(poi_fg_2_substr[fg]) for fg in poi_fgs]
        e3_samples = [np.random.choice(e3_fg_2_substr[fg]) for fg in e3_fgs]

        generated_protacs = []

        for poi_smiles, poi_fg, e3_smiles, e3_fg in zip(poi_samples, poi_fgs, e3_samples, e3_fgs):
            linkers = set(substr_fg_2_linker.get(poi_fg, [])) & set(substr_fg_2_linker.get(e3_fg, []))
            if not linkers:
                continue

            linker_smiles = np.random.choice(list(linkers))

            # Get the PROTAC by combining the POI, linker, and E3
            protac_smiles = '.'.join([poi_smiles, linker_smiles, e3_smiles])
            protac = Chem.MolFromSmiles(protac_smiles)

            if protac is None:
                continue
            try:
                protac = Chem.molzip(protac)
            except:
                continue

            # Sanitize molecule
            zero_on_success = Chem.SanitizeMol(protac, catchErrors=True)
            if zero_on_success != 0:
                continue

            protac_smiles = Chem.MolToSmiles(protac, canonical=True)

            if original_df is not None and protac_smiles in original_df['PROTAC SMILES'].values:
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
        for _ in tqdm(range(total_batches), desc="Generating Batches"):
            futures.append(executor.submit(generate_protac_batch, batch_size))

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

    # Save to file if specified
    if filename_generated_df:
        final_df.to_csv(filename_generated_df, index=False)
        print(f"Generated PROTACs saved to: {filename_generated_df}")

    return final_df.head(num_samples)