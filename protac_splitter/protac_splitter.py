from typing import Tuple, Optional
import logging

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
)
from rdkit import Chem

# from .evaluation import (
#     is_valid_smiles,
#     has_three_substructures,
#     has_all_attachment_points,
#     check_substructs,
# )
# from .llms.evaluation import split_prediction


# def fix_prediction(
#         protac_smiles: str,
#         pred_smiles: str,
#         poi_attachment_id: int = 1,
#         e3_attachment_id: int = 2,
# ) -> Optional[str]:
    
#     substructs = split_prediction(pred_smiles)

#     if substructs is None:
#         logging.warning(f'Invalid prediction for "{pred_smiles}"')
#         return None
    
#     if check_substructs(
#         protac_smiles,
#         substructs['poi'],
#         substructs['linker'],
#         substructs['e3'],
#     ):
#         return pred_smiles
    
#     # TODO: Check if removing stereochemistry results in a valid prediction
    
#     protac_mol = Chem.MolFromSmiles(protac_smiles)
#     substructs = {k: {'smiles': v, 'mol': Chem.MolFromSmiles(v)} for k, v in substructs.items()}

#     # Check if any of the substructures is NOT a substruction of the PROTAC
#     num_matches = 0
#     for sub in ['poi', 'linker', 'e3']:
#         if protac_mol.HasSubstructMatch(substructs[sub]['mol']):
#             substructs[sub]['match'] = True
#             num_matches += 1
#         else:
#             substructs[sub]['match'] = False
    
#     if num_matches < 2:
#         logging.warning(f'Prediction "{pred_smiles}" does not contain at least two substructures of the PROTAC')
#         return None
    
#     # Get the mis-matching substructure
#     for sub in ['poi', 'linker', 'e3']:
#         if not substructs[sub]['match']:
#             non_matching_mol = substructs[sub]['mol']
#             break

#     matching_mol = []
#     for sub in ['poi', 'linker', 'e3']:
#         if substructs[sub]['match']:
#             matching_mol.append(substructs[sub]['smiles'])
#     matching_mol = Chem.MolFromSmiles('.'.join(matching_mol))

#     non_matching_mol_dir = Chem.ReplaceCore(protac_mol, matching_mol, labelByIndex=False, replaceDummies=False)

#     # Check if the non-matching substructure is now a substructure of the PROTAC
#     if protac_mol.HasSubstructMatch(non_matching_mol_dir):
#         logging.info(f'Prediction "{pred_smiles}" has been fixed')
#         return pred_smiles



def split_protac(
        protac_smiles: str,
        llm_pipeline: Optional[str] = None,
        use_llm_prediction: bool = True,
) -> Tuple[str, str, str]:
    """
    Split a PROTAC SMILES into the two ligands and the linker.

    Dummy implementation that simply splits the SMILES string by '.'.

    Args:
        protac_smiles (str): A string containing the SMILES of the PROTAC.
    
    Returns:
        Tuple[str, str, str]: A tuple containing the SMILES of the first ligand, the linker, and the second ligand.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if use_llm_prediction:
        if llm_pipeline is None:
            tokenizer = AutoTokenizer.from_pretrained("ailab-bio/PROTAC-Splitter-standard_rand_recombined-ChemBERTa-zinc-base")
            llm_pipeline = llm_pipeline = pipeline(
                "text2text-generation",
                model="ailab-bio/PROTAC-Splitter-standard_rand_recombined-ChemBERTa-zinc-base",
                tokenizer=tokenizer,
                device=device,
            )
        return llm_pipeline(protac_smiles)['generated_text']


    # Split the PROTAC SMILES into the two ligands and the linker
    ligand1, linker, ligand2 = protac_smiles.split('.')
    return ligand1, linker, ligand2