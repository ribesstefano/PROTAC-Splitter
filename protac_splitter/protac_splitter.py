from typing import Tuple


def split_protac(smiles: str) -> Tuple[str, str, str]:
    """
    Split a PROTAC SMILES into the two ligands and the linker.

    Dummy implementation that simply splits the SMILES string by '.'.

    Args:
        smiles (str): A string containing the SMILES of the PROTAC.
    
    Returns:
        Tuple[str, str, str]: A tuple containing the SMILES of the first ligand, the linker, and the second ligand.
    """
    # Split the PROTAC SMILES into the two ligands and the linker
    ligand1, linker, ligand2 = smiles.split('.')
    return ligand1, linker, ligand2