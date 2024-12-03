from typing import List, Optional
from rdkit import Chem
from multiprocessing import Process, Queue

def worker(q, mol, substruct, useChirality, maxMatches):
    q.put(list(mol.GetSubstructMatches(substruct, useChirality=useChirality, maxMatches=maxMatches)))

def GetSubstructMatchesWithTimeout(
    mol: Chem.Mol,
    substruct: Chem.Mol,
    useChirality: bool = True,
    maxMatches: int = 50,
    timeout: int | float = 10,
) -> Optional[List[List[int]]]:
    """ Get substructure matches with a timeout.

    Args:
        mol (Chem.Mol): The molecule to search for substructure matches.
        substruct (Chem.Mol): The substructure to search for in the molecule.
        useChirality (bool, optional): Whether to use chirality in the substructure search. Defaults to True.
        maxMatches (int, optional): The maximum number of matches to return. Defaults to 50.
        timeout (int | float, optional): The timeout in seconds. Defaults to 10.
    
    Returns:
        Optional[List[List[int]]]: A list of lists containing the atom indices of the substructure matches. Returns None if the search times out or failed.
    """
    q = Queue()
    p = Process(target=worker, args=(q, mol, substruct, useChirality, maxMatches))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None
    else:
        return q.get()