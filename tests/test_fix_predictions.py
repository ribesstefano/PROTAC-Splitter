from typing import List, Tuple
import random
from typing import Dict, Optional
import logging

import pytest
from rdkit import Chem
from rdkit.Chem import Draw

from protac_splitter.evaluation import (
    check_reassembly,
    split_prediction,
)
from protac_splitter.chemoinformatics import (
    dummy2query,
    canonize,
)

from protac_splitter.protac_splitter import fix_prediction


def protac_examples() -> List[Tuple[str, str]]:
    return [
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(N4CCN(CCCCCNc5ccc6c(c5)C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]CCCCCN2CCN(c1ccc(C([*:1])=O)cc1)CC2.[*:2]Nc3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'CN(c1ccc(C#N)c(Cl)c1)[C@H]1CC[C@H](NC(=O)c2ccc(N3CC(CN4CCN(c5ccc6c(c5)C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)C3)cc2)CC1',
            '[*:1]N[C@@H]2CC[C@@H](N(C)c1ccc(C#N)c(Cl)c1)CC2.[*:1]C(=O)c3ccc(N2CC(CN1CCN([*:2])CC1)C2)cc3.[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'CN1C(=O)CCc2cc3cc(c21)OCCOCC1CN(C(=O)CCC(=O)NCCCOCCOCCOc2cccc4c2C(=O)N(C2CCC(=O)NC2=O)C4=O)CCN1c1ncc(Cl)c(n1)N3',
            '[*:1]N5CCN4c1ncc(Cl)c(n1)Nc3cc2CCC(=O)N(C)c2c(c3)OCCOCC4C5.[*:2]OCCOCCOCCCNC(=O)CCC([*:1])=O.[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'N#CC1(CNc2cccc(-c3cc(N[C@H]4CC[C@H](NCC(=O)NCCOCCOCCOCCNc5cccc6c5C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)ncc3Cl)n2)CCOCC1',
            '[*:1]C(=O)CN[C@@H]4CC[C@@H](Nc3cc(c2cccc(NCC1(C#N)CCOCC1)n2)c(Cl)cn3)CC4.[*:2]NCCOCCOCCOCCN[*:1].[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'O=C1CCC(N2C(=O)c3ccc(OCCOCCOCCOCCN4CCN(Cc5ccc6nc(NC(=O)c7cccc(C(F)(F)F)c7)n([C@H]7CC[C@@H](CO)CC7)c6c5)CC4)cc3C2=O)C(=O)N1',
            '[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3.[*:1]CN1CCN(CCOCCOCCOCCO[*:2])CC1.[*:1]c4ccc3nc(NC(=O)c1cccc(C(F)(F)F)c1)n([C@@H]2CC[C@H](CO)CC2)c3c4',
        ],
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(N4CCN(CCCCNc5ccc6c(c5)C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]NCCCCN2CCN(c1ccc(C([*:1])=O)cc1)CC2.[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)COCCOCCOCCNC(=O)CCC(=O)N2CCN([C@H]3CC[C@@H](Nc4ncnn5ccc(C(C)C)c45)CC3)CC2)C(C)(C)C)cc1',
            '[*:2]N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)NCc3ccc(c2scnc2C)cc3)C(C)(C)C.[*:1]C(=O)CCC(=O)NCCOCCOCCOCC([*:2])=O.[*:1]N4CCN([C@@H]3CC[C@H](Nc1ncnn2ccc(C(C)C)c12)CC3)CC4',
        ],
        [
            'Cc1ncsc1-c1ccc(CNC(=O)C2CC(O)CN2C(=O)C(NC(=O)CNC(=O)c2cccc(-c3ccc(N4CCN(C)CC4)c(NC(=O)c4c[nH]c(=O)cc4C(F)(F)F)c3)c2)C(C)(C)C)cc1',
            '[*:2]NC(C(=O)N1CC(O)CC1C(=O)NCc3ccc(c2scnc2C)cc3)C(C)(C)C.[*:1]NCC([*:2])=O.[*:1]C(=O)c4cccc(c3ccc(N1CCN(C)CC1)c(NC(=O)c2c[nH]c(=O)cc2C(F)(F)F)c3)c4',
        ],
        [
            'CN1C(=O)CCc2cc3cc(c21)OCCOC[C@H]1CN(C(=O)CCC(=O)NCCCOCCOCCOc2cccc4c2C(=O)N(C2CCC(=O)NC2=O)C4=O)CCN1c1ncc(Cl)c(n1)N3',
            '[*:1]N5CCN4c1ncc(Cl)c(n1)Nc3cc2CCC(=O)N(C)c2c(c3)OCCOC[C@H]4C5.[*:2]OCCOCCOCCCNC(=O)CCC([*:1])=O.[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'CC(C)Nc1cc(-n2ccc3cc(C#N)cnc32)ncc1C(=O)N[C@H]1CC[C@H](C(=O)NCCOCCOCCOCCNC(=O)COc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1',
            '[*:1]C(=O)[C@@H]4CC[C@@H](NC(=O)c3cnc(n2ccc1cc(C#N)cnc12)cc3NC(C)C)CC4.[*:1]NCCOCCOCCOCCNC(=O)CO[*:2].[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],

        [
            'CC(C)Nc1cc(-n2ccc3cc(C#N)cnc32)ncc1C(=O)N[C@H]1CC[C@H](C(=O)NCCOCCOCCNC(=O)COc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1',
            '[*:1]C(=O)[C@@H]4CC[C@@H](NC(=O)c3cnc(n2ccc1cc(C#N)cnc12)cc3NC(C)C)CC4.[*:1]NCCOCCOCCNC(=O)CO[*:2].[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)CCCNC(=O)CCC(=O)N2CCN([C@H]3CC[C@@H](Nc4ncnn5ccc(C(C)C)c45)CC3)CC2)C(C)(C)C)cc1',
            '[*:2]N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)NCc3ccc(c2scnc2C)cc3)C(C)(C)C.[*:2]C(=O)CCCNC(=O)CCC([*:1])=O.[*:1]N4CCN([C@@H]3CC[C@H](Nc1ncnn2ccc(C(C)C)c12)CC3)CC4',
        ],
        [
            'CC(C)Nc1cc(-n2ccc3cc(C#N)cnc32)ncc1C(=O)N[C@H]1CC[C@H](C(=O)NCCOCCNC(=O)COc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1',
            '[*:1]C(=O)c3cnc(n2ccc1cc(C#N)cnc12)cc3NC(C)C.[*:1]N[C@@H]1CC[C@@H](C(=O)NCCOCCNC(=O)CO[*:2])CC1.[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'O=C1CCC(N2C(=O)c3ccc(OCCOCCOCCN4CCN(Cc5ccc6nc(NC(=O)c7cccc(C(F)(F)F)c7)n([C@H]7CC[C@@H](CO)CC7)c6c5)CC4)cc3C2=O)C(=O)N1',
            '[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3.[*:1]CCOCCOCCO[*:2].[*:1]N5CCN(Cc4ccc3nc(NC(=O)c1cccc(C(F)(F)F)c1)n([C@@H]2CC[C@H](CO)CC2)c3c4)CC5',
        ],
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(NCCCNc4ccc5c(c4)C(=O)N(C4CCC(=O)NC4=O)C5=O)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]NCCCNc1ccc(C([*:1])=O)cc1.[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'CC(C)Nc1cc(-n2ccc3cc(C#N)cnc32)ncc1C(=O)N[C@H]1CC[C@H](C(=O)NCCCCNC(=O)COc2cccc3c2C(=O)N(C2CCC(=O)NC2=O)C3=O)CC1',
            '[*:1]C(=O)c3cnc(n2ccc1cc(C#N)cnc12)cc3NC(C)C.[*:1]N[C@@H]1CC[C@@H](C(=O)NCCCCNC(=O)CO[*:2])CC1.[*:2]c2cccc3c(=O)n(C1CCC(=O)NC1=O)c(=O)c23',
        ],
        [
            'Cc1ncsc1-c1ccc([C@H](C)NC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)CN2CCN(CCN3CCC(O[C@H]4C[C@H](Oc5ccc6c(c5)Sc5cc([N+](=O)[O-])ccc5N6)C4)CC3)CC2)C(C)(C)C)cc1',
            '[*:2]N[C@H](C(=O)N1C[C@H](O)C[C@H]1C(=O)N[C@@H](C)c3ccc(c2scnc2C)cc3)C(C)(C)C.[*:1]O[C@@H]3C[C@@H](OC2CCN(CCN1CCN(CC([*:2])=O)CC1)CC2)C3.[*:1]c3ccc2[nH]c1ccc(N(=O)=O)cc1sc2c3',
        ],
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(NCCCCCCCCCNc4ccc5c(c4)C(=O)N(C4CCC(=O)NC4=O)C5=O)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]NCCCCCCCCCNc1ccc(C([*:1])=O)cc1.[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
        [
            'N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(NCCCCCCCCCNc4ccc5c(c4)C(=O)N(C4CCC(=O)NC4=O)C5=O)cc3)CC2)cc1Cl',
            '[*:1]N[C@@H]2CC[C@@H](Oc1ccc(C#N)c(Cl)c1)CC2.[*:2]CCC[*:1].[*:2]c3ccc2c(=O)n(C1CCC(=O)NC1=O)c(=O)c2c3',
        ],
    ]

# Set logging level so that we can debug the code
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def remove_attach_atom(mol: Chem.Mol, attach_id: int) -> Chem.Mol:
    """ Removes the atom with the specified attachment id from the molecule.

    Example:
    
    >>> remove_attach_atom(Chem.MolFromSmiles('CC[*:1]'), 1)
    CC

    There are no checks on the molecule, so it is assumed it is not None.

    Args:
        mol (Chem.Mol): The molecule.
        attach_id (int): The attachment id of the atom to remove.

    Returns:
        (Chem.Mol) The molecule with the atom removed.
    """
    atoms_to_remove = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:  # Dummy atom
            map_num = atom.GetAtomMapNum()
            if map_num == attach_id:  # Targeting only [*:attach_id]
                atoms_to_remove.append(atom.GetIdx())

    # Remove atoms using an EditableMol
    editable_mol = Chem.EditableMol(mol)
    for idx in sorted(atoms_to_remove, reverse=True):  # Remove from highest index to avoid shifting
        editable_mol.RemoveAtom(idx)

    # Convert back to a molecule
    new_mol = editable_mol.GetMol()
    Chem.SanitizeMol(new_mol)
    return new_mol


def make_sub_none(substructs: Dict[str, str]) -> Dict[str, str]:
    """ Makes a random substructure None in the substructs dictionary. """
    sub = random.choice(list(substructs.keys()))
    substructs[sub] = 'wrong_substructure'
    return substructs

def add_atom(substructs: Dict[str, str]) -> Dict[str, str]:
    """ Adds a random atom to a random substructure in the substructs dictionary. """
    subs = ['e3', 'linker', 'poi']
    random.shuffle(subs)
    for sub in subs:
        if 'CC' in substructs[sub]:
            print(f'Removing one atom from substructure: {sub.upper()}')
            substructs[sub] = substructs[sub].replace('CC', 'C', 1)
            break
    return substructs

def add_extra_atoms(substructs: Dict[str, str]) -> Dict[str, str]:
    subs = ['e3', 'linker', 'poi']
    random.shuffle(subs)
    for sub in subs:
        if 'CC' in substructs[sub]:
            num_errors = random.choice([1, 2])
            error_atoms = 'CC' + 'C' * num_errors
            substructs[sub] = substructs[sub].replace('CC', error_atoms, 1)
            print(f'Adding N.{num_errors} atom to substructure: {sub.upper()}')
            break
    return substructs

def alter_atom(substructs: Dict[str, str]) -> Dict[str, str]:
    subs = ['e3', 'linker', 'poi']
    random.shuffle(subs)
    for sub in subs:
        if 'N' in substructs[sub]:
            print(f'Altering one atom from substructure: {sub.upper()}')
            # Randomly select a "N" atom and replace it with "C"
            n_atoms = [i for i, c in enumerate(substructs[sub]) if c == 'N']
            if len(n_atoms) == 0:
                continue
            n_atom = random.choice(n_atoms)
            substructs[sub] = substructs[sub][:n_atom] + 'C' + substructs[sub][n_atom + 1:]
            break
    return substructs

def test_fix_prediction():
    for i in range(5):
        random.seed(42 + i)

        error_functions = [
            make_sub_none,
            add_atom,
            add_extra_atoms,
            alter_atom,
        ]
        for error_function in error_functions:
            print('-' * 100)
            print(f"Testing error function: {error_function.__name__}")
            print('-' * 100)
            for protac_smiles, pred_smiles in protac_examples():
                protac_smiles = canonize(protac_smiles)
                pred_smiles = canonize(pred_smiles)
                
                protac_mol = Chem.MolFromSmiles(pred_smiles)
                protac_mol = canonize(Chem.molzip(protac_mol))
                protac_smiles = Chem.MolToSmiles(protac_mol, canonical=True)

                substructs = split_prediction(pred_smiles)
                label_smiles = f"{substructs['e3']}.{substructs['linker']}.{substructs['poi']}"

                substructs = error_function(substructs)
                pred_smiles = '.'.join([substructs[s] for s in ['e3', 'linker', 'poi']])
                
                print(f'PROTAC: {protac_smiles}')
                print(f'Label:  {label_smiles}')
                print(f'Pred:   {pred_smiles}')

                fixed_smiles = fix_prediction(protac_smiles, pred_smiles)

                print(f'PROTAC: {protac_smiles}')
                print(f'Pred:   {pred_smiles}')
                print(f'Label:  {label_smiles}')
                print(f'Fixed:  {fixed_smiles}')

                if fixed_smiles is None:
                    # display(Chem.MolFromSmiles(protac_smiles))
                    print(f'Failed to fix prediction for "{pred_smiles}"')
                assert fixed_smiles is not None, f'Failed to fix prediction for "{pred_smiles}"'
                assert fixed_smiles == label_smiles, f'Fixed prediction is not the same as the original prediction for "{pred_smiles}"'
                print('-' * 80)
