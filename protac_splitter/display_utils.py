import os
import sys
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Draw


def safe_display(*args):
    """Displays content only if running in a Jupyter notebook."""
    if 'ipykernel' in sys.modules:
        display(*args)
    else:
        print(*args)


def display_mol(
        mol: Chem.Mol,
        w: int = 800,
        h: int = 300,
        legend: Optional[str] = None,
        use_smiles_as_legend: bool = True,
        display_svg: bool = True,
):
    """ Display a molecule in a Jupyter notebook. Useful for having """
    if mol is None:
        print('Molecule is None')
        return None
    if use_smiles_as_legend and legend is None:
        legend = Chem.MolToSmiles(mol)
    if display_svg:
        mol.SetProp("_Name", Chem.MolToSmiles(mol, canonical=True))
        d = Draw.rdMolDraw2D.MolDraw2DSVG(w, h, noFreetype=True)
        font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
        if os.path.exists(font_path):
            d.fontFile = font_path
        d.DrawMolecule(mol, legend=legend)
        d.FinishDrawing()
        svg = d.GetDrawingText()
        # Check if in Jupyter notebook
        if sys.modules.get('ipykernel', None):
            from IPython.display import SVG
            safe_display(SVG(svg))
    else:
        img = Draw.MolToImage(mol, size=(w, h))
        safe_display(img)