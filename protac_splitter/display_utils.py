import os
import sys
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Draw

if 'ipykernel' in sys.modules:
    from IPython.display import SVG

from .chemoinformatics import get_atom_idx_at_attachment, canonize

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


def get_mapped_protac_img(
        protac_smiles: str,
        poi_smiles: str,
        linker_smiles: str,
        e3_smiles: str,
        w: int = 1000,
        h: int = 1000,
        useSVG: bool = False,
        display_image: bool = False,
        legend: Optional[str] = None,
):
    """ Display a PROTAC molecule with the POI, linker, and E3 ligase highlighted.
    
    If `useSVG` is True, then the POI-Linker bond is highlighted in purple, whereas the E3-Linker bond is highlighted in green.
    If `useSVG` is False, then both splitting points are highlighted in purple.
    
    Args:
        protac_smiles: The SMILES string of the PROTAC.
        poi_smiles: The SMILES string of the POI.
        linker_smiles: The SMILES string of the linker.
        e3_smiles: The SMILES string of the E3 ligase.
        w: The width of the image.
        h: The height of the image.
        useSVG: Whether to use SVG format.
        display_image: Whether to display the image.
        legend: The legend to display.
    """
    protac_smiles = canonize(protac_smiles)
    e3_smiles = canonize(e3_smiles)
    poi_smiles = canonize(poi_smiles)
    linker_smiles = canonize(linker_smiles)

    # Check if any of the canonicalized SMILES is None
    if None in [protac_smiles, e3_smiles, poi_smiles, linker_smiles]:
        return None

    protac_mol = Chem.MolFromSmiles(protac_smiles)
    e3_mol = Chem.MolFromSmiles(e3_smiles)
    poi_mol = Chem.MolFromSmiles(poi_smiles)
    linker_mol = Chem.MolFromSmiles(linker_smiles)

    if None in [protac_mol, e3_mol, poi_mol, linker_mol]:
        return None

    if linker_smiles in ['[*:1][*:2]', '[*:2][*:1]']:
        print('WARNING. Linker is empty.')
        poi_attachment_idx = get_atom_idx_at_attachment(protac_mol, poi_mol, e3_mol)
        e3_attachment_idx = get_atom_idx_at_attachment(protac_mol, e3_mol, poi_mol)
    else:
        poi_attachment_idx = get_atom_idx_at_attachment(protac_mol, poi_mol, linker_mol)
        e3_attachment_idx = get_atom_idx_at_attachment(protac_mol, e3_mol, linker_mol)

    cyan = (0, 1, 1, 0.5)
    red = (1, 0, 0, 0.5)
    green = (0, 1, 0, 0.5)
    blue = (0, 0, 1, 0.5)
    purple = (1, 0, 1, 0.3)

    highlight_atoms = []
    highlight_bonds = []
    atom_colors = {}
    bond_colors = {}

    if poi_attachment_idx is not None:
        if len(poi_attachment_idx) != 2:
            if linker_smiles in ['[*:1][*:2]', '[*:2][*:1]']:
                print(f'WARNING. Linker is empty, no highlighting will be showed for the POI.')
            else:
                print(f'WARNING. POI attachment points must be only two, got instead: {poi_attachment_idx}')
        else:
            poi_bond_idx = protac_mol.GetBondBetweenAtoms(*poi_attachment_idx).GetIdx()
            highlight_atoms += poi_attachment_idx
            highlight_bonds.append(poi_bond_idx)
            atom_colors[poi_attachment_idx[0]] = purple
            atom_colors[poi_attachment_idx[1]] = purple
            bond_colors[poi_bond_idx] = purple

    if e3_attachment_idx is not None:
        if len(e3_attachment_idx) != 2:
            if linker_smiles in ['[*:1][*:2]', '[*:2][*:1]']:
                print(f'WARNING. Linker is empty, no highlighting will be showed for the E3.')
            else:
                print(f'WARNING. E3 attachment points must be only two, got instead: {e3_attachment_idx}')
        else:
            e3_bond_idx = protac_mol.GetBondBetweenAtoms(*e3_attachment_idx).GetIdx()
            highlight_atoms += e3_attachment_idx
            highlight_bonds.append(e3_bond_idx)
            atom_colors[e3_attachment_idx[0]] = green
            atom_colors[e3_attachment_idx[1]] = green
            bond_colors[e3_bond_idx] = green

    if useSVG:
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(w, h, noFreetype=True)
        options = drawer.drawOptions()
        options.fontFile = '/System/Library/Fonts/Supplemental/Arial.ttf'

        if legend is None:
            legend = '.'.join([e3_smiles, linker_smiles, poi_smiles])

        drawer.DrawMolecule(
            protac_mol,
            legend=legend,
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
        )

        drawer.FinishDrawing()
        svg_text = drawer.GetDrawingText()

        if display_image:
            safe_display(SVG(svg_text))

        return svg_text
    else:
        img = Draw.MolToImage(
            protac_mol,
            size=(w, h),
            highlightColor=purple,
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
        )

        if display_image:
            safe_display(img)

        return img