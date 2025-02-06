import re

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Draw,
    rdFMCS,
    rdMolAlign,
)


def save_as_svg(svg_content, filename, num_mols):
    """Save SVG content to a file."""
    with open(filename, 'w') as file:
        data = str(svg_content.data)
        data = data.replace('1500', str(500*num_mols))
        file.write(data)


def align_molecules_2D(ref_mol, to_align_mol):
    AllChem.Compute2DCoords(ref_mol)
    AllChem.Compute2DCoords(to_align_mol)
    # Find the maximum common substructure and use it to align molecules
    mcs = rdFMCS.FindMCS([ref_mol, to_align_mol])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    ref_match = ref_mol.GetSubstructMatch(mcs_mol)
    align_match = to_align_mol.GetSubstructMatch(mcs_mol)
    atom_map = list(zip(align_match, ref_match))
    rdMolAlign.AlignMol(to_align_mol, ref_mol, atomMap=atom_map)
    return to_align_mol


def align_molecules_by_coordinates(ref_mol, to_align_mol):
    # Find the maximum common substructure
    AllChem.Compute2DCoords(to_align_mol)
    mcs = rdFMCS.FindMCS([ref_mol, to_align_mol])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    ref_match = ref_mol.GetSubstructMatch(mcs_mol)
    align_match = to_align_mol.GetSubstructMatch(mcs_mol)

    # Copy the coordinates from the reference molecule to the molecule to be aligned
    ref_conf = ref_mol.GetConformer()
    align_conf = to_align_mol.GetConformer()
    for ref_idx, align_idx in zip(ref_match, align_match):
        ref_pos = ref_conf.GetAtomPosition(ref_idx)
        align_conf.SetAtomPosition(align_idx, ref_pos)

    return to_align_mol


def draw_molecule_to_svg(mol, size=(500, 500), scale=1.0):
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().fixedBondLength = scale
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = re.sub(r'\<\?xml.*?\?\>', '', svg)  # Remove XML declaration
    svg = svg.replace('<svg', '<g').replace(
        '</svg>', '</g>')  # Replace svg tags with g tags
    return svg


def combine_svgs(svgs, output_filename, dimensions=None, size=(500, 500), xy_shifts=None):
    if dimensions is None:
        dimensions = (len(svgs), 1)
    if xy_shifts is None:
        xy_shifts = [(0, 0) for i in range(dimensions[0]*dimensions[1])]

    width, height = size
    grid_width, grid_height = dimensions
    # Include only one XML declaration and the opening <svg> tag
    combined_svg = f'<?xml version="1.0" standalone="no"?>\n'
    combined_svg += f'<svg width="{grid_width * width}px" height="{grid_height * height}px" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">\n'

    # Arrange SVGs in a grid
    for i, (svg, xy_shift) in enumerate(zip(svgs, xy_shifts)):
        x = (i % grid_width) * width
        y = (i // grid_width) * height
        combined_svg += f'<g transform="translate({x+xy_shift[0]},{y-xy_shift[1]})">{svg}</g>\n'

    combined_svg += '</svg>'
    with open(output_filename, 'w') as file:
        file.write(combined_svg)


def draw_molecule_with_highlighted_bonds(mol, bonds_to_highlight):
    """
    Draws a molecule with specified atoms and bonds highlighted.

    Parameters:
    - smiles (str): SMILES string for the molecule.
    - atoms_to_highlight (set): Set of atom indices to highlight.
    - bonds_to_highlight (list): List of bond indices to highlight.
    - highlight_bond_colors (dict): Dictionary mapping bond indices to colors.
    """
    # Create molecule from SMILES

    # Initialize drawer
    d2d = Draw.rdMolDraw2D.MolDraw2DSVG(350*2, 300*2)

    # Set drawing options
    d2d.drawOptions().useBWAtomPalette()
    d2d.drawOptions().continuousHighlight = False
    d2d.drawOptions().highlightBondWidthMultiplier = 24
    d2d.drawOptions().setHighlightColour((0, 0, 1))
    d2d.drawOptions().fillHighlights = False

    # Draw the molecule with highlights
    d2d.DrawMolecule(mol,
                     highlightAtoms=[],
                     highlightBonds=bonds_to_highlight)
    d2d.FinishDrawing()

    # Convert drawing to image and display
    svg = d2d.GetDrawingText()
    svg = svg.replace('svg:', '')

    return svg


def align_mol_2D_ver2(template, query):
    mcs = rdFMCS.FindMCS([template, query])
    patt = Chem.MolFromSmarts(mcs.smartsString)

    query_match = query.GetSubstructMatch(patt)
    template_match = template.GetSubstructMatch(patt)

    rms = AllChem.AlignMol(query, template, atomMap=list(
        zip(query_match, template_match)))
    return template, query


def transform_molecule(mol, degrees, translate_x=0, translate_y=0, flip_x_axis=False):
    """Apply rotation, translation, and optionally flip the molecule."""
    radians = np.deg2rad(degrees)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians),  np.cos(radians), 0],
        [0,               0,               1]
    ])
    AllChem.Compute2DCoords(mol)

    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(i))
        new_pos = np.dot(rotation_matrix, pos)
        new_pos[0] += translate_x  # Translate along the x-axis
        new_pos[1] += translate_y  # Translate along the y-axis
        if flip_x_axis:
            new_pos[1] = -new_pos[1]  # Flip along the x-axis
        conf.SetAtomPosition(i, new_pos)


def tailored_framework_example(mol_ms):
    # remove lone atoms
    # define all atoms to be atom number 1
    # define all bonds to be single bonds

    mol_ms_w = Chem.RWMol(mol_ms)
    atom_idx_to_remove = []
    for atom in mol_ms_w.GetAtoms():
        # lone atom. Need to remove it to create the generic framework.
        if atom.GetDegree() == 1:
            atom_idx_to_remove.append(atom.GetIdx())
            continue
        atom.SetAtomicNum(0)

    for bond in mol_ms_w.GetBonds():
        bond.SetBondType(Chem.rdchem.BondType.SINGLE)

    atom_idx_to_remove.sort(reverse=True)
    for atom_idx in atom_idx_to_remove:
        mol_ms_w.RemoveAtom(atom_idx)

    mol_ms_new = mol_ms_w.GetMol()
    return mol_ms_new
