#!/usr/bin/env python3
"""Recombine PROTAC components using empirical, oriented junction environments.

Requires Python >=3.10 and RDKit:  python -m pip install rdkit

Examples:
  python enumerate_protacs.py input.csv -o enumerated.csv --workers 8
  python enumerate_protacs.py input.csv -o sampled.csv --num-protacs 100000 \\
      --extra-linkers linkers.smi --workers 8 --seed 42

Input: CSV with PROTAC, warhead, linker and E3 SMILES columns. Common column
names are detected; explicit --*-col options override detection. Warheads must
have one [*:1], E3 ligands one [*:2], and linkers one of each. Other atom maps
are discarded. Salt/disconnected components and direct dummy-dummy linkers
are rejected. Extra linkers: .smi/.txt (one SMILES per nonempty line, optional
whitespace-delimited name) or CSV (--extra-linker-col, default autodetected).
Both orientations of EVERY linker are included and canonicalized/deduplicated.

Method:
  Each joining-bond environment is represented by its two oriented endpoint
  neighborhoods: the endpoint plus its immediate neighbors, explicit Hs and
  bond orders. The partner is represented by an unmapped dummy; the endpoint
  is marked :9. These are local environments, not named functional groups.
  Each junction is also reported as ONE canonical SMARTS per side, built by
  joining the two oriented neighborhoods across the formed bond and dropping
  the two zipped dummies; this matches the notebook's functional groups
  (`Chem.MolFragmentToSmarts` over a 1-hop neighborhood, then canonized), so
  the two columns are comparable with `functional_group_distributions.json`.
  An attachment point of the partner component that is itself within one hop
  survives as `[#0]`, marking where the third component attaches.
  Learn counts c_W(w,l1), c_E(e,l2) from distinct canonical input decompositions.
  Exact duplicate rows do not inflate counts. Allow a triple
  only if BOTH oriented junction environments were observed. Extra linkers
  acquire compatibility from their local environments, without annotations.
  Without --num-protacs, visit ALL compatible component triples. Exhaustive
  enumeration enforces support, not empirical frequency proportions.
  With --num-protacs N, allocate largest-remainder quotas across feasible
  junction-pair strata with weight c_W(w,l1)*c_E(e,l2). This follows the
  notebook's independent-end sampling, conditioned on available linker end
  pairs. Sample components in a seeded pseudorandom order within each stratum.
  A constant-memory affine permutation traverses each Cartesian pool exactly
  once; this is not a uniformly random permutation of all possible triples.
  Stratum quotas preserve the conditioned distribution to rounding, with NO
  silent redistribution when a stratum is exhausted by exclusions/deduplication.
  If quotas cannot be filled, save the partial CSV and exit 2; lower N or use
  exhaustive mode. Conditioning on linker availability can change the original
  marginal distributions; this is reported as total variation distance.

Assembly uses RDKit molzip, preserves specified stereochemistry, sanitizes,
requires one connected product without dummy atoms, re-perceives stereo so
that dependent centers (e.g. cyclobutane-1,3-diyl) get one stable descriptor
pair, and writes canonical isomeric SMILES. Input decompositions must
reassemble to their supplied PROTAC, compared after the same standardization
on both sides; rows that do not are dropped and counted in the report.
Original PROTACs are excluded by default (--include-originals overrides).
A disk-backed SQLite primary key deduplicates by product structure globally,
including molecules arising from different decompositions/orientations. This
is canonical-SMILES identity, not tautomer/protonation-state normalization.
Workers assemble cached molecules in bounded batches; only the parent writes.
Output order and selection are reproducible for fixed seed and RDKit version,
independent of worker count. The first accepted decomposition is retained.

Reference implementation studied:
https://github.com/ribesstefano/PROTAC-Splitter/blob/main/notebooks/generate_train_data.ipynb
https://github.com/ribesstefano/PROTAC-Splitter/tree/main/protac_splitter/data/generation
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
from typing import Optional

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

RDLogger.DisableLog("rdApp.*")

_SALT_REMOVER = SaltRemover()
_UNCHARGER = rdMolStandardize.Uncharger()
_ATTACHMENT_MAP_NUMS = (1, 2)


def _has_attachment_points(mol: Chem.Mol) -> bool:
    """ Whether the molecule contains a "[*:1]" or "[*:2]" attachment point. """
    return any(
        atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() in _ATTACHMENT_MAP_NUMS
        for atom in mol.GetAtoms()
    )

def standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """ Drops salts, uncharges, and keeps the largest fragment of a molecule.

    The largest fragment is NOT selected if the molecule contains either of the
    "[*:1]" or "[*:2]" attachment points, since multi-fragment strings with
    attachment points are valid (e.g., "e3.linker.poi" predictions).

    Args:
        mol (Chem.Mol): The input molecule. It is not modified.

    Returns:
        Chem.Mol or None: The standardized molecule, or None if it failed.
    """
    if mol is None:
        return None
    try:
        # Drop salts, but never strip the molecule down to nothing
        mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
        # Uncharge the molecule (i.e., remove formal charges)
        mol = _UNCHARGER.uncharge(mol)
        # Keep only the largest fragment, unless there are attachment points
        if not _has_attachment_points(mol):
            mol = rdMolStandardize.FragmentParent(mol)
        return mol
    except Exception as e:
        print(f"Warning: error standardizing molecule: {e}", file=sys.stderr)
        return None


def canonicalize(smiles: str) -> Optional[str]:
    """ Canonizes a SMILES string by converting it to canonical SMILES representation.

    Salts are dropped, the molecule is uncharged, and only the largest fragment
    is kept, unless the SMILES contains "[*:1]" or "[*:2]" attachment points (in
    which case all fragments are kept).

    Args:
        smiles (str): The input SMILES string.

    Returns:
        str or None: The canonized SMILES string, or None if it failed.
    """
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(f"Error: {e}")
        return None
    if mol is None:
        return None
    mol = standardize_mol(mol)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        return None

def canonize_smarts(smarts: str) -> Optional[str]:
    """ Cleans a SMARTS string by converting it to canonical SMARTS representation.

    Mirrors ``protac_splitter.chemoinformatics.canonize_smarts`` so that this
    script stays importable with nothing but RDKit.

    NOTE: It might not work for complex patterns: https://github.com/rdkit/rdkit/discussions/6929

    Args:
        smarts (str): The input SMARTS string.

    Returns:
        str or None: The canonical SMARTS string, or None if it failed.
    """
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        return None
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol), sanitize=False)
    if mol is None:
        return None
    return Chem.MolToSmarts(mol)


ALIASES = {
    'protac': ('PROTAC SMILES', 'protac_smiles', 'protac'),
    'warhead': ('Warhead SMILES', 'POI Ligand SMILES with direction', 'warhead_smiles', 'warhead'),
    'linker': ('Linker SMILES', 'Linker SMILES with direction', 'linker_smiles', 'linker'),
    'e3': ('E3 Ligase Ligand SMILES', 'E3 Binder SMILES with direction', 'e3_smiles', 'e3'),
}
FIELDS = ['PROTAC SMILES', 'Warhead SMILES', 'Linker SMILES', 'E3 Ligase Ligand SMILES',
          'Warhead Junction SMARTS', 'E3 Junction SMARTS']


def column(headers: Sequence[str], explicit: Optional[str], role: str) -> str:
    """ Resolves which CSV column holds a given role, by name or by alias.

    Args:
        headers: The column names present in the CSV.
        explicit: A column name given on the command line, which wins if set.
        role: A key of ALIASES, i.e. 'protac', 'warhead', 'linker' or 'e3'.

    Returns:
        str: The matching header, as spelled in the file.

    Raises:
        ValueError: If the explicit column is absent, or no alias matches.
    """
    if explicit:
        if explicit not in headers:
            raise ValueError(f'Missing column {explicit!r}; available: {headers}')
        return explicit
    lookup = {h.strip().lower(): h for h in headers}
    for name in ALIASES[role]:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise ValueError(f'Cannot identify {role} column; use --{role}-col. Available: {headers}')


def molecule(smiles: str) -> Chem.Mol:
    """ Parses one standardized, single-fragment molecule from a SMILES string.

    Args:
        smiles: The input SMILES string.

    Returns:
        Chem.Mol: The parsed molecule.

    Raises:
        ValueError: If the SMILES is unparseable, empty or disconnected.
    """
    # Ensure the SMILES is canonicalized before creating the molecule
    mol = Chem.MolFromSmiles(canonicalize(smiles.strip()))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f'Disconnected SMILES unsupported: {smiles!r}')
    return mol


def canonical_product(smiles: str) -> str:
    """ Canonical SMILES of a finished PROTAC, with atom maps cleared.

    Both sides of the reassembly check pass through here, so a normalization
    the standardizer applies cannot by itself make the two sides differ.

    Args:
        smiles: The input SMILES string.

    Returns:
        str: The canonical isomeric SMILES.

    Raises:
        ValueError: If the molecule still carries dummy atoms.
    """
    mol = molecule(smiles)
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError('PROTAC contains dummy atoms')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=200000)
def component(smiles: str, role: str) -> str:
    """ Validates one PROTAC component and returns its canonical SMILES.

    Attachment points must be exactly those the role calls for: [*:1] on a
    warhead, [*:2] on an E3 ligand, one of each on a linker. Atom maps on real
    atoms are discarded, so only the attachment points stay labelled.

    Args:
        smiles: The input SMILES string.
        role: 'warhead', 'linker' or 'e3'.

    Returns:
        str: The canonical isomeric SMILES, attachment points included.

    Raises:
        ValueError: If a dummy is isotopic, is not terminal on a real atom, or
            if the set of attachment maps is not the one the role requires.
    """
    mol = molecule(smiles)
    labels = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            labels.append(atom.GetAtomMapNum())
            if atom.GetDegree() != 1 or atom.GetNeighbors()[0].GetAtomicNum() == 0:
                raise ValueError('Each dummy must have exactly one non-dummy neighbor')
            if atom.GetIsotope():
                raise ValueError('Isotopic dummy atoms unsupported')
        else:
            atom.SetAtomMapNum(0)
    expected = {'warhead': [1], 'linker': [1, 2], 'e3': [2]}[role]
    if sorted(labels) != expected:
        raise ValueError(f'{role}: expected attachment maps {expected}, got {labels}: {smiles}')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def flipped(smiles: str) -> str:
    """ Swaps a linker's two attachment points, giving its other orientation.

    Args:
        smiles: A canonical linker SMILES carrying [*:1] and [*:2].

    Returns:
        str: The canonical SMILES of the linker with the labels exchanged. A
            symmetric linker canonizes back to the input, so adding this to a
            set of linkers cannot duplicate one.
    """
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(3 - atom.GetAtomMapNum())
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=200000)
def endpoint(smiles: str, label: int) -> str:
    """ Canonical rooted local environment; attachment direction is explicit.

    The environment is the atom bearing the attachment point plus its immediate
    neighbors, with explicit Hs and bond orders. The attachment itself becomes
    an unmapped dummy and the central atom is marked :9, so the string encodes
    which side of the junction it describes. This is the key compatibility is
    learned and grouped on; `junction_smarts` reports the joined form.

    Args:
        smiles: A canonical component SMILES.
        label: The attachment map number to describe, 1 or 2.

    Returns:
        str: The fragment SMILES of the environment. Note it is not always a
            parseable SMILES on its own (an aromatic neighbor has no ring to
            close), which is why the CSV reports SMARTS instead.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    dummy = next(a for a in mol.GetAtoms()
                 if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == label)
    center = dummy.GetNeighbors()[0]
    indices = [center.GetIdx()] + [a.GetIdx() for a in center.GetNeighbors()]
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    center.SetAtomMapNum(9)
    # Local stereo can depend on atoms outside this neighborhood. Preserve it
    # in full components/products, but do not restrict junction compatibility.
    return Chem.MolFragmentToSmiles(mol, atomsToUse=indices, canonical=True,
                                   isomericSmiles=False, allBondsExplicit=True)


@lru_cache(maxsize=16384)
def parsed(smiles: str) -> Chem.Mol:
    """ Cached parse of a component SMILES, shared by every assembly using it.

    Callers must not mutate the result; it is one object per distinct SMILES.
    """
    return Chem.MolFromSmiles(smiles)


@lru_cache(maxsize=200000)
def junction_smarts(ligand: str, label: int, linker: str) -> str:
    """ Canonical SMARTS of the 1-hop environment of the bond formed at a junction.

    The two neighborhoods that `endpoint` reports separately are joined across
    the bond that molzip will create, and the two zipped dummies are dropped,
    so the result describes the environment as it exists in the product. Both
    members of a junction stratum share it, so it is computed once per stratum.
    This reproduces the notebook's functional-group vocabulary, which makes the
    output columns comparable with `functional_group_distributions.json`.

    Args:
        ligand: A canonical warhead or E3 ligand SMILES. The parameter is not
            named `component` so as not to shadow the function of that name.
        label: The attachment map joined here, 1 for a warhead, 2 for an E3.
        linker: The canonical linker SMILES joined to it.

    Returns:
        str: The canonical SMARTS, or the uncanonized fragment SMARTS if
            canonization gives up. An attachment point of the linker that is
            itself within one hop stays as [#0], marking the far junction.
    """
    mols, centers, dummies, order = [], [], [], None
    for smiles in (ligand, linker):
        mol = Chem.AddHs(parsed(smiles))
        dummy = next(a for a in mol.GetAtoms()
                     if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == label)
        center = dummy.GetNeighbors()[0]
        if order is None:
            # `assemble` refuses a junction whose two dummy bonds disagree, so
            # the component's order is the order of the formed bond.
            order = mol.GetBondBetweenAtoms(dummy.GetIdx(), center.GetIdx()).GetBondType()
        mols.append(mol); centers.append(center); dummies.append(dummy)
    # Indices of the second molecule shift by the size of the first once combined
    merged = Chem.RWMol(Chem.CombineMols(*mols))
    offsets = (0, mols[0].GetNumAtoms())
    merged.AddBond(centers[0].GetIdx(), centers[1].GetIdx() + offsets[1], order)
    # Keep both centers and their neighbors, minus the two dummies being zipped
    indices = []
    for center, dummy, offset in zip(centers, dummies, offsets):
        indices.append(center.GetIdx() + offset)
        indices += [a.GetIdx() + offset for a in center.GetNeighbors()
                    if a.GetIdx() != dummy.GetIdx()]
    for atom in merged.GetAtoms():
        atom.SetAtomMapNum(0)
    smarts = Chem.MolFragmentToSmarts(merged, atomsToUse=indices)
    # canonize_smarts round-trips through SMILES and can give up on a pattern;
    # the uncanonized fragment SMARTS is still correct, just not canonical.
    return canonize_smarts(smarts) or smarts


@lru_cache(maxsize=16384)
def attachment_bonds(smiles: str) -> dict[int, Chem.BondType]:
    """ Maps each attachment point of a component to the order of its one bond. """
    return {a.GetAtomMapNum(): a.GetBonds()[0].GetBondType()
            for a in parsed(smiles).GetAtoms() if a.GetAtomicNum() == 0}


def assemble(w: str, l: str, e: str) -> str:
    """ Zips a warhead, linker and E3 ligand into one product SMILES.

    Args:
        w: Canonical warhead SMILES, carrying [*:1].
        l: Canonical linker SMILES, carrying [*:1] and [*:2].
        e: Canonical E3 ligand SMILES, carrying [*:2].

    Returns:
        str: The canonical isomeric SMILES of the product. This string is both
            the output and the deduplication key, so it has to be stable
            across the different decompositions that reach one molecule.

    Raises:
        ValueError: If the two dummy bonds at a junction have different orders,
            or if zipping leaves dummy atoms or more than one fragment.
    """
    if (attachment_bonds(w)[1] != attachment_bonds(l)[1]
            or attachment_bonds(e)[2] != attachment_bonds(l)[2]):
        raise ValueError('The two dummy bonds at a junction must have matching orders')
    mol = Chem.CombineMols(Chem.CombineMols(parsed(w), parsed(l)), parsed(e))
    mol = Chem.molzip(mol)
    Chem.SanitizeMol(mol)
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()) or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('Incomplete assembly')
    # molzip keeps the tags the fragments carried, which for dependent centers
    # (a cyclobutane-1,3-diyl linker, a 1,4-disubstituted ring) leaves a valid
    # but input-order-dependent descriptor pair: the same molecule reached from
    # two decompositions would then get two SMILES and escape deduplication.
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def read_inputs(
        args: argparse.Namespace,
) -> tuple[list[str], list[str], list[str], set[str], Counter, Counter, dict[str, int]]:
    """ Reads the input CSV into component pools and junction-environment counts.

    Only decompositions that reassemble to their supplied PROTAC contribute.
    Exact duplicate rows are collapsed, so a popular PROTAC does not inflate
    the counts the quotas are built from. Extra linkers, if given, join the
    linker pool without contributing counts, and every linker is then added in
    both orientations.

    Args:
        args: Parsed command-line arguments.

    Returns:
        tuple: Sorted warheads, oriented linkers and E3 ligands; the set of
            original PROTAC SMILES to exclude; the c_W and c_E counters keyed
            by (ligand environment, linker environment); and a dict of input
            row statistics for the report.

    Raises:
        ValueError: If a column cannot be resolved, a row cannot be parsed, or
            no input PROTAC survives.
    """
    ws, ls, es, originals = set(), set(), set(), set()
    decompositions = set()
    rows = dropped = 0
    left, right = Counter(), Counter()
    with args.input.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        cols = {r: column(reader.fieldnames or [], getattr(args, r + '_col'), r) for r in ALIASES}
        for line, row in enumerate(reader, 2):
            try:
                rows += 1
                p = canonical_product(row[cols['protac']])
                w, l, e = (component(row[cols[r]], r) for r in ('warhead', 'linker', 'e3'))
                # Both sides go through the same standardization: comparing a
                # standardized PROTAC against a raw assembly rejects rows over
                # normalizations the components never saw (a sulfoxide written
                # S(=O) vs [S+][O-]) rather than over real disagreement.
                if canonical_product(assemble(w, l, e)) != p:
                    # Drop the row if the components do not reassemble to the supplied PROTAC
                    dropped += 1
                    continue
                if (p, w, l, e) in decompositions:
                    continue
                decompositions.add((p, w, l, e))
                originals.add(p)
                ws.add(w); ls.add(l); es.add(e)
                left[endpoint(w, 1), endpoint(l, 1)] += 1
                right[endpoint(e, 2), endpoint(l, 2)] += 1
            except (ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
                raise ValueError(f'Input row {line}: {exc}') from exc
    if dropped:
        print(f'{dropped} of {rows} input rows dropped: components do not reassemble '
              f'to the supplied PROTAC', file=sys.stderr)
    if not originals:
        raise ValueError('Input contains no PROTACs')
    if args.extra_linkers:
        with args.extra_linkers.open(newline='', encoding='utf-8-sig') as handle:
            if args.extra_linkers.suffix.lower() == '.csv':
                reader = csv.DictReader(handle)
                col = column(reader.fieldnames or [], args.extra_linker_col, 'linker')
                records = ((i, row[col]) for i, row in enumerate(reader, 2))
            else:
                # .smi/.txt: first whitespace-delimited field, skipping blanks
                records = ((i, line.split()[0]) for i, line in enumerate(handle, 1)
                           if line.strip() and not line.lstrip().startswith('#'))
            for line, smi in records:
                try:
                    ls.add(component(smi, 'linker'))
                except (ValueError, RuntimeError) as exc:
                    raise ValueError(f'Extra linker row {line}: {exc}') from exc
    # Every linker is usable in both directions; a symmetric one dedups away
    ls |= {flipped(l) for l in ls}
    return sorted(ws), sorted(ls), sorted(es), originals, left, right, dict(rows=rows, dropped=dropped)


@dataclass
class Pool:
    """ One junction stratum: every triple sharing a pair of junctions.

    A stratum is the Cartesian product of the warheads, linkers and E3 ligands
    whose environments match, so its triples are enumerated without being
    materialized. Strata partition the triple space: a component belongs to
    exactly one environment group, so no triple is visited twice.
    """

    w: tuple[int, ...]            # Indices into the sorted warhead list
    l: tuple[int, ...]            # Indices into the sorted oriented-linker list
    e: tuple[int, ...]            # Indices into the sorted E3 ligand list
    left: tuple[str, str]         # (warhead environment, linker [*:1] environment)
    right: tuple[str, str]        # (E3 environment, linker [*:2] environment)
    weight: int                   # c_W * c_E, the stratum's share of the quota
    left_smarts: str = ''         # Canonical SMARTS of the warhead junction
    right_smarts: str = ''        # Canonical SMARTS of the E3 junction
    size: int = 0                 # Number of triples in the stratum
    cursor: int = 0               # How many have been handed out so far
    offset: int = 0               # Affine permutation start
    stride: int = 1               # Affine permutation step, coprime with size
    target: int = 0               # Quota of unique products wanted
    accepted: int = 0             # Unique products written so far

    def initialize(self, rng: random.Random, sample: bool) -> None:
        """ Sizes the stratum and, when sampling, picks its traversal order.

        A stride coprime with the size makes i -> offset + i * stride mod size
        a permutation, so the stratum can be walked exactly once in a shuffled
        order at constant memory. It is not a uniform permutation of triples.

        Args:
            rng: The shared seeded generator, drawn from in stratum order.
            sample: False for exhaustive runs, which walk in natural order.
        """
        self.size = len(self.w) * len(self.l) * len(self.e)
        if sample:
            self.offset = rng.randrange(self.size)
            self.stride = rng.randrange(1, self.size + 1)
            while math.gcd(self.stride, self.size) != 1:
                self.stride = rng.randrange(1, self.size + 1)

    def take(self, n: int) -> list[tuple[int, int, int]]:
        """ Hands out up to n not-yet-visited triples, advancing the cursor.

        Args:
            n: The most triples to return.

        Returns:
            list: (warhead, linker, E3) index triples, empty once exhausted.
        """
        triples = []
        for _ in range(min(n, self.size - self.cursor)):
            i = (self.offset + self.cursor * self.stride) % self.size
            self.cursor += 1
            # Decode the flat index into one index per component list
            i, ei = divmod(i, len(self.e))
            wi, li = divmod(i, len(self.l))
            triples.append((self.w[wi], self.l[li], self.e[ei]))
        return triples


def make_pools(
        ws: list[str],
        ls: list[str],
        es: list[str],
        left: Counter,
        right: Counter,
        args: argparse.Namespace,
) -> list[Pool]:
    """ Builds the junction strata and, when sampling, their quotas.

    Components are grouped by junction environment, and a stratum is created
    for every (linker end pair, observed warhead junction, observed E3
    junction) combination. With --num-protacs the requested total is split over
    strata by largest remainder, in proportion to c_W * c_E.

    Args:
        ws: Sorted warhead SMILES.
        ls: Sorted oriented linker SMILES.
        es: Sorted E3 ligand SMILES.
        left: c_W counts keyed by (warhead environment, linker environment).
        right: c_E counts keyed by (E3 environment, linker environment).
        args: Parsed command-line arguments.

    Returns:
        list: The strata, initialized and with targets assigned.

    Raises:
        ValueError: If no linker end pair has both junctions supported.
    """
    wgroups, egroups, lgroups = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, s in enumerate(ws): wgroups[endpoint(s, 1)].append(i)
    for i, s in enumerate(es): egroups[endpoint(s, 2)].append(i)
    for i, s in enumerate(ls): lgroups[endpoint(s, 1), endpoint(s, 2)].append(i)
    lallowed, rallowed = defaultdict(list), defaultdict(list)
    for (w, l), count in sorted(left.items()): lallowed[l].append((w, count))
    for (e, l), count in sorted(right.items()): rallowed[l].append((e, count))
    pools = []
    used_linkers = set()
    for (l1, l2), indices in sorted(lgroups.items()):
        # Any member represents the stratum's junctions; they share environments
        linker = ls[indices[0]]
        for w, wc in lallowed[l1]:
            for e, ec in rallowed[l2]:
                pools.append(Pool(tuple(wgroups[w]), tuple(indices), tuple(egroups[e]),
                                  (w, l1), (e, l2), wc * ec,
                                  junction_smarts(ws[wgroups[w][0]], 1, linker),
                                  junction_smarts(es[egroups[e][0]], 2, linker)))
                used_linkers.update(indices)
    if not pools:
        raise ValueError('No compatible junction pairs')
    rng = random.Random(args.seed)
    for pool in pools: pool.initialize(rng, args.num_protacs is not None)
    if args.num_protacs:
        total_weight = sum(p.weight for p in pools)
        remainders = []
        for i, pool in enumerate(pools):
            pool.target, rem = divmod(args.num_protacs * pool.weight, total_weight)
            remainders.append((rem, i))
        # Largest remainder, ties by stratum index, so the split is reproducible
        for _, i in sorted(remainders, key=lambda t: (-t[0], t[1]))[:args.num_protacs - sum(p.target for p in pools)]:
            pools[i].target += 1
        # A quota above a stratum's capacity cannot be filled however long the
        # run goes, and quotas are never redistributed. Report it now rather
        # than after enumerating for hours.
        overcapacity = sum(1 for p in pools if p.target > p.size)
        feasible = min(p.size * total_weight // p.weight for p in pools)
    else:
        for pool in pools: pool.target = pool.size
    print(f'{len(ws)} warheads, {len(ls)} oriented linkers, {len(es)} E3 ligands; '
          f'{len(ls) - len(used_linkers)} linkers have no compatible end pair', file=sys.stderr)
    print(f'{len(pools)} junction strata; {sum(p.size for p in pools):,} compatible triples', file=sys.stderr)
    if args.num_protacs:
        # Report the shift caused by conditioning independent junction draws on
        # the presence of a compatible linker. No molecules are enumerated here.
        for name, counts, attr in [('warhead', left, 'left'), ('E3', right, 'right')]:
            conditioned = Counter()
            for pool in pools: conditioned[getattr(pool, attr)] += pool.weight
            z, n = sum(conditioned.values()), sum(counts.values())
            tv = sum(abs(counts[k] / n - conditioned[k] / z) for k in counts) / 2
            print(f'{name} junction marginal: conditioning TV distance = {tv:.4f}', file=sys.stderr)
        if overcapacity:
            print(f'WARNING: {overcapacity} of {len(pools)} junction strata have a quota larger '
                  f'than the number of triples they contain, so this run will end with a shortfall '
                  f'and exit 2. The largest --num-protacs every stratum can fill is about '
                  f'{feasible:,}; omit --num-protacs to enumerate exhaustively.', file=sys.stderr)
    return pools


_COMPONENTS = None


def initialize_worker(components: tuple[list[str], list[str], list[str]]) -> None:
    """ Installs the component lists in a worker process (or in the parent).

    Workers receive the lists once at start-up and are then sent only index
    triples, which keeps the per-job payload small.

    Args:
        components: The (warheads, linkers, E3 ligands) SMILES lists.
    """
    global _COMPONENTS
    _COMPONENTS = components
    RDLogger.DisableLog('rdApp.*')


def assemble_batch(
        job: tuple[int, list[tuple[int, int, int]]],
) -> tuple[int, list[tuple[str, int, int, int]], int]:
    """ Assembles one batch of index triples, skipping those that fail.

    Args:
        job: The stratum index and its index triples.

    Returns:
        tuple: The stratum index, the (product SMILES, w, l, e) rows in input
            order, and how many triples failed to assemble. Deduplication is
            left to the parent, which owns the only database handle.
    """
    pool_id, triples = job
    ws, ls, es = _COMPONENTS
    rows = []
    invalid = 0
    for wi, li, ei in triples:
        try:
            rows.append((assemble(ws[wi], ls[li], es[ei]), wi, li, ei))
        except (ValueError, RuntimeError):
            invalid += 1
    return pool_id, rows, invalid


def enumerate_to_csv(
        args: argparse.Namespace,
        components: tuple[list[str], list[str], list[str]],
        originals: set[str],
        pools: list[Pool],
        inputs: dict[str, int],
) -> int:
    """ Enumerates the strata, writes the unique products, and reports.

    Jobs are cut in fixed-size waves and their results consumed in order, so
    which products are kept does not depend on the number of workers.

    Args:
        args: Parsed command-line arguments.
        components: The (warheads, linkers, E3 ligands) SMILES lists.
        originals: Input PROTACs, pre-seeded as seen unless --include-originals.
        pools: The junction strata.
        inputs: Input row statistics from `read_inputs`.

    Returns:
        int: 0 on success, or 2 if a stratum could not fill its quota, in which
            case the partial CSV is still written.
    """
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = duplicates = invalid = 0
    # The output remains untouched until a complete or explicitly partial run
    # finishes. SQLite enforces exact deduplication without an enormous RAM set.
    with tempfile.TemporaryDirectory(prefix='.enumerate-', dir=args.output.parent) as temp:
        db = sqlite3.connect(str(Path(temp) / 'seen.sqlite'))
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        db.execute('CREATE TABLE seen (smiles TEXT PRIMARY KEY) WITHOUT ROWID')
        if not args.include_originals:
            db.executemany('INSERT INTO seen VALUES (?)', ((p,) for p in sorted(originals)))
        db.commit()
        tmp_csv = Path(temp) / 'output.csv'
        executor = None
        try:
            if args.workers > 1:
                executor = ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                                               initializer=initialize_worker, initargs=(components,))
            else:
                initialize_worker(components)
            with tmp_csv.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(FIELDS)
                # Fixed-size waves make selection independent of worker count,
                # including when different strata yield the same product.
                window = 32
                while True:
                    jobs = []
                    for i, pool in enumerate(pools):
                        # Quotas count unique products, so a stratum losing rows
                        # to deduplication is revisited until its triples run out
                        need = pool.target - pool.accepted if args.num_protacs else pool.size - pool.cursor
                        if need <= 0 or pool.cursor == pool.size:
                            continue
                        slots = window - len(jobs)
                        count = min(need, slots * args.batch_size)
                        while count > 0:
                            n = min(count, args.batch_size)
                            triples = pool.take(n)
                            if not triples: break
                            jobs.append((i, triples))
                            count -= len(triples)
                        if len(jobs) >= window: break
                    if not jobs: break
                    results = executor.map(assemble_batch, jobs) if executor else map(assemble_batch, jobs)
                    for i, rows, failed in results:
                        pool = pools[i]
                        invalid += failed
                        for p, wi, li, ei in rows:
                            # The primary key is the filter: a row that inserts
                            # nothing is a product some earlier triple produced
                            if db.execute('INSERT OR IGNORE INTO seen VALUES (?)', (p,)).rowcount == 0:
                                duplicates += 1
                                continue
                            writer.writerow([p, components[0][wi], components[1][li], components[2][ei],
                                             pool.left_smarts, pool.right_smarts])
                            pool.accepted += 1
                            total += 1
                    db.commit()
                    print(f'\rUnique products: {total:,}; tested: {sum(p.cursor for p in pools):,}',
                          end='', file=sys.stderr, flush=True)
            print(file=sys.stderr)
        finally:
            if executor: executor.shutdown(cancel_futures=True)
            db.close()
        # Atomic publish: a crashed run leaves any previous output in place
        os.replace(tmp_csv, args.output)
    shortfalls = [dict(stratum=i, requested=p.target, written=p.accepted)
                  for i, p in enumerate(pools) if args.num_protacs and p.accepted < p.target]
    stats = dict(rdkit_version=rdBase.rdkitVersion, seed=args.seed, written=total,
                 input_rows=inputs['rows'], input_rows_not_reassembling=inputs['dropped'],
                 tested_triples=sum(p.cursor for p in pools),
                 duplicate_or_original_products=duplicates, invalid_assemblies=invalid,
                 requested=args.num_protacs, shortfalls=shortfalls)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats), file=sys.stderr)
    if shortfalls:
        print('Requested distribution quotas could not be filled. Partial CSV saved. '
              'Lower --num-protacs or omit it for exhaustive enumeration.', file=sys.stderr)
        return 2
    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """ Parses and validates the command line.

    Args:
        argv: Arguments to parse, or None to read sys.argv.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--output', required=True, type=Path)
    parser.add_argument('--extra-linkers', type=Path)
    parser.add_argument('--extra-linker-col')
    for role in ALIASES: parser.add_argument(f'--{role}-col')
    parser.add_argument('--num-protacs', type=int, help='Unique product count with empirical distribution quotas; default: exhaustive')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include-originals', action='store_true')
    parser.add_argument('--report', type=Path, help='Optional JSON run statistics')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.batch_size < 1 or (args.num_protacs is not None and args.num_protacs < 1):
        parser.error('workers, batch-size and num-protacs must be positive')
    # Reject aliasing, so a run cannot overwrite the file it is reading
    paths = [args.input, args.extra_linkers, args.output, args.report]
    resolved = [p.resolve() for p in paths if p is not None]
    if len(set(resolved)) != len(resolved):
        parser.error('Input, extra-linker, output and report paths must differ')
    return args


def main(argv: Optional[list[str]] = None) -> int:
    """ Entry point.

    Args:
        argv: Arguments to parse, or None to read sys.argv.

    Returns:
        int: 0 on success, 1 on a handled error, 2 on an unfillable quota.
    """
    args = parse_args(argv)
    RDLogger.DisableLog('rdApp.*')
    try:
        ws, ls, es, originals, left, right, inputs = read_inputs(args)
        pools = make_pools(ws, ls, es, left, right, args)
        return enumerate_to_csv(args, (ws, ls, es), originals, pools, inputs)
    except (ValueError, OSError, csv.Error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
