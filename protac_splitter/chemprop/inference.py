"""ChemProp 2.3.1 bond-classification inference: featurize, predict, and decode a split.

Ported from the training pipeline (``chemprop/pipeline.py``), keeping only what
inference needs. The model predicts one of three classes per undirected RDKit bond:
0 = other, 1 = the ``[*:1]`` (POI side) attachment bond, 2 = the ``[*:2]`` (E3 side)
attachment bond.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem

# Keys of a ``ChemPropSplitter`` config, i.e. the constructor arguments of the
# message-passing encoder and the bond head that determine the weight shapes.
MESSAGE_PASSING_KEYS = ('d_v', 'd_e', 'd_h', 'bias', 'depth', 'dropout', 'activation', 'undirected')
BOND_PREDICTOR_KEYS = ('n_tasks', 'n_classes', 'input_dim', 'hidden_dim', 'n_layers', 'dropout', 'activation')


def parse_protac(smiles: str) -> Optional[Chem.Mol]:
    """Parse a model input: one connected molecule, no dummy atoms or atom maps; else None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if any(a.GetAtomicNum() == 0 or a.GetAtomMapNum() for a in mol.GetAtoms()):
        return None
    return mol


def make_dataset(mols: Sequence[Chem.Mol]):
    from chemprop import data, featurizers
    # Passing the SAME mol avoids hidden atom/bond reordering by a SMILES parser, so
    # predicted bond indices refer to the mol the caller holds.
    points = [data.MolAtomBondDatapoint(mol=mol) for mol in mols]
    return data.MolAtomBondDataset(points, featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer())


def build_model(config: Dict):
    """Build an untrained ``MolAtomBondMPNN`` with the architecture described by ``config``."""
    from chemprop import nn, models
    from protac_splitter.chemprop.model_utils import WeightedCrossEntropy
    mp = nn.MABBondMessagePassing(**config['message_passing'], return_vertex_embeddings=False)
    head = nn.MulticlassClassificationFFN(
        **config['bond_predictor'],
        criterion=WeightedCrossEntropy(config.get('class_weights', (1., 1., 1.))))
    return models.MolAtomBondMPNN(message_passing=mp, bond_predictor=head, batch_norm=False)


def infer(model, loader) -> List[np.ndarray]:
    """Per-molecule bond probabilities on the model's current device; restores train/eval mode."""
    import torch
    was_training = model.training
    model.eval()
    result = []
    with torch.inference_mode():
        for batch in loader:
            batch.bmg.to(model.device)
            probs = model(batch.bmg, batch.V_d, batch.E_d, batch.X_d)[2][:, 0].cpu().numpy()
            # ChemProp already merges the two directed edges into one chemical bond.
            counts = np.bincount(batch.bmg.bond_batch[::2].cpu().numpy())
            result.extend(np.split(probs, np.cumsum(counts)[:-1]))
    model.train(was_training)
    return result


def predict_arrays(model, mols: Sequence[Chem.Mol], batch_size: int = 16) -> List[np.ndarray]:
    """Bond-class probabilities, shape ``(mol.GetNumBonds(), 3)``, for each molecule."""
    from chemprop import data
    if not mols:
        return []
    # drop_last=False: ChemProp otherwise may drop a final singleton batch.
    loader = data.build_dataloader(make_dataset(mols), batch_size=batch_size,
                                   shuffle=False, num_workers=0, drop_last=False)
    result = infer(model, loader)
    if len(result) != len(mols):
        raise RuntimeError('Prediction count mismatch')
    for mol, p in zip(mols, result):
        if len(p) != mol.GetNumBonds():
            raise RuntimeError('Bond count mismatch between prediction and molecule')
    return result


def decode_pair(mol: Chem.Mol, probs: np.ndarray) -> Optional[Dict]:
    """Maximum-likelihood one-of-each assignment, restricted to cuts yielding three parts.

    Picks distinct non-ring bonds ``(i, j)`` maximizing
    ``log(p[i,1]/p[i,0]) + log(p[j,2]/p[j,0])`` such that cutting them gives a
    ``[*:1]`` part, a ``[*:1]``/``[*:2]`` linker, and a ``[*:2]`` part. It always
    forces a pair when one exists, so it is not evidence of a correct split.
    Returns None when no such pair exists.
    """
    logp = np.log(np.clip(probs, 1e-12, 1))
    gain = logp[:, 1:] - logp[:, :1]
    bridges = [b.GetIdx() for b in mol.GetBonds() if not b.IsInRing()]
    candidates = sorted(((gain[i, 0] + gain[j, 1], i, j)
                         for i in bridges for j in bridges if i != j), reverse=True)
    for _, i, j in candidates:
        cut = Chem.FragmentOnBonds(mol, [i, j], dummyLabels=[(1, 1), (2, 2)])
        parts = Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True)
        maps = [{a.GetIsotope() for a in p.GetAtoms() if a.GetAtomicNum() == 0} for p in parts]
        if len(parts) == 3 and sorted(map(sorted, maps)) == [[1], [1, 2], [2]]:
            output = {}
            for part, labels in zip(parts, maps):
                for a in part.GetAtoms():
                    if a.GetAtomicNum() == 0:
                        a.SetAtomMapNum(a.GetIsotope())
                        a.SetIsotope(0)
                output[{frozenset({1}): 'part_1', frozenset({1, 2}): 'linker',
                        frozenset({2}): 'part_2'}[frozenset(labels)]] = Chem.MolToSmiles(part)
            return {'bond_1': int(i), 'bond_2': int(j), 'parts': output}
    return None
