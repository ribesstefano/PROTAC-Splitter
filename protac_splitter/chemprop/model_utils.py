import torch
from torch.nn import functional as F

from protac_splitter.chemprop._deferred import import_chemprop

import_chemprop()  # before `chemprop.nn`, so chemprop's optional dependencies are deferred
from chemprop.nn import CrossEntropyLoss  # noqa: E402


class WeightedCrossEntropy(CrossEntropyLoss):
    """One task, three class weights; task weights are NOT class weights.

    Only the training loss: inference ignores it, but the bond head needs a
    criterion to be built, and training checkpoints pickle it by reference.
    """
    def __init__(self, class_weights=(1., 1., 1.)):
        super().__init__()
        self.register_buffer('class_weights', torch.as_tensor(class_weights, dtype=torch.float32))

    def _calc_unreduced_loss(self, preds, targets, *args):
        return F.cross_entropy(preds.transpose(1, 2), targets.long(),
                               weight=self.class_weights, reduction='none')
