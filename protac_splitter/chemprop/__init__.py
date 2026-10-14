"""ChemProp bond-classification model for splitting PROTACs (needs the ``[chemprop]`` extra)."""
from protac_splitter.chemprop.model import DEFAULT_REPO_ID, ChemPropSplitter

__all__ = ["ChemPropSplitter", "DEFAULT_REPO_ID"]
