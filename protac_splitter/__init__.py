""" PROTAC Splitter package for splitting PROTAC SMILES into substructures."""
import importlib

__version__ = "1.5.0"
__author__ = "Stefano Ribes and Anders Källberg"

# Public names are resolved on first access (PEP 562), not imported here: split_protac
# pulls in the XGBoost, graph, and datasets stacks, which take minutes to import from a
# cold shared filesystem and are not needed by e.g. `protac_splitter.chemprop`.
_LAZY_ATTRS = {
    "split_protac": "protac_splitter.protac_splitter",
    "fix_prediction": "protac_splitter.fixing_functions",
    "split_protac_graph_based": "protac_splitter.graphs.splitting_algorithms",
    "check_reassembly": "protac_splitter.evaluation",
    "split_prediction": "protac_splitter.evaluation",
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name):
    if name in _LAZY_ATTRS:
        value = getattr(importlib.import_module(_LAZY_ATTRS[name]), name)
        globals()[name] = value  # cache: later lookups skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
