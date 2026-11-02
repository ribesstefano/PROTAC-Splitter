import importlib

# Resolved on first access (PEP 562), not imported here: importing any single submodule,
# e.g. `substructure_extraction` from `fixing_functions`, runs this file first, and loading
# the whole curation pipeline (curation.py and everything it imports) just to reach one
# helper is slow on a cold shared filesystem.
_LAZY_ATTRS = {
    'update_dictionary': 'protac_splitter.data.curation.mapping_utils',
    'split_protacs': 'protac_splitter.data.curation.curation',
    'iterative_protac_splitting': 'protac_splitter.data.curation.curation',
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
