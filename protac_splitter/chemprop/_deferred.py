"""Import ``chemprop`` without loading its optional heavy dependencies.

chemprop's package ``__init__`` imports every submodule, and with them three optional
dependencies that inference never uses but that take minutes to load from a cold
network filesystem: ``cuik_molmaker`` (GPU featurizer), ``astartes`` (dataset splitting;
pulls in aimsim, mordred, and matplotlib) and ``descriptastorus`` (RDKit2D descriptor
features). chemprop only touches them inside function bodies, so while chemprop is
being imported they are stood in for by placeholders that import the real thing on
first use; afterwards the placeholders are removed from ``sys.modules``, so any later
``import astartes`` gets the real package.
"""
import importlib
import sys
import types

# Module chemprop imports -> names it binds from it with ``from module import name``.
# A placeholder name is "module" for a module and "module:attr" for any other object.
_DEFERRED = {
    "cuik_molmaker": (),
    "astartes": ("train_test_split", "train_val_test_split"),
    "astartes.molecules": ("train_test_split_molecules", "train_val_test_split_molecules"),
    "descriptastorus.descriptors": ("rdDescriptors", "rdNormalizedDescriptors"),
}
_SUBMODULES = {"descriptastorus.descriptors"}  # its names are submodules, not functions


class _Deferred(types.ModuleType):
    """Stands in for a module (attribute access) or a function (calls) until first use."""

    def _resolve(self):
        module, _, attr = self.__name__.partition(":")
        real = importlib.import_module(module)
        return getattr(real, attr) if attr else real

    def __getattr__(self, name):
        # Dunders stay unresolved: the import system probes e.g. __path__ and __spec__.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)


def import_chemprop():
    """Import and return ``chemprop``, deferring the dependencies listed in ``_DEFERRED``."""
    if "chemprop" in sys.modules:
        return sys.modules["chemprop"]
    installed = {}
    for module, names in _DEFERRED.items():
        if module in sys.modules:  # already really imported: nothing to save
            continue
        placeholder = _Deferred(module)
        for name in names:
            sep = "." if module in _SUBMODULES else ":"
            setattr(placeholder, name, _Deferred(f"{module}{sep}{name}"))
        installed[module] = placeholder
    sys.modules.update(installed)
    try:
        return importlib.import_module("chemprop")
    finally:
        for module, placeholder in installed.items():
            if sys.modules.get(module) is placeholder:
                del sys.modules[module]
