"""Graph-based PROTAC splitting algorithms and edge classifiers."""
import importlib

# Resolved on first access (PEP 562) so that importing any `protac_splitter.graphs.*`
# submodule does not also load the XGBoost / scikit-learn stack of `edge_classifier`.
_LAZY_ATTRS = {
    "split_protac_graph_based": "protac_splitter.graphs.splitting_algorithms",
    "split_protac_with_betweenness_centrality": "protac_splitter.graphs.splitting_algorithms",
    "split_protac_with_graphs_parallel": "protac_splitter.graphs.splitting_algorithms",
    "GraphEdgeClassifier": "protac_splitter.graphs.edge_classifier",
    "get_representative_e3s_fp": "protac_splitter.graphs.clustering",
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
