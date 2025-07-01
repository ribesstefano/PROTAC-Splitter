""" PROTAC Splitter package for splitting PROTAC SMILES into substructures."""
from protac_splitter.protac_splitter import split_protac
from protac_splitter.fixing_functions import fix_prediction
from protac_splitter.graphs.splitting_algorithms import split_protac_graph_based
from protac_splitter.evaluation import (
    check_reassembly,
    split_prediction,
)

__version__ = "1.0.0"
__author__ = "Stefano Ribes and Anders Källberg"
