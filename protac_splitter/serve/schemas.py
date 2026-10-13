"""Request and response models for the PROTAC-Splitter server."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

_EXAMPLE_SMILES = (
    "N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(N4CCN(CCCCCNc5ccc6c(c5)"
    "C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)cc3)CC2)cc1Cl"
)

_EXAMPLE_RESULT = {
    "SMILES": _EXAMPLE_SMILES,
    "default_pred_n0": (
        "O=C1CCC(N2C(=O)c3ccc([*:2])cc3C2=O)C(=O)N1."
        "O=C(c1ccc(N2CCN(CCCCCN[*:2])CC2)cc1)[*:1]."
        "N#Cc1ccc(O[C@H]2CC[C@H](N[*:1])CC2)cc1Cl"
    ),
    "model_name": "Heuristic",
    "e3": "O=C1CCC(N2C(=O)c3ccc([*:2])cc3C2=O)C(=O)N1",
    "linker": "O=C(c1ccc(N2CCN(CCCCCN[*:2])CC2)cc1)[*:1]",
    "poi": "N#Cc1ccc(O[C@H]2CC[C@H](N[*:1])CC2)cc1Cl",
}


class SplitRequest(BaseModel):
    """PROTAC SMILES to split with the strategy the server was started with."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"smiles": [_EXAMPLE_SMILES]}})

    smiles: List[str] = Field(..., description="PROTAC SMILES to split, in order.")


class SplitResponse(BaseModel):
    """One entry per input SMILES, in order.

    Each entry is the dict ``protac_splitter.split_protac`` returns for that molecule
    (so ``model="adaptive"`` also carries ``heuristic_params``, ``n_flags`` and
    ``review_reasons``), plus the parsed ``e3``, ``linker`` and ``poi`` substructures.
    An entry is ``null`` where the SMILES could not be parsed or split, so the list
    always lines up with the request.
    """

    model_config = ConfigDict(json_schema_extra={"example": {
        "results": [_EXAMPLE_RESULT], "n_invalid": 0}})

    results: List[Optional[Dict[str, Any]]] = Field(
        ..., description="Per input SMILES, the split result, or `null` if it failed.")
    n_invalid: int = Field(0, description="How many entries are `null`.", examples=[0])


class InfoResponse(BaseModel):
    """What this server is serving, for a client to check it reached the right one."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "model": "adaptive",
        "options": {"betweenness_threshold": 0.4, "use_capacity_weight": False},
        "output_keys": ["SMILES", "default_pred_n0", "model_name", "heuristic_params",
                        "n_flags", "review_reasons", "e3", "linker", "poi"],
        "protac_splitter_version": "1.5.0",
    }})

    model: str = Field(..., description="Splitting strategy, as passed to `split_protac`.")
    options: Dict[str, Any] = Field(
        ..., description="Every other `split_protac` argument the server applies.")
    output_keys: List[str] = Field(
        ..., description="Keys of a non-null entry in `results`, taken from the "
                         "start-up warm-up split.")
    protac_splitter_version: str = Field(..., description="Installed package version.")
