"""FastAPI application serving ``split_protac`` over HTTP.

Design notes:

* **One worker, one lock.** The strategy is fixed when the server starts, and the
  models it loads are cached in this process. Extra uvicorn workers would each load
  their own copy, so scale with batch size instead. The endpoints are plain ``def`` so
  starlette runs them in a thread pool, and a lock serialises the calls into
  ``split_protac`` (the Transformer pipeline is not safe to share between threads).
* **A bad molecule is not an error.** An unparseable or unsplittable SMILES yields
  ``null`` in place, not a 4xx for the whole batch, so the result list always lines up with the request.
* **Fail at start-up, not per request.** ``load_service`` runs one warm-up split, so a
  missing ``[transformer]`` extra or an unreachable model download stops the server
  before it listens, and the first real request does not pay for model loading.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

from protac_splitter.serve.schemas import InfoResponse, SplitRequest, SplitResponse

logger = logging.getLogger("protac_splitter.serve")

# A PROTAC any strategy splits cleanly; only used to warm the models up.
WARMUP_SMILES = (
    "N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(N4CCN(CCCCCNc5ccc6c(c5)"
    "C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)cc3)CC2)cc1Cl"
)


def _json_safe(value: Any) -> Any:
    """Map non-finite floats to ``None``: ``NaN`` is not valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _with_parts(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a result with its ``e3``/``linker``/``poi`` parsed out of the prediction.

    Every key ``split_protac`` returned is kept as is; the three substructures are
    added so a client does not have to parse the dot-separated format itself.
    """
    from protac_splitter.evaluation import split_prediction

    out = {k: _json_safe(v) for k, v in result.items()}
    pred = result.get("default_pred_n0")
    parts = (split_prediction(pred) if pred else None) or {}
    for key in ("e3", "linker", "poi"):
        out[key] = parts.get(key)
    return out


class Service:
    """A fixed ``split_protac`` configuration, plus the state shared across requests.

    Args:
        split_kwargs: Keyword arguments for ``split_protac`` (``model`` included),
            applied to every request.
    """

    def __init__(self, split_kwargs: Dict[str, Any]):
        self.split_kwargs = dict(split_kwargs)
        self.lock = threading.Lock()
        self.output_keys: List[str] = []

    def _split_batch(self, smiles: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Split already-validated SMILES, raising on any failure.

        A valid molecule the strategy cannot split comes back from ``split_protac`` as a
        dict whose ``default_pred_n0`` is ``None``; it is mapped to ``None`` here so it
        is reported and counted like any other failure.
        """
        from protac_splitter import split_protac

        return [_with_parts(r) if r.get("default_pred_n0") else None
                for r in split_protac(smiles, **self.split_kwargs)]

    def _split(self, smiles: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Split already-validated SMILES; fall back to one at a time if the batch fails.

        A molecule that makes the whole batch raise would otherwise take the other
        molecules down with it.
        """
        try:
            return self._split_batch(smiles)
        except Exception as e:
            if len(smiles) == 1:
                logger.warning(f"Could not split {smiles[0]!r}: {e}")
                return [None]
            logger.warning(f"Batch of {len(smiles)} failed ({e}); retrying one by one.")
            return [self._split([s])[0] for s in smiles]

    def split(self, smiles: List[str]) -> SplitResponse:
        """Split many PROTACs, with ``None`` in place of any that cannot be split."""
        from protac_splitter.chemoinformatics import canonize

        # `split_protac` raises for the whole list on one bad SMILES, so filter first.
        valid = [i for i, s in enumerate(smiles) if s.strip() and canonize(s) is not None]
        results: List[Optional[Dict[str, Any]]] = [None] * len(smiles)
        if valid:
            for i, r in zip(valid, self._split([smiles[i] for i in valid])):
                results[i] = r
        return SplitResponse(results=results, n_invalid=results.count(None))

    def info(self) -> InfoResponse:
        """Describe the loaded strategy."""
        try:
            from importlib.metadata import version
            package_version = version("protac-splitter")
        except Exception:
            package_version = "unknown"
        options = {k: v for k, v in self.split_kwargs.items() if k != "model"}
        return InfoResponse(model=self.split_kwargs["model"], options=options,
                            output_keys=self.output_keys,
                            protac_splitter_version=package_version)


def load_service(split_kwargs: Dict[str, Any]) -> Service:
    """Build a :class:`Service` and warm it up.

    Raises:
        RuntimeError: If the warm-up split fails or yields no split, with the cause
            attached. The usual
            causes are a missing ``[transformer]`` extra or a model that could not be
            downloaded.
    """
    service = Service(split_kwargs)
    try:
        warmup = service._split_batch([WARMUP_SMILES])[0]
    except Exception as e:
        raise RuntimeError(f"Could not load model={split_kwargs['model']!r}: {e}") from e
    if warmup is None:
        raise RuntimeError(
            f"Could not load model={split_kwargs['model']!r}: it failed to split the "
            "warm-up PROTAC.")
    service.output_keys = list(warmup)
    return service


def create_app(service: Service) -> FastAPI:
    """Build the FastAPI app around an already-loaded :class:`Service`."""
    app = FastAPI(
        title="PROTAC-Splitter",
        description=__doc__,
        version=service.info().protac_splitter_version,
    )
    app.state.service = service

    @app.get("/health")
    def health() -> dict:
        """Liveness probe: cheap, and touches no model."""
        return {"status": "ok"}

    @app.get("/info", response_model=InfoResponse)
    def info() -> InfoResponse:
        """What this server is serving."""
        return service.info()

    @app.post("/split", response_model=SplitResponse)
    def split(request: SplitRequest) -> SplitResponse:
        """Split PROTAC SMILES into E3 ligand, linker and POI warhead."""
        with service.lock:
            return service.split(request.smiles)

    return app
