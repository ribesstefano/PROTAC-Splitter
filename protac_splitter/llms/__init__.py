"""LLM-based PROTAC splitting: Transformer inference and training utilities."""

__all__ = ["get_pipeline", "run_pipeline"]


def __getattr__(name: str):
    if name in __all__:
        from protac_splitter.llms.model_utils import get_pipeline, run_pipeline
        globals()["get_pipeline"] = get_pipeline
        globals()["run_pipeline"] = run_pipeline
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
