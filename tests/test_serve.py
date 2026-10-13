"""Tests for `protac-splitter serve` (needs the [serve] extra and httpx)."""
import subprocess
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from protac_splitter.serve.app import WARMUP_SMILES, create_app, load_service  # noqa: E402

PYTHON = sys.executable


@pytest.fixture(scope="module")
def client():
    # The heuristic needs no model download, so this stays fast and offline.
    service = load_service({"model": "heuristic", "betweenness_threshold": 0.4})
    return TestClient(create_app(service))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_info(client):
    body = client.get("/info").json()
    assert body["model"] == "heuristic"
    assert body["options"]["betweenness_threshold"] == 0.4
    assert {"default_pred_n0", "e3", "linker", "poi"} <= set(body["output_keys"])


def test_split_aligns_with_request_and_nulls_invalid(client):
    # "C" is valid but unsplittable: split_protac returns it with no prediction.
    smiles = [WARMUP_SMILES, "not_a_smiles", "", WARMUP_SMILES, "C"]
    body = client.post("/split", json={"smiles": smiles}).json()
    results = body["results"]
    assert len(results) == len(smiles)
    assert results[1] is None and results[2] is None and results[4] is None
    assert body["n_invalid"] == 3
    for i in (0, 3):
        assert results[i]["model_name"] == "Heuristic"
        assert all(results[i][k] for k in ("e3", "linker", "poi"))
    assert results[0] == results[3]


def test_split_empty_batch(client):
    assert client.post("/split", json={"smiles": []}).json() == {"results": [], "n_invalid": 0}


def test_split_rejects_malformed_body(client):
    assert client.post("/split", json={"smiles": "CCO"}).status_code == 422
    assert client.post("/split", json={"smiles": [], "extra": 1}).status_code == 422


def test_load_service_fails_fast_on_unloadable_model(monkeypatch):
    import protac_splitter

    def boom(*args, **kwargs):
        raise ImportError("no transformer")

    monkeypatch.setattr(protac_splitter, "split_protac", boom)
    with pytest.raises(RuntimeError, match="Could not load model='heuristic'"):
        load_service({"model": "heuristic"})


def test_serve_help_works():
    r = subprocess.run([PYTHON, "-m", "protac_splitter.cli", "serve", "--help"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "--port-file" in r.stdout and "--betweenness-threshold" in r.stdout


def test_serve_names_the_extra_when_missing():
    code = (
        "import sys; sys.modules['uvicorn'] = None; sys.argv = ['protac-splitter', 'serve'];"
        "from protac_splitter.cli import main; main()"
    )
    r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    assert "protac-splitter[serve]" in r.stderr
