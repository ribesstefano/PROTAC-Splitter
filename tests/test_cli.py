"""Smoke tests for the protac-splitter CLI entry point."""
import subprocess
import sys
import pytest

EXAMPLE_SMILES = (
    "N#Cc1ccc(O[C@H]2CC[C@H](NC(=O)c3ccc(N4CCN(CCCCCNc5ccc6c(c5)"
    "C(=O)N(C5CCC(=O)NC5=O)C6=O)CC4)cc3)CC2)cc1Cl"
)
PYTHON = sys.executable


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, "-m", "protac_splitter.cli"] + args,
        capture_output=True, text=True, timeout=timeout
    )


def test_cli_help():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "smiles" in r.stdout.lower()


def test_cli_heuristic():
    r = _run(["--smiles", EXAMPLE_SMILES, "--model", "heuristic"])
    assert r.returncode == 0, r.stderr
    assert "E3" in r.stdout or "Linker" in r.stdout or "POI" in r.stdout


def test_cli_heuristic_csv_format():
    r = _run(["--smiles", EXAMPLE_SMILES, "--model", "heuristic", "--output-format", "csv"])
    assert r.returncode == 0, r.stderr
    lines = [l for l in r.stdout.strip().splitlines() if l]
    assert len(lines) == 2  # header + 1 result


def test_cli_heuristic_betweenness_threshold():
    r = _run(["--smiles", EXAMPLE_SMILES, "--model", "heuristic",
              "--betweenness-threshold", "0.6"])
    assert r.returncode == 0, r.stderr


def test_cli_heuristic_capacity_weight():
    r = _run(["--smiles", EXAMPLE_SMILES, "--model", "heuristic",
              "--use-capacity-weight"])
    assert r.returncode == 0, r.stderr


def test_cli_xgboost():
    r = _run(["--smiles", EXAMPLE_SMILES, "--model", "xgboost"], timeout=120)
    assert r.returncode == 0, r.stderr
    assert "E3" in r.stdout or "Linker" in r.stdout or "POI" in r.stdout


def test_cli_missing_input_errors():
    r = _run([])
    assert r.returncode != 0
