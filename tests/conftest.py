"""Shared fixtures: the synthetic inputs, and a reproducible CPU search (see run_cli.py)."""

import pytest
import run_cli
import synthetic


@pytest.fixture(autouse=True)
def reproducible_cpu_search(monkeypatch):
    run_cli.reproducible(monkeypatch.setattr)


@pytest.fixture(scope="session")
def inputs(tmp_path_factory):
    """Folder holding a.mrc, b.mrc, c.mrc, target.mrc and model.pdb (see synthetic.py)."""
    directory = tmp_path_factory.mktemp("inputs")
    synthetic.write_inputs(str(directory))
    return directory
