"""Standalone fixes: CPU results in angle order, zero score spread, empty rotation set, input checks."""

import concurrent.futures
import re
import shutil
import time
from types import SimpleNamespace

import click
import numpy as np
import pytest
from test_multi_map import _grid
from typer.testing import CliRunner

from vesper import cli
from vesper.cli import app, validate_search_args
from vesper.fitter import MapFitter


def _fitter(inputs, **kwargs):
    (ref,), tgt = _grid(inputs, ["a"])
    args = (ref, tgt, 120.0, "V", True, None, None, None, 2, False, None)
    return MapFitter(*args, **kwargs)


def _backwards(futures):
    """as_completed finishing in reverse submission order (the old collection loop)."""
    return iter(reversed(list(futures)))


def _slow_early_first(fitter, angles):
    delay = {tuple(a): 0.005 * (len(angles) - i) for i, a in enumerate(angles)}

    def search(rot_ang, return_data, ref_ids=None):
        time.sleep(delay[tuple(rot_ang)])  # early angles finish last
        return [(1.0, (0, 0, 0))]

    return search


def test_cpu_results_come_back_in_angle_order(inputs, monkeypatch):
    monkeypatch.setattr(concurrent.futures, "as_completed", _backwards)
    fitter = _fitter(inputs)
    angles = [tuple(a) for a in fitter.angle_comb]
    seen = []
    monkeypatch.setattr(
        fitter, "_rot_and_search_fft", _slow_early_first(fitter, angles)
    )
    monkeypatch.setattr(
        fitter, "_add_search_results", lambda lists, rot_ang, res: seen.append(rot_ang)
    )
    monkeypatch.setattr(fitter, "_finish_selected_map", lambda: None)
    fitter.fit()
    assert [tuple(a) for a in seen] == angles


def test_cpu_refinement_takes_ties_in_angle_order(inputs, monkeypatch):
    monkeypatch.setattr(concurrent.futures, "as_completed", _backwards)
    fitter = _fitter(inputs)
    coarse = {"angle": (30.0, 30.0, 30.0), "score": 0.0, "vox_trans": (0, 0, 0)}
    fitter.result_list = [coarse]
    fitter.ref_map.new_data = np.zeros(1)  # set by fit(), which is skipped here
    # every neighbour scores 1.0, so the first one in the list wins the tie
    neighbours = [
        np.array([x, y, z], dtype=np.float32)
        for x in range(25, 36, 2)
        for y in range(25, 36, 2)
        for z in range(25, 36, 2)
    ]
    monkeypatch.setattr(
        fitter, "_rot_and_search_fft", _slow_early_first(fitter, neighbours)
    )
    fitter.refine(2, 1)
    assert tuple(fitter.refined_list[0]["angle"]) == tuple(neighbours[0])


@pytest.mark.parametrize("std", [0.0, 2.0])
def test_normalized_score_with_no_spread(capsys, std):
    fitter = SimpleNamespace(score_ave=1.0, score_std=std, ldp_recall_mode=False)
    item = {
        "angle": (0, 0, 0),
        "real_trans": (0.0, 0.0, 0.0),
        "score": 3.0,
        "vox_trans": (0, 0, 0),
    }
    MapFitter._print_result_item(fitter, item, 0)
    expected = "0.000000" if std == 0 else "1.000000"
    assert f"Normalized Score= {expected}\n" in capsys.readouterr().out


def test_empty_rotation_set_is_refused(inputs):
    with pytest.raises(ValueError, match=r"^No rotations to search: check -A "):
        _fitter(inputs, confine_angles=-1)


def _message(text):
    return pytest.raises(click.UsageError, match=f"^{re.escape(text)}$")


def _check(**changes):
    args = {
        "angle_spacing": 30.0,
        "refine_top": 10,
        "angle_limit": None,
        "batch_size": None,
        "pdbin": None,
    }
    validate_search_args(**{**args, **changes})


def test_default_search_args_pass():
    _check()
    _check(angle_limit=0.0, batch_size=1, refine_top=1)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"angle_spacing": 0.0}, "-A (angle spacing) must be > 0; got 0.0"),
        ({"angle_spacing": -5.0}, "-A (angle spacing) must be > 0; got -5.0"),
        ({"refine_top": 0}, "-N (models to refine) must be >= 1; got 0"),
        ({"batch_size": 0}, "-batch must be >= 1; got 0"),
        ({"angle_limit": -1.0}, "-al (angle limit) must be >= 0; got -1.0"),
        ({"pdbin": "gone.pdb"}, "-pdbin gone.pdb does not exist"),
    ],
)
def test_bad_search_args(change, message):
    with _message(message):
        _check(**change)


def test_cli_refuses_a_missing_pdbin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        ["orig", "-a", "a.mrc", "-b", "t.mrc", "-pdbin", "gone.pdb"],
    )
    assert result.exit_code == 2
    assert "-pdbin gone.pdb does not exist" in result.stderr


class _NoSearch:
    """Stands in for MapFitter: searches nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def fit(self):
        pass


@pytest.mark.parametrize("out", [[], ["-o", "out"]])
def test_pdbin_stays_optional(inputs, tmp_path, monkeypatch, out):
    for name in ("a.mrc", "target.mrc"):
        shutil.copy(inputs / name, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "MapFitter", _NoSearch)
    args = ["orig", "-a", "a.mrc", "-b", "target.mrc", "-t", "0.05", "-T", "0.05"]
    result = CliRunner().invoke(app, [*args, "-s", "3", "-g", "8", *out])
    assert result.exit_code == 0, result.output
    assert "No input PDB file, skipping transformation" in result.stdout
