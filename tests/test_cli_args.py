"""-a, -labels, -ldp, -ca and --direct_fit of `vesper orig`: parsing and checks."""

import os
import re
import shutil
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from typer.testing import CliRunner

from vesper import cli
from vesper.cli import app, check_ref_grids, validate_ref_args


def _raises(message):
    return pytest.raises(ValueError, match=f"^{re.escape(message)}$")


@pytest.mark.parametrize("maps", ["", ",", ",,"])
def test_empty_a(maps):
    with _raises("Empty -a"):
        validate_ref_args(maps)


def test_one_map_needs_no_label():
    assert validate_ref_args("a.mrc") == (["a.mrc"], None, None)


def test_empty_entries_are_dropped():
    assert validate_ref_args("a.mrc,,b.mrc,", "x,,y,") == (
        ["a.mrc", "b.mrc"],
        ["x", "y"],
        None,
    )


def test_several_maps_need_labels():
    with _raises("-labels required when -a has multiple paths"):
        validate_ref_args("a.mrc,b.mrc")


def test_one_label_per_map():
    with _raises("-labels has 1 entries but -a has 2 refs"):
        validate_ref_args("a.mrc,b.mrc", "x")


@pytest.mark.parametrize(
    ("labels", "bad"),
    [("x/y,z", "['x/y']"), ("..,z", "['..']"), ("a..b,z", "['a..b']")],
)
def test_labels_stay_inside_the_output_folder(labels, bad):
    with _raises(f"-labels must not contain '/' or '..': {bad}"):
        validate_ref_args("a.mrc,b.mrc", labels)


def test_labels_are_unique():
    with _raises("-labels must be unique; got ['x', 'x']"):
        validate_ref_args("a.mrc,b.mrc", "x,x")


def test_label_of_one_map_is_checked_too():
    with _raises("-labels must not contain '/' or '..': ['x/y']"):
        validate_ref_args("a.mrc", "x/y")


@pytest.mark.parametrize(
    ("ldp", "ca"), [("l.pdb", None), (None, "ca.pdb"), ("l.pdb", "ca.pdb")]
)
def test_direct_fit_refuses_ldp_and_ca(ldp, ca):
    with _raises("-ldp/-ca incompatible with --direct_fit"):
        validate_ref_args("a.mrc,b.mrc", "x,y", ldp, ca, direct_fit=True)


def test_direct_fit_alone_is_fine():
    assert validate_ref_args("a.mrc,b.mrc", "x,y", direct_fit=True)[0] == [
        "a.mrc",
        "b.mrc",
    ]


@pytest.mark.parametrize("ldp", ["l.pdb", "l1.pdb,l2.pdb"])
def test_ldp_is_one_file_or_one_per_map(ldp):
    _, _, ldp_paths = validate_ref_args("a.mrc,b.mrc", "x,y", ldp, "ca.pdb")
    assert ldp_paths == ldp.split(",")


@pytest.mark.parametrize(("ldp", "n"), [("l1,l2,l3", 3), (",", 0)])
def test_ldp_count(ldp, n):
    with _raises(f"-ldp must have 1 entry or 2 entries; got {n}"):
        validate_ref_args("a.mrc,b.mrc", "x,y", ldp, "ca.pdb")


def _ref(cent, xwidth=2.0):
    return SimpleNamespace(new_cent=np.array(cent, dtype=float), xwidth=xwidth)


def test_maps_on_one_grid_pass():
    check_ref_grids([_ref([0, 0, 0]), _ref([1.4, 0, -1.4])], voxel_spacing=3.0)


def test_maps_centred_apart_are_refused():
    with pytest.raises(ValueError, match=r"^ref 1 is centred differently from ref 0 "):
        check_ref_grids([_ref([0, 0, 0]), _ref([0, 1.6, 0])], voxel_spacing=3.0)


def test_maps_of_other_voxel_size_are_refused():
    with _raises(
        "ref 1 has voxel size 1.0 but ref 0 has 2.0: multiple -a maps must share one grid"
    ):
        check_ref_grids([_ref([0, 0, 0]), _ref([0, 0, 0], 1.0)], voxel_spacing=3.0)


def test_missing_maps_exit_1_naming_each(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["orig", "-a", "gone.mrc,also.mrc", "-labels", "x,y", "-b", "t.mrc"]
    )
    assert result.exit_code == 1
    assert "[ERROR] ref map missing or unreadable: gone.mrc" in result.stderr
    assert "[ERROR] ref map missing or unreadable: also.mrc" in result.stderr


def test_cli_raises_the_check_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["orig", "-a", "a.mrc,b.mrc", "-b", "t.mrc"])
    assert isinstance(result.exception, ValueError)
    assert str(result.exception) == "-labels required when -a has multiple paths"


class _RecordingFitter:
    """Stands in for MapFitter: keeps the arguments, searches nothing."""

    calls: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def fit(self):
        pass


@pytest.mark.parametrize("ldp", ["l1.pdb,l2.pdb", "l1.pdb"])
def test_cli_hands_maps_labels_and_ldp_files_to_the_fitter(
    inputs, tmp_path, monkeypatch, ldp
):
    for name in ("a.mrc", "b.mrc", "target.mrc", "model.pdb"):
        shutil.copy(inputs / name, tmp_path)
    for name in ("l1.pdb", "l2.pdb"):
        shutil.copy(inputs / "model.pdb", tmp_path / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "MapFitter", _RecordingFitter)
    _RecordingFitter.calls.clear()

    result = CliRunner().invoke(
        app,
        [
            *("orig", "-a", "a.mrc,b.mrc", "-b", "target.mrc", "-labels", "x,y"),
            *("-ldp", ldp, "-ca", "model.pdb"),
            *("-t", "0.05", "-T", "0.05", "-s", "3", "-g", "8"),
        ],
    )

    assert result.exit_code == 0, result.output
    (args, kwargs), *_ = _RecordingFitter.calls
    ref_maps, _, _, _, _, ldp_paths, ca_file = args[:7]
    assert [r.mrcfile_path for r in ref_maps] == ["a.mrc", "b.mrc"]
    assert ldp_paths == ldp.split(",")
    assert ca_file == "model.pdb"
    assert kwargs["ref_labels"] == ["x", "y"]
    out = result.stdout
    assert "LDP Recall Reranking Enabled" in out
    abs_ldp = ",".join(os.path.abspath(p) for p in ldp.split(","))
    assert f"LDP_PDB_file: {abs_ldp}\n" in out
    here = os.getcwd()
    assert f"Reference_Map_Paths: {here}/a.mrc,{here}/b.mrc\n" in out
    assert "Reference_Map_Labels: x,y\n" in out
    assert "Reference_Map_Path:" not in out
