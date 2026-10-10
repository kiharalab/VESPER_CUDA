"""`vesper orig` takes DiffModeler's command lines, as the VESPER_CUDA copy in DiffModeler_ET did.

The lines are DiffModeler_ET's (modeling/fit_structure_chain.py: a chain fit with one or several
maps, and the global/local refit of one map), with printf numbers and `vesper` in place of
`python VESPER_CUDA/main.py`. The GPU is stubbed and MapFitter recorded, so this tests the
parsing and what reaches the fitter.
"""

import shlex
import shutil
from typing import ClassVar

import pytest
from typer.testing import CliRunner

from vesper import cli

CHAIN_FIT = (
    "orig -a {maps} {labels} -t 0.010000 -b target.mrc -T 0.050000 -g 8.000000 "
    "-s 3.000000 -A 30.000000 -N 3 -M {mode} -gpu 0 -o out -pdbin model.pdb {ldp} "
    "-c 2 -res 5.000000 {direct_fit}"
)
REFIT = (
    "orig -a a.mrc -t 0.010000 -b target.mrc -T 0.050000 -g 8.000000 -s 3.000000 "
    "-A 5.000000 -N 4 -M {mode} -gpu 0 -o out -pdbin model.pdb {ldp} -c 2 {al} "
    "-res 5.000000"
)
CASES = {
    "several maps, an LDP file each": (
        CHAIN_FIT.format(
            maps="a.mrc,b.mrc",
            labels="-labels a,b",
            mode="V",
            ldp="-ldp l1.pdb,l2.pdb -ca model.pdb",
            direct_fit="",
        ),
        dict(maps=["a.mrc", "b.mrc"], labels=["a", "b"], ldp=["l1.pdb", "l2.pdb"]),
    ),
    "several maps, direct fit": (
        CHAIN_FIT.format(
            maps="a.mrc,b.mrc",
            labels="-labels a,b",
            mode="C",
            ldp="",
            direct_fit="--direct_fit",
        ),
        dict(maps=["a.mrc", "b.mrc"], labels=["a", "b"], mode="C"),
    ),
    "one map": (
        CHAIN_FIT.format(
            maps="a.mrc",
            labels="",
            mode="V",
            ldp="-ldp l1.pdb -ca model.pdb",
            direct_fit="",
        ),
        dict(ldp=["l1.pdb"]),
    ),
    "global refit": (
        REFIT.format(mode="V", ldp="-ca model.pdb -ldp l1.pdb", al=""),
        dict(ldp=["l1.pdb"], angle=5.0, topn=4),
    ),
    "local refit": (
        REFIT.format(mode="V", ldp="", al="-al 30.000000"),
        dict(angle=5.0, topn=4, angle_limit=30.0),
    ),
}


class _RecordingFitter:
    """Stands in for MapFitter: keeps the arguments, searches nothing."""

    calls: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def fit(self):
        pass


@pytest.mark.parametrize("case", CASES)
def test_diffmodeler_command_line(inputs, tmp_path, monkeypatch, case):
    for name in ("a.mrc", "b.mrc", "target.mrc", "model.pdb"):
        shutil.copy(inputs / name, tmp_path)
    for name in ("l1.pdb", "l2.pdb"):
        shutil.copy(inputs / "model.pdb", tmp_path / name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "setup_gpu", lambda gpu_id: (gpu_id is not None, None))
    monkeypatch.setattr(cli, "MapFitter", _RecordingFitter)
    _RecordingFitter.calls.clear()
    line, expected = CASES[case]
    expected = {
        **dict(maps=["a.mrc"], labels=None, mode="V", ldp=None, angle=30.0, topn=3),
        **dict(angle_limit=None),
        **expected,
    }

    result = CliRunner().invoke(cli.app, shlex.split(line))

    assert result.exit_code == 0, result.output
    [(args, kwargs)] = _RecordingFitter.calls
    ref_maps, _, angle, mode, _, ldp, ca, pdbin, threads, gpu = args[:10]
    assert [ref_map.mrcfile_path for ref_map in ref_maps] == expected["maps"]
    assert kwargs["ref_labels"] == expected["labels"]
    assert (angle, mode, ldp, pdbin, threads, gpu) == (
        expected["angle"],
        expected["mode"],
        expected["ldp"],
        "model.pdb",
        2,
        True,
    )
    assert ca == ("model.pdb" if expected["ldp"] else None)
    assert kwargs["topn"] == expected["topn"]
    assert kwargs["outdir"] == "out"
    assert kwargs["confine_angles"] == expected["angle_limit"]
