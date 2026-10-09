"""-R sets the refinement step (1 or 2 degrees) within the fixed +-5 degrees."""

import os

import pytest
import synthetic
from test_golden import CASES, COMMON, golden, run
from typer.testing import CliRunner

from vesper.cli import app
from vesper.fitter import MapFitter


def refine_searches(tmp_path, monkeypatch, *extra):
    """Run `vesper orig -N 2`; return the searches made inside refine() and the poses."""
    synthetic.write_inputs(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    counts, poses = [], []
    search, refine = MapFitter._rot_and_search_fft, MapFitter.refine
    state = {"in_refine": 0}

    def count_search(self, rot_ang, *args, **kwargs):
        if state["in_refine"]:
            poses.append(tuple(rot_ang))
        return search(self, rot_ang, *args, **kwargs)

    def count_refine(self, *args, **kwargs):
        state["in_refine"] = 1
        before = len(poses)
        refine(self, *args, **kwargs)
        counts.append(len(poses) - before)
        state["in_refine"] = 0

    monkeypatch.setattr(MapFitter, "_rot_and_search_fft", count_search)
    monkeypatch.setattr(MapFitter, "refine", count_refine)
    args = ["orig", "-a", "a.mrc", "-b", "target.mrc", *COMMON, *extra]
    args[args.index("-N") + 1] = "2"
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    return counts, poses, result.output


@pytest.mark.parametrize(
    ("extra", "per_top"), [([], 216), (["-R", "2"], 216), (["-R", "1"], 1331)]
)
def test_candidates_per_top_pose(tmp_path, monkeypatch, extra, per_top):
    counts, poses, _ = refine_searches(tmp_path, monkeypatch, *extra)
    assert counts == [2 * per_top]
    # +-5 degrees around each coarse pose, in steps of -R (mod 360)
    step = 1 if "1" in extra else 2
    first = poses[:per_top]
    assert len({p[0] for p in first}) == 10 // step + 1


@pytest.mark.parametrize("bad", ["0", "3"])
def test_step_must_be_1_or_2(tmp_path, bad):
    result = CliRunner().invoke(app, ["orig", "-a", "a.mrc", "-b", "b.mrc", "-R", bad])
    assert result.exit_code != 0


def test_explicit_default_matches_golden(tmp_path):
    assert run(tmp_path, [*CASES["A_V_nodup"], "-R", "2"]) == golden("A_V_nodup")


def test_step_1_changes_only_the_refined_poses(tmp_path):
    printout = run(tmp_path, [*CASES["A_V_nodup"], "-R", "1"])
    assert "Refine_step: 1\n" in printout
    gold = golden("A_V_nodup")
    marker = "###Preliminary Results Summary###"
    assert (
        printout.split(marker)[1].split("###Start Refining###")[0]
        == gold.split(marker)[1].split("###Start Refining###")[0]
    )
    assert sorted(os.listdir(tmp_path / "out")) == ["PDB", "score.pkl"]
