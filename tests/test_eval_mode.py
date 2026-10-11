"""-E scores the target where it sits, prints the six scores and exits without a search."""

import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import synthetic

from vesper.data.map import EMmap, unify_dims
from vesper.utils.utils import get_score

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = ["-t", "0.05", "-T", "0.05", "-s", "3", "-g", "8", "-E"]
SCORES = re.compile(
    r"Overlap:\s+(\S+) CC:\s+(\S+) PCC:\s+(\S+) N:\s+(\S+) Total:\s+(\S+) Dot:\s+(\S+)"
)


def write_moved_maps(directory):
    """A's blobs where they are in a.mrc, in a bigger box (moved.mrc), and 300 A away (far.mrc)"""
    # the bigger box starts whole voxels away, so both grids sample the same points
    origin = np.array(synthetic.REF_ORIGIN) + np.array((-8.0, -4.0, -6.0))
    centre = synthetic._centre(synthetic.REF_BOX, synthetic.REF_ORIGIN)
    offset = centre - synthetic._centre(40, origin)
    moved = [(np.asarray(o, dtype=float) + offset, h) for o, h in synthetic.BLOBS["A"]]
    synthetic.write_map(f"{directory}/moved.mrc", moved, 40, tuple(origin))
    far = np.array(synthetic.REF_ORIGIN) + np.array((300.0, 0.0, 0.0))
    blobs, box = synthetic.BLOBS["A"], synthetic.REF_BOX
    synthetic.write_map(f"{directory}/far.mrc", blobs, box, tuple(far))


def run_eval(directory, target, *extra):
    synthetic.write_inputs(str(directory))
    write_moved_maps(str(directory))
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "run_cli.py"), "orig", "-a", "a.mrc"]
        + ["-b", target, *ARGS, *extra],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout, [float(v) for v in SCORES.search(proc.stdout).groups()]


def test_map_on_itself_scores_perfectly(tmp_path):
    stdout, (overlap, cc, pcc, n, total, dot) = run_eval(tmp_path, "a.mrc")
    assert overlap == 1.0 and n == total > 0
    assert abs(cc - 1.0) < 1e-4 and abs(pcc - 1.0) < 1e-4
    assert dot > 0
    assert "Start Refining" not in stdout and "Final Results" not in stdout


def test_same_density_in_another_box_scores_perfectly(tmp_path):
    _, (overlap, cc, pcc, n, total, _) = run_eval(tmp_path, "moved.mrc")
    assert overlap == 1.0 and n == total > 0
    assert abs(cc - 1.0) < 1e-3 and abs(pcc - 1.0) < 1e-3


def test_target_300_angstroms_away_does_not_overlap(tmp_path):
    _, (overlap, cc, pcc, n, total, dot) = run_eval(tmp_path, "far.mrc")
    assert overlap == 0 and n == 0 and total > 0
    assert cc == 0 and pcc == 0 and dot == 0


def test_misplaced_target_scores_lower_and_json_matches(tmp_path):
    _, (overlap, cc, _, n, total, _) = run_eval(tmp_path, "b.mrc", "-o", "out")
    assert 0 < overlap < 1 and n < total and cc < 0.9
    with open(tmp_path / "out" / "eval.json") as f:
        record = json.load(f)
    assert record["overlap"] == overlap
    assert record["cc"] == pytest.approx(cc, rel=1e-6)  # printed as float32
    assert record["n"] == n and record["total"] == total
    assert record["ref"] == str(tmp_path / "a.mrc")
    assert record["target"] == str(tmp_path / "b.mrc")
    assert "mode" not in record
    assert set(os.listdir(tmp_path / "out")) == {"eval.json"}


def test_structure_target_is_recorded_by_its_own_path(tmp_path):
    # With -res the target is simulated into a temporary map; eval.json names the -b file.
    run_eval(tmp_path, "model.pdb", "-res", "6", "-T", "0", "-o", "out")
    with open(tmp_path / "out" / "eval.json") as f:
        record = json.load(f)
    assert record["target"] == str(tmp_path / "model.pdb")


def test_printed_scores_are_get_score_on_the_reference_grid(tmp_path):
    _, printed = run_eval(tmp_path, "moved.mrc")
    ref, tgt = EMmap(str(tmp_path / "a.mrc")), EMmap(str(tmp_path / "moved.mrc"))
    ref.set_vox_size(thr=0.05, voxel_size=3.0)
    tgt.set_vox_size(thr=0.05, voxel_size=3.0)
    unify_dims([ref, tgt], voxel_size=3.0)
    tgt.new_cent, tgt.new_orig = ref.new_cent, ref.new_orig
    for m in (ref, tgt):
        m.resample_and_vec(dreso=8.0)
    _, overlap, cc, pcc, n, total, dot = get_score(
        ref, tgt.new_data, tgt.vec, np.array((0, 0, 0))
    )
    assert printed == pytest.approx([overlap, cc, pcc, n, total, dot], rel=1e-5)
