"""-nodup drops a pose within 30 degrees (of rotation) and a short walk of a better kept one."""

from types import SimpleNamespace

import numpy as np

from vesper.fitter import MapFitter

DIM = 32


def kept(*poses):
    """Run the duplicate removal on (angle, translation) poses, best first."""
    fit = object.__new__(MapFitter)
    fit.ang_interval = 10
    fit.tgt_map = SimpleNamespace(new_dim=DIM)
    fit.result_list = [
        {"angle": np.array(a, dtype=float), "vox_trans": np.array(t)} for a, t in poses
    ]
    fit._remove_dup_results()
    return [tuple(r["angle"]) for r in fit.result_list]


def test_euler_equivalent_pair_is_one_pose():
    # (x, y, z) = (x + 180, 180 - y, z + 180), here with z = 360 wrapped to 0
    a, b = (60, 40, 180), (240, 140, 0)
    assert kept((a, (5, 5, 5)), (b, (6, 5, 5))) == [a]
    assert kept((a, (5, 5, 5)), (b, (30, 30, 30))) == [a, b]


def test_gimbal_lock_pose_written_two_ways_is_one_pose():
    # at y = 90 only x - z counts
    a, b = (30, 90, 60), (90, 90, 120)
    assert kept((a, (5, 5, 5)), (b, (6, 5, 5))) == [a]


def test_thirty_degrees_is_a_duplicate_forty_is_not():
    a = (60, 0, 0)
    assert kept((a, (5, 5, 5)), ((90, 0, 0), (6, 5, 5))) == [a]
    assert kept((a, (5, 5, 5)), ((100, 0, 0), (6, 5, 5))) == [a, (100, 0, 0)]


def test_a_pose_is_compared_with_every_kept_pose():
    a, b = ((60, 40, 30), (5, 5, 5)), ((70, 40, 30), (60, 5, 5))
    c = ((60, 40, 30), (5, 6, 5))  # a's rotation, near a's translation
    assert kept(a, b, c) == [a[0], b[0]]
