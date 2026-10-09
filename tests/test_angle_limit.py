"""-al searches -al..al degrees about every axis, each negative angle wrapped by 360."""

import types

import numpy as np
import pytest

from vesper.fitter import MapFitter


def _angles(interval, limit):
    fitter = types.SimpleNamespace(
        ang_interval=interval, confine_angles=limit, angle_comb=[]
    )
    MapFitter._calc_angle_comb(fitter)
    return np.asarray(fitter.angle_comb)


@pytest.mark.parametrize(("interval", "limit"), [(10, 20), (5, 15), (30, 60)])
def test_every_axis_is_searched_from_minus_al_to_al(interval, limit):
    angles = _angles(interval, limit)
    axis = np.arange(-limit, limit + 1, interval) % 360
    assert set(angles.flatten()) <= set(axis)
    assert angles.min() >= 0
    assert all(set(angles[:, i]) <= set(axis) for i in range(3))
    # one pose per distinct rotation: a 0..limit and a -limit..0 angle both occur
    assert {0, limit, 360 - limit} <= set(angles.flatten())
