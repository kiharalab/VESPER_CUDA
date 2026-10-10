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


def _old_angles(interval, limit):
    """Frozen copy of c8561d2's _calc_angle_comb (x, y and z share one -al array)."""
    from scipy.spatial.transform import Rotation as R

    if limit is not None:
        x_angle = y_angle = z_angle = np.arange(-limit, limit + 1, interval)
    else:
        x_angle = np.arange(0, 360, interval)
        y_angle = np.arange(0, 360, interval)
        z_angle = np.arange(0, 181, interval)
    x_angle[x_angle < 0] += 360
    y_angle[y_angle < 0] += 360
    z_angle[z_angle < 0] += 180
    angle_comb = np.array(np.meshgrid(x_angle, y_angle, z_angle)).T.reshape(-1, 3)
    out, seen = [], set()
    for ang in angle_comb:
        quat = tuple(np.round(R.from_euler("xyz", ang, degrees=True).as_quat(), 4))
        if quat not in seen:
            seen.add(quat)
            out.append(ang)
    return np.asarray(out)


# No behaviour change is intended: this passes on c8561d2 and on the clearer code.
@pytest.mark.parametrize(
    ("interval", "limit"),
    [(10, 20), (5, 15), (30, 60), (10, 0), (30, 200), (7, 20), (12, 190), (30, None)],
)
def test_ordered_angles_match_c8561d2(interval, limit):
    np.testing.assert_array_equal(
        _angles(interval, limit), _old_angles(interval, limit)
    )
