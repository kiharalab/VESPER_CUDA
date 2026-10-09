"""Standalone fixes: CPU results in angle order, zero score spread, empty rotation set, input checks."""

import concurrent.futures
import time
from types import SimpleNamespace

import pytest
from test_multi_map import _grid

from vesper.fitter import MapFitter


def _fitter(inputs, **kwargs):
    (ref,), tgt = _grid(inputs, ["a"])
    args = (ref, tgt, 120.0, "V", True, None, None, None, 2, False, None)
    return MapFitter(*args, **kwargs)


def test_cpu_results_come_back_in_angle_order(inputs, monkeypatch):
    # the harness hands results back in submission order; undo that
    monkeypatch.setattr(
        concurrent.futures, "as_completed", concurrent.futures._base.as_completed
    )
    fitter = _fitter(inputs)
    angles = [tuple(a) for a in fitter.angle_comb]
    delay = {a: 0.005 * (len(angles) - i) for i, a in enumerate(angles)}

    def slow_first(rot_ang, return_data, ref_ids=None):
        time.sleep(delay[tuple(rot_ang)])  # early angles finish last
        return [(1.0, (0, 0, 0))]

    seen = []
    monkeypatch.setattr(fitter, "_rot_and_search_fft", slow_first)
    monkeypatch.setattr(
        fitter, "_add_search_results", lambda lists, rot_ang, res: seen.append(rot_ang)
    )
    monkeypatch.setattr(fitter, "_finish_selected_map", lambda: None)
    fitter.fit()
    assert [tuple(a) for a in seen] == angles


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
