"""Standalone fixes: CPU results in angle order, zero score spread, empty rotation set, input checks."""

import concurrent.futures
import time

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
