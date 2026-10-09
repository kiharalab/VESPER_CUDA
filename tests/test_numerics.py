"""FFT numerics: reference spectra are complex64 and vector channels are summed in frequency space."""

import numpy as np
import pytest
from test_multi_map import BANDWIDTH, _grid

from vesper.fitter import MapFitter


def _fitter(inputs, mode):
    refs, tgt = _grid(inputs, ["a"])
    for em_map in [*refs, tgt]:
        em_map.resample_and_vec(dreso=BANDWIDTH)
    return MapFitter(refs, tgt, 30.0, mode, True, None, None, None, 2, False, None)


@pytest.mark.parametrize("mode", ["V", "C", "P", "O", "L"])
def test_reference_spectra_are_complex64(inputs, mode):
    fitter = _fitter(inputs, mode)
    assert len(fitter.ref_map_fft_lists[0]) == (3 if mode == "V" else 1)
    assert {f.dtype for f in fitter.ref_map_fft_lists[0]} == {np.dtype(np.complex64)}
