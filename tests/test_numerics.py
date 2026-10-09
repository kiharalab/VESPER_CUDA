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


@pytest.mark.parametrize("mode", ["V", "C"])
def test_channels_summed_before_one_inverse_transform(inputs, mode):
    fitter = _fitter(inputs, mode)
    ref = fitter.ref_map_fft_lists[0]
    rng = np.random.default_rng(0)
    tgt = [
        (rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)).astype(np.complex64)
        for f in ref
    ]
    channels = fitter._fft_get_prod_list(ref, tgt)
    (summed,) = fitter._fft_get_prod_list(ref, tgt, sum_channels=True)
    assert len(channels) == len(ref)
    np.testing.assert_allclose(
        summed, np.sum(channels, axis=0), rtol=1e-4, atol=1e-4 * np.abs(summed).max()
    )
