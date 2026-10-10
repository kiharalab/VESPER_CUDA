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


@pytest.mark.parametrize("mode", ["V", "C", "P", "O", "L"])
def test_target_spectra_are_complex64(inputs, mode):
    fitter = _fitter(inputs, mode)
    seen = []
    fft_list = fitter._fft_list
    fitter._fft_list = lambda pre: seen.append(fft_list(pre)) or seen[-1]
    fitter._rot_and_search_fft([10.0, 20.0, 30.0], False)
    assert len(seen[0]) == (3 if mode == "V" else 1)
    assert {f.dtype for f in seen[0]} == {np.dtype(np.complex64)}


@pytest.mark.parametrize("mode", ["V", "C"])
def test_channels_summed_before_one_inverse_transform(inputs, mode):
    fitter = _fitter(inputs, mode)
    ref = fitter.ref_map_fft_lists[0]
    rng = np.random.default_rng(0)
    tgt = [
        (rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)).astype(np.complex64)
        for f in ref
    ]
    # the old way: one inverse transform per channel, summed in real space
    channels = fitter._fft_get_prod_list(ref, tgt)
    (summed,) = fitter._fft_get_prod_list(ref, tgt, sum_channels=True)
    assert len(channels) == len(ref)
    old = np.sum(channels, axis=0)
    np.testing.assert_allclose(summed, old, rtol=0, atol=1e-6 * np.abs(old).max())
