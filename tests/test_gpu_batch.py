"""GPU batch size: sized from free memory, halved on out-of-memory, never changing results."""

import pytest
import torch
from test_multi_map import BANDWIDTH, _grid, _poses

from vesper.fitter import MapFitter, auto_batch_size, fit_batch_size

GB = 1024**3


def fake_allocator(capacity):
    """try_batch that runs out of memory above `capacity` rotations, like cuda would"""
    tried = []

    def try_batch(n):
        tried.append(n)
        if n > capacity:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    return tried, try_batch


def test_auto_batch_is_70_percent_of_free_memory_per_stream_after_fixed_cost():
    assert auto_batch_size(8 * GB, per_rotation=GB // 10, fixed=GB, streams=4) == 4
    assert auto_batch_size(80 * GB, per_rotation=GB // 10, fixed=GB, streams=4) == 130


def test_auto_batch_is_capped_and_at_least_one():
    assert auto_batch_size(10000 * GB, GB // 10, 0, 4) == 256
    assert auto_batch_size(GB, GB, 10 * GB, 4) == 1


def test_batch_halves_until_it_fits_and_says_so(capsys):
    tried, try_batch = fake_allocator(capacity=40)
    assert fit_batch_size(256, try_batch) == 32
    assert tried == [256, 128, 64, 32]
    out = capsys.readouterr().out
    assert "GPU out of memory at batch 256: retrying with 128" in out
    assert "GPU out of memory at batch 64: retrying with 32" in out


def test_batch_that_fits_is_kept_silently(capsys):
    tried, try_batch = fake_allocator(capacity=256)
    assert fit_batch_size(37, try_batch) == 37
    assert tried == [37]
    assert capsys.readouterr().out == ""


def test_cufft_plan_failure_counts_as_out_of_memory():
    calls = []

    def try_batch(n):
        calls.append(n)
        if n > 3:
            raise RuntimeError("cuFFT error: CUFFT_ALLOC_FAILED")

    assert fit_batch_size(10, try_batch) == 2
    assert calls == [10, 5, 2]


def test_batch_of_one_that_does_not_fit_is_an_error():
    tried, try_batch = fake_allocator(capacity=0)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        fit_batch_size(4, try_batch)
    assert tried == [4, 2, 1]


def test_other_errors_are_not_retried():
    def try_batch(n):
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        fit_batch_size(8, try_batch)


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _gpu_fit(inputs, outdir, batch_size, max_rotations=None, monkeypatch=None):
    """Search two maps on the GPU; rotations above max_rotations per batch run out of memory"""
    refs, tgt = _grid(inputs, ["a", "c"])
    for em_map in [*refs, tgt]:
        em_map.resample_and_vec(dreso=BANDWIDTH)
    if max_rotations is not None:
        batch = MapFitter._rot_and_search_fft_batch

        def limited(self, angles, *args, **kwargs):
            if len(angles) > max_rotations:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return batch(self, angles, *args, **kwargs)

        monkeypatch.setattr(MapFitter, "_rot_and_search_fft_batch", limited)
    fitter = MapFitter(
        refs, tgt, 30.0, "V", True, None, None, str(inputs / "model.pdb"), 2,
        True, torch.device("cuda:0"), topn=3, outdir=outdir, ref_labels=["a", "c"],
        batch_size=batch_size,
    )  # fmt: skip
    fitter.fit()
    return [_poses(final) for final in fitter.final_lists]


@gpu
def test_gpu_results_do_not_depend_on_batch_size(inputs, tmp_path):
    small = _gpu_fit(inputs, str(tmp_path / "b3"), 3)
    large = _gpu_fit(inputs, str(tmp_path / "b64"), 64)
    auto = _gpu_fit(inputs, str(tmp_path / "auto"), None)
    assert small == large == auto


@gpu
def test_gpu_out_of_memory_halves_and_completes(inputs, tmp_path, monkeypatch, capsys):
    expected = _gpu_fit(inputs, str(tmp_path / "ok"), 3)
    capsys.readouterr()
    halved = _gpu_fit(inputs, str(tmp_path / "oom"), 64, 5, monkeypatch)
    out = capsys.readouterr().out
    assert "GPU out of memory at batch 64: retrying with 32" in out
    assert "GPU out of memory at batch 8: retrying with 4" in out
    assert halved == expected
