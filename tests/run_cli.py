"""Run the vesper CLI with a reproducible CPU search, as the tests do.

    python tests/run_cli.py orig -a ... -b ...

Upstream's CPU search is not reproducible from run to run, and on the tests' small grids its
near-ties then decide which poses come out. reproducible() changes how it runs, not what it
computes:
- MapFitter takes rotations as their threads finish, so equal scores are ranked by timing; here
  they are taken in submission order, as on the GPU.
- FFTW plans with FFTW_MEASURE, which picks among algorithms by timing them, so the (float32)
  scores move in their last bits; here it plans with FFTW_ESTIMATE, which always picks the same.
- MapFitter gives FFTW all CPUs but two (os.cpu_count() - 2), which on these grids costs far
  more than it saves; here it sees 4 CPUs, so FFTW runs on 2 threads.
"""

import concurrent.futures
import functools
import os


def _in_submission_order(fs, timeout=None):
    return list(fs)


def reproducible(setattr=setattr, in_order=True):
    """Patch the CPU search as described above (pytest passes monkeypatch.setattr)."""
    from pyfftw.interfaces import numpy_fft

    if in_order:
        setattr(concurrent.futures, "as_completed", _in_submission_order)
    setattr(os, "cpu_count", lambda: 4)
    for name in ("rfftn", "irfftn"):
        transform = functools.partial(
            getattr(numpy_fft, name), planner_effort="FFTW_ESTIMATE"
        )
        setattr(numpy_fft, name, transform)


if __name__ == "__main__":
    reproducible()

    from vesper.cli import app

    app()
