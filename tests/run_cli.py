"""Run the vesper CLI with a reproducible CPU search, as the tests do.

    python tests/run_cli.py orig -a ... -b ...

Upstream's CPU search ranks rotations as their threads finish, so equal scores are ordered by
timing and, on the tests' small grids, near-ties then decide which poses come out. reproducible()
takes them in submission order instead, as on the GPU; it changes how the search runs, not what it
computes. (FFTW is planned without timing and runs one thread per transform, set by MapFitter itself.)
"""

import concurrent.futures


def _in_submission_order(fs, timeout=None):
    return list(fs)


def reproducible(setattr=setattr, in_order=True):
    """Patch the CPU search as described above (pytest passes monkeypatch.setattr)."""
    if in_order:
        setattr(concurrent.futures, "as_completed", _in_submission_order)


if __name__ == "__main__":
    reproducible()

    from vesper.cli import app

    app()
