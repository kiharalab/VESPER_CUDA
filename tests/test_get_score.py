"""get_score compares the target's own densities with the reference's."""

from types import SimpleNamespace

import numpy as np

from vesper.utils.utils import get_score


def test_cc_uses_the_target_densities():
    rng = np.random.default_rng(0)
    ref = np.zeros((6, 6, 6), dtype=np.float32)
    ref[1:5, 1:5, 1:5] = rng.uniform(0.5, 1.0, (4, 4, 4))
    vec = np.zeros(ref.shape + (3,), dtype=np.float32)
    ref_map = SimpleNamespace(
        ave=float(ref[ref > 0].mean()),
        std=float(np.linalg.norm(ref[ref > 0])),
        std_norm_ave=float(np.linalg.norm(ref[ref > 0] - ref[ref > 0].mean())),
        new_data=ref,
        vec=vec,
    )
    # A target proportional to the reference correlates perfectly; trimming it with the
    # reference's values instead gave 0.5 here.
    _, _, cc, _, _, _, _ = get_score(ref_map, 2 * ref, vec, (0, 0, 0))
    assert np.isclose(cc, 1.0)
