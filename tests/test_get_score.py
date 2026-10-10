"""get_score compares the target's own densities with the reference's."""

from types import SimpleNamespace

import numpy as np

from vesper.utils.utils import get_score


def make_ref(ref):
    vec = np.zeros(ref.shape + (3,), dtype=np.float32)
    pos = ref[ref > 0]
    ref_map = SimpleNamespace(
        ave=float(pos.mean()),
        std=float(np.linalg.norm(pos)),
        std_norm_ave=float(np.linalg.norm(pos - pos.mean())),
        new_data=ref,
        vec=vec,
    )
    return ref_map, vec


def make_block():
    rng = np.random.default_rng(0)
    ref = np.zeros((6, 6, 6), dtype=np.float32)
    ref[1:5, 1:5, 1:5] = rng.uniform(0.5, 1.0, (4, 4, 4))
    return ref


def test_cc_uses_the_target_densities():
    ref = make_block()
    ref_map, vec = make_ref(ref)
    # A target proportional to the reference correlates perfectly; trimming it with the
    # reference's values instead gave 0.5 here.
    _, overlap, cc, pcc, Nm, total, _ = get_score(ref_map, 2 * ref, vec, (0, 0, 0))
    assert np.isclose(cc, 1.0)
    assert np.isclose(pcc, 1.0)
    assert (overlap, Nm, total) == (1.0, 64, 64)


def test_counts_of_a_shifted_target():
    ref = make_block()
    ref_map, vec = make_ref(ref)
    tgt = np.zeros_like(ref)
    tgt[1:] = ref[:-1]  # the reference moved one voxel along the first axis
    _, overlap, _, _, Nm, total, _ = get_score(ref_map, tgt, vec, (0, 0, 0))
    # 3 of the 4 layers overlap; the 16 voxels of the extra layer are in the union.
    assert (Nm, total) == (48, 80)
    assert np.isclose(overlap, 48 / 80)
    # Translating by the shift recovers the reference.
    _, overlap, cc, pcc, Nm, total, _ = get_score(ref_map, tgt, vec, (1, 0, 0))
    assert (Nm, total) == (64, 64)
    assert np.allclose([overlap, cc, pcc], 1.0)


def test_overlap_ignores_negative_reference_voxels():
    ref = make_block()
    ref[0, 0, 0] = -1.0
    ref_map, vec = make_ref(ref)
    _, overlap, _, _, Nm, total, _ = get_score(ref_map, 2 * ref, vec, (0, 0, 0))
    # Nm counts positive voxels only, so the total must too (it was 65 here).
    assert (overlap, Nm, total) == (1.0, 64, 64)
