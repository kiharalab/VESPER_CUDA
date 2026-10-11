"""get_score compares the target's own densities with the reference's."""

import warnings
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


def test_target_density_on_a_negative_reference_voxel_is_in_the_total():
    ref = make_block()
    ref[0, 0, 0] = -1.0
    ref_map, vec = make_ref(ref)
    tgt = 2 * ref
    tgt[0, 0, 0] = 1.0
    _, overlap, _, _, Nm, total, _ = get_score(ref_map, tgt, vec, (0, 0, 0))
    # The target voxel is in the union: the reference has no density there.
    assert (Nm, total) == (64, 65)
    assert np.isclose(overlap, 64 / 65)


def test_overlap_is_zero_without_positive_voxels():
    ref = np.zeros((6, 6, 6), dtype=np.float32)
    ref[0, 0, 0] = -1.0
    vec = np.zeros(ref.shape + (3,), dtype=np.float32)
    ref_map = SimpleNamespace(ave=0.0, std=0.0, std_norm_ave=0.0, new_data=ref, vec=vec)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, overlap, _, _, Nm, total, _ = get_score(
            ref_map, np.zeros_like(ref), vec, (0, 0, 0)
        )
    assert (overlap, Nm, total) == (0.0, 0, 0)


def test_dot_and_score_array_of_known_vectors():
    ref = make_block()
    ref_map, ref_vec = make_ref(ref)
    ref_vec[:] = (1.0, 2.0, 3.0)
    tgt_vec = np.zeros_like(ref_vec)
    tgt_vec[..., 0] = np.arange(6)[:, None, None]  # x index
    tgt_vec[..., 1] = 1.0
    # Shifting by one along x pairs reference voxel x with target voxel x + 1, so the
    # product there is 1 * (x + 1) + 2 * 1 + 3 * 0 = x + 3 for x in 0..4.
    sco_arr, _, _, _, _, _, dot = get_score(ref_map, ref, tgt_vec, (1, 0, 0))
    expected = np.zeros(ref.shape)
    expected[1:] = (np.arange(1, 6) + 2)[:, None, None]
    assert np.allclose(sco_arr, expected)
    # 36 voxels per layer: 36 * (3 + 4 + 5 + 6 + 7) = 900.
    assert np.isclose(dot, 900.0)
