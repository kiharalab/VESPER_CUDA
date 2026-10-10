"""-v DIR saves both maps' resampled vectors as .npy files and exits without a search."""

import os
import subprocess
import sys

import numpy as np
import synthetic

from vesper.data.map import EMmap, unify_dims

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import pymol_vec  # noqa: E402

ARGS = ["-t", "0.05", "-T", "0.05", "-s", "3", "-g", "8"]


def test_cli_writes_coords_and_vecs_of_both_maps_and_exits(tmp_path):
    synthetic.write_inputs(str(tmp_path))
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "run_cli.py"), "orig", "-a", "a.mrc"]
        + ["-b", "target.mrc", *ARGS, "-v", "vec/sub"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Start Refining" not in proc.stdout
    names = [f"{m}_map_{k}.npy" for m in ("ref", "tgt") for k in ("coords", "vecs")]
    assert sorted(os.listdir(tmp_path / "vec" / "sub")) == sorted(names)

    # the files hold what the in-process maps hold
    ref, tgt = (EMmap(str(tmp_path / f)) for f in ("a.mrc", "target.mrc"))
    ref.set_vox_size(thr=0.05, voxel_size=3.0)
    tgt.set_vox_size(thr=0.05, voxel_size=3.0)
    unify_dims([ref, tgt], voxel_size=3.0)
    for name, emmap in (("ref_map", ref), ("tgt_map", tgt)):
        emmap.resample_and_vec(dreso=8.0)
        coords = np.load(tmp_path / "vec" / "sub" / f"{name}_coords.npy")
        vecs = np.load(tmp_path / "vec" / "sub" / f"{name}_vecs.npy")
        index = np.nonzero(emmap.new_data)
        assert coords.shape == vecs.shape == (len(index[0]), 3)
        np.testing.assert_allclose(vecs, emmap.vec[index])
        np.testing.assert_allclose(
            coords, emmap.new_orig + np.array(index).T * emmap.new_width
        )


def test_save_vectors_coordinates_are_real_space(tmp_path):
    synthetic.write_inputs(str(tmp_path))
    emmap = EMmap(str(tmp_path / "target.mrc"))
    emmap.set_vox_size(thr=0.05, voxel_size=3.0)
    unify_dims([emmap], voxel_size=3.0)
    emmap.resample_and_vec(dreso=8.0)
    emmap.save_vectors(str(tmp_path / "t"))
    coords = np.load(tmp_path / "t_coords.npy")
    # on the search grid, inside the map's box, and around the blobs' centre
    grid = (coords - emmap.new_orig) / emmap.new_width
    np.testing.assert_allclose(grid, np.round(grid))
    assert grid.min() >= 0 and grid.max() < emmap.new_dim
    centre = np.array(synthetic.TGT_ORIGIN) + 0.5 * synthetic.TGT_BOX * synthetic.VOXEL
    assert np.linalg.norm(coords.mean(axis=0) - centre) < 15


def test_saved_vectors_point_to_an_off_centre_blob(tmp_path):
    # one blob off-centre by different amounts along x, y, z: a swapped or
    # sign-flipped component of a coordinate or a vector cannot pass
    offset = np.array([8.0, -6.0, 4.0])
    path = str(tmp_path / "blob.mrc")
    synthetic.write_map(path, [(offset, 1.0)], synthetic.REF_BOX, synthetic.REF_ORIGIN)
    emmap = EMmap(path)
    emmap.set_vox_size(thr=0.05, voxel_size=3.0)
    unify_dims([emmap], voxel_size=3.0)
    emmap.resample_and_vec(dreso=8.0)
    emmap.save_vectors(str(tmp_path / "b"))
    coords = np.load(tmp_path / "b_coords.npy")
    vecs = np.load(tmp_path / "b_vecs.npy")
    blob = synthetic._centre(synthetic.REF_BOX, synthetic.REF_ORIGIN) + offset

    # the densest voxel: its coordinate is origin + index * width in x, y, z order
    index = np.unravel_index(np.argmax(emmap.new_data), emmap.new_data.shape)
    top = emmap.new_orig + np.array(index) * emmap.new_width
    assert np.any(np.all(np.isclose(coords, top), axis=1))
    assert np.all(np.abs(top - blob) <= emmap.new_width)

    # every voxel more than 2 voxels from the blob has a vector toward its centre
    to_blob = blob - coords
    far = np.linalg.norm(to_blob, axis=1) > 2 * emmap.new_width
    assert far.sum() > 10
    cosine = np.sum(vecs[far] * to_blob[far], axis=1) / (
        np.linalg.norm(vecs[far], axis=1) * np.linalg.norm(to_blob[far], axis=1)
    )
    assert cosine.min() > 0.9


def test_cli_with_two_refs_names_files_by_label(tmp_path):
    synthetic.write_inputs(str(tmp_path))
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "run_cli.py"), "orig", "-a", "a.mrc,c.mrc"]
        + ["-labels", "x,y", "-b", "target.mrc", *ARGS, "-v", "vec"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    names = [f"ref_{m}_{k}.npy" for m in "xy" for k in ("coords", "vecs")]
    names += [f"tgt_map_{k}.npy" for k in ("coords", "vecs")]
    assert sorted(os.listdir(tmp_path / "vec")) == sorted(names)


def test_arrow_tip_is_at_x2_and_zero_length_is_skipped():
    tail, (base, tip) = pymol_vec.arrow_parts([1, 2, 3], [4, 0, 0], cut=0.25)
    assert np.allclose(tip, [4, 2, 3])
    assert np.allclose(np.linalg.norm(tip - base), 0.7)
    assert np.allclose(tail[0], [1, 2, 3])
    assert pymol_vec.arrow_parts([1, 2, 3], [0, 0, 0]) is None


def test_arrow_head_is_clamped_to_a_short_vector():
    tail, (base, tip) = pymol_vec.arrow_parts([0, 0, 0], [0.3, 0, 0])
    assert tail is None
    assert np.allclose(base, [0, 0, 0])
    assert np.allclose(tip, [0.3, 0, 0])


def test_notail_zero_string_keeps_the_tail():
    # PyMOL hands over "0"; showvectors converts it with int() before use
    assert pymol_vec.arrow_parts([0, 0, 0], [5, 0, 0], notail=int("0"))[0] is not None
    assert pymol_vec.arrow_parts([0, 0, 0], [5, 0, 0], notail=int("1"))[0] is None
