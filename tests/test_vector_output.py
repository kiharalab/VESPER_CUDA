"""-v DIR saves both maps' resampled vectors as .npy files and exits without a search."""

import os
import subprocess
import sys

import numpy as np
import synthetic

from vesper.data.map import EMmap, unify_dims

HERE = os.path.dirname(os.path.abspath(__file__))
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
