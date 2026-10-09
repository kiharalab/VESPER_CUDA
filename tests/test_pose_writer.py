"""Poses written for .pdb and .cif input.

fitter passes rot_mtx = inv(M) for the pose x' = M x + t; the .pdb path (Biopython's
structure.transform) gives that, and the .cif path has to give the same coordinates.
"""

import numpy as np
import pytest
import synthetic
from scipy.spatial.transform import Rotation as R

from vesper.data.io import save_rotated_pdb


def _xyz(path):
    """x, y, z of each atom line (split into fields: the layouts of the two writers differ)."""
    with open(path) as f:
        rows = [line.split() for line in f if line.startswith("ATOM")]
    return np.array([[float(v) for v in fields[-6:-3]] for fields in rows])


@pytest.mark.parametrize("ext", ["pdb", "cif"])
def test_pose_is_m_x_plus_t(tmp_path, ext):
    coords = synthetic.model_coords()
    model = str(tmp_path / f"model.{ext}")
    getattr(synthetic, "write_model_cif" if ext == "cif" else "write_model")(
        model, coords
    )
    m = R.from_euler("xyz", [30.0, 50.0, 70.0], degrees=True)
    t = np.array([1.5, -2.0, 3.0])
    pose = str(tmp_path / "pose")
    save_rotated_pdb(model, m.inv().as_matrix(), t, pose, 0)

    np.testing.assert_allclose(_xyz(pose + ".pdb"), m.apply(coords) + t, atol=1e-3)


def test_pdb_and_cif_poses_agree(tmp_path):
    coords = synthetic.model_coords()
    synthetic.write_model(str(tmp_path / "model.pdb"), coords)
    synthetic.write_model_cif(str(tmp_path / "model.cif"), coords)
    rot = R.from_euler("xyz", [30.0, 50.0, 70.0], degrees=True).inv().as_matrix()
    t = np.array([1.5, -2.0, 3.0])
    for ext in ("pdb", "cif"):
        save_rotated_pdb(str(tmp_path / f"model.{ext}"), rot, t, str(tmp_path / ext), 0)

    np.testing.assert_allclose(
        _xyz(str(tmp_path / "cif.pdb")), _xyz(str(tmp_path / "pdb.pdb")), atol=1e-3
    )
