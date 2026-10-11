"""Poses written for .pdb and .cif input.

fitter passes rot_mtx = inv(M) for the pose x' = M x + t; the .pdb path (Biopython's
structure.transform) gives that, and the .cif path has to give the same coordinates.
"""

import numpy as np
import pytest
import synthetic
from Bio.PDB import PDBParser
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


def test_cif_pose_lines_are_pdb_columns(tmp_path):
    coords = synthetic.model_coords()
    synthetic.write_model_cif(str(tmp_path / "model.cif"), coords)
    rot = R.from_euler("xyz", [30.0, 50.0, 70.0], degrees=True).inv().as_matrix()
    t = np.array([1.5, -2.0, 3.0])
    pose = str(tmp_path / "pose")
    save_rotated_pdb(str(tmp_path / "model.cif"), rot, t, pose, 0, occupancy=0.5)

    with open(pose + ".pdb") as f:
        lines = [line.rstrip("\n") for line in f if line.startswith("ATOM")]
    x, y, z = R.from_matrix(rot).inv().apply(coords[0]) + t
    assert lines[0] == (
        f"ATOM      1  CA  ALA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  0.50  0.00           C"
    )
    # the same structure through Biopython: same atoms, same coordinates
    atoms = list(PDBParser(QUIET=True).get_structure("p", pose + ".pdb").get_atoms())
    assert [a.get_name() for a in atoms] == ["CA"] * len(coords)
    assert [a.get_parent().get_id()[1] for a in atoms] == [1, 2, 3, 4]
    assert [a.get_parent().get_parent().get_id() for a in atoms] == ["A"] * 4
    np.testing.assert_allclose(
        [a.coord for a in atoms], R.from_matrix(rot).inv().apply(coords) + t, atol=1e-3
    )
    assert {a.get_occupancy() for a in atoms} == {0.5}


def _cif_pose(tmp_path, atom_row):
    """Write a one-atom cif from atom_row (a dict over the columns) and pose it."""
    cols = dict(
        group_PDB="ATOM", id="1", type_symbol="C", label_atom_id="CA",
        label_comp_id="ALA", label_asym_id="A", label_seq_id="1", Cartn_x="1.0",
        Cartn_y="2.0", Cartn_z="3.0", auth_seq_id="1", auth_asym_id="A",
    )  # fmt: skip
    cols.update(atom_row)
    model = tmp_path / "m.cif"
    with open(model, "w") as f:
        f.write("data_m\nloop_\n")
        f.writelines(f"_atom_site.{k}\n" for k in cols)
        f.write(" ".join(cols.values()) + "\n")
    save_rotated_pdb(str(model), np.eye(3), np.zeros(3), str(tmp_path / "p"), 0)
    with open(tmp_path / "p.pdb") as f:
        return next(line for line in f if line.startswith(("ATOM", "HETATM")))


@pytest.mark.parametrize(
    "name, element, field",
    [
        ("CA", "C", " CA "),
        ("1HB", "H", "1HB "),
        ("CA", "CA", "CA  "),
        ("HE21", "H", "HE21"),
    ],
)
def test_cif_atom_name_column_as_biopython(tmp_path, name, element, field):
    line = _cif_pose(tmp_path, dict(label_atom_id=name, type_symbol=element))
    assert line[12:16] == field


def test_cif_hetatm_record_kept(tmp_path):
    line = _cif_pose(tmp_path, dict(group_PDB="HETATM", label_comp_id="HOH"))
    assert line.startswith("HETATM    1")
    assert line[17:20] == "HOH"


@pytest.mark.parametrize(
    "bad",
    [
        dict(auth_asym_id="AA"),
        dict(auth_seq_id="10000"),
        dict(label_comp_id="ALAX"),
        dict(label_atom_id="CAXXX"),
    ],
)
def test_cif_pose_refuses_fields_that_do_not_fit(tmp_path, bad):
    with pytest.raises(ValueError, match="exceeds PDB format limit"):
        _cif_pose(tmp_path, bad)
    assert not (tmp_path / "p.pdb").exists()


def test_cif_pose_writes_nothing_when_a_later_atom_does_not_fit(tmp_path):
    model = tmp_path / "m.cif"
    model.write_text(
        "data_m\nloop_\n"
        + "".join(
            f"_atom_site.{k}\n"
            for k in "group_PDB id type_symbol label_atom_id label_comp_id label_asym_id "
            "label_seq_id Cartn_x Cartn_y Cartn_z auth_seq_id auth_asym_id".split()
        )
        + "ATOM 1 C CA ALA A 1 1.0 2.0 3.0 1 A\n"
        + "ATOM 2 C CA ALA A 2 1.0 2.0 3.0 10000 A\n"
    )
    with pytest.raises(ValueError, match="exceeds PDB format limit"):
        save_rotated_pdb(str(model), np.eye(3), np.zeros(3), str(tmp_path / "p"), 0)
    assert not (tmp_path / "p.pdb").exists()


def test_cif_resseq_9999_accepted(tmp_path):
    line = _cif_pose(tmp_path, dict(auth_seq_id="9999"))
    assert line[22:26] == "9999"


def test_cif_serial_limit(tmp_path):
    def pose(n):
        model = tmp_path / "m.cif"
        rows = "".join(
            f"ATOM {i} C CA ALA A 1 1.0 2.0 3.0 1 A\n" for i in range(1, n + 1)
        )
        model.write_text(
            "data_m\nloop_\n"
            + "".join(
                f"_atom_site.{k}\n"
                for k in "group_PDB id type_symbol label_atom_id label_comp_id "
                "label_asym_id label_seq_id Cartn_x Cartn_y Cartn_z auth_seq_id "
                "auth_asym_id".split()
            )
            + rows
        )
        save_rotated_pdb(str(model), np.eye(3), np.zeros(3), str(tmp_path / "p"), 0)

    pose(99999)
    (tmp_path / "p.pdb").unlink()
    with pytest.raises(ValueError, match="serial number exceeds"):
        pose(100000)
    assert not (tmp_path / "p.pdb").exists()


def test_cif_altloc_and_insertion_code_written(tmp_path):
    line = _cif_pose(
        tmp_path, dict(label_alt_id="B", pdbx_PDB_ins_code="A", auth_seq_id="52")
    )
    assert line[16] == "B"
    assert line[22:27] == "  52A"
    atom = next(
        PDBParser(QUIET=True).get_structure("p", str(tmp_path / "p.pdb")).get_atoms()
    )
    assert atom.get_altloc() == "B"
    assert atom.get_parent().get_id() == (" ", 52, "A")


@pytest.mark.parametrize("blank", ["?", "."])
def test_cif_blank_altloc_and_insertion_code(tmp_path, blank):
    line = _cif_pose(tmp_path, dict(label_alt_id=blank, pdbx_PDB_ins_code=blank))
    assert line[16] == " "
    assert line[26] == " "
