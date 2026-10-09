"""Small asymmetric test inputs, written from fixed formulas (no random numbers).

Maps A and B share one box (same origin, size and voxel size) and their two outer blobs, so they get
the same search grid; map C sits in the same box but is smaller, so it would get a smaller grid on
its own. The target holds A's four core blobs turned by 120 degrees about (1, 1, 1) in a box of its
own, and the model has one CA atom on each of them. No blob arrangement has a symmetry, so the
search has one clear best pose and no ties among the top poses.
"""

import mrcfile
import numpy as np

VOXEL = 2.0
REF_BOX, REF_ORIGIN = 32, (10.0, -6.0, 4.0)
TGT_BOX, TGT_ORIGIN = 24, (100.0, 50.0, -20.0)
SIGMA = 3.0

# (offset from the box centre in angstroms, height)
CORE = [((-10, -4, 2), 1.0), ((2, -8, -6), 0.8), ((8, 6, -2), 1.2), ((-2, 10, 8), 0.6)]
OUTER = [((12, -12, 12), 0.7), ((-14, 8, -10), 0.9)]
BLOBS = {
    "A": [*CORE, *OUTER],
    "B": [
        ((-6, -6, 4), 0.9),
        ((0, 0, 0), 1.0),
        ((6, -2, 10), 0.5),
        ((4, 8, -6), 1.1),
        *OUTER,
    ],
    "C": CORE,
}
# 120 degrees about (1, 1, 1): x -> y -> z -> x, exact in floating point
TURN = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
TGT_SHIFT = np.array([3.0, -2.0, 1.0])


def _centre(box, origin):
    return np.array(origin) + 0.5 * box * VOXEL


def write_map(path, blobs, box, origin):
    """Write a sum of Gaussian blobs (offsets from the box centre) as an MRC file."""
    axis = np.arange(box) * VOXEL
    x, y, z = np.meshgrid(*(axis + o for o in origin), indexing="ij")
    centre = _centre(box, origin)
    data = np.zeros((box, box, box))
    for offset, height in blobs:
        cx, cy, cz = centre + np.asarray(offset, dtype=float)
        r2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        data += height * np.exp(-r2 / (2 * SIGMA**2))
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.ascontiguousarray(data.transpose(2, 1, 0), dtype=np.float32))
        mrc.voxel_size = VOXEL
        mrc.header.origin = origin


def target_blobs():
    """A's core blobs turned and shifted, as offsets from the target box centre."""
    return [(TURN @ np.asarray(o, dtype=float) + TGT_SHIFT, h) for o, h in CORE]


def write_inputs(directory):
    """Write a.mrc, b.mrc, c.mrc, target.mrc and model.pdb into directory."""
    for name, blobs in BLOBS.items():
        write_map(f"{directory}/{name.lower()}.mrc", blobs, REF_BOX, REF_ORIGIN)
    write_map(f"{directory}/target.mrc", target_blobs(), TGT_BOX, TGT_ORIGIN)
    write_model(f"{directory}/model.pdb", model_coords())


def model_coords():
    """One atom on each target blob, in the target's frame."""
    centre = _centre(TGT_BOX, TGT_ORIGIN)
    return [centre + offset for offset, _ in target_blobs()]


def write_model(path, coords):
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(coords, start=1):
            f.write(
                f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.0:6.2f}{0.0:6.2f}           C\n"
            )
        f.write("END\n")


def write_model_cif(path, coords):
    with open(path, "w") as f:
        f.write("data_model\nloop_\n")
        for key in (
            "group_PDB id type_symbol label_atom_id label_comp_id label_asym_id "
            "label_seq_id Cartn_x Cartn_y Cartn_z auth_seq_id auth_asym_id"
        ).split():
            f.write(f"_atom_site.{key}\n")
        for i, (x, y, z) in enumerate(coords, start=1):
            f.write(f"ATOM {i} C CA ALA A {i} {x:.3f} {y:.3f} {z:.3f} {i} A\n")
