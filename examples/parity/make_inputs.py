"""Write the inputs of the parity examples into a folder, from the shipped EMD-17127 and 8orj files.

    python make_inputs.py OUT_DIR

map.mrc     EMD-17127, uncompressed
half.mrc    the same map with every voxel of x index < nx/2 set to 0 (same grid, for -a with two maps
            and for -E)
moved.mrc   half.mrc with its origin moved by 10 A along x (for -E: the density sits 10 A away)
ldp.pdb     stand-in LDP points: the centres of every 3rd voxel along each axis with density above
            0.0289 (DiffModeler builds LDP points from the map by mean shift; these only exercise
            the LDP recall reranking)
model.pdb   model.cif as PDB, written by VESPER's writer (the -b/-pdbin structure)
model.cif   8orj_stage_1.cif with its chain "A0" renamed "A" (a PDB chain id has one character)
ca.pdb      the CA atoms of model.pdb (the -ca backbone file)
"""

import gzip
import os
import sys

import mrcfile
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO

from vesper.data.io import save_rotated_pdb

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_GZ = os.path.join(HERE, "..", "emd_17127.map.gz")
CIF = os.path.join(HERE, "..", "8orj_stage_1.cif")


def write_maps(out):
    map_path = os.path.join(out, "map.mrc")
    with gzip.open(MAP_GZ, "rb") as f_in, open(map_path, "wb") as f_out:
        f_out.write(f_in.read())
    with (
        open(map_path, "rb") as f_in,
        open(os.path.join(out, "half.mrc"), "wb") as f_out,
    ):
        f_out.write(f_in.read())
    with mrcfile.open(os.path.join(out, "half.mrc"), mode="r+") as mrc:
        data = mrc.data.copy()
        data[:, :, : data.shape[2] // 2] = 0  # mrcfile arrays are (z, y, x)
        mrc.set_data(data)
        origin = mrc.header.origin
        moved = (float(origin.x) + 10.0, float(origin.y), float(origin.z))
        with mrcfile.new(os.path.join(out, "moved.mrc"), overwrite=True) as moved_mrc:
            moved_mrc.set_data(data)
            moved_mrc.voxel_size = mrc.voxel_size
            moved_mrc.header.origin = moved

    with mrcfile.open(map_path) as mrc:
        voxel = float(mrc.voxel_size.x)
        origin = np.array(mrc.header.origin.tolist())
        z, y, x = np.nonzero(mrc.data[::3, ::3, ::3] > 0.0289)
    points = origin + 3 * voxel * np.stack((x, y, z), axis=1)
    with open(os.path.join(out, "ldp.pdb"), "w") as f:
        for i, (px, py, pz) in enumerate(points, start=1):
            f.write(
                f"ATOM  {i % 100000:5d}  CA  ALA A{i % 10000:4d}    "
                f"{px:8.3f}{py:8.3f}{pz:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")


def write_structures(out):
    # the shipped .cif names its chain "A0", which a PDB file cannot hold: rename it "A"
    cif = MMCIF2Dict(CIF)
    for key in ("_atom_site.auth_asym_id", "_atom_site.label_asym_id"):
        cif[key] = ["A"] * len(cif[key])
    io = MMCIFIO()
    io.set_dict(cif)
    model_cif = os.path.join(out, "model.cif")
    io.save(model_cif)
    # VESPER's own .cif reader and writer: this .cif has no B-factors, which Biopython needs
    save_rotated_pdb(model_cif, np.eye(3), np.zeros(3), os.path.join(out, "model"), 0)

    with open(os.path.join(out, "model.pdb")) as f:
        ca_lines = [
            line
            for line in f
            if line.startswith("ATOM") and line[12:16].strip() == "CA"
        ]
    with open(os.path.join(out, "ca.pdb"), "w") as f:
        f.writelines(ca_lines)
        f.write("END\n")


if __name__ == "__main__":
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    write_maps(out_dir)
    write_structures(out_dir)
