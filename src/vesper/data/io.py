import gzip
import os
import pickle
import tempfile

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO

from vesper.utils.utils import get_file_extension


def _atom_name_field(name, element):
    """Columns 13-16: the name starts in column 14 unless it has four characters, does
    not start with a letter, or its element has two (as Biopython's PDBIO)."""
    if len(name) < 4 and name[:1].isalpha() and len(element) == 1:
        name = " " + name
    return f"{name[:4]:<4s}"


def save_rotated_pdb(
    input_pdb, rot_mtx, real_trans, save_path, model_num, occupancy=None
):
    """
    Save a rotated and translated PDB file.

    Handles .gz compressed files and CIF files without B-factors. occupancy, if given,
    replaces every atom's occupancy.
    """
    input_format, is_compressed = get_file_extension(input_pdb)

    # Handle .gz compressed files by decompressing to temp file
    temp_path = None
    if is_compressed:
        with gzip.open(input_pdb, "rb") as f:
            with tempfile.NamedTemporaryFile(
                suffix=f".{input_format}", delete=False
            ) as tmp:
                tmp.write(f.read())
                temp_path = tmp.name
        actual_path = temp_path
    else:
        actual_path = input_pdb

    try:
        if input_format == "pdb":
            parser = PDBParser(QUIET=True)
            io = PDBIO()
            structure = parser.get_structure("target_pdb", actual_path)
            structure.transform(rot_mtx, real_trans)
            if occupancy is not None:
                for atom in structure.get_atoms():
                    atom.set_occupancy(occupancy)
            io.set_structure(structure)
            io.save(save_path + ".pdb")
        elif input_format == "cif":
            # For CIF files without B-factors, directly write transformed PDB
            from Bio.PDB.MMCIF2Dict import MMCIF2Dict

            mmcif_dict = MMCIF2Dict(actual_path)

            # Get atom coordinates and types
            x_list = mmcif_dict["_atom_site.Cartn_x"]
            y_list = mmcif_dict["_atom_site.Cartn_y"]
            z_list = mmcif_dict["_atom_site.Cartn_z"]
            type_symbol_list = mmcif_dict["_atom_site.type_symbol"]
            label_atom_id = mmcif_dict.get(
                "_atom_site.label_atom_id", ["CA"] * len(x_list)
            )
            label_comp_id = mmcif_dict.get(
                "_atom_site.label_comp_id", ["ALA"] * len(x_list)
            )
            label_seq_id = mmcif_dict.get(
                "_atom_site.label_seq_id", ["1"] * len(x_list)
            )
            label_asym_id = mmcif_dict.get(
                "_atom_site.label_asym_id", ["A"] * len(x_list)
            )
            auth_seq_id = mmcif_dict.get("_atom_site.auth_seq_id", ["1"] * len(x_list))
            auth_asym_id = mmcif_dict.get(
                "_atom_site.auth_asym_id", ["A"] * len(x_list)
            )

            group_pdb = mmcif_dict.get("_atom_site.group_PDB", ["ATOM"] * len(x_list))

            if occupancy is None:
                occupancy = 1.0

            # Apply transformation
            n_atoms = len(x_list)
            coords = np.zeros((n_atoms, 3))
            coords[:, 0] = [float(x) for x in x_list]
            coords[:, 1] = [float(y) for y in y_list]
            coords[:, 2] = [float(z) for z in z_list]

            # Same as Biopython's structure.transform(rot_mtx, real_trans) above
            transformed = coords @ rot_mtx + real_trans

            # Write PDB format
            with open(save_path + ".pdb", "w") as f:
                f.write("REMARK   VESPER transformed structure\n")
                f.write("REMARK   Rotation matrix:\n")
                for i in range(3):
                    f.write(
                        f"REMARK   [{rot_mtx[i, 0]:10.6f}, {rot_mtx[i, 1]:10.6f}, {rot_mtx[i, 2]:10.6f}]\n"
                    )
                f.write(
                    f"REMARK   Translation: [{real_trans[0]:10.4f}, {real_trans[1]:10.4f}, {real_trans[2]:10.4f}]\n"
                )

                for i in range(n_atoms):
                    x, y, z = transformed[i]
                    atom_name = label_atom_id[i] if i < len(label_atom_id) else "CA"
                    res_name = label_comp_id[i] if i < len(label_comp_id) else "ALA"
                    chain = auth_asym_id[i] if i < len(auth_asym_id) else "A"
                    res_seq = auth_seq_id[i] if i < len(auth_seq_id) else str(i + 1)

                    if i + 1 > 99999:
                        raise ValueError("Atom serial number exceeds PDB format limit")
                    if len(chain) > 1:
                        raise ValueError(f"Chain id {chain!r} exceeds PDB format limit")
                    if len(res_seq) > 4:
                        raise ValueError(
                            f"Residue number {res_seq!r} exceeds PDB format limit"
                        )

                    # Format PDB ATOM record (PDB columns, as Biopython's PDBIO)
                    line = (
                        f"{group_pdb[i]:<6.6s}{i + 1:5d} {_atom_name_field(atom_name, type_symbol_list[i])}"
                        f" {res_name:>3.3s} {chain:1s}{res_seq:>4s}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{0.0:6.2f}"
                        f"          {type_symbol_list[i]:>2s}\n"
                    )
                    f.write(line)
                f.write("END\n")
        else:
            raise Exception("Input file format not supported. Use .pdb or .cif")
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def save_score_pkl(pose_scores, save_path):
    """
    Pickle {pose file: score}, best first.

    The file is written under a temporary name and renamed, so that a score.pkl that exists
    is complete: DiffModeler takes it as the sign that a fit is done.
    """
    pose_scores = dict(sorted(pose_scores.items(), key=lambda kv: kv[1], reverse=True))
    with open(save_path + ".tmp", "wb") as f:
        pickle.dump(pose_scores, f)
    os.replace(save_path + ".tmp", save_path)


def save_vec_as_pdb(
    origin,
    sampled_mrc_vec,
    sampled_mrc_data,
    score_arr,
    score,
    sample_width,
    trans,
    save_path,
    rank,
):
    dim = sampled_mrc_data.shape[0]

    origin = np.array([origin[0], origin[1], origin[2]])
    trans = np.array(trans)

    trans = np.where(trans > 0.5 * dim, trans - dim, trans)

    add = origin - trans * sample_width

    natm = 1
    nres = 1

    with open(save_path, "w") as pdb_file:
        non_zero_dens_index = np.transpose(np.nonzero(sampled_mrc_data))
        for position in non_zero_dens_index:
            real_coord = position * sample_width + add  # center of voxel
            atom_header = f"ATOM{natm:>7d}  CA  ALA{nres:>6d}    "
            atom_content = f"{real_coord[0]:8.3f}{real_coord[1]:8.3f}{real_coord[2]:8.3f}{1.0:6.2f}{score_arr[position[0], position[1], position[2]]:6.2f}"
            pdb_file.write(atom_header + atom_content + "\n")
            natm += 1

            real_coord = (
                position + sampled_mrc_vec[position[0]][position[1]][position[2]]
            ) * sample_width + add  # center of voxel plus unit vector
            atom_header = f"ATOM{natm:>7d}  CB  ALA{nres:>6d}    "
            atom_content = f"{real_coord[0]:8.3f}{real_coord[1]:8.3f}{real_coord[2]:8.3f}{1.0:6.2f}{score_arr[position[0], position[1], position[2]]:6.2f}"
            pdb_file.write(atom_header + atom_content + "\n")
            natm += 1
            nres += 1


def save_map_as_pdb(origin, data, sample_width, trans, save_path):
    dim = data.shape[0]  # dimension of map
    origin = np.array([origin[0], origin[1], origin[2]])
    trans = np.array(trans)
    trans = np.where(trans > 0.5 * dim, trans - dim, trans)
    offset = origin - trans * sample_width

    natm = 1  # atom number
    nres = 1  # residue number

    with open(save_path, "w") as pdb_file:
        non_zero_dens_index = np.transpose(np.nonzero(data))
        for position in non_zero_dens_index:
            real_coord = position * sample_width + offset
            atom_header = f"ATOM{natm:>7d}  CA  ALA{nres:>6d}    "
            atom_content = f"{real_coord[0]:8.3f}{real_coord[1]:8.3f}{real_coord[2]:8.3f}{1.0:6.2f}{data[position[0], position[1], position[2]]:6.2f}"
            pdb_file.write(atom_header + atom_content + "\n")
            natm += 1
            nres += 1
