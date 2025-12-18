import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.mmcifio import MMCIFIO
import os


def save_rotated_pdb(input_pdb, rot_mtx, real_trans, save_path, model_num):
    # check input file format
    parser = None
    input_format = input_pdb.split(".")[-1]

    if input_format == "pdb":
        parser = PDBParser(QUIET=True)
        io = PDBIO()
    elif input_format == "cif":
        parser = MMCIFParser(QUIET=True)
        io = MMCIFIO()
    else:
        raise Exception("Input PDB/mmCIF file format not supported.")

    structure = parser.get_structure("target_pdb", input_pdb)
    structure.transform(rot_mtx, real_trans)

    io.set_structure(structure)
    io.save(save_path + f".{input_format}")


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
