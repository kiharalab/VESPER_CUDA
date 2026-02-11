import argparse
import os
import pathlib

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from scipy.spatial.transform import Rotation as R


def calc_rmsd(search_pdb_path, ref_pdb_path, rot_mtx, trans_vec, verbose=False):
    parser = PDBParser(QUIET=True)
    search_pdb = parser.get_structure("search", search_pdb_path)
    search_pdb.transform(rot_mtx, trans_vec)

    ref_pdb = parser.get_structure("ref", ref_pdb_path)

    ref_atoms = [np.array(atom.coord) for atom in ref_pdb.get_atoms()]
    sample_atoms = [np.array(atom.coord) for atom in search_pdb.get_atoms()]

    ref_atoms_array = np.array(ref_atoms)
    sample_atoms_array = np.array(sample_atoms)

    if len(ref_atoms_array) != len(sample_atoms_array):
        i, j = 0, 0
        cnt = 0

        common_atom_ref = []
        common_atom_sample = []

        atom_list_ref = [res for res in ref_pdb.get_atoms()]
        atom_list_sample = [res for res in search_pdb.get_atoms()]

        if verbose:
            print(
                "Warning: Number of residues in reference and sample structures do not match"
            )
            print("Reference: " + str(len(atom_list_ref)) + " atoms")
            print("Sample: " + str(len(atom_list_sample)) + " atoms")

        while i < len(atom_list_ref) and j < len(atom_list_sample):
            if (
                atom_list_ref[i].get_full_id()[3][1]
                < atom_list_sample[j].get_full_id()[3][1]
            ):
                i += 1
            elif (
                atom_list_ref[i].get_full_id()[3][1]
                > atom_list_sample[j].get_full_id()[3][1]
            ):
                j += 1
            else:
                common_atom_ref.append(atom_list_ref[i])
                common_atom_sample.append(atom_list_sample[j])
                cnt += 1
                i += 1
                j += 1

        if verbose:
            print("Number of common atoms: " + str(cnt))

        ref_atoms_array = np.array([atom.coord for atom in common_atom_ref])
        sample_atoms_array = np.array([atom.coord for atom in common_atom_sample])

    rmsd = np.sqrt(
        np.sum((ref_atoms_array - sample_atoms_array) ** 2) / len(ref_atoms_array)
    )

    return rmsd, search_pdb


def calc_rmsd_list(
    search_pdb_path,
    ref_pdb_path_list,
    vesper_output,
    verbose=False,
    gen_pdb=False,
    topn=10,
):
    print("Calculating RMSD...")
    print("Search PDB: " + search_pdb_path)
    print("Reference PDB: " + str(ref_pdb_path_list))
    print("VESPER Output: " + vesper_output)
    print("Top N: " + str(topn))

    if gen_pdb:
        os.makedirs("./rmsd_pdb_results/", exist_ok=True)

    pdbio = PDBIO()

    with open(vesper_output) as f:
        model_line_start = []
        for i in range(topn):
            model_line_start.append("#" + str(i))

        print("Model line start: " + str(model_line_start))

        rmsd_list = []

        lines = f.readlines()
        result_line = [line for line in lines if line.split(" ")[0] in model_line_start]
        del lines

        for line in result_line:
            print(line)
            model_num = line.split(" ")[0][1:]
            print(model_num)
            # r_info = line.split()[1:4] # for legacy
            r_info = line.split()[2:5]
            print(r_info)
            # t_info = line.split()[13:16] # for legacy
            t_info = line.split()[6:9]
            print(t_info)

            rot_vec = np.array(
                [float(r_info[0][3:]), float(r_info[1]), float(r_info[2][:-1])]
            )
            trans_vec = np.array(
                [float(t_info[0][3:]), float(t_info[1]), float(t_info[2][:-1])]
            )

            if verbose:
                print("Rotation Vector: ", rot_vec)
                print("Translation Vector: ", trans_vec)
                print("Model " + model_num + " rotation vector: " + str(rot_vec))
                print("Model " + model_num + " translation vector: " + str(trans_vec))

            rotation = R.from_euler("xyz", rot_vec, degrees=True)
            rotation = rotation.inv().as_matrix()

            min_rmsd = 9999

            for ref_pdb_path in ref_pdb_path_list:
                rmsd, search_pdb = calc_rmsd(
                    search_pdb_path, ref_pdb_path, rotation, trans_vec, verbose=verbose
                )
                if rmsd < min_rmsd:
                    min_rmsd = rmsd

            if gen_pdb:
                pdbio.set_structure(search_pdb)
                pdbio.save(
                    "./rmsd_pdb_results/model_"
                    + pathlib.Path(search_pdb_path).stem
                    + "_#"
                    + model_num
                    + ".pdb"
                )

            if rmsd is None:
                print("Error: RMSD calculation failed")
                return None

            if verbose:
                print("Model " + model_num + " RMSD: " + str(rmsd))

            rmsd_list.append(min_rmsd)

    return rmsd_list


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate RMSD of VESPER output")
    parser.add_argument("search_pdb", help="Path to the PDB file to be searched")
    parser.add_argument("ref_pdb", help="Path to the reference PDB file")
    parser.add_argument("vesper_output", help="Path to the VESPER output file")
    parser.add_argument(
        "-n", "--topn", help="Top N models to calculate RMSD", type=int, default=10
    )
    parser.add_argument(
        "-v", "--verbose", help="Print verbose output", action="store_true"
    )
    parser.add_argument(
        "-g",
        "--gen_pdb",
        help="Generate PDB files of the RMSD calculations",
        action="store_true",
    )

    args = parser.parse_args()

    print("Calculating RMSD...")
    print("Search PDB: " + args.search_pdb)
    print("Reference PDB: " + args.ref_pdb)
    print("VESPER Output: " + args.vesper_output)

    rmsd = calc_rmsd_list(
        args.search_pdb,
        [args.ref_pdb],
        args.vesper_output,
        args.verbose,
        args.gen_pdb,
        args.topn,
    )
    print("RMSD: " + str(rmsd))
