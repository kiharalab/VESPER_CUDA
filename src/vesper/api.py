"""Programmatic API for running VESPER alignment."""

import os
import tempfile

import numpy as np

from .data.map import EMmap, unify_dims
from .fitter import MapFitter
from .utils.pdb2vol import pdb2vol


def run_vesper_fit(
    map_path: str,
    input_pdb: str,
    contour_level: float,
    resolution: float,
    num_conformation: int = 10,
    gpu_id: int = 0,
    angle_spacing: float = 5.0,
    voxel_spacing: float = 2.0,
    gaussian_bandwidth: float = 1.0,
    num_threads: int = 8,
    only_best: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run VESPER orig mode alignment.

    Equivalent to CLI:
        vesper orig -a map -t $contour_level -b $input_pdb -T 0.01 -gpu 0
                    -N $num_conformation -A 5 -s 2 -pdbin $input_pdb
                    -res $resolution -nodup -g 1 -M C -c 8

    Args:
        map_path: Path to the reference MRC map
        input_pdb: Path to the input PDB/CIF structure file
        contour_level: Density threshold for reference map
        resolution: Resolution for simulating map from structure
        num_conformation: Number of top conformations to refine (default: 10)
        gpu_id: GPU device ID (default: 0)
        angle_spacing: Angular sampling interval in degrees (default: 5.0)
        voxel_spacing: Voxel spacing for sampling (default: 2.0)
        gaussian_bandwidth: Gaussian filter bandwidth (default: 1.0)
        num_threads: Number of CPU threads (default: 8)
        only_best: If True, return only the best conformation (default: False)

    Returns:
        Tuple of (rotations, translations):
            - If only_best=False: rotations (num_conformation, 3, 3), translations (num_conformation, 3)
            - If only_best=True: rotation (3, 3), translation (3,)
    """
    import random
    import string

    import torch

    mode = "C"
    threshold_map2 = 0.01
    remove_duplicates = True

    rand_str = "".join(random.choices(string.ascii_letters + string.digits, k=8))

    assert os.path.exists(map_path), f"Reference map not found: {map_path}"
    assert os.path.exists(input_pdb), f"Input PDB not found: {input_pdb}"

    sim_map_path = os.path.join(tempfile.gettempdir(), f"simu_map_{rand_str}.mrc")
    pdb2vol(input_pdb, resolution, sim_map_path, backbone_only=False)
    assert os.path.exists(sim_map_path), (
        "Failed to create simulated map from structure."
    )

    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise ValueError("GPU is specified but CUDA is not available.")
    device = torch.device(f"cuda:{gpu_id}")

    ref_map = EMmap(map_path)
    tgt_map = EMmap(sim_map_path)

    ref_map.set_vox_size(thr=contour_level, voxel_size=voxel_spacing)
    tgt_map.set_vox_size(thr=threshold_map2, voxel_size=voxel_spacing)

    unify_dims([ref_map, tgt_map], voxel_size=voxel_spacing)

    ref_map.resample_and_vec(dreso=gaussian_bandwidth)
    tgt_map.resample_and_vec(dreso=gaussian_bandwidth)

    fitter = MapFitter(
        ref_map,
        tgt_map,
        angle_spacing,
        mode,
        remove_duplicates,
        None,
        None,
        None,
        num_threads,
        True,
        device,
        topn=num_conformation,
        outdir=None,
        save_mrc=False,
        alpha=None,
        confine_angles=None,
        save_vec=False,
        batch_size=None,
    )
    fitter.fit()

    from scipy.spatial.transform import Rotation as R

    rotations = []
    translations = []
    for item in fitter.final_list:  # ty:ignore[not-iterable]
        rot_mtx = R.from_euler("xyz", item["angle"], degrees=True).inv().as_matrix()
        rotations.append(rot_mtx)
        translations.append(item["real_trans"])

    if only_best:
        return np.array(rotations[0]), np.array(translations[0])
    return np.array(rotations), np.array(translations)
