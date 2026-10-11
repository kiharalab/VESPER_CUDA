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


def fit(
    ref_maps: str | list[str],
    target: str,
    *,
    labels: list[str] | None = None,
    ref_threshold: float = 0.0,
    target_threshold: float = 0.0,
    bandwidth: float = 16.0,
    voxel_spacing: float = 7.0,
    angle_spacing: float = 30.0,
    top_n: int = 10,
    refine_step: int = 2,
    mode: str = "V",
    remove_duplicates: bool = False,
    ldp: str | list[str] | None = None,
    ca: str | None = None,
    num_threads: int = 2,
    gpu_id: int | None = None,
    angle_limit: float | None = None,
    resolution: float | None = None,
    backbone_only: bool = False,
    batch_size: int | None = None,
    direct_fit: bool = False,
) -> list[list[dict]]:
    """Search the target in one or several maps, as `vesper orig` does, without writing files

    The arguments are `vesper orig`'s: -a (a path or a list of paths on one grid), -b, -labels,
    -t, -T, -g, -s, -A, -N, -R, -M, -nodup, -ldp (one file, or one per map), -ca, -c, -gpu
    (None: CPU), -al, -res, -bbonly, -batch and --direct_fit. A structure target (.pdb/.cif)
    is simulated at `resolution` first.

    Returns, per map in the order given, its final poses in rank order, each a dict:
    - "angle": the Euler angles searched ("xyz", degrees), as in the pose file names;
    - "rotation", "translation": the pose of the -b structure, x -> rotation.T @ x +
      translation (the arguments `vesper.data.io.save_rotated_pdb` takes, which `vesper orig`
      uses to write each pose file);
    - "score": the search score; "fit_score": DiffModeler's fit score (score.pkl's value);
    - "ldp_recall", with -ldp and -ca.
    """
    from scipy.spatial.transform import Rotation as R

    from .cli import (
        prepare_maps,
        setup_gpu,
        simulate_target,
        validate_input_files,
        validate_ref_args,
        validate_search_args,
    )
    from .utils.utils import get_file_extension

    ref_arg = ref_maps if isinstance(ref_maps, str) else ",".join(ref_maps)
    labels_arg = None if labels is None else ",".join(labels)
    ldp_arg = ldp if ldp is None or isinstance(ldp, str) else ",".join(ldp)
    ref_paths, ref_labels, ldp_paths = validate_ref_args(
        ref_arg, labels_arg, ldp_arg, ca, direct_fit
    )
    validate_search_args(angle_spacing, top_n, angle_limit, batch_size, None)
    validate_input_files(target, resolution, ldp_arg, ca, ldp_paths)

    sim_map = None
    if get_file_extension(target)[0] in ["pdb", "cif"]:
        sim_map = target = simulate_target(target, resolution, backbone_only)
    try:
        use_gpu, device = setup_gpu(gpu_id)
        refs, tgt_map = prepare_maps(
            ref_paths,
            ref_labels,
            target,
            ref_threshold,
            target_threshold,
            voxel_spacing,
            bandwidth,
        )
        fitter = MapFitter(
            refs,
            tgt_map,
            angle_spacing,
            mode,
            remove_duplicates,
            ldp_paths,
            ca,
            None,
            num_threads,
            use_gpu,
            device,
            topn=top_n,
            outdir=None,
            save_mrc=False,
            alpha=None,
            confine_angles=angle_limit,
            save_vec=False,
            batch_size=batch_size,
            ref_labels=ref_labels,
            refine_step=refine_step,
        )
        fitter.fit()
        poses = [
            [
                {
                    "angle": np.array(item["angle"], dtype=float),
                    "rotation": R.from_euler("xyz", item["angle"], degrees=True)
                    .inv()
                    .as_matrix(),
                    "translation": np.array(item["real_trans"], dtype=float),
                    "score": float(item["score"]),
                    "fit_score": fit_score,
                    **(
                        {"ldp_recall": float(item["ldp_recall"])}
                        if "ldp_recall" in item
                        else {}
                    ),
                }
                for item, fit_score in zip(final_list, fit_scores)
            ]
            for final_list, fit_scores in zip(
                fitter.final_lists, fitter.fit_score_lists
            )
        ]
    finally:
        if sim_map is not None and os.path.exists(sim_map):
            os.remove(sim_map)
    del fitter
    if use_gpu:
        import torch

        torch.cuda.empty_cache()
    return poses
