import os
import sys
import tempfile
import time
from enum import Enum

import numpy as np
import typer

from .data.map import EMmap, unify_dims
from .fitter import MapFitter
from .utils.pdb2vol import pdb2vol
from .utils.utils import get_file_extension

app = typer.Typer(help="VESPER - CUDA accelerated version")


class Mode(str, Enum):
    VEC_PRODUCT = "V"
    OVERLAP = "O"
    CROSS_CORRELATION = "C"
    PEARSON_CORRELATION = "P"
    LAPLACIAN = "L"


def print_command_summary(**kwargs) -> None:
    """Print command summary"""
    print("### Command ###")
    print(" ".join(sys.argv))
    print("\n### Input Params Summary ###")
    for key, value in kwargs.items():
        if value is not None:
            print(f"{key}: {value}")


def setup_gpu(gpu_id: int | None) -> tuple[bool, object | None]:
    """Setup GPU device"""
    device = None
    if gpu_id is not None:
        use_gpu = True
        import torch

        torch.set_grad_enabled(False)
        if not torch.cuda.is_available():
            raise ValueError("GPU is specified but CUDA is not available.")
        else:
            device = torch.device(f"cuda:{gpu_id}")
            print(
                f"Using GPU {gpu_id} for CUDA acceleration. GPU Name: {torch.cuda.get_device_name(gpu_id)}"
            )
    else:
        print("Using FFTW3 for CPU.")
        use_gpu = False

    return use_gpu, device


@app.command("orig")
def orig_command(
    map1: str = typer.Option(..., "-a", help="MAP1.mrc (large)"),
    map2: str = typer.Option(..., "-b", help="MAP2.mrc (small)"),
    threshold_map1: float = typer.Option(0.0, "-t", help="Threshold of density map1"),
    threshold_map2: float = typer.Option(0.0, "-T", help="Threshold of density map2"),
    gaussian_bandwidth: float = typer.Option(
        16.0,
        "-g",
        help="Bandwidth of the Gaussian filter def=16.0, sigma = 0.5*[value entered]",
    ),
    voxel_spacing: float = typer.Option(
        7.0, "-s", help="Sampling voxel spacing def=7.0"
    ),
    angle_spacing: float = typer.Option(
        30.0, "-A", help="Sampling angle spacing def=30.0"
    ),
    refine_top: int = typer.Option(10, "-N", help="Refine Top [int] models def=10"),
    save_models: bool = typer.Option(
        False, "-S", help="Show topN models in PDB format def=false"
    ),
    mode: Mode = typer.Option(
        Mode.VEC_PRODUCT,
        "-M",
        help="V: vector product mode (default)\n"
        "O: overlap mode\n"
        "C: Cross Correlation Coefficient Mode\n"
        "P: Pearson Correlation Coefficient Mode\n"
        "L: Laplacian Filtering Mode",
    ),
    eval_mode: bool = typer.Option(
        False, "-E", help="Evaluation mode of the current position def=false"
    ),
    output_dir: str | None = typer.Option(None, "-o", help="Output folder name"),
    gpu_id: int | None = typer.Option(
        None, "-gpu", help="GPU ID to use for CUDA acceleration def=0"
    ),
    remove_duplicates: bool = typer.Option(
        False, "-nodup", help="Remove duplicate models using heuristics def=false"
    ),
    ldp_file: str | None = typer.Option(
        None, "-ldp", help="Path to the local dense point file def=None"
    ),
    ca_file: str | None = typer.Option(
        None, "-ca", help="Path to the CA file def=None"
    ),
    pdbin: str | None = typer.Option(
        None, "-pdbin", help="Input PDB file to be transformed def=None"
    ),
    backbone_only: bool = typer.Option(
        False,
        "-bbonly",
        help="Whether to only use backbone atoms for simulated map def=false",
    ),
    mrcout: bool = typer.Option(
        False, "-mrcout", help="Output the transformed query map def=false"
    ),
    num_threads: int = typer.Option(2, "-c", help="Number of threads to use def=2"),
    angle_limit: float | None = typer.Option(
        None, "-al", help="Angle limit for searching def=None"
    ),
    resolution: float | None = typer.Option(
        None,
        "-res",
        help="Resolution of the experimental map used to create simulated map from structure",
    ),
    batch_size: int | None = typer.Option(
        None, "-batch", help="Override batch size for GPU processing def=auto-detect"
    ),
) -> None:
    """Original VESPER command"""
    import random
    import string

    mode_val = mode.value if isinstance(mode, Mode) else mode

    if save_models and mode_val != "V":
        print(
            "Warning: -S option is only available in VecProduct mode, ignoring -S option"
        )
        save_models = False

    # generate random string for temp folder
    rand_str = "".join(random.choices(string.ascii_letters + string.digits, k=8))

    # check if the second input is a structure file
    ext, _ = get_file_extension(map2)
    if ext in ["pdb", "cif"]:
        assert resolution is not None, (
            "Please specify resolution when using structure as input."
        )
        # simulate the map at target resolution
        sim_map_path = os.path.join(tempfile.gettempdir(), f"simu_map_{rand_str}.mrc")
        pdb2vol(map2, resolution, sim_map_path, backbone_only=backbone_only)
        assert os.path.exists(sim_map_path), (
            "Failed to create simulated map from structure."
        )
        map2 = sim_map_path

    assert os.path.exists(map1), "Reference map not found, please check -a option"
    assert os.path.exists(map2), "Target map not found, please check -b option"

    # Setup GPU
    use_gpu, device = setup_gpu(gpu_id)

    # Print summary
    print_command_summary(
        Reference_Map_Path=os.path.abspath(map1),
        Target_Map_Path=os.path.abspath(map2),
        Threshold_of_Reference_Map=threshold_map1,
        Threshold_of_Target_Map=threshold_map2,
        Bandwidth_of_the_Gaussian_filter=gaussian_bandwidth,
        Search_voxel_spacing=voxel_spacing,
        Search_angle_spacing=angle_spacing,
        Refine_Top=f"{refine_top} models",
        Remove_duplicates=remove_duplicates,
        Scoring_mode=mode_val,
        Show_topN_models_in_PDB_format=save_models,
        Number_of_threads_to_use=num_threads,
        Output_folder=os.path.abspath(output_dir) if output_dir else None,
        Using_GPU_ID=gpu_id,
        Transform_PDB_file=os.path.abspath(pdbin)
        if pdbin and os.path.exists(pdbin)
        else None,
        LDP_Recall_Reranking=(ldp_file and ca_file),
        LDP_PDB_file=os.path.abspath(ldp_file) if ldp_file else None,
        Backbone_PDB_file=os.path.abspath(ca_file) if ca_file else None,
        Angle_limit_for_searching=angle_limit,
    )

    if ldp_file or ca_file:
        assert ldp_file and ca_file, "Please specify both -ldp and -ca options"
    if ldp_file:
        assert os.path.exists(ldp_file), "LDP file not found, please check -ldp option"
    if ca_file:
        assert os.path.exists(ca_file), "CA file not found, please check -ca option"

    if not pdbin or not os.path.exists(pdbin):
        print("No input PDB file, skipping transformation")
        pdbin = None
    else:
        print(f"Transform PDB file: {os.path.abspath(pdbin)}")

    if ldp_file and ca_file and os.path.exists(ldp_file) and os.path.exists(ca_file):
        print("LDP Recall Reranking Enabled")
    else:
        print("LDP Recall Reranking Disabled")

    start_time = time.time()

    # construct mrc objects
    ref_map = EMmap(map1)
    tgt_map = EMmap(map2)

    # set voxel size
    ref_map.set_vox_size(thr=threshold_map1, voxel_size=voxel_spacing)
    tgt_map.set_vox_size(thr=threshold_map2, voxel_size=voxel_spacing)

    # unify dimensions
    unify_dims([ref_map, tgt_map], voxel_size=voxel_spacing)

    # resample the maps using mean-shift with Gaussian kernel and calculate the vector representation
    print("\n###Processing Reference Map Resampling###")
    ref_map.resample_and_vec(dreso=gaussian_bandwidth)
    print("\n###Processing Target Map Resampling###")
    tgt_map.resample_and_vec(dreso=gaussian_bandwidth)
    print()

    end_resample = time.time()
    print(f"Resample time: {end_resample - start_time:.2f} s")

    fitter = MapFitter(
        ref_map,
        tgt_map,
        angle_spacing,
        mode_val,
        remove_duplicates,
        ldp_file,
        ca_file,
        pdbin,
        num_threads,
        use_gpu,
        device,
        topn=refine_top,
        outdir=output_dir,
        save_mrc=mrcout,
        alpha=None,
        confine_angles=angle_limit,
        save_vec=save_models,
        batch_size=batch_size,
    )
    fitter.fit()


@app.command("ss")
def ss_command(
    map1: str = typer.Option(..., "-a", help="MAP1.mrc (large)"),
    map2: str = typer.Option(..., "-b", help="MAP2.mrc (small)"),
    ss_predictions_map2: str | None = typer.Option(
        None, "-npb", help="numpy array for Predictions for map 2"
    ),
    ss_predictions_map1: str = typer.Option(
        ..., "-npa", help="numpy array for Predictions for map 1"
    ),
    alpha: float = typer.Option(
        0.5,
        "-alpha",
        help="The weighting parameter for secondary structure in score mixing def=0.5",
    ),
    threshold_map1: float = typer.Option(0.0, "-t", help="Threshold of density map1"),
    threshold_map2: float = typer.Option(0.0, "-T", help="Threshold of density map2"),
    gaussian_bandwidth: float = typer.Option(
        16.0,
        "-g",
        help="Bandwidth of the Gaussian filter def=16.0, sigma = 0.5*[value entered]",
    ),
    voxel_spacing: float = typer.Option(
        7.0, "-s", help="Sampling voxel spacing def=7.0"
    ),
    angle_spacing: float = typer.Option(
        30.0, "-A", help="Sampling angle spacing def=30.0"
    ),
    refine_top: int = typer.Option(10, "-N", help="Refine Top [int] models def=10"),
    save_models: bool = typer.Option(
        False, "-S", help="Show topN models in PDB format def=false"
    ),
    eval_mode: bool = typer.Option(
        False, "-E", help="Evaluation mode of the current position def=false"
    ),
    output_dir: str | None = typer.Option(None, "-o", help="Output folder name"),
    remove_duplicates: bool = typer.Option(
        False, "-nodup", help="Remove duplicate models using heuristics def=false"
    ),
    gpu_id: int | None = typer.Option(
        None, "-gpu", help="GPU ID to use for CUDA acceleration def=0"
    ),
    pdbin: str | None = typer.Option(
        None, "-pdbin", help="Input PDB file to be transformed def=None"
    ),
    backbone_only: bool = typer.Option(
        False,
        "-bbonly",
        help="Whether to only use backbone atoms for simulated map def=false",
    ),
    num_threads: int = typer.Option(2, "-c", help="Number of threads to use def=2"),
    resolution: float | None = typer.Option(
        None,
        "-res",
        help="Resolution of the experimental map used to create simulated map from structure",
    ),
    score_file: str | None = typer.Option(
        None, "-score", help="Path to a list of transformations to score"
    ),
    batch_size: int | None = typer.Option(
        None, "-batch", help="Override batch size for GPU processing def=auto-detect"
    ),
) -> None:
    """Secondary structure matching command"""
    import random
    import string

    from .utils.pdb2ss import gen_npy

    # generate random string for temp folder
    rand_str = "".join(random.choices(string.ascii_letters + string.digits, k=8))

    # Print summary
    print_command_summary(
        Reference_Map_Path=os.path.abspath(map1),
        Target_Map_Path=os.path.abspath(map2),
        Reference_Map_Secondary_Structure_Path=os.path.abspath(ss_predictions_map1),
        Target_Map_Secondary_Structure_Path=os.path.abspath(ss_predictions_map2)
        if ss_predictions_map2
        else "Assigned from input structure",
        Threshold_of_Reference_Map=threshold_map1,
        Threshold_of_Target_Map=threshold_map2,
        Bandwidth_of_the_Gaussian_filter=gaussian_bandwidth,
        Search_space_voxel_spacing=voxel_spacing,
        Search_space_angle_spacing=angle_spacing,
        Refine_Top=f"{refine_top} models",
        Show_topN_models_in_PDB_format=save_models,
        Remove_duplicates=remove_duplicates,
        Number_of_threads_to_use=num_threads,
        Output_folder=os.path.abspath(output_dir) if output_dir else None,
        Using_GPU_ID=gpu_id,
        Transform_PDB_file=os.path.abspath(pdbin)
        if pdbin and os.path.exists(pdbin)
        else None,
    )

    tgt_ss = None
    # check if the second input is a structure file
    ext, _ = get_file_extension(map2)
    if ext in ["pdb", "cif"]:
        assert resolution is not None, (
            "Please specify resolution when using structure as input."
        )
        # simulate the map at target resolution
        sim_map_path = os.path.join(tempfile.gettempdir(), f"simu_map_{rand_str}.mrc")
        pdb2vol(map2, resolution, sim_map_path, backbone_only=backbone_only)
        assert os.path.exists(sim_map_path), (
            "Failed to create simulated map from structure."
        )

        # generate secondary structure assignment for the simulated map
        print("Generating secondary structure assignment for input structure...")
        tgt_ss = gen_npy(map2, resolution, verbose=True)

        map2 = sim_map_path

    if not pdbin or not os.path.exists(pdbin):
        print("No input PDB file, skipping transformation")
        pdbin = None
    else:
        print(f"Transform PDB file: {os.path.abspath(pdbin)}")

    # Setup GPU
    use_gpu, device = setup_gpu(gpu_id)

    start_check = time.time()

    # preliminary checks for input secondary structure predictions:
    ss_ref_pred = np.load(ss_predictions_map1).astype(np.float32)
    ss_ref_pred_max = np.max(ss_ref_pred, axis=-1)
    confidence = np.count_nonzero(ss_ref_pred_max > 0.9) / np.count_nonzero(
        ss_ref_pred_max
    )  # percent of voxels with probability > 0.9

    # Extract non-zero values and get argmax
    nonzero_vals = ss_ref_pred[ss_ref_pred > 0].reshape(-1, ss_ref_pred.shape[-1])
    preds = np.argmax(nonzero_vals, axis=-1)

    # Calculate frequency of each unique value
    unique_vals, counts = np.unique(preds, return_counts=True)
    normalized_counts = counts / counts.sum()

    norm_count_dict = dict(zip(unique_vals, normalized_counts))

    drna_content = norm_count_dict[3] if 3 in norm_count_dict else 0.0

    # set fallback to True if confidence is low or if there is a high amount of double-stranded RNA
    fallback = confidence < 0.05 or drna_content > 0.4

    start_resample_map = time.time()
    print(f"Check time: {start_resample_map - start_check:.2f} seconds")

    # construct mrc objects
    ref_map = EMmap(map1, ss_data=ss_ref_pred)
    # generate numpy array for target map if input is a PDB file
    if tgt_ss is not None:
        tgt_map = EMmap(map2, ss_data=tgt_ss.astype(np.float32))
    else:
        tgt_map = EMmap(map2, ss_data=np.load(ss_predictions_map2).astype(np.float32))

    # set voxel size
    ref_map.set_vox_size(thr=threshold_map1, voxel_size=voxel_spacing)
    tgt_map.set_vox_size(thr=threshold_map2, voxel_size=voxel_spacing)

    # unify dimensions
    unify_dims([ref_map, tgt_map], voxel_size=voxel_spacing)

    print("\n###Processing Reference Map Resampling###")
    ref_map.resample_and_vec(dreso=gaussian_bandwidth)
    print("\n###Processing Target Map Resampling###")
    tgt_map.resample_and_vec(dreso=gaussian_bandwidth)
    print()

    start_init_fitter = time.time()
    print(f"Resample time: {start_init_fitter - start_resample_map:.2f} seconds")

    fitter = MapFitter(
        ref_map,
        tgt_map,
        angle_spacing,
        "VecProduct",
        remove_duplicates,
        None,
        None,
        pdbin,
        num_threads,
        use_gpu,
        device,
        topn=refine_top,
        outdir=output_dir,
        save_mrc=False,
        alpha=alpha,
        save_vec=save_models,
        batch_size=batch_size,
    )

    start_fit = time.time()
    print(f"Init fitter time: {start_fit - start_init_fitter:.2f} seconds")

    if fallback:
        print("The prediction quality is low, falling back to original mode.")
        print(f"Confidence: {confidence}, Predicted DNA/RNA content: {drna_content}")
        fitter.fit()
    else:
        fitter.fit_ss()

    end_fit = time.time()
    print(f"Fit time: {end_fit - start_fit:.2f} seconds")
    print(f"Total time: {end_fit - start_check:.2f} seconds")


if __name__ == "__main__":
    app()
