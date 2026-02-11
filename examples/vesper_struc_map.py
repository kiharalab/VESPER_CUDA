"""Run VESPER alignment on CIF structure and MAP density map, save transformed PDB."""

import os

from vesper.api import run_vesper_fit
from vesper.data.io import save_rotated_pdb


def run_vesper_on_structure_map(
    map_path: str,
    cif_path: str,
    output_dir: str = ".",
    contour_level: float = 0.0289,
    resolution: float = 5.0,
    num_conformation: int = 10,
    gpu_id: int = 0,
    angle_spacing: float = 5.0,
    voxel_spacing: float = 2.0,
    gaussian_bandwidth: float = 1.0,
    num_threads: int = 8,
) -> None:
    """
    Run VESPER alignment on a CIF structure against an MRC/MAP density map.

    Args:
        map_path: Path to the reference MRC/MAP file (can be .gz compressed)
        cif_path: Path to the input CIF structure file (can be .gz compressed)
        output_dir: Directory to save output files
        contour_level: Density threshold for reference map
        resolution: Resolution for simulating map from structure
        num_conformation: Number of top conformations to refine
        gpu_id: GPU device ID
        angle_spacing: Angular sampling interval in degrees
        voxel_spacing: Voxel spacing for sampling
        gaussian_bandwidth: Gaussian filter bandwidth
        num_threads: Number of CPU threads
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get base filename for output
    base_name = os.path.splitext(os.path.basename(cif_path))[0]
    if base_name.endswith(".gz"):
        base_name = os.path.splitext(base_name)[0]

    print(f"Running VESPER alignment...")
    print(f"  Map: {map_path}")
    print(f"  Structure: {cif_path}")
    print(f"  Contour level: {contour_level}")
    print(f"  Resolution: {resolution}")

    # Run VESPER fitting
    rotation, translation = run_vesper_fit(
        map_path=map_path,
        input_pdb=cif_path,
        contour_level=contour_level,
        resolution=resolution,
        num_conformation=num_conformation,
        gpu_id=gpu_id,
        angle_spacing=angle_spacing,
        voxel_spacing=voxel_spacing,
        gaussian_bandwidth=gaussian_bandwidth,
        num_threads=num_threads,
        only_best=True,
    )

    print(f"\nVESPER fitting succeeded!")
    print(f"\nBest rotation matrix:")
    print(f"  [{rotation[0, 0]:.6f}, {rotation[0, 1]:.6f}, {rotation[0, 2]:.6f}]")
    print(f"  [{rotation[1, 0]:.6f}, {rotation[1, 1]:.6f}, {rotation[1, 2]:.6f}]")
    print(f"  [{rotation[2, 0]:.6f}, {rotation[2, 1]:.6f}, {rotation[2, 2]:.6f}]")
    print(f"\nBest translation: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]")

    # Save transformed PDB (without .pdb extension - save_rotated_pdb adds it)
    output_pdb = os.path.join(output_dir, f"{base_name}_vesper_transformed")
    save_rotated_pdb(cif_path, rotation, translation, output_pdb, model_num=0)
    print(f"\nTransformed PDB saved to: {output_pdb}.pdb")

    # Also save transformation parameters
    params_file = os.path.join(output_dir, f"{base_name}_vesper_params.txt")
    with open(params_file, 'w') as f:
        f.write(f"# VESPER transformation parameters\n")
        f.write(f"# Map: {map_path}\n")
        f.write(f"# Structure: {cif_path}\n")
        f.write(f"# Contour level: {contour_level}\n")
        f.write(f"# Resolution: {resolution}\n")
        f.write(f"\n# Rotation matrix:\n")
        f.write(f"{rotation[0, 0]:.6f} {rotation[0, 1]:.6f} {rotation[0, 2]:.6f}\n")
        f.write(f"{rotation[1, 0]:.6f} {rotation[1, 1]:.6f} {rotation[1, 2]:.6f}\n")
        f.write(f"{rotation[2, 0]:.6f} {rotation[2, 1]:.6f} {rotation[2, 2]:.6f}\n")
        f.write(f"\n# Translation:\n")
        f.write(f"{translation[0]:.6f} {translation[1]:.6f} {translation[2]:.6f}\n")
    print(f"Transformation parameters saved to: {params_file}")


if __name__ == "__main__":
    # Example usage with the provided files
    # Use absolute paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Note: VESPER now supports .gz compressed files automatically
    MAP_PATH = os.path.join(SCRIPT_DIR, "emd_17127.map.gz")
    CIF_PATH = os.path.join(SCRIPT_DIR, "8orj_stage_1.cif")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "examples_output")
    VESPER_CONTOUR_LEVEL = 0.0289

    run_vesper_on_structure_map(
        map_path=MAP_PATH,
        cif_path=CIF_PATH,
        output_dir=OUTPUT_DIR,
        contour_level=VESPER_CONTOUR_LEVEL,
        resolution=5.0,
        num_conformation=10,
        gpu_id=0,
        angle_spacing=5.0,
        voxel_spacing=2.0,
        gaussian_bandwidth=1.0,
        num_threads=8,
    )
