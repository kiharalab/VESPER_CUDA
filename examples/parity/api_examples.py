"""Run parity examples through the Python API (vesper.api.fit), laid out as run_all.sh's.

    python api_examples.py OUT_DIR [GPU_ID] [EXAMPLE...]

Run run_all.sh first: it writes OUT_DIR/inputs. Each example here writes OUT_DIR/api_<example>/
as `vesper orig -pdbin model.pdb -o` would: PDB/#<rank>_rx.._ry.._rz.._tx.._ty.._tz...pdb from the
returned transforms (vesper.data.io.save_rotated_pdb, occupancy = fit score / 100) and
score.pkl, in <label>/ for several maps. summarize.py then gives api_<example> the same rows as
<example>, which is what must hold.
"""

import os
import pickle
import sys
import time

from vesper.api import fit
from vesper.data.io import save_rotated_pdb

# run_all.sh's examples: their -b, search and options, as fit() arguments
QUICK = dict(
    ref_threshold=0.0289,
    target_threshold=0.01,
    resolution=5.0,
    voxel_spacing=2.0,
    bandwidth=2.0,
    angle_spacing=30.0,
    top_n=5,
    num_threads=8,
)
EXAMPLES = {
    "search_V": dict(QUICK, mode="V"),
    "search_C": dict(QUICK, mode="C"),
    "nodup": dict(QUICK, mode="V", top_n=10, remove_duplicates=True),
    "multi_map": dict(
        QUICK, maps=["map.mrc", "half.mrc"], labels=["full", "half"], mode="C"
    ),
    "ldp": dict(QUICK, mode="C", ldp="ldp.pdb", ca="ca.pdb"),
    "direct_fit": dict(QUICK, mode="C", direct_fit=True),
    "local_fit": dict(
        QUICK,
        mode="C",
        angle_spacing=1.0,
        angle_limit=20.0,
        top_n=1,
        ldp="ldp.pdb",
        ca="ca.pdb",
    ),
    "diffmodeler": dict(
        maps=["map.mrc", "half.mrc"],
        labels=["full", "half"],
        ref_threshold=0.05,
        target_threshold=0.01,
        bandwidth=2.0,
        voxel_spacing=2.0,
        angle_spacing=10.0,
        top_n=100,
        mode="C",
        ldp="ldp.pdb",
        ca="ca.pdb",
        num_threads=8,
        resolution=1.0,
    ),
}


def write_poses(folder, poses, model):
    """Pose files and score.pkl, as `vesper orig -pdbin model` writes them"""
    os.makedirs(os.path.join(folder, "PDB"), exist_ok=True)
    scores = {}
    for i, pose in enumerate(poses):
        angle = "_".join(f"r{a}{int(v)}" for a, v in zip("xyz", pose["angle"]))
        trans = "_".join(f"t{a}{v:.3f}" for a, v in zip("xyz", pose["translation"]))
        path = os.path.join(folder, "PDB", f"#{i}_{angle}_{trans}")
        occupancy = pose["fit_score"] / 100
        save_rotated_pdb(
            model, pose["rotation"], pose["translation"], path, i, occupancy
        )
        scores[os.path.abspath(path + ".pdb")] = pose["fit_score"]
    with open(os.path.join(folder, "score.pkl"), "wb") as f:
        pickle.dump(scores, f)


def run(name, inputs, out_dir, gpu_id):
    kwargs = dict(EXAMPLES[name])
    maps = [os.path.join(inputs, m) for m in kwargs.pop("maps", ["map.mrc"])]
    for key in ("ldp", "ca"):
        if key in kwargs:
            kwargs[key] = os.path.join(inputs, kwargs[key])
    model = os.path.join(inputs, "model.pdb")
    folder = os.path.join(out_dir, f"api_{name}")
    os.makedirs(folder, exist_ok=True)
    start = time.time()
    status = 0
    try:
        poses = fit(maps, model, gpu_id=gpu_id, **kwargs)
        labels = kwargs.get("labels")
        for i, map_poses in enumerate(poses):
            label_folder = os.path.join(folder, labels[i]) if labels else folder
            write_poses(label_folder, map_poses, model)
    except Exception as error:
        status = 1
        print(f"{name}: {error!r}")
    seconds = int(time.time() - start)
    with open(os.path.join(folder, "status.txt"), "w") as f:
        f.write(f"exit={status} seconds={seconds}\n")
    print(f"api_{name}: exit {status}, {seconds} s")


if __name__ == "__main__":
    out = sys.argv[1]
    gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for example in sys.argv[3:] or EXAMPLES:
        run(example, os.path.join(out, "inputs"), out, gpu)
