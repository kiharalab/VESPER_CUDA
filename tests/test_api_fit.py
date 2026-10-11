"""vesper.api.fit returns the poses `vesper orig` writes: same order, scores and pose files."""

import os
import pickle
import shutil
import tempfile

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from vesper import api
from vesper.cli import app
from vesper.data.io import save_rotated_pdb

SEARCH = dict(
    ref_threshold=0.05,
    target_threshold=0.05,
    bandwidth=8.0,
    voxel_spacing=3.0,
    angle_spacing=30.0,
    top_n=3,
)
SEARCH_ARGS = "-t 0.05 -T 0.05 -g 8 -s 3 -A 30 -N 3"

CASES = {
    "one map": (dict(ref_maps="a.mrc"), "-a a.mrc -M V", [""]),
    "two maps": (
        dict(ref_maps=["a.mrc", "b.mrc"], labels=["a", "b"], mode="C"),
        "-a a.mrc,b.mrc -labels a,b -M C",
        ["a", "b"],
    ),
    "ldp": (
        dict(ref_maps="a.mrc", ldp="model.pdb", ca="model.pdb", mode="C"),
        "-a a.mrc -M C -ldp model.pdb -ca model.pdb",
        [""],
    ),
    "nodup": (
        dict(ref_maps="a.mrc", remove_duplicates=True),
        "-a a.mrc -M V -nodup",
        [""],
    ),
}


def _case(name):
    if name == "ldp":  # LDP recall runs on the GPU only
        gpu = torch.cuda.is_available()
        return pytest.param(name, marks=pytest.mark.skipif(not gpu, reason="GPU"))
    return name


@pytest.mark.parametrize("case", [_case(name) for name in CASES])
def test_api_poses_are_the_cli_pose_files(inputs, tmp_path, monkeypatch, case):
    for name in ("a.mrc", "b.mrc", "target.mrc", "model.pdb"):
        shutil.copy(inputs / name, tmp_path)
    monkeypatch.chdir(tmp_path)
    kwargs, cli_args, folders = CASES[case]

    gpu = "ldp" in kwargs
    gpu_args = " -gpu 0" if gpu else ""
    result = CliRunner().invoke(
        app,
        f"orig {cli_args} -b target.mrc {SEARCH_ARGS} -pdbin model.pdb -o out{gpu_args}".split(),
    )
    assert result.exit_code == 0, result.output
    poses = api.fit(target="target.mrc", gpu_id=0 if gpu else None, **SEARCH, **kwargs)

    assert len(poses) == len(folders)
    for map_poses, folder in zip(poses, folders):
        out = os.path.join("out", folder)
        with open(os.path.join(out, "score.pkl"), "rb") as f:
            scores = pickle.load(f)
        assert len(map_poses) == len(scores) > 0
        for i, (pose, (path, fit_score)) in enumerate(zip(map_poses, scores.items())):
            angle = "_".join(f"r{a}{int(v)}" for a, v in zip("xyz", pose["angle"]))
            trans = "_".join(f"t{a}{v:.3f}" for a, v in zip("xyz", pose["translation"]))
            assert os.path.basename(path) == f"#{i}_{angle}_{trans}.pdb"
            assert pose["fit_score"] == fit_score
            # the pose written from the returned transform is the CLI's pose file
            mine = str(tmp_path / f"mine_{folder}_{i}")
            occupancy = pose["fit_score"] / 100
            save_rotated_pdb(
                "model.pdb", pose["rotation"], pose["translation"], mine, i, occupancy
            )
            with open(mine + ".pdb") as f, open(path) as g:
                assert f.read() == g.read()
        if "ldp" in kwargs:
            assert all("ldp_recall" in pose for pose in map_poses)


def test_structure_target_is_simulated_and_its_map_removed(inputs):
    before = set(os.listdir(tempfile.gettempdir()))
    poses = api.fit(
        str(inputs / "a.mrc"),
        str(inputs / "model.pdb"),
        resolution=6.0,
        **{**SEARCH, "target_threshold": 0.0},
    )
    assert len(poses) == 1 and len(poses[0]) == 3
    assert np.allclose(np.linalg.det(poses[0][0]["rotation"]), 1.0)
    after = set(os.listdir(tempfile.gettempdir()))
    assert not {name for name in after - before if name.startswith("simu_map_")}
