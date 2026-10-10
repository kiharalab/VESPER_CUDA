"""Several maps searched together: each map gets what a search of it alone gives on the same grid."""

import os
import pickle

import pytest

from vesper.data.map import EMmap, unify_dims
from vesper.fitter import MapFitter

VOXEL, BANDWIDTH, THRESHOLD = 3.0, 8.0, 0.05


def _grid(inputs, names):
    """The named maps and the target with their search grid unified over all of them."""
    refs = [EMmap(str(inputs / f"{name}.mrc")) for name in names]
    tgt = EMmap(str(inputs / "target.mrc"))
    for em_map in [*refs, tgt]:
        em_map.set_vox_size(thr=THRESHOLD, voxel_size=VOXEL)
    unify_dims([*refs, tgt], voxel_size=VOXEL)
    return refs, tgt


def _fit(refs, tgt, mode, outdir, model, labels=None):
    fitter = MapFitter(
        refs,
        tgt,
        30.0,
        mode,
        True,
        None,
        None,
        model,
        2,
        False,
        None,
        topn=3,
        outdir=outdir,
        ref_labels=labels,
    )
    fitter.fit()
    return fitter


def _poses(final_list):
    return [
        (
            tuple(item["angle"]),
            tuple(int(v) for v in item["vox_trans"]),
            item["score"],
            tuple(item["real_trans"]),
        )
        for item in final_list
    ]


def _map_blocks(out):
    """Each map's part of a several-map printout, without its header line."""
    blocks = out.split("\n###Reference Map ")[1:]
    return [block.split("\n", 1)[1] for block in blocks]


def _outputs(directory):
    """Pose files (name -> content) and score.pkl (pose file name -> score)."""
    pdb_dir = os.path.join(directory, "PDB")
    poses = {}
    for name in sorted(os.listdir(pdb_dir)):
        with open(os.path.join(pdb_dir, name)) as f:
            poses[name] = f.read()
    with open(os.path.join(directory, "score.pkl"), "rb") as f:
        scores = [(os.path.basename(k), v) for k, v in pickle.load(f).items()]
    return poses, scores


@pytest.mark.parametrize(("mode", "labels"), [("V", ["a", "c"]), ("C", None)])
def test_each_map_gets_its_own_search_on_the_shared_grid(
    inputs, tmp_path, capsys, mode, labels
):
    refs, tgt = _grid(inputs, ["a", "c"])
    # c alone would be searched on a smaller grid; the shared grid is a's
    c_alone, tgt_alone = _grid(inputs, ["c"])
    assert max(c_alone[0].new_dim, tgt_alone.new_dim) < tgt.new_dim
    for em_map in [*refs, tgt]:
        em_map.resample_and_vec(dreso=BANDWIDTH)
    model = str(inputs / "model.pdb")
    dirs = labels or ["ref0", "ref1"]

    capsys.readouterr()
    together = _fit(refs, tgt, mode, str(tmp_path / "together"), model, labels)
    blocks = _map_blocks(capsys.readouterr().out)

    assert sorted(os.listdir(tmp_path / "together")) == sorted(dirs)
    assert len(blocks) == 2
    for i, ref in enumerate(refs):
        alone = _fit(ref, tgt, mode, str(tmp_path / f"alone{i}"), model)
        out = capsys.readouterr().out
        assert "###Reference Map" not in out

        assert _poses(together.final_lists[i]) == _poses(alone.final_list)
        assert blocks[i] == out[out.index("###Preliminary Results Summary###") :]
        # one map writes into outdir itself, several into outdir/<label>
        assert sorted(os.listdir(tmp_path / f"alone{i}")) == ["PDB", "score.pkl"]
        assert _outputs(tmp_path / "together" / dirs[i]) == _outputs(
            tmp_path / f"alone{i}"
        )


def test_rerun_removes_the_old_score_pkl_before_the_search_ends(
    inputs, tmp_path, monkeypatch
):
    """A rerun that dies must not leave the last run's score.pkl, which reads as fit done"""
    refs, tgt = _grid(inputs, ["a", "c"])
    for em_map in [*refs, tgt]:
        em_map.resample_and_vec(dreso=BANDWIDTH)
    model = str(inputs / "model.pdb")
    out = tmp_path / "out"
    for sub in ("a", "c"):
        (out / sub).mkdir(parents=True)
        for name in ("score.pkl", "score.pkl.tmp"):
            (out / sub / name).write_bytes(b"old")

    def die(self, *args, **kwargs):
        raise RuntimeError("rerun died")

    monkeypatch.setattr(MapFitter, "_save_topn_pdb", die)
    with pytest.raises(RuntimeError, match="rerun died"):
        _fit(refs, tgt, "V", str(out), model, ["a", "c"])

    assert [sorted(os.listdir(out / sub)) for sub in ("a", "c")] == [[], []]


def test_score_pkl_holds_the_ldp_recall_times_100(inputs, tmp_path, monkeypatch):
    """LDP recall needs a GPU; its values are stubbed, the path to score.pkl is not"""

    def stub_recall(self, results, sort=False, progress_bar=True):
        for result in results:
            result["ldp_recall"] = 0.8123456

    monkeypatch.setattr(MapFitter, "_calc_ldp_recall", stub_recall)
    refs, tgt = _grid(inputs, ["a"])
    for em_map in [*refs, tgt]:
        em_map.resample_and_vec(dreso=BANDWIDTH)
    fitter = MapFitter(
        refs, tgt, 30.0, "V", True, None, None, str(inputs / "model.pdb"), 2,
        False, None, topn=3, outdir=str(tmp_path),
    )  # fmt: skip
    fitter.ldp_recall_mode = True
    fitter.fit()

    with open(tmp_path / "score.pkl", "rb") as f:
        scores = list(pickle.load(f).values())
    assert scores == [float("0.812346") * 100] * 3
