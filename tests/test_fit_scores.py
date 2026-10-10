"""DiffModeler's outputs: the fit score, score.pkl and the occupancy of saved poses.

DiffModeler's read_score() takes a pose's fit score from the printout: its LDP recall, or else its
"Normalized Score" (6 decimals), x100. score.pkl and the occupancy column carry that same number.
"""

import os
import pickle

import numpy as np
import pytest
import synthetic

from vesper.data.io import save_rotated_pdb, save_score_pkl
from vesper.fitter import dm_fit_scores

SEARCH = np.array([1.0, 2.0, 3.0, 6.0])  # mean 3, population std sqrt(3.5)
AVE, STD = SEARCH.mean(), SEARCH.std()


def test_normalized_score_rounded_as_printed_times_100():
    top = [{"score": 6.0}, {"score": 3.0}]
    assert dm_fit_scores(top, AVE, STD) == [float("1.603567") * 100, 0.0]


def test_ldp_recall_when_any_top_pose_has_one():
    top = [{"score": 6.0, "ldp_recall": 0.8123456}, {"score": 3.0, "ldp_recall": 0.0}]
    assert dm_fit_scores(top, AVE, STD) == [float("0.812346") * 100, 0.0]


def test_all_zero_ldp_recall_falls_back_to_normalized_score():
    top = [{"score": 6.0, "ldp_recall": 0.0}]
    assert dm_fit_scores(top, AVE, STD) == [float("1.603567") * 100]


def test_constant_search_scores_give_zero():
    assert dm_fit_scores([{"score": 2.0}], 2.0, 0.0) == [0.0]


def test_score_pkl_is_best_first_and_appears_complete(tmp_path, monkeypatch):
    path = str(tmp_path / "score.pkl")
    scores = {"/p/a.pdb": 1.0, "/p/b.pdb": 3.0, "/p/c.pdb": 2.0}
    best_first = [("/p/b.pdb", 3.0), ("/p/c.pdb", 2.0), ("/p/a.pdb", 1.0)]
    renames = []
    replace = os.replace

    def watch_replace(src, dst):
        # the rename is the only write of score.pkl, and its source is complete
        with open(src, "rb") as f:
            renames.append(
                (src, dst, os.path.exists(dst), list(pickle.load(f).items()))
            )
        replace(src, dst)

    monkeypatch.setattr(os, "replace", watch_replace)
    save_score_pkl(scores, path)

    assert renames == [(path + ".tmp", path, False, best_first)]
    with open(path, "rb") as f:
        assert list(pickle.load(f).items()) == best_first
    assert os.listdir(tmp_path) == ["score.pkl"]


def _write_model(path):
    if path.endswith(".cif"):
        synthetic.write_model_cif(path, synthetic.model_coords())
    else:
        synthetic.write_model(path, synthetic.model_coords())


def _atom_fields(path):
    """Fields of each atom line; occupancy is the third from the end."""
    with open(path) as f:
        return [line.split() for line in f if line.startswith(("ATOM", "HETATM"))]


@pytest.mark.parametrize("ext", ["pdb", "cif"])
def test_occupancy_is_the_given_score(tmp_path, ext):
    model = str(tmp_path / f"model.{ext}")
    _write_model(model)
    pose = str(tmp_path / "pose")
    save_rotated_pdb(model, np.eye(3), np.zeros(3), pose, 0, occupancy=-0.853)

    assert [fields[-3] for fields in _atom_fields(pose + ".pdb")] == ["-0.85"] * 4


@pytest.mark.parametrize("ext", ["pdb", "cif"])
def test_occupancy_changes_nothing_else(tmp_path, ext):
    model = str(tmp_path / f"model.{ext}")
    _write_model(model)
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    save_rotated_pdb(model, rot, np.ones(3), str(tmp_path / "plain"), 0)
    save_rotated_pdb(model, rot, np.ones(3), str(tmp_path / "scored"), 0, occupancy=0.5)

    plain = _atom_fields(str(tmp_path / "plain.pdb"))
    scored = _atom_fields(str(tmp_path / "scored.pdb"))
    assert [fields[-3] for fields in plain] == ["1.00"] * 4
    assert [fields[-3] for fields in scored] == ["0.50"] * 4
    assert [f[:-3] + f[-2:] for f in scored] == [f[:-3] + f[-2:] for f in plain]
