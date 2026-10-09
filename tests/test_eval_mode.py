"""-E scores the target where it sits, prints the six scores and exits without a search."""

import json
import os
import re
import subprocess
import sys

import pytest
import synthetic

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = ["-t", "0.05", "-T", "0.05", "-s", "3", "-g", "8", "-E"]
SCORES = re.compile(
    r"Overlap:\s+(\S+) CC:\s+(\S+) PCC:\s+(\S+) N:\s+(\S+) Total:\s+(\S+) Dot:\s+(\S+)"
)


def run_eval(directory, target, *extra):
    synthetic.write_inputs(str(directory))
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "run_cli.py"), "orig", "-a", "a.mrc"]
        + ["-b", target, *ARGS, *extra],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout, [float(v) for v in SCORES.search(proc.stdout).groups()]


def test_map_on_itself_scores_perfectly(tmp_path):
    stdout, (overlap, cc, pcc, n, total, dot) = run_eval(tmp_path, "a.mrc")
    assert overlap == 1.0 and n == total > 0
    assert abs(cc - 1.0) < 1e-4 and abs(pcc - 1.0) < 1e-4
    assert dot > 0
    assert "Start Refining" not in stdout and "Final Results" not in stdout


def test_misplaced_target_scores_lower_and_json_matches(tmp_path):
    _, (overlap, cc, _, n, total, _) = run_eval(
        tmp_path, "target.mrc", "-o", "out", "-M", "C"
    )
    assert 0 < overlap < 1 and n < total and cc < 0.9
    with open(tmp_path / "out" / "eval.json") as f:
        record = json.load(f)
    assert record["overlap"] == overlap
    assert record["cc"] == pytest.approx(cc, rel=1e-6)  # printed as float32
    assert record["n"] == n and record["total"] == total
    assert record["ref"] == str(tmp_path / "a.mrc")
    assert record["target"] == str(tmp_path / "target.mrc")
    assert record["mode"] == "C"
    assert set(os.listdir(tmp_path / "out")) == {"eval.json"}
