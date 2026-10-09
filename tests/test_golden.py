"""Against 880f7d9: one map without the new flags gives the same printout and pose files.

data/golden_880f7d9 holds 880f7d9's runs (CPU, through tests/run_cli.py) on the synthetic inputs,
with the printout normalized as below. Pose files now carry DiffModeler's fit score in their
occupancy column, and score.pkl is new; everything else must match byte for byte, except -nodup's kept count (and B's #2),
which changed when duplicates were found by rotation angle (test_dedup.py). Maps a and b
get the same search grid, so searched together each must give its own golden.
"""

import os
import pickle
import re
import subprocess
import sys

import pytest
import synthetic

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "data", "golden_880f7d9")
COMMON = ["-b", "target.mrc", "-t", "0.05", "-T", "0.05", "-s", "3", "-g", "8"]
COMMON += ["-A", "30", "-N", "3", "-c", "2", "-pdbin", "model.pdb", "-o", "out"]
CASES = {
    "A_V_nodup": ["-a", "a.mrc", "-nodup"],
    "B_V_nodup": ["-a", "b.mrc", "-nodup"],
    "A_C": ["-a", "a.mrc", "-M", "C"],
}
NORMALIZED = re.compile(r"Normalized Score= (\S+)")


def normalize(stdout, directory):
    """Drop the command line and the timing, and name the run folder <DIR>."""
    for path in sorted({os.path.realpath(directory), str(directory)}, key=len)[::-1]:
        stdout = stdout.replace(path, "<DIR>")
    lines = stdout.split("\n")
    del lines[lines.index("### Command ###") + 1]
    return "\n".join(line for line in lines if not line.startswith("Resample time:"))


def run(directory, args):
    """Run `vesper orig` on the synthetic inputs in directory; return the normalized stdout."""
    synthetic.write_inputs(str(directory))
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "run_cli.py"), "orig", *args, *COMMON],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return normalize(proc.stdout, directory)


def golden(case):
    with open(os.path.join(GOLDEN, case, "stdout.txt")) as f:
        return f.read()


def check_poses(out_dir, case, printout):
    """Pose files as in the golden but for occupancy, which is score.pkl's fit score / 100."""
    pdb_dir = os.path.join(out_dir, "PDB")
    names = sorted(os.listdir(pdb_dir))
    assert names == sorted(os.listdir(os.path.join(GOLDEN, case, "PDB")))

    # score.pkl: absolute pose path -> printed normalized score x 100, best first
    with open(os.path.join(out_dir, "score.pkl"), "rb") as f:
        scores = pickle.load(f)
    printed = [float(v) * 100 for v in NORMALIZED.findall(printout)]
    by_rank = sorted(names, key=lambda name: int(name[1:].split("_")[0]))
    assert len(printed) == len(by_rank)
    expected = {
        os.path.abspath(os.path.join(pdb_dir, name)): score
        for name, score in zip(by_rank, printed)
    }
    assert scores == expected
    assert list(scores.values()) == sorted(scores.values(), reverse=True)

    for name in names:
        with open(os.path.join(pdb_dir, name)) as f:
            lines = f.read().split("\n")
        with open(os.path.join(GOLDEN, case, "PDB", name)) as f:
            golden_lines = f.read().split("\n")
        occupancy = f"{scores[os.path.abspath(os.path.join(pdb_dir, name))] / 100:6.2f}"
        assert len(lines) == len(golden_lines)
        for line, golden_line in zip(lines, golden_lines):
            if line.startswith(("ATOM", "HETATM")):
                assert line[54:60] == occupancy
                assert line[:54] + line[60:] == golden_line[:54] + golden_line[60:]
            else:
                assert line == golden_line


@pytest.mark.parametrize("case", CASES)
def test_one_map_matches_880f7d9(tmp_path, case):
    printout = run(tmp_path, CASES[case])

    assert printout == golden(case)
    assert sorted(os.listdir(tmp_path / "out")) == ["PDB", "score.pkl"]
    check_poses(str(tmp_path / "out"), case, printout)


def test_two_maps_on_one_grid_each_match_880f7d9(tmp_path):
    # a and b get the same grid alone, so together they search on it too
    nvox = [golden(case).split("Nvox= ")[1].split("\n")[0] for case in CASES]
    assert nvox[0] == nvox[1]

    printout = run(tmp_path, ["-a", "a.mrc,b.mrc", "-labels", "a,b", "-nodup"])

    assert "Reference_Map_Paths: <DIR>/a.mrc,<DIR>/b.mrc\n" in printout
    assert "Reference_Map_Labels: a,b\n" in printout
    assert sorted(os.listdir(tmp_path / "out")) == ["a", "b"]
    blocks = printout.split("\n###Reference Map ")[1:]
    for label, case, block in zip("ab", ["A_V_nodup", "B_V_nodup"], blocks):
        header, block = block.split("\n", 1)
        assert header == f"{'ab'.index(label)}: {label}###"
        expected = golden(case)
        assert block == expected[expected.index("###Preliminary Results Summary###") :]
        check_poses(str(tmp_path / "out" / label), case, block)
