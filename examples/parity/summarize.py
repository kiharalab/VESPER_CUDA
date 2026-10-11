"""Summarize the outputs of run_all.sh as one tab-separated table, to compare two engines.

    python summarize.py OUT_DIR > summary.tsv
    diff <(python summarize.py VESPER_OUT) <(python summarize.py OTHER_OUT)

Columns: example, what, value. For every example: its exit status, and in each output folder the
pose files in rank order (their names carry the angles and the translation), every pose's
occupancy (DiffModeler's fit score / 100), score.pkl, eval.json, and each .npy file's shape and
sum. Scores are rounded to 4 decimals, so last-digit float differences between GPUs do not show.
"""

import glob
import json
import os
import pickle
import re
import sys

import numpy as np


def rows(example_dir):
    name = os.path.basename(example_dir)
    with open(os.path.join(example_dir, "status.txt")) as f:
        yield name, "status", f.read().split()[0]
    for root, _, files in sorted(os.walk(example_dir)):
        rel = os.path.relpath(root, example_dir)
        rel = "" if rel == "." else rel + "/"
        poses = sorted(
            (
                f
                for f in files
                if f.endswith(".pdb") and os.path.basename(root) == "PDB"
            ),
            key=lambda f: int(re.match(r"#(\d+)", f).group(1)),
        )
        for pose in poses:
            yield name, f"{rel}pose", pose
            with open(os.path.join(root, pose)) as f:
                occupancies = {line[54:60] for line in f if line.startswith("ATOM")}
            yield name, f"{rel}{pose}/occupancy", ",".join(sorted(occupancies))
        if "score.pkl" in files:
            with open(os.path.join(root, "score.pkl"), "rb") as f:
                scores = pickle.load(f)
            for path, score in scores.items():
                yield name, f"{rel}score.pkl", f"{os.path.basename(path)}\t{score:.4f}"
        if "eval.json" in files:
            with open(os.path.join(root, "eval.json")) as f:
                record = json.load(f)
            for key, value in record.items():
                if isinstance(value, float):
                    value = f"{value:.4f}"
                elif isinstance(value, str):
                    value = os.path.basename(value)
                yield name, f"{rel}eval.json/{key}", value
        for npy in sorted(f for f in files if f.endswith(".npy")):
            array = np.load(os.path.join(root, npy))
            yield name, f"{rel}{npy}", f"shape={array.shape} sum={array.sum():.4f}"


if __name__ == "__main__":
    out_dir = sys.argv[1]
    for status in sorted(glob.glob(os.path.join(out_dir, "*", "status.txt"))):
        for row in rows(os.path.dirname(status)):
            print("\t".join(str(v) for v in row))
