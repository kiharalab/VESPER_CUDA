# Parity examples

One run per CLI feature, on the EMD-17127 map and the 8orj model shipped in `examples/`, so that
another engine (FlashVESPER) can be checked against this one feature by feature.

```bash
pip install -e .                      # or uv sync; the `vesper` command must be on PATH
bash examples/parity/run_all.sh OUT_DIR [GPU_ID] [EXAMPLE...]
python examples/parity/summarize.py OUT_DIR > OUT_DIR.tsv
```

`run_all.sh` writes the inputs into `OUT_DIR/inputs/` (see `make_inputs.py`), then runs each example
into `OUT_DIR/<example>/`: the `-o` folder, `stdout.txt` and `status.txt` (exit status, seconds).
Set `VESPER` to the command that runs another engine's CLI, e.g.
`VESPER="python -m flashvesper.cli" bash run_all.sh OTHER_DIR`, and compare the two tables:

```bash
diff <(python summarize.py OUT_DIR) <(python summarize.py OTHER_DIR)
```

The table lists, per example, the pose files in rank order (their names carry the Euler angles and
the translation), each pose's occupancy, `score.pkl`, `eval.json` and every `.npy` file's shape and
sum, with scores rounded to 4 decimals.

## Examples

Unless a row says otherwise, each example searches the structure, simulated at 5 A (`-res 5`), in
the map: `-a map.mrc -t 0.0289 -b model.pdb -T 0.01 -res 5 -s 2 -g 2 -A 30 -N 5 -c 8 -gpu GPU_ID`.

| Example | Options | Feature | What must hold |
|---|---|---|---|
| `search_V`, `search_C`, `search_P`, `search_O` | `-M V`, `C`, `P`, `O`, `-pdbin model.pdb` | the four score modes; pose files, `score.pkl` and the fit score / 100 as each atom's occupancy | |
| `search_L` | `-M L` | mode L: Laplacian filter on both maps, on every path | |
| `refine_R1` | `-M V -R 1` | refinement in 1-degree steps (default 2) | |
| `nodup` | `-M V -N 10 -nodup` | duplicate removal by rotation angle: a pose within 30 degrees of a kept one (and close in translation) is dropped | |
| `pdbin_cif` | `-M V -pdbin model.cif` | `.cif` poses rotated and laid out as `.pdb` poses | same poses and scores as `search_V` |
| `batch_3` | `-M V -batch 3` | GPU batch size override | same poses and scores as `search_V` |
| `multi_map` | `-a map.mrc,half.mrc -labels full,half -M C` | several maps on one grid in one call; outputs in `<o>/<label>/` | `full/` as `direct_fit`, `half/` as `half_alone` |
| `half_alone` | `-a half.mrc -M C` | one map alone, for `multi_map` | |
| `ldp` | `-M C -ldp ldp.pdb -ca ca.pdb` | LDP recall reranking; the fit score is the LDP recall | |
| `direct_fit` | `-M C --direct_fit` | DiffModeler's direct fit (no LDP reranking) | |
| `local_fit` | `-M C -A 1 -al 20 -N 1 -ldp ldp.pdb -ca ca.pdb` | DiffModeler's local re-fit: angles within 20 degrees, 1-degree steps | |
| `eval_half` | `-E -b half.mrc` | `-E`: score the target where it sits, print the scores, write `eval.json` | |
| `eval_moved` | `-E -b moved.mrc` | `-E` with the same density 10 A away (origin moved): scored where it sits | lower than `eval_half` |
| `vectors` | `-v <o>/vectors` | `-v`: both maps' resampled vectors as `.npy` files, then exit | |
| `diffmodeler` | `-a map.mrc,half.mrc -labels full,half -t 0.05 -T 0.01 -g 2 -s 2 -A 10 -N 100 -M C -ldp ldp.pdb -ca ca.pdb -res 1.0 -c 8` | DiffModeler's global fit of one chain (its `config/diffmodeler.json`) | |
| `cpu` | no `-gpu`, `-s 4 -M V` | the CPU search (FFTW planned without timing: same result on every run) | |

## The Python API

`api_examples.py OUT_DIR [GPU_ID] [EXAMPLE...]` (after `run_all.sh`, which writes the inputs) runs
`search_V`, `search_C`, `nodup`, `multi_map`, `ldp`, `direct_fit`, `local_fit` and `diffmodeler`
through `vesper.api.fit` and writes each pose from the returned transform as `vesper orig` would,
into `OUT_DIR/api_<example>/`. What must hold: `api_<example>` has the same rows as `<example>` in
the summary table (on the reference run the pose files are also byte-identical). Another engine's
API is checked by swapping the `fit` import.

## Inputs

`make_inputs.py` writes them from the shipped files:

- `map.mrc` is EMD-17127.
- `half.mrc` is that map with half its voxels set to 0. It has the same grid, so it can go into one `-a` call with `map.mrc`.
- `moved.mrc` is `half.mrc` with its origin moved by 10 A.
- `model.cif` is `8orj_stage_1.cif` with its chain `A0` renamed `A`. A PDB chain id has one character, so a pose of the original chain cannot be written.
- `model.pdb` is the same model as PDB.
- `ca.pdb` holds its CA atoms.
- `ldp.pdb` holds stand-in LDP points: voxel centres above the contour, every 3rd voxel along each axis. DiffModeler builds its LDP points from the map by mean shift.

## Reference run

`reference_summary.tsv` is `summarize.py` on a run on an RTX A6000 (torch 2.10, CUDA 13.0) of the
CLI examples at 8e464fa and the API examples at the commit that added `vesper.api.fit`. Every
example exits 0, every row of "What must hold" holds, and every `api_<example>` equals `<example>`.
Seconds per example:

| search_V | search_C | search_P | search_O | search_L | refine_R1 | nodup | pdbin_cif | batch_3 | multi_map |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 36 | 32 | 46 | 94 | 35 | 33 | 39 | 27 | 40 |

| half_alone | ldp | direct_fit | local_fit | eval_half | eval_moved | vectors | diffmodeler | cpu |
|---|---|---|---|---|---|---|---|---|
| 28 | 69 | 32 | 159 | 26 | 23 | 27 | 236 | 36 |

Through the API (the CLI's `-pdbin` files included):

| api_search_V | api_search_C | api_nodup | api_multi_map | api_ldp | api_direct_fit | api_local_fit | api_diffmodeler |
|---|---|---|---|---|---|---|---|
| 19 | 7 | 10 | 11 | 10 | 7 | 143 | 199 |
