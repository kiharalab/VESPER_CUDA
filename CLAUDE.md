# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VESPER is a CUDA-accelerated computational tool using local vector-based algorithms for accurately identifying global and local alignment of cryo-electron microscopy (EM) maps. This is a reimplementation in Python of the original VESPER tool.

**Key Features:**

- Vector-based alignment algorithm for EM maps
- CUDA acceleration for GPU processing
- Two modes: Original (`orig`) and Secondary Structure matching (`ss`)
- Supports various scoring modes: Vector Product, Overlap, Cross Correlation, Pearson Correlation, Laplacian

## Architecture

The project uses a modern Python package structure with `src/vesper/` layout:

```
src/vesper/
├── cli.py              # Typer-based CLI with two commands: orig and ss
├── fitter.py           # MapFitter class - core alignment algorithm
├── data/
│   ├── __init__.py
│   ├── io.py           # PDB/MRC file I/O operations
│   └── map.py          # EMmap class - MRC file handling and vector representation
└── utils/              # Utility modules
    ├── pdb2vol.py      # PDB to volume conversion
    ├── pdb2ss.py       # Secondary structure generation
    ├── segmap.py       # Map segmentation
    ├── unify.py        # Map dimension unification
    ├── rmsd.py         # RMSD calculations
    └── utils.py        # General utilities (euler angles, scoring)
```

### Core Classes

- **EMmap** (`src/vesper/data/map.py`): Represents MRC density map data, handles resampling, vector representation, and Gaussian filtering
- **MapFitter** (`src/vesper/fitter.py`): Main alignment engine that performs rotational and translational search using FFT-based optimization

## Development Workflow

### Environment Setup

The project uses `uv` for Python package management. Python 3.12+ is required.

```bash
# Install all dependencies (including dev) from GitHub
uv pip install git+https://github.com/kiharalab/VESPER_CUDA.git

# Or clone and install locally in editable mode
git clone https://github.com/kiharalab/VESPER_CUDA.git
cd VESPER_CUDA
uv sync --all-packages --all-extras --dev
```

### Running Commands

Always use `uv run` to execute commands:

```bash
# Run the CLI
uv run vesper --help
uv run vesper orig --help
uv run vesper ss --help

# Or use Python module syntax
uv run python -m vesper --help
```

### Code Quality

```bash
# Lint code
uv run ruff check src/vesper

# Lint and auto-fix
uv run ruff check --fix src/vesper

# Format code
uv run ruff format src/vesper

# Type checking (if using ty)
uv run ty check src/vesper
```

### Building and Installing

```bash
# Install package in editable mode (done automatically by uv sync)
uv sync

# The CLI command 'vesper' is registered via pyproject.toml [project.scripts]
# Can be invoked as: uv run vesper <command>
```

## CLI Usage

### Original Mode (Vector-based alignment)

```bash
uv run vesper orig -a MAP1.mrc -b MAP2.mrc [options]
```

Common options:

- `-t` / `-T`: Density thresholds for MAP1/MAP2
- `-g`: Gaussian filter bandwidth (default: 16.0)
- `-s`: Voxel spacing (default: 7.0)
- `-A`: Angle spacing (default: 30.0)
- `-N`: Number of top models to refine (default: 10)
- `-gpu`: GPU ID for CUDA acceleration
- `-M`: Scoring mode (V=Vector Product, O=Overlap, C=Cross Correlation, P=Pearson, L=Laplacian)

### Secondary Structure Mode

```bash
uv run vesper ss -a MAP1.mrc -b MAP2.mrc -npa predictions1.npy -npb predictions2.npy [options]
```

Requires pre-computed secondary structure predictions in numpy format (from Emap2sec+).

### Input from Structure Files

Both modes support PDB/CIF input for map2 (simulates map at specified resolution):

```bash
uv run vesper orig -a MAP1.mrc -b structure.pdb -res 3.5 [options]
```

## Key Dependencies

- **PyTorch**: CUDA acceleration (with custom index for CPU/CUDA variants)
- **mrcfile**: MRC file format I/O
- **numba**: JIT compilation for performance
- **scipy**: Scientific computing (FFT, rotations)
- **typer**: Modern CLI framework
- **biopython**: PDB file processing
- **gemmi**: CIF file processing

## Implementation Notes

### GPU Processing

The MapFitter class automatically detects CUDA availability. GPU processing is enabled with `-gpu` flag and uses batch processing for efficient memory management. Rotation matrices are cached for performance.

### Scoring Modes

Five scoring algorithms are implemented (see Mode enum in `cli.py`):

1. Vector Product (default, fastest)
2. Overlap
3. Cross Correlation Coefficient
4. Pearson Correlation Coefficient
5. Laplacian Filtering

### Vector Representation

Maps are converted to vector representations using:

1. Gaussian filtering (bandwidth via `-g` parameter)
2. Mean-shift resampling (voxel spacing via `-s` parameter)
3. Gradient calculation for vector field

### Secondary Structure Integration

The `ss` mode combines density-based scoring with secondary structure probability maps, weighted by the `-alpha` parameter (default: 0.5). Falls back to original mode if prediction quality is low.

## Common Tasks

### Adding a New Scoring Mode

1. Add to `Mode` enum in `src/vesper/cli.py`
2. Implement scoring logic in `MapFitter._score()` method in `src/vesper/fitter.py`
3. Update help text in CLI options

### Modifying GPU Batch Processing

The batch size can be controlled via `-batch` parameter. Auto-detection uses available GPU memory. Key code in `MapFitter._rot_and_search_fft_gpu()`.

### Working with PDB/CIF Files

The `pdb2vol()` function in `src/vesper/utils/pdb2vol.py` simulates EM maps from atomic structures. Supports:

- Backbone-only mode (`-bbonly` flag)
- Custom resolution (`-res` parameter)
- Automatic chain ID handling

## Recent Changes

Recent commits show active development:

- GPU batch processing optimization
- Rotation matrix caching
- PDB saving refactoring
- Vector visualization features
- CIF file handling improvements

Check `git log --oneline` for full history.
