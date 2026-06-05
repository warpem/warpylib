# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

warpylib is a PyTorch-based Python library for cryo-electron tomography data processing, replicating core functionality from WarpLib (a C# framework). It operates heavily on GPU tensors and MRC/STAR file formats.

## Commands

```bash
pip install -e ".[dev]"   # development install
pytest                    # run all tests
pytest tests/ctf/         # run a specific module
black .                   # format
ruff check .              # lint
ruff check --fix .        # lint + auto-fix
python -m build           # build distribution
```

## Code style

- Line length: **100 characters** (configured in both black and ruff)
- Python minimum: **3.8** — avoid syntax not supported by 3.8
- Type hints are used in parts of the codebase; follow the surrounding style

## Gotchas

- **setuptools-scm**: version is derived from git tags. Always clone with full history (`git fetch --unshallow` if needed). Shallow clones break the build.
- **torch-projectors**: this dependency must be manually sourced for the correct CUDA version — it cannot be resolved automatically by pip. Users are responsible for installing the matching build.
- **testdata/**: contains large `.mrc` files (binary scientific data). Do not modify or delete them; tests depend on specific byte-level content.

## Release process

Releases are published to PyPI automatically by GitHub Actions when a version tag is pushed:

```bash
git tag v1.2.3
git push origin v1.2.3
```

The workflow verifies the tag matches the package version, builds, and publishes via PyPI trusted publishing. Do not bump version numbers manually — setuptools-scm derives them from git tags.
