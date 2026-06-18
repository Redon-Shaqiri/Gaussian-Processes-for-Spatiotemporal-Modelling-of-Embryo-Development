# gpembryos

Gaussian process methods for Spatiotemporal Modelling of Embryo Development.

## Install

```bash
pip install -e .
```

## Repo structure

```
gpembryos/      core source code (importable package)
experiments/    numbered scripts; each saves to artifacts/expNNN/<timestamp>/
artifacts/      ephemeral run outputs (gitignored)
data/           raw/processed inputs (gitignored)
tests/          unit tests
```

## Running experiments

```bash
# Fresh run
python experiments/exp001__template.py

# Resume the latest run
python experiments/exp001__template.py --resume

# Resume a specific run
python experiments/exp001__template.py --resume 2026-06-18_120000
```

Each experiment maintains a `latest` symlink under `artifacts/expNNN/`.

## Tests

```bash
pytest
```

## Formatting

```bash
isort .
black .
```
