# CLAUDE.md

## Project Overview

nested-pandas is a pandas extension for efficiently representing nested/hierarchical datasets. It packs nested dataframes into top-level dataframe columns using PyArrow internally, enabling hierarchical column access and operations without costly groupby operations.

## Development Setup

```bash
# Recommended: use the setup script
./.setup_dev.sh

# Or manually:
pip install -e '.[dev]'
pre-commit install
```

## Common Commands

### Testing
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=nested_pandas --cov-report=xml
```

### Linting & Formatting
```bash
# Run all pre-commit checks
pre-commit run --all-files

# Ruff lint only
ruff check src/ tests/

# Ruff format only
ruff format src/ tests/

# Type checking
mypy src/ tests/
```

### Documentation
```bash
cd docs && make html
```

## Code Style

- Line length: 110 characters
- Formatter: ruff-format
- Linter: ruff (rules: pycodestyle, pyflakes, pep8-naming, pyupgrade, flake8-bugbear, isort, numpy v2 compat)
- Type checker: mypy (Python 3.10+)
- Pre-commit hooks enforce all of the above before commits

## Architecture

```
src/nested_pandas/
  nestedframe/   # Core NestedFrame class (pandas DataFrame subclass)
  series/        # NestedSeries and PyArrow storage backend
  datasets/      # Data generation utilities
  utils/         # Shared utilities
tests/           # pytest test suite
docs/            # Sphinx documentation
benchmarks/      # ASV performance benchmarks
```

## Key Notes

- Python 3.10–3.13 supported
- CI runs tests on Linux and Windows
- Benchmarks run via ASV (airspeed velocity) on PRs and main branch
- Version is managed by `setuptools_scm` from git tags — do not set version manually
