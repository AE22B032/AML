# Flame: A Flower / PyTorch app

## Install

The dependencies are listed in `pyproject.toml`. Install in editable mode:

```bash
pip install -e .
```

## Run a local simulation

From the project root (directory containing `pyproject.toml`), run:

```bash
flwr run .
```

This uses the Simulation Runtime with defaults configured in `pyproject.toml`.

## Notes

- Data file expected at `data/PS_20174392719_1491204439457_log.csv`.
- Model input dimension is inferred from the fitted preprocessor.
- Metrics aggregated server-side via weighted average.
