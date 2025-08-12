# Flame: Federated Learning with Flower and PyTorch (PaySim tabular)

This project implements a lightweight federated learning (FL) pipeline with Flower and PyTorch on the PaySim tabular fraud dataset. It simulates two clients (e.g., US/EU) and a server coordinating training using FedAvg, with robust logging and optional privacy/efficiency features.

## Key features
- Two-client FL simulation with a central server (FedAvg) and server-side evaluation on a held-out test set.
- End-to-end metrics: ROC-AUC, precision, recall, F1, accuracy — aggregated across clients and logged per round.
- Logging: CSV and TensorBoard built-in; MLflow integration available (can be toggled off for low-memory runs).
- Privacy and communication efficiency toggles: Differential Privacy (Opacus DP-SGD), quantization, sparsification, additive noise, adversarial client simulation (optional).
- Memory-constrained execution: heavy sampling, small batches, single-threaded execution, low Ray resources — runs on limited CPU/RAM.
- Clean Flower app components: `ServerApp` and `ClientApp` for deployment, plus a `main.py` simulation entrypoint.

## Repository layout

```
flame/
	pyproject.toml         # Project metadata, deps, and Flower app config
	README.md              # This file
	main.py                # Local FL simulation entrypoint (two clients)
	flame/
		__init__.py
		client_app.py        # ClientApp (deployment) + FlowerClient implementation
		server_app.py        # ServerApp (deployment) + FedAvg strategy config
		data.py              # Data loading, preprocessing, splitting, DataLoaders
		task.py              # Model (MLP) and parameter (de)serialization utils
		logging_utils.py     # CSV, TensorBoard, MLflow logging helpers
	data/
		PS_2017...csv        # PaySim CSV (ignored by Git); add your copy here
```

## Prerequisites
- Python 3.10+
- PyTorch CPU build
- For MLflow UI (optional): `mlflow`

## Installation

From the project root (directory containing `pyproject.toml`):

```bash
pip install -e .
```

This installs dependencies listed in `pyproject.toml` (Flower, PyTorch, pandas, scikit-learn, Opacus, MLflow, etc.).

## Data
- Expected dataset: PaySim `PS_20174392719_1491204439457_log.csv` under `data/`.
- The file is ignored by Git. Place it manually at `data/PS_20174392719_1491204439457_log.csv`.

## Running the local simulation

Use the Flower Simulation Runtime defaults configured in `pyproject.toml`:

```bash
flwr run .
```

Or run the custom simulation entrypoint:

```bash
python main.py
```

Notes on low-memory mode:
- `main.py` uses aggressive sampling (by default loads <=1000 rows and subsamples to ~500) and small batches to avoid OOM.
- Ray resources are minimized (CPU and object store) to fit constrained environments.

## Configuration

The `pyproject.toml` defines the Flower app components and runtime knobs:

```toml
[tool.flwr.app.components]
serverapp = "flame.server_app:app"
clientapp = "flame.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 3
fraction-fit = 1.0
local-epochs = 2
```

Runtime toggles in code (see `main.py` and `client_app.py`):
- Number of clients (default 2)
- Batch size and number of local epochs
- Client sampling fractions: `fraction_fit`, `fraction_evaluate`
- Optional DP, quantization, sparsification, noise, adversarial client

## Logging and experiment tracking

- CSV logs: written under `runs/<timestamp>/metrics.csv` with fields time/kind/round/loss/precision/recall/f1/roc_auc/accuracy.
- TensorBoard: if `tensorboard` is available via PyTorch, scalars are logged under `runs/<timestamp>`.
- MLflow: if installed, metrics and params can be logged; there’s also a helper script:

```bash
python scripts/start_mlflow_ui.py
# Visit http://localhost:5000
```

You can disable MLflow in `main.py` for ultra-low memory runs.

## Privacy and security options

- Differential Privacy (DP-SGD) via Opacus: configurable gradient clipping and noise multiplier.
- Secure/efficient comms (experimental): quantization, sparsification, Gaussian noise on updates.
- Adversarial toggle (simulation): perturbs client behavior for robustness testing.

## Troubleshooting

- Ray OOM: reduce batch size, use fewer rows (sampling), lower `num_rounds`, and minimize `client_resources` and `ray_init_args` in `main.py`.
- Deprecation (NumPyClient vs Client): code returns `FlowerClient(...).to_client()` to satisfy current Flower API.
- Missing dataset: place the PaySim CSV under `data/` as noted above.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
