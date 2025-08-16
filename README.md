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
	README.md              # Project docs under the Flame app folder
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

From the Flame app directory (contains `pyproject.toml`):

```bash
cd flame
pip install -e .
```

This installs dependencies listed in `pyproject.toml` (Flower, PyTorch, pandas, scikit-learn, Opacus, MLflow, etc.).

## Data
- Expected dataset: PaySim `PS_20174392719_1491204439457_log.csv` under `flame/data/`.
- The file is ignored by Git. Place it manually at `flame/data/PS_20174392719_1491204439457_log.csv`.

## Running the local simulation

Use the Flower Simulation Runtime defaults configured in `pyproject.toml`:

```bash
cd flame
flwr run .
```

Or run the custom simulation entrypoint:

```bash
cd flame
python main.py
```

### Notes on low-memory mode
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
cd flame
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
- Missing dataset: place the PaySim CSV under `flame/data/` as noted above.

## How it works

Federated learning proceeds in synchronous rounds coordinated by the server:
1. Server initializes or updates the global model (FedAvg).
2. A subset of clients is selected (configurable via `fraction_fit`).
3. Each selected client trains locally for `local_epochs` on its partition and reports updated weights and metrics.
4. Server aggregates client updates (weighted by local sample count) and logs aggregated metrics.
5. Optionally, the server evaluates the updated global model on a central test set and logs results.

## Data pipeline
- Loader: reads PaySim CSV and applies aggressive sampling for low-RAM runs (<=1000 rows read, subsampled to ~500).
- Preprocessing: scikit-learn `ColumnTransformer` with `StandardScaler` for numerics and `OneHotEncoder` for categoricals.
- Splits: deterministic train/val per client (two partitions) and a global test split for server-side evaluation.
- DataLoaders: small batches, single-threaded (`num_workers=0`) to minimize memory.

## Model
- Simple MLP (PyTorch) for tabular fraud detection.
- Hidden layers with dropout; sigmoid output for binary classification.
- Parameter helpers convert between NumPy and PyTorch to interoperate with Flower.

## Metrics and aggregation
- Per client: loss, ROC-AUC, precision, recall, F1, accuracy.
- Server aggregates client metrics using weighted average by sample count.
- Server-side evaluation computes the same metrics on the global test set each round.

## Configuration knobs
- In `flame/pyproject.toml` (Flower runtime):
	- `num-server-rounds`, `fraction-fit`, `local-epochs`.
- In `flame/main.py` (simulation):
	- Number of clients, batch size, sampling fraction, local epochs, Ray resource limits.
	- Privacy/efficiency toggles per client (DP-SGD, quantization, sparsification, Gaussian noise, adversarial simulation).

## Scaling up (performance tips)
- Increase sampling fraction and batch size gradually; raise `num_rounds` once stable.
- Allow more Ray CPUs and object store memory if resources permit.
- Disable MLflow when memory-constrained; rely on CSV/TensorBoard only.
- Consider feature pruning or using sparse encodings to reduce memory footprint.

## Roadmap
- Add centralized baseline training for comparison.
- Expose more run-time knobs via `pyproject.toml` for deployment.
- Robust aggregation (e.g., median, Krum) and anomaly detection for adversarial clients.
- Extended heterogeneity scenarios (feature/label skew) and personalization.

## References
- Flower: https://flower.ai
- PyTorch: https://pytorch.org
- PaySim dataset: https://www.kaggle.com/datasets/ealaxi/paysim1

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
