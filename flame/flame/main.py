import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import flwr as fl
import torch
from functools import partial
from pathlib import Path
from typing import List, Tuple

from flame.data import (
    load_and_preprocess_data,
    prepare_two_client_data,
    train_test_split_df,
    build_test_loader,
)
from flame.task import MLP
from flame.client_app import FlowerClient
from flame.server_app import weighted_average, make_server_evaluate_fn
from flame.logging_utils import CombinedLogger

# Constants and defaults
NUM_CLIENTS = 2
BATCH_SIZE = 128  # small to reduce memory
DATA_PATH = "data/PS_20174392719_1491204439457_log.csv"
INPUT_DIM = 20
LOG_DIR = Path("runs")
SAMPLE_FRAC = 0.1  # use a fraction of data to reduce memory

# Experiment toggles per simulated client
CLIENT_CFGS = [
    {"local_epochs": 2},
    {"local_epochs": 2},
]


def _calc_input_dim(preprocessor, sample_df):
    try:
        X_sample = sample_df.drop("isFraud", axis=1).iloc[:1]
        transformed = preprocessor.transform(X_sample)
        # Handle sparse
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return transformed.shape[1]
    except Exception:
        return 20


def client_fn(cid: str, dataloaders):
    cfg = CLIENT_CFGS[int(cid)] if int(cid) < len(CLIENT_CFGS) else {}
    model = MLP(input_dim=INPUT_DIM).to(torch.device("cpu"))
    train_loader, val_loader = dataloaders[int(cid)]
    # NumPyClient -> Client per new API
    return FlowerClient(model, train_loader, val_loader, cfg).to_client()


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that logs aggregated client metrics after each evaluate round."""

    def __init__(self, logger: CombinedLogger, **kwargs):
        super().__init__(**kwargs)
        self._logger = logger

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated is not None:
            loss, metrics = aggregated
            self._logger.log("aggregate", int(server_round), loss if loss is not None else 0.0, metrics or {})
        return aggregated


def main():
    global INPUT_DIM

    # Load and preprocess (sample to reduce memory)
    df, preprocessor = load_and_preprocess_data(DATA_PATH, sample_frac=SAMPLE_FRAC)
    df_train, df_test = train_test_split_df(df, test_frac=0.2)
    INPUT_DIM = _calc_input_dim(preprocessor, df)

    # Build loaders for two clients and server-side test
    dataloaders = prepare_two_client_data(df_train, preprocessor, BATCH_SIZE)
    test_loader = build_test_loader(df_test, preprocessor, BATCH_SIZE)

    client_fn_with_data = partial(client_fn, dataloaders=dataloaders)

    # Server-side evaluation on central test set
    model_for_eval = MLP(input_dim=INPUT_DIM)
    server_eval_fn = make_server_evaluate_fn(model_for_eval, test_loader)

    # Logging
    logger = CombinedLogger(LOG_DIR)

    def eval_and_log(server_round: int, parameters, config):
        loss, metrics = server_eval_fn(server_round, parameters, config)
        logger.log("server", server_round, loss, metrics)
        return loss, metrics

    # Strategy with reduced sampling and fit metrics aggregation
    strategy = LoggingFedAvg(
        logger=logger,
        fraction_fit=0.5,            # sample fewer clients per round (1 of 2)
        fraction_evaluate=0.5,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=lambda results: weighted_average(results) if results else {},
        on_fit_config_fn=lambda rnd: {
            "server_round": rnd,
            "local_epochs": CLIENT_CFGS[0].get("local_epochs", 1),
            "sample-frac": SAMPLE_FRAC,
            "batch-size": BATCH_SIZE,
        },
        evaluate_fn=eval_and_log,
    )

    # Limit Ray resources to avoid OOM
    fl.simulation.start_simulation(
        client_fn=client_fn_with_data,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources={"num_cpus": 1},
        ray_init_args={"include_dashboard": False, "ignore_reinit_error": True, "num_cpus": 2},
    )


if __name__ == "__main__":
    main()