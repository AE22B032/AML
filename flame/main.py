# main.py

import os
import gc
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
from flame.server_app import weighted_average, fit_config, make_server_evaluate_fn
from flame.logging_utils import CombinedLogger

NUM_CLIENTS = 2
BATCH_SIZE = 16  # Reduced from 64 to 16
DATA_PATH = "data/PS_20174392719_1491204439457_log.csv"
INPUT_DIM = 20
LOG_DIR = Path("runs")

# Experiment toggles
CLIENT_CFGS = [
    {"local_epochs": 1, "dp": {"noise_multiplier": 0.0, "max_grad_norm": 1.0}, "quant_bits": 0, "sparsity": 0.0},  # Reduced from 2 to 1
    {"local_epochs": 1, "dp": {"noise_multiplier": 0.0, "max_grad_norm": 1.0}, "quant_bits": 0, "sparsity": 0.0},  # Reduced from 2 to 1
]


def _calc_input_dim(preprocessor, sample_df):
    """Infer model input dimension from a fitted ColumnTransformer and sample frame."""
    try:
        # Build a single-row sample
        X_sample = sample_df.drop("isFraud", axis=1).iloc[:1]
        transformed = preprocessor.transform(X_sample)
        return transformed.shape[1]
    except Exception:
        # Fallback: try attributes if available
        num = 0
        if hasattr(preprocessor, "transformers_"):
            for name, trans, cols in preprocessor.transformers_:
                if name == "cat" and hasattr(trans, "get_feature_names_out"):
                    num += len(trans.get_feature_names_out(cols))
                elif name == "num":
                    num += len(cols)
        return num


def client_fn(cid: str, dataloaders):
    """Create a Flower client instance for a given client ID."""
    cfg = CLIENT_CFGS[int(cid)] if int(cid) < len(CLIENT_CFGS) else {}
    model = MLP(input_dim=INPUT_DIM).to(torch.device("cpu"))
    train_loader, val_loader = dataloaders[int(cid)]
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
            # metrics already aggregated via evaluate_metrics_aggregation_fn
            self._logger.log("aggregate", int(server_round), loss if loss is not None else 0.0, metrics or {})
        return aggregated


def main():
    """Load data, start Flower simulation."""
    global INPUT_DIM
    
    # Aggressive memory management settings
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Force garbage collection
    gc.collect()

    # Load and preprocess data with extreme sampling to reduce memory
    df, preprocessor = load_and_preprocess_data(DATA_PATH, sample_frac=0.001)  # Use only 0.1% of data
    df_train, df_test = train_test_split_df(df, test_frac=0.2)

    # Determine input dimension after preprocessing
    INPUT_DIM = _calc_input_dim(preprocessor, df)
    print(f"Input dimension for MLP: {INPUT_DIM}")

    # Prepare federated data
    dataloaders = prepare_two_client_data(df_train, preprocessor, BATCH_SIZE)
    test_loader = build_test_loader(df_test, preprocessor, BATCH_SIZE)

    # Create a partial function for client_fn to pass dataloaders
    client_fn_with_data = partial(client_fn, dataloaders=dataloaders)

    # Model for evaluation
    model_for_eval = MLP(input_dim=INPUT_DIM)
    server_eval_fn = make_server_evaluate_fn(model_for_eval, test_loader)

    logger = CombinedLogger(LOG_DIR, experiment_name="flame_minimal_memory")
    
    # Log minimal hyperparameters to reduce overhead
    hyperparams = {
        "num_clients": NUM_CLIENTS,
        "batch_size": BATCH_SIZE,
        "sample_fraction": 0.001,
        "num_rounds": 1,
    }
    # Comment out MLflow logging to save memory
    # logger.log_params(hyperparams)

    # Wrap evaluate_fn to also log to CSV/TensorBoard
    def eval_and_log(server_round: int, parameters, config):
        loss, metrics = server_eval_fn(server_round, parameters, config)
        logger.log("server", server_round, loss, metrics)
        return loss, metrics

    # Define strategy
    strategy = LoggingFedAvg(
        logger=logger,
        fraction_fit=0.5,  # Train on 50% of clients per round (reduced from 100%)
        fraction_evaluate=0.5,  # Evaluate on 50% of clients per round (reduced from 100%)
        min_fit_clients=1,  # Reduced from NUM_CLIENTS
        min_evaluate_clients=1,  # Reduced from NUM_CLIENTS
        min_available_clients=1,  # Reduced from NUM_CLIENTS
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,  # Configure clients for training
        evaluate_fn=eval_and_log,  # Server-side evaluation function
    )

    # Start simulation
    try:
        fl.simulation.start_simulation(
            client_fn=client_fn_with_data,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=1),  # Reduced to 1 round only
            strategy=strategy,
            client_resources={"num_cpus": 0.5},  # Reduced from 1 to 0.5
            ray_init_args={"num_cpus": 1, "object_store_memory": 100000000},  # 100MB object store (reduced from 200MB)
        )
        
        # Comment out model logging to save memory
        # final_model = MLP(input_dim=INPUT_DIM)
        # logger.log_model(final_model, "final_federated_model")
        
        # Force cleanup
        gc.collect()
        
    finally:
        # Always end the MLflow run
        logger.end_run()
        gc.collect()

    print("Simulation finished with minimal memory usage.")


if __name__ == "__main__":
    main()