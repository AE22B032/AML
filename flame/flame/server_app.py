# src/server.py

from typing import Any, Dict, List, Tuple, Callable
import flwr as fl

from .client_app import test as client_test
from .task import set_parameters


def weighted_average(results):
    """Aggregate evaluation metrics with weights = num_examples.

    Accepts a list of (num_examples, metrics_dict) pairs.
    """
    if not results:
        return {}
    total_examples = sum(num for num, _ in results) or 1

    def _avg(key: str) -> float:
        return sum(num * float(metrics.get(key, 0.0)) for num, metrics in results) / total_examples

    return {
        "precision": _avg("precision"),
        "recall": _avg("recall"),
        "f1": _avg("f1"),
        "roc_auc": _avg("roc_auc"),
        "accuracy": _avg("accuracy"),
    }


def fit_config(server_round: int):
    return {"server_round": int(server_round), "local_epochs": 1}


def make_server_evaluate_fn(model, test_loader) -> Callable:
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config):
        set_parameters(model, parameters)
        loss, metrics = client_test(model, test_loader)
        return loss, metrics

    return evaluate


# Robust fit metrics aggregation that handles different result shapes
# - Newer API: List[(ClientProxy, FitRes)] where FitRes.metrics and FitRes.num_examples exist
# - Older/simple API: List[(num_examples, metrics_dict)]

def _fit_metrics_agg(results):
    if not results:
        return {}

    # Detect form
    first = results[0]
    acc: Dict[str, float] = {}
    total = 0

    def _accumulate(metrics: Dict[str, Any], n: int):
        nonlocal acc, total
        if n <= 0:
            n = 1
        total += n
        for k, v in (metrics or {}).items():
            if isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0.0) + float(v) * n

    if isinstance(first, tuple) and len(first) == 2:
        a, b = first
        # Case 1: (ClientProxy, FitRes)
        if hasattr(b, "metrics"):
            for _, fit_res in results:
                m = getattr(fit_res, "metrics", {}) or {}
                n = int(getattr(fit_res, "num_examples", 0) or m.get("num_examples", 0) or 0)
                _accumulate(m, n)
        # Case 2: (num_examples, metrics)
        else:
            for n, m in results:
                _accumulate(m, int(n))
    return {k: (v / total if total else 0.0) for k, v in acc.items()}


try:
    from flwr.server import ServerApp, ServerAppComponents
except Exception:  # pragma: no cover
    ServerApp = object  # type: ignore
    ServerAppComponents = object  # type: ignore


def server_fn(context: Any):
    run_cfg = getattr(context, "run_config", {}) or {}
    fraction_fit = float(run_cfg.get("fraction-fit", 0.5))
    fraction_evaluate = float(run_cfg.get("fraction-evaluate", 0.5))
    local_epochs = int(run_cfg.get("local-epochs", 1))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=_fit_metrics_agg,
        on_fit_config_fn=lambda rnd: {"server_round": rnd, "local_epochs": local_epochs},
    )

    return ServerAppComponents(strategy=strategy)


app = ServerApp(server_fn=server_fn)  # type: ignore
