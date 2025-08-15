# src/client.py

import os
import flwr as fl
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import numpy as np
from typing import Optional, Dict, Any
from copy import deepcopy

# Force minimal threads for memory efficiency
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from flame.task import get_parameters, set_parameters, MLP
from flame.utils import quantize_dequantize, sparsify, add_gaussian_noise
from flame.data import (
    load_and_preprocess_data,
    prepare_two_client_data,
    train_test_split_df,
)

DEVICE = torch.device("cpu")  # Force CPU to save memory


def _maybe_wrap_dp(model: torch.nn.Module, loader, max_grad_norm: float, noise_multiplier: float):
    """Optionally wrap training with Opacus DP-SGD and return (engine, optimizer, private_loader)."""
    if noise_multiplier <= 0:
        return None, None, None
    try:
        from opacus import PrivacyEngine  # type: ignore
    except Exception:
        print("Opacus not installed; proceeding without DP")
        return None, None, None

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    engine = PrivacyEngine()
    model, optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return engine, optimizer, private_loader


def train(net, trainloader, epochs, dp_cfg: Optional[Dict[str, Any]] = None):
    criterion = torch.nn.BCELoss()

    engine = None
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    if dp_cfg:
        engine, opt_dp, dp_loader = _maybe_wrap_dp(
            net,
            trainloader,
            dp_cfg.get("max_grad_norm", 1.0),
            dp_cfg.get("noise_multiplier", 0.0),
        )
        if opt_dp is not None:
            optimizer = opt_dp
        if dp_loader is not None:
            trainloader = dp_loader

    net.train()
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test(net, valloader):
    """Validate the model on the validation set."""
    criterion = torch.nn.BCELoss()
    loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    net.eval()
    with torch.no_grad():
        for features, labels in valloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = net(features)
            loss += criterion(outputs, labels).item()

            probs = outputs.detach().cpu().numpy().ravel()
            preds = (probs > 0.5).astype(np.float32)
            labels_np = labels.detach().cpu().numpy().ravel()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_np.tolist())

    if len(valloader) > 0:
        loss /= len(valloader)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        roc_auc = 0.0

    return loss, {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "accuracy": float(acc),
    }


class FlowerClient(fl.client.NumPyClient):
    """Flower client with optional DP/compression/personalization/attack toggles."""

    def __init__(self, model, trainloader, valloader, cfg: Optional[Dict[str, Any]] = None):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.cfg = cfg or {}

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        local_epochs = int(config.get("local_epochs", self.cfg.get("local_epochs", 1)))

        # Simulate adversarial client if enabled
        if self.cfg.get("adversarial", False):
            # Apply a random perturbation to parameters before training
            current = get_parameters(self.model)
            perturbed = [p + np.random.normal(0, 0.05, size=p.shape).astype(p.dtype) for p in current]
            set_parameters(self.model, perturbed)

        dp_cfg = self.cfg.get("dp")  # e.g., {"noise_multiplier": 1.0, "max_grad_norm": 1.0}
        train(self.model, self.trainloader, epochs=local_epochs, dp_cfg=dp_cfg)

        params = get_parameters(self.model)

        # Apply compression/noise if configured
        num_bits = int(self.cfg.get("quant_bits", 0))
        sparsity = float(self.cfg.get("sparsity", 0.0))
        agg_noise = float(self.cfg.get("agg_noise_std", 0.0))

        if num_bits > 0 or sparsity > 0.0 or agg_noise > 0.0:
            comp = []
            for p in params:
                out = p
                if num_bits > 0:
                    out = quantize_dequantize(out, num_bits=num_bits)
                if sparsity > 0.0:
                    out = sparsify(out, sparsity=sparsity)
                if agg_noise > 0.0:
                    out = add_gaussian_noise(out, std=agg_noise)
                comp.append(out)
            params = comp

        return params, len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        # Optional personalization: fine-tune locally for evaluation only
        if self.cfg.get("personalize", False):
            backup = deepcopy(self.model.state_dict())
            try:
                p_epochs = int(self.cfg.get("personalize_epochs", 1))
                train(self.model, self.trainloader, epochs=p_epochs, dp_cfg=None)
                loss, metrics = test(self.model, self.valloader)
            finally:
                # Restore global model state
                self.model.load_state_dict(backup)
            return loss, len(self.valloader.dataset), metrics

        loss, metrics = test(self.model, self.valloader)
        return loss, len(self.valloader.dataset), metrics


# Replace placeholder with a real ClientApp for Flower runtime
try:
    from flwr.client import ClientApp  # type: ignore

    def _calc_input_dim(preprocessor, df):
        try:
            X_sample = df.drop("isFraud", axis=1).iloc[:1]
            return preprocessor.transform(X_sample).shape[1]
        except Exception:
            return 20

    def client_fn(context):  # pragma: no cover
        run_cfg = getattr(context, "run_config", {}) or {}
        data_path = run_cfg.get("data-path", "data/PS_20174392719_1491204439457_log.csv")
        batch_size = int(run_cfg.get("batch-size", 256))
        local_epochs = int(run_cfg.get("local-epochs", 1))

        # Load and split data
        df, preprocessor = load_and_preprocess_data(data_path)
        df_train, _ = train_test_split_df(df, test_frac=0.2)
        loaders = prepare_two_client_data(df_train, preprocessor, batch_size)

        # Pick partition deterministically per node
        node_str = getattr(context, "node_id", "0")
        try:
            idx = int(str(node_str)) % 2
        except Exception:
            idx = abs(hash(str(node_str))) % 2

        train_loader, val_loader = loaders[idx]

        input_dim = _calc_input_dim(preprocessor, df)
        model = MLP(input_dim=input_dim).to(torch.device("cpu"))

        # Minimal cfg from run_config
        cfg = {"local_epochs": local_epochs}
        return FlowerClient(model, train_loader, val_loader, cfg).to_client()

    app = ClientApp(client_fn=client_fn)  # noqa: F401
except Exception:  # pragma: no cover
    app = None  # type: ignore