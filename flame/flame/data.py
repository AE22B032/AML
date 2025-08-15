# src/data.py

import math
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


KAGGLE_DATASET = "ealaxi/paysim1"
KAGGLE_FILE_NAME = "PS_20174392719_1491204439457_log.csv"


def ensure_dataset_present(csv_path: str | os.PathLike) -> Path:
    """Ensure the PaySim CSV exists locally; download via kagglehub if missing.

    Returns the Path to the CSV. Raises with a helpful message on failure.
    """
    csv_path = Path(csv_path)
    if csv_path.exists():
        return csv_path

    target_dir = csv_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kagglehub  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "kagglehub is not installed. Install it with `pip install kagglehub`"
        ) from e

    try:
        dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as e:
        raise RuntimeError(
            "Failed to download dataset from Kaggle. Ensure your Kaggle API credentials are set.\n"
            "See: https://github.com/Kaggle/kaggle-api#api-credentials"
        ) from e

    # Find the expected CSV inside the downloaded directory
    candidate = dataset_dir / KAGGLE_FILE_NAME
    if not candidate.exists():
        # search recursively in case layout differs
        matches = list(dataset_dir.rglob(KAGGLE_FILE_NAME))
        if not matches:
            raise FileNotFoundError(
                f"Could not locate {KAGGLE_FILE_NAME} within {dataset_dir}"
            )
        candidate = matches[0]

    shutil.copyfile(candidate, csv_path)
    return csv_path


class TransactionDataset(Dataset):
    """Custom PyTorch Dataset for transaction data."""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_preprocess_data(filepath: str):
    """Loads CSV, drops IDs, fits preprocessing pipeline, and returns df + fitted preprocessor."""
    # Ensure the data exists, auto-downloading if needed
    csv_path = ensure_dataset_present(filepath)
    df = pd.read_csv(csv_path)

    # Drop non-informative columns if present
    for col in ["nameOrig", "nameDest"]:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Separate features and labels
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    # Identify categorical and numerical features
    categorical_features = [c for c in X.columns if X[c].dtype == "object"]
    numerical_features = [c for c in X.columns if c not in categorical_features]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Fit the preprocessor on the entire dataset to ensure consistent scaling/encoding
    preprocessor.fit(X)

    return df, preprocessor


essential_cols = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def _safe_concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([d for d in dfs if d is not None and len(d) > 0], ignore_index=True)


def prepare_federated_data(
    df: pd.DataFrame, preprocessor: ColumnTransformer, num_clients: int, batch_size: int
):
    """Partitions the data into semi non-IID sets for federated learning.

    - Client 0 biased towards TRANSFER fraud
    - Client 1 biased towards CASH_OUT fraud
    - Others mostly non-fraud with random sampling
    """
    rng = np.random.default_rng(42)

    # Ensure required columns exist
    missing = [c for c in ["isFraud", "type"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    fraud_df = df[df["isFraud"] == 1]
    non_fraud_df = df[df["isFraud"] == 0]

    fraud_transfer = fraud_df[fraud_df["type"] == "TRANSFER"]
    fraud_cashout = fraud_df[fraud_df["type"] == "CASH_OUT"]

    # Split non-fraud roughly evenly
    non_fraud_indices = rng.permutation(len(non_fraud_df))
    splits = np.array_split(non_fraud_indices, num_clients)

    client_dfs: list[pd.DataFrame] = []
    for i in range(num_clients):
        nf_slice = non_fraud_df.iloc[splits[i]] if len(splits[i]) > 0 else non_fraud_df.sample(0)
        special = None
        if i == 0 and len(fraud_transfer) > 0:
            special = fraud_transfer
        elif i == 1 and len(fraud_cashout) > 0:
            special = fraud_cashout
        client_df = _safe_concat([nf_slice, special]) if special is not None else nf_slice.copy()
        client_dfs.append(client_df.reset_index(drop=True))

    client_loaders = []
    for client_df in client_dfs:
        if "isFraud" not in client_df.columns or len(client_df) == 0:
            # Create a tiny empty dataset to avoid crashes
            X_client = df.drop("isFraud", axis=1).iloc[:0]
            y_client = np.array([], dtype=np.float32)
        else:
            X_client = client_df.drop("isFraud", axis=1)
            y_client = client_df["isFraud"].to_numpy(dtype=np.float32)

        X_client_transformed = preprocessor.transform(X_client)
        dataset = TransactionDataset(X_client_transformed, y_client)

        # 80/20 split with min 1 in val if possible
        n = len(dataset)
        val_size = max(1 if n > 1 else 0, int(round(0.2 * n)))
        train_size = n - val_size
        if train_size == 0 and n > 0:
            train_size, val_size = n, 0

        if n > 0 and val_size > 0:
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        else:
            train_dataset, val_dataset = dataset, torch.utils.data.Subset(dataset, [])

        train_loader = DataLoader(train_dataset, batch_size=batch_size or max(1, n), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size or max(1, n))
        client_loaders.append((train_loader, val_loader))

    return client_loaders


def split_us_eu(df: pd.DataFrame, frac_us: float = 0.5, seed: int = 42):
    """Simulate geo split to two clients (US/EU) by random partition with fixed seed.
    Returns (df_us, df_eu).
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < frac_us
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)


def prepare_two_client_data(df: pd.DataFrame, preprocessor: ColumnTransformer, batch_size: int):
    """Prepare loaders for exactly two clients (US/EU) with per-client val split."""
    from .data import TransactionDataset  # self-import-safe

    client_loaders = []
    for part in split_us_eu(df, 0.5):
        X = part.drop("isFraud", axis=1)
        y = part["isFraud"].to_numpy(dtype=np.float32)
        X_t = preprocessor.transform(X)
        dataset = TransactionDataset(X_t, y)
        n = len(dataset)
        val_size = max(1 if n > 1 else 0, int(round(0.2 * n)))
        train_size = n - val_size
        if n > 0 and val_size > 0:
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
        else:
            train_dataset, val_dataset = dataset, torch.utils.data.Subset(dataset, [])
        client_loaders.append(
            (
                DataLoader(train_dataset, batch_size=batch_size or max(1, n), shuffle=True),
                DataLoader(val_dataset, batch_size=batch_size or max(1, n)),
            )
        )
    return client_loaders


def train_test_split_df(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Split full dataframe into train and global test sets."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) >= test_frac
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)


def build_test_loader(df_test: pd.DataFrame, preprocessor: ColumnTransformer, batch_size: int):
    """Build a DataLoader for the global test set."""
    X = df_test.drop("isFraud", axis=1)
    y = df_test["isFraud"].to_numpy(dtype=np.float32)
    X_t = preprocessor.transform(X)
    dataset = TransactionDataset(X_t, y)
    return DataLoader(dataset, batch_size=batch_size or max(1, len(dataset)))