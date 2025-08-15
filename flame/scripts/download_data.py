#!/usr/bin/env python3
"""Download the PaySim dataset to data/ using kagglehub.

This is optional, as load_and_preprocess_data() auto-downloads when missing.
"""
from pathlib import Path
from flame.data import ensure_dataset_present, KAGGLE_FILE_NAME

if __name__ == "__main__":
    target = Path("data") / KAGGLE_FILE_NAME
    path = ensure_dataset_present(target)
    print("Dataset downloaded to:", path)
