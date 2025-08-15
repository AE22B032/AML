"""Logging utilities for round-wise metrics.

Provides CSV logging and optional TensorBoard logging if tensorboard is installed.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import datetime as _dt


def _now_ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class CSVLogger:
    logdir: Path
    filename: str = "metrics.csv"

    def __post_init__(self) -> None:
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._file = (self.logdir / self.filename)
        if not self._file.exists():
            with self._file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "kind", "round", "loss", "precision", "recall", "f1", "roc_auc", "accuracy"])

    def log(self, kind: str, server_round: int, loss: float, metrics: Dict[str, float]) -> None:
        row = [
            _dt.datetime.now().isoformat(timespec="seconds"),
            kind,
            int(server_round),
            float(loss) if loss is not None else None,
            float(metrics.get("precision", 0.0)),
            float(metrics.get("recall", 0.0)),
            float(metrics.get("f1", 0.0)),
            float(metrics.get("roc_auc", 0.0)),
            float(metrics.get("accuracy", 0.0)),
        ]
        with (self._file).open("a", newline="") as f:
            csv.writer(f).writerow(row)


class TensorBoardLogger:
    def __init__(self, logdir: Path) -> None:
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except Exception:
            return
        self._writer = SummaryWriter(str(logdir))

    def log(self, kind: str, server_round: int, loss: float, metrics: Dict[str, float]) -> None:
        if self._writer is None:
            return
        tag_prefix = f"{kind}" if kind else "metrics"
        if loss is not None:
            self._writer.add_scalar(f"{tag_prefix}/loss", loss, server_round)
        for k, v in metrics.items():
            self._writer.add_scalar(f"{tag_prefix}/{k}", v, server_round)


class CombinedLogger:
    def __init__(self, base_dir: Path, subdir: Optional[str] = None) -> None:
        run_dir = base_dir / (subdir or _now_ts())
        self.csv = CSVLogger(run_dir)
        self.tb = TensorBoardLogger(run_dir)

    def log(self, kind: str, server_round: int, loss: float, metrics: Dict[str, float]) -> None:
        self.csv.log(kind, server_round, loss, metrics)
        self.tb.log(kind, server_round, loss, metrics)
