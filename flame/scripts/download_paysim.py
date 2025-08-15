"""Download the PaySim dataset using kagglehub and place CSV under data/."""
from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    import kagglehub  # type: ignore

    print("Downloading ealaxi/paysim via kagglehub...")
    path = kagglehub.dataset_download("ealaxi/paysim")
    print("Dataset downloaded to:", path)

    # Find the main CSV (there are a couple variants; prefer the big log one)
    src_dir = Path(path)
    candidates = [
        "PS_20174392719_1491204439457_log.csv",
        "PS_20174392719_1491204439457_log.csv.gz",
        "PS_20174392719_1491204439457.csv",
    ]
    src = None
    for name in candidates:
        p = src_dir / name
        if p.exists():
            src = p
            break
    if src is None:
        raise FileNotFoundError(f"Could not find PaySim CSV in {src_dir}")

    dest_dir = Path(__file__).resolve().parents[1] / "data"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "PS_20174392719_1491204439457_log.csv"

    # If gz, just copy as-is; preprocessing can handle gz with pandas if needed
    shutil.copy2(src, dest)
    print("Copied to:", dest)


if __name__ == "__main__":
    main()
