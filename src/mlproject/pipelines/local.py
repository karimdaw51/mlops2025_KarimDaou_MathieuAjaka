from __future__ import annotations

import subprocess
from mlproject.utils.paths import get_project_paths


def train_cli() -> None:
    paths = get_project_paths()

    subprocess.check_call(
        [
            "python",
            "scripts/preprocess.py",
            "--train_csv",
            str(paths.data_dir / "train.csv"),
            "--test_csv",
            str(paths.data_dir / "test.csv"),
            "--out_train",
            str(paths.data_dir / "clean_train.parquet"),
            "--out_test",
            str(paths.data_dir / "clean_test.parquet"),
        ],
        cwd=str(paths.root),
    )

    print("Preprocess stage completed (next: features -> train)")


def inference_cli() -> None:
    print("INFERENCE CLI OK ✅ (next: implement batch inference)")

