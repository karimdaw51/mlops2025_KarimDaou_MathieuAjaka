from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    artifacts_dir: Path
    outputs_dir: Path


def get_project_paths() -> ProjectPaths:
    root = Path(__file__).resolve().parents[3]
    return ProjectPaths(
        root=root,
        data_dir=root / "src" / "mlproject" / "data",
        artifacts_dir=root / "artifacts",
        outputs_dir=root / "outputs",
    )


def dated_predictions_filename() -> str:
    return datetime.now().strftime("%Y%m%d") + "_predictions.csv"

