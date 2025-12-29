from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlproject.features.build import build_features
from mlproject.utils.paths import get_project_paths


def fe(in_path: Path, out_path: Path) -> None:
    df = pd.read_parquet(in_path)
    df = build_features(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def main() -> None:
    paths = get_project_paths()
    p = argparse.ArgumentParser()
    p.add_argument("--clean_train", default=str(paths.data_dir / "clean_train.parquet"))
    p.add_argument("--clean_test", default=str(paths.data_dir / "clean_test.parquet"))
    p.add_argument("--out_train", default=str(paths.data_dir / "features_train.parquet"))
    p.add_argument("--out_test", default=str(paths.data_dir / "features_test.parquet"))
    args = p.parse_args()

    fe(Path(args.clean_train), Path(args.out_train))
    fe(Path(args.clean_test), Path(args.out_test))
    print(f"✅ Wrote: {args.out_train}")
    print(f"✅ Wrote: {args.out_test}")


if __name__ == "__main__":
    main()

