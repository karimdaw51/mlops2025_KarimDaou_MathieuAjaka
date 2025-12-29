from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlproject.preprocess.cleaning import basic_clean
from mlproject.utils.paths import get_project_paths


def preprocess(input_csv: Path, output_parquet: Path) -> None:
    df = pd.read_csv(input_csv)
    df = basic_clean(df)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)


def main() -> None:
    paths = get_project_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=str(paths.data_dir / "train.csv"))
    parser.add_argument("--test_csv", default=str(paths.data_dir / "test.csv"))
    parser.add_argument("--out_train", default=str(paths.data_dir / "clean_train.parquet"))
    parser.add_argument("--out_test", default=str(paths.data_dir / "clean_test.parquet"))
    args = parser.parse_args()

    preprocess(Path(args.train_csv), Path(args.out_train))
    preprocess(Path(args.test_csv), Path(args.out_test))

    print(f"✅ Wrote: {args.out_train}")
    print(f"✅ Wrote: {args.out_test}")


if __name__ == "__main__":
    main()

