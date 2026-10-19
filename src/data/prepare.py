"""Download the Arabic reviews dataset and write the DVC-tracked train/val/test splits.

Run as: python -m src.data.prepare
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config import (
    HF_DATASET_NAME,
    ID_TO_LABEL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
)


def download_raw() -> pd.DataFrame:
    dataset = load_dataset(HF_DATASET_NAME)["train"]
    df = dataset.to_pandas()
    df["label_name"] = df["label"].map(ID_TO_LABEL)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_DIR / "reviews.csv", index=False)
    return df


def make_splits(df: pd.DataFrame, seed: int = 42) -> None:
    total = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    sample, _ = train_test_split(
        df, train_size=total, stratify=df["label"], random_state=seed
    )

    train, rest = train_test_split(
        sample, train_size=TRAIN_SIZE, stratify=sample["label"], random_state=seed
    )
    val, test = train_test_split(
        rest, train_size=VAL_SIZE, stratify=rest["label"], random_state=seed
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train[["text", "label", "label_name"]].to_csv(
        PROCESSED_DATA_DIR / "train.csv", index=False
    )
    val[["text", "label", "label_name"]].to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test[["text", "label", "label_name"]].to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    make_splits(download_raw())
