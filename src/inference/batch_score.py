"""Batch-score a CSV of Arabic reviews and write predictions + scoring metadata.

Run as: python -m src.inference.batch_score --input data/scoring/input/batch.csv
Defaults to sampling from data/raw/reviews.csv when no --input is given.
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

import pandas as pd

from src.config import BASE_MODEL_NAME, RAW_DATA_DIR, SCORING_DIR
from src.models.sentiment_model import ArabicSentimentModel

MODEL_PATH = os.environ.get("MODEL_PATH", BASE_MODEL_NAME)


def load_batch(input_path: str | None, n_samples: int) -> pd.DataFrame:
    if input_path:
        return pd.read_csv(input_path)
    df = pd.read_csv(RAW_DATA_DIR / "reviews.csv")
    return df.sample(n=n_samples, random_state=0).reset_index(drop=True)


def run(input_path: str | None, n_samples: int) -> str:
    batch = load_batch(input_path, n_samples)
    model = ArabicSentimentModel(MODEL_PATH)

    predictions = model.predict_batch(batch["text"].tolist())
    batch["predicted_label"] = [p.label for p in predictions]
    batch["confidence_score"] = [p.confidence for p in predictions]
    batch["model_version"] = [p.model_version for p in predictions]
    batch["text_length"] = batch["text"].str.len()

    output_dir = SCORING_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"scored_{timestamp}.csv"
    batch.to_csv(output_path, index=False)

    print(f"Scored {len(batch)} rows -> {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=1000)
    args = parser.parse_args()
    run(args.input, args.n_samples)
