"""Fine-tune AraBERT on the Arabic sentiment splits, tracked with MLflow.

Run a single experiment:
    python -m src.training.train --lr 2e-5 --batch-size 16 --epochs 1

Run the >=5-run sweep required by the rubric and register the best model:
    python -m src.training.sweep
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import (
    BASE_MODEL_NAME,
    ID_TO_LABEL,
    MAX_SEQ_LENGTH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
)


class ReviewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer) -> None:
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_split(name: str, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA_DIR / f"{name}.csv")
    return df.head(limit) if limit else df


@torch.inference_mode()
def evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
    }


def train_one_run(
    lr: float,
    batch_size: int,
    epochs: int,
    train_limit: int | None,
    val_limit: int | None,
    run_name: str | None = None,
    model_name: str = BASE_MODEL_NAME,
    seed: int = 42,
) -> dict[str, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(ID_TO_LABEL)
    ).to(device)

    train_ds = ReviewsDataset(_load_split("train", train_limit), tokenizer)
    val_ds = ReviewsDataset(_load_split("val", val_limit), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_name": model_name,
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "train_size": len(train_ds),
                "val_size": len(val_ds),
            }
        )

        start = time.time()
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = model(**batch)
                outputs.loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += outputs.loss.item()
            mlflow.log_metric("train_loss", epoch_loss / len(train_loader), step=epoch)

        metrics = evaluate(model, val_loader, device)
        metrics["train_seconds"] = time.time() - start
        mlflow.log_metrics(metrics)

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            pip_requirements=["torch", "transformers"],
        )

        run_id = mlflow.active_run().info.run_id

    return {"run_id": run_id, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Write {accuracy, f1_macro} as JSON here (used by the DVC 'train' stage)",
    )
    args = parser.parse_args()

    result = train_one_run(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        run_name=args.run_name,
    )
    print(result)

    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(
            json.dumps({"accuracy": result["accuracy"], "f1_macro": result["f1_macro"]}, indent=2)
        )


if __name__ == "__main__":
    main()
