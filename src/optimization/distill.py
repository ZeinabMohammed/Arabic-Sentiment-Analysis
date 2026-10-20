"""Distill the 12-layer AraBERT teacher into a 6-layer student.

Student init follows the DistilBERT trick: copy the teacher's embeddings and
every other encoder layer (0, 2, 4, 6, 8, 10), then fine-tune with a loss that
combines hard-label cross-entropy and KL divergence against the teacher's
softened logits.

Run as: python -m src.optimization.distill
"""

from __future__ import annotations

import argparse
import os
import time

import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig

from src.config import (
    BASE_MODEL_NAME,
    ID_TO_LABEL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    STUDENT_NUM_LAYERS,
)
from src.training.train import ReviewsDataset, _load_split, evaluate, set_seed

TEACHER_PATH = os.environ.get("MODEL_PATH", BASE_MODEL_NAME)
TEMPERATURE = 2.0
ALPHA = 0.5  # weight on the distillation (KL) loss vs. the hard-label loss


def build_student(
    teacher: AutoModelForSequenceClassification,
) -> AutoModelForSequenceClassification:
    teacher_layers = teacher.config.num_hidden_layers
    stride = teacher_layers // STUDENT_NUM_LAYERS
    keep = list(range(0, teacher_layers, stride))[:STUDENT_NUM_LAYERS]

    student_config = BertConfig.from_dict(teacher.config.to_dict())
    student_config.num_hidden_layers = STUDENT_NUM_LAYERS
    student = AutoModelForSequenceClassification.from_config(student_config)

    student.bert.embeddings.load_state_dict(teacher.bert.embeddings.state_dict())
    for student_idx, teacher_idx in enumerate(keep):
        student.bert.encoder.layer[student_idx].load_state_dict(
            teacher.bert.encoder.layer[teacher_idx].state_dict()
        )
    student.bert.pooler.load_state_dict(teacher.bert.pooler.state_dict())
    student.classifier.load_state_dict(teacher.classifier.state_dict())

    return student


def distill(
    train_limit: int | None = None,
    val_limit: int | None = None,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 3e-5,
) -> dict[str, float]:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_PATH)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_PATH, num_labels=len(ID_TO_LABEL)
    ).to(device)
    teacher.eval()

    student = build_student(teacher).to(device)

    train_ds = ReviewsDataset(_load_split("train", train_limit), tokenizer)
    val_ds = ReviewsDataset(_load_split("val", val_limit), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="distill-12to6"):
        mlflow.log_params(
            {
                "teacher_layers": teacher.config.num_hidden_layers,
                "student_layers": STUDENT_NUM_LAYERS,
                "temperature": TEMPERATURE,
                "alpha": ALPHA,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
            }
        )

        start = time.time()
        student.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()

                student_logits = student(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
                with torch.no_grad():
                    teacher_logits = teacher(
                        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                    ).logits

                hard_loss = F.cross_entropy(student_logits, batch["labels"])
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / TEMPERATURE, dim=-1),
                    F.softmax(teacher_logits / TEMPERATURE, dim=-1),
                    reduction="batchmean",
                ) * (TEMPERATURE**2)
                loss = ALPHA * soft_loss + (1 - ALPHA) * hard_loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            mlflow.log_metric("train_loss", epoch_loss / len(train_loader), step=epoch)

        metrics = evaluate(student, val_loader, device)
        metrics["train_seconds"] = time.time() - start
        mlflow.log_metrics(metrics)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        student_dir = MODELS_DIR / "distilled"
        student.save_pretrained(student_dir)
        tokenizer.save_pretrained(student_dir)
        mlflow.log_artifacts(str(student_dir), artifact_path="distilled_student")

    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    distill(args.train_limit, args.val_limit, args.epochs, args.batch_size)
