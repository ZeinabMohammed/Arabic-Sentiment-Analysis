"""Benchmark original vs distilled vs INT8-quantized models: accuracy, p95
single-request latency, and file size. Writes reports/benchmark.md.

Run as: python -m src.optimization.benchmark
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import BASE_MODEL_NAME, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.models.onnx_sentiment_model import OnnxSentimentModel
from src.models.sentiment_model import ArabicSentimentModel

TEACHER_PATH = os.environ.get("MODEL_PATH", BASE_MODEL_NAME)
N_LATENCY_SAMPLES = 30


def _dir_size_mb(path: str | Path) -> float:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def _eval(model, texts: list[str], labels: list[int], label_to_id: dict[str, int]) -> dict:
    predictions = model.predict_batch(texts)
    preds = [label_to_id[p.label] for p in predictions]
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def _p95_latency_ms(model, texts: list[str]) -> float:
    # Warm up, then time single-text calls (the latency a live /predict sees).
    for text in texts[:3]:
        model.predict(text)

    latencies = []
    for text in texts[:N_LATENCY_SAMPLES]:
        start = time.perf_counter()
        model.predict(text)
        latencies.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(latencies, 95))


def run_benchmark(val_limit: int = 100) -> pd.DataFrame:
    val = pd.read_csv(PROCESSED_DATA_DIR / "val.csv").head(val_limit)
    texts = val["text"].tolist()
    labels = val["label"].tolist()
    label_to_id = {"Positive": 0, "Mixed": 1, "Negative": 2}

    rows = []

    original = ArabicSentimentModel(TEACHER_PATH)
    rows.append(
        {
            "model": "original (fp32, torch)",
            # filled in below from the exported .onnx file size, if present
            "size_mb": None,
            "p95_latency_ms": round(_p95_latency_ms(original, texts), 1),
            **_eval(original, texts, labels, label_to_id),
        }
    )

    distilled_dir = MODELS_DIR / "distilled"
    if distilled_dir.exists():
        distilled = ArabicSentimentModel(distilled_dir)
        rows.append(
            {
                "model": "distilled (6-layer, torch)",
                "size_mb": round(_dir_size_mb(distilled_dir), 1),
                "p95_latency_ms": round(_p95_latency_ms(distilled, texts), 1),
                **_eval(distilled, texts, labels, label_to_id),
            }
        )

    int8_path = MODELS_DIR / "model_int8.onnx"
    if int8_path.exists():
        quantized = OnnxSentimentModel(str(int8_path), tokenizer_path=TEACHER_PATH)
        rows.append(
            {
                "model": "INT8 (ONNX Runtime)",
                "size_mb": round(_dir_size_mb(int8_path), 1),
                "p95_latency_ms": round(_p95_latency_ms(quantized, texts), 1),
                **_eval(quantized, texts, labels, label_to_id),
            }
        )

    fp32_onnx_path = MODELS_DIR / "model_fp32.onnx"
    if fp32_onnx_path.exists():
        rows[0]["size_mb"] = round(_dir_size_mb(fp32_onnx_path), 1)

    df = pd.DataFrame(rows)[["model", "accuracy", "f1_macro", "p95_latency_ms", "size_mb"]]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "benchmark.md").write_text(
        "# Optimization benchmark\n\n" + df.to_markdown(index=False) + "\n"
    )
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_benchmark()
