"""Run the >=5-run MLflow sweep, register the best model, and promote it to Production.

Run as: python -m src.training.sweep
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient

from src.config import (
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    REPORTS_DIR,
)
from src.training.train import train_one_run

# Grid kept small so this sweep finishes on a laptop CPU; widen it for a
# real GPU training run (see README "Scaling up training").
GRID = [
    {"lr": 2e-5, "batch_size": 8, "epochs": 1},
    {"lr": 3e-5, "batch_size": 8, "epochs": 1},
    {"lr": 5e-5, "batch_size": 8, "epochs": 1},
    {"lr": 2e-5, "batch_size": 16, "epochs": 1},
    {"lr": 2e-5, "batch_size": 8, "epochs": 2},
]


def run_sweep(train_limit: int | None = None, val_limit: int | None = None) -> list[dict]:
    results = []
    for i, params in enumerate(GRID):
        print(f"--- run {i + 1}/{len(GRID)}: {params} ---")
        result = train_one_run(
            **params,
            train_limit=train_limit,
            val_limit=val_limit,
            run_name=f"sweep-{i + 1}",
        )
        results.append({**params, **result})
    return results


def plot_comparison(results: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [f"run{i + 1}" for i in range(len(results))]
    f1_scores = [r["f1_macro"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, f1_scores, color="#4C72B0")
    ax.set_ylabel("f1_macro (validation)")
    ax.set_title("MLflow sweep comparison — Arabic Sentiment")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "mlflow_comparison.png", dpi=150)
    plt.close(fig)


def register_best(results: list[dict]) -> None:
    best = max(results, key=lambda r: r["f1_macro"])
    print(f"Best run: {best['run_id']} (f1_macro={best['f1_macro']:.4f})")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri, MLFLOW_REGISTERED_MODEL_NAME)
    client.transition_model_version_stage(
        name=MLFLOW_REGISTERED_MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Registered {MLFLOW_REGISTERED_MODEL_NAME} v{mv.version} -> Production")


if __name__ == "__main__":
    all_results = run_sweep()
    plot_comparison(all_results)
    register_best(all_results)
