"""Shared constants for the Arabic sentiment project."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SCORING_DIR = DATA_DIR / "scoring"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"

HF_DATASET_NAME = "arbml/arabic_100k_reviews"

# "Mixed" from the source dataset is used as the Neutral class.
LABEL_NAMES = ["Positive", "Mixed", "Negative"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID_TO_LABEL = dict(enumerate(LABEL_NAMES))

BASE_MODEL_NAME = "aubmindlab/bert-base-arabertv02"
TEACHER_NUM_LAYERS = 12
STUDENT_NUM_LAYERS = 6

MLFLOW_EXPERIMENT_NAME = "arabic-sentiment"
MLFLOW_REGISTERED_MODEL_NAME = "ArabicSentiment"
# Defaults to a local sqlite store so training/CI work without the MLflow
# server running; point MLFLOW_TRACKING_URI at http://localhost:5000 (the
# docker-compose "mlflow" service, same mlflow.db) to browse the same runs
# in the UI.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")

MAX_SEQ_LENGTH = 128

# Training subsample sizes — kept small so the pipeline runs end to end on a
# laptop CPU; scale these up when training on a GPU box.
TRAIN_SIZE = 4000
VAL_SIZE = 600
TEST_SIZE = 600

DRIFT_PSI_ALERT_THRESHOLD = 0.25
RAGAS_NOT_APPLICABLE = None  # project 1 (DL track) has no RAG component
