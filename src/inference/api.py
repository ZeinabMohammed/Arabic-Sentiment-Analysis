"""FastAPI service exposing /predict and /health for the Arabic sentiment model.

MODEL_PATH env var selects the checkpoint: a local fine-tuned directory
(produced by src.training.train), an MLflow model URI such as
"models:/ArabicSentiment/Production", or the base AraBERT checkpoint for
pre-training smoke tests. Defaults to the base checkpoint.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.config import BASE_MODEL_NAME
from src.inference.schemas import HealthResponse, PredictRequest, PredictResponse
from src.models.sentiment_model import ArabicSentimentModel

MODEL_PATH = os.environ.get("MODEL_PATH", BASE_MODEL_NAME)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "dev")
# "torch" (default) or "onnx" — see src/optimization/export_onnx.py for how the
# .onnx file referenced by MODEL_PATH is produced.
MODEL_ENGINE = os.environ.get("MODEL_ENGINE", "torch")

_model: ArabicSentimentModel | None = None


def get_model() -> ArabicSentimentModel:
    global _model
    if _model is None:
        if MODEL_ENGINE == "onnx":
            from src.models.onnx_sentiment_model import OnnxSentimentModel

            _model = OnnxSentimentModel(MODEL_PATH, version=MODEL_VERSION)
        else:
            _model = ArabicSentimentModel(MODEL_PATH, version=MODEL_VERSION)
    return _model


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_model()
    yield


app = FastAPI(title="Arabic Sentiment API", lifespan=lifespan)

REQUEST_LATENCY = Histogram(
    "predict_latency_seconds",
    "Latency of /predict requests",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
REQUEST_COUNT = Counter("predict_requests_total", "Total /predict requests", ["label"])


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    result = get_model().predict(request.text)
    REQUEST_LATENCY.observe(time.perf_counter() - start)
    REQUEST_COUNT.labels(label=result.label).inc()
    return PredictResponse(
        label=result.label, confidence=result.confidence, model_version=result.model_version
    )


@app.get("/metrics")
def metrics() -> Response:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
