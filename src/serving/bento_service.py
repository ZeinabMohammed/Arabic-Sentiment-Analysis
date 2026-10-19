"""BentoML serving wrapper around ArabicSentimentModel, with a batchable /predict API.

Build and serve:
    bentoml serve src.serving.bento_service:ArabicSentimentService --reload
"""

from __future__ import annotations

import os

import bentoml

from src.config import BASE_MODEL_NAME
from src.models.sentiment_model import ArabicSentimentModel

MODEL_PATH = os.environ.get("MODEL_PATH", BASE_MODEL_NAME)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "dev")


@bentoml.service(name="arabic_sentiment", resources={"cpu": "2"})
class ArabicSentimentService:
    def __init__(self) -> None:
        self.model = ArabicSentimentModel(MODEL_PATH, version=MODEL_VERSION)

    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=32, max_latency_ms=200)
    def predict(self, texts: list[str]) -> list[dict]:
        predictions = self.model.predict_batch(texts)
        return [
            {"label": p.label, "confidence": p.confidence, "model_version": p.model_version}
            for p in predictions
        ]

    @bentoml.api
    def health(self) -> dict:
        return {"status": "healthy"}
