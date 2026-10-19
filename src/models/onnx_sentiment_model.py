"""ONNX Runtime backend for the sentiment model — the serving-optimization path
used in place of TensorRT (see src/optimization/export_onnx.py for why)."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from src.config import BASE_MODEL_NAME, ID_TO_LABEL, MAX_SEQ_LENGTH
from src.models.sentiment_model import Prediction


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class OnnxSentimentModel:
    """Same interface as ArabicSentimentModel, backed by an ONNX Runtime session."""

    def __init__(
        self,
        onnx_path: str,
        tokenizer_path: str = BASE_MODEL_NAME,
        version: str = "onnx",
    ) -> None:
        self.version = version
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
        )

    def _logits(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=True,
            return_tensors="np",
        )
        return self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )[0]

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        logits = self._logits(texts)
        probs = _softmax(logits)
        label_ids = np.argmax(probs, axis=-1)

        return [
            Prediction(
                label=ID_TO_LABEL[int(label_id)],
                confidence=round(float(probs[i, label_id]), 4),
                model_version=self.version,
            )
            for i, label_id in enumerate(label_ids)
        ]
