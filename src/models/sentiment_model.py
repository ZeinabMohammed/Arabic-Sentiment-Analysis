"""AraBERT sentiment classifier wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import BASE_MODEL_NAME, ID_TO_LABEL, MAX_SEQ_LENGTH


@dataclass
class Prediction:
    label: str
    confidence: float
    model_version: str


class ArabicSentimentModel:
    """Loads an AraBERT (or compatible) checkpoint once and serves predictions."""

    def __init__(self, model_path: str | Path = BASE_MODEL_NAME, version: str = "dev") -> None:
        self.model_path = str(model_path)
        self.version = version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=len(ID_TO_LABEL)
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> Prediction:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        label_id = int(torch.argmax(probs).item())

        return Prediction(
            label=ID_TO_LABEL[label_id],
            confidence=round(float(probs[label_id]), 4),
            model_version=self.version,
        )

    @torch.inference_mode()
    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        label_ids = torch.argmax(probs, dim=-1)

        return [
            Prediction(
                label=ID_TO_LABEL[int(label_id)],
                confidence=round(float(probs[i, label_id]), 4),
                model_version=self.version,
            )
            for i, label_id in enumerate(label_ids)
        ]
