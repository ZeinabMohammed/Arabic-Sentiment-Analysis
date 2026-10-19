import pytest

from src.config import LABEL_NAMES
from src.models.sentiment_model import ArabicSentimentModel


@pytest.fixture(scope="module")
def model() -> ArabicSentimentModel:
    return ArabicSentimentModel()


def test_predict_returns_known_label(model: ArabicSentimentModel) -> None:
    result = model.predict("المنتج رائع جدا")
    assert result.label in LABEL_NAMES
    assert 0.0 <= result.confidence <= 1.0


def test_predict_batch_matches_input_length(model: ArabicSentimentModel) -> None:
    texts = ["جيد", "سيء جدا", "عادي"]
    results = model.predict_batch(texts)
    assert len(results) == len(texts)
    assert all(r.label in LABEL_NAMES for r in results)
