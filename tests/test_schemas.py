import pytest

from src.inference.schemas import PredictRequest


def test_predict_request_rejects_blank_text() -> None:
    with pytest.raises(ValueError):
        PredictRequest(text="   ")


def test_predict_request_accepts_text() -> None:
    request = PredictRequest(text="ممتاز")
    assert request.text == "ممتاز"
