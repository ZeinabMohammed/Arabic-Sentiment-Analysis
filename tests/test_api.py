from fastapi.testclient import TestClient

from src.inference.api import app


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_predict_valid_text() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "المنتج رائع"})
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"label", "confidence", "model_version"}


def test_predict_rejects_empty_text() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422
