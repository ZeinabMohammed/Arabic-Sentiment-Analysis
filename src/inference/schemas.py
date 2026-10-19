"""Pydantic request/response models for the sentiment API."""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Arabic review text")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
