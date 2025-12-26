from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


LabelName = Literal[
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


class SingleTextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    predictions: dict[LabelName, float]


