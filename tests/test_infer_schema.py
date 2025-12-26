from __future__ import annotations

from jigsaw_toxic_classification_ml.data.schema import PredictionResponse


def test_infer_schema() -> None:
    resp = PredictionResponse(
        predictions={
            "toxic": 0.1,
            "severe_toxic": 0.2,
            "obscene": 0.3,
            "threat": 0.4,
            "insult": 0.5,
            "identity_hate": 0.6,
        }
    )
    out = resp.model_dump()
    assert "predictions" in out
    assert set(out["predictions"].keys()) == {
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    }
