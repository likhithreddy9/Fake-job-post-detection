from pathlib import Path
import re

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fake_job_pipeline.pkl"
FAKE_THRESHOLD = 0.50
REVIEW_MARGIN = 0.15

pipeline = joblib.load(MODEL_PATH)


def _requires_upfront_payment(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return bool(
        re.search(
            r"\b(?:required to|must|need to|asked to)\s+(?:pay|send|transfer)\b.{0,100}\b(?:fee|deposit|payment|money)\b",
            normalized,
        )
        or re.search(
            r"\b(?:fee|deposit|payment)\b.{0,100}\b(?:before|prior to|in order to receive|for receiving)\b",
            normalized,
        )
    )


def _has_high_risk_scam_pattern(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return (
        bool(re.search(r"\bguaranteed\s+(?:income|earnings|salary)\b", normalized))
        and bool(re.search(r"\bno\s+(?:prior\s+|previous\s+|professional\s+)?experience\s+(?:required|needed)\b", normalized))
        and bool(re.search(r"\b(?:work|working)\s+(?:completely\s+)?from\s+home\b", normalized))
    ) or (
        bool(re.search(
            r"\b(?:send|provide|submit)\b.{0,80}\b(?:government[- ]issued|national)\s+id(?:entification)?\b", normalized))
        and bool(re.search(r"\b(?:before|prior to)\b.{0,80}\b(?:contract|employment)\b", normalized))
    )


def predict(text: str) -> dict:
    proba = pipeline.predict_proba([text])[0]
    fake_probability = float(proba[list(pipeline.classes_).index(1)])
    payment_scam = _requires_upfront_payment(text)
    high_risk_pattern = _has_high_risk_scam_pattern(text)
    if payment_scam or high_risk_pattern:
        prediction = "fake"
        confidence = 0.99
    elif abs(fake_probability - FAKE_THRESHOLD) < REVIEW_MARGIN:
        prediction = "review"
        confidence = max(fake_probability, 1 - fake_probability)
    else:
        prediction = "fake" if fake_probability >= FAKE_THRESHOLD else "real"
        confidence = fake_probability if prediction == "fake" else 1 - fake_probability

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }
