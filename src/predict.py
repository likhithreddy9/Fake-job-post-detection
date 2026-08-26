from pathlib import Path
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fake_job_pipeline.pkl"

pipeline = joblib.load(MODEL_PATH)


def predict(text: str) -> dict:
    label = int(pipeline.predict([text])[0])

    proba = pipeline.predict_proba([text])[0]

    confidence = float(max(proba))

    return {
        "prediction": "fake" if label == 1 else "real",
        "confidence": round(confidence, 4)
    }
