from pathlib import Path
import json
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.predict import predict


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
FEEDBACK_PATH = BASE_DIR.parent / "data" / "feedback" / "predictions.jsonl"

app = FastAPI(
    title="Fake Job Detection API",
    description="AI-powered fake job posting detection system"
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


class JobPosting(BaseModel):
    text: str


class PredictionFeedback(BaseModel):
    text: str
    prediction: str
    correct: bool
    actual_label: str | None = None


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict")
def predict_job(posting: JobPosting):
    return predict(posting.text)


@app.post("/feedback")
def save_feedback(feedback: PredictionFeedback):
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = feedback.model_dump()
    record["created_at"] = datetime.now(timezone.utc).isoformat()
    with FEEDBACK_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
    return {"status": "saved"}


@app.get("/health")
def health():
    return {"status": "ok"}
