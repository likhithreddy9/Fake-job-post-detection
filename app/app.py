from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.predict import predict


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

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


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict")
def predict_job(posting: JobPosting):
    return predict(posting.text)


@app.get("/health")
def health():
    return {"status": "ok"}
