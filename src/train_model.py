from pathlib import Path
import argparse

import joblib
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data_preprocessing import load_training_data


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_job_postings.csv"
MODEL_PATH = BASE_DIR / "models" / "fake_job_pipeline.pkl"


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                            max_features=12000,
                            ngram_range=(1, 2),
                            stop_words="english",
                            sublinear_tf=True,
                            ),
        ),
        (
            "character",
            TfidfVectorizer(
                analyzer="char",
                max_features=15000,
                min_df=2,
                ngram_range=(3, 5),
                sublinear_tf=True,
            ),
        ),
    ])
    return Pipeline([
        ("features", features),
        (
            "clf",
            LogisticRegression(
                C=2.0,
                class_weight="balanced",
                max_iter=1500,
            ),
        ),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Training CSV; supports the raw dataset or a text/fraudulent dataset.",
    )
    args = parser.parse_args()

    df = load_training_data(args.data)
    df["fraudulent"] = df["fraudulent"].astype(int)
    feedback_path = BASE_DIR / "data" / "feedback" / "predictions.jsonl"
    if feedback_path.exists():
        feedback_rows = []
        for line in feedback_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("actual_label") in {"fake", "real"}:
                feedback_rows.append({
                    "text": record["text"],
                    "fraudulent": int(record["actual_label"] == "fake"),
                })
        if feedback_rows:
            df = pd.concat([df, pd.DataFrame(feedback_rows)],
                           ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["fraudulent"],
        test_size=0.2,
        random_state=42,
        stratify=df["fraudulent"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print(classification_report(y_test, pipeline.predict(X_test), zero_division=0))
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
