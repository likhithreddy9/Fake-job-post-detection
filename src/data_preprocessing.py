import pandas as pd
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "fake_job_postings.csv"

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]
STRUCTURED_COLUMNS = [
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "location",
]


def load_training_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates()
    if {"text", "fraudulent"}.issubset(df.columns):
        df["text"] = df["text"].fillna("").astype(str).str.strip()
        df["fraudulent"] = pd.to_numeric(df["fraudulent"], errors="coerce")
        df = df[df["text"] != ""]
        return df[df["fraudulent"].isin([0, 1])][["text", "fraudulent"]].copy()

    columns = TEXT_COLUMNS + STRUCTURED_COLUMNS
    df[columns] = df[columns].fillna("").astype(
        str).apply(lambda column: column.str.strip())
    df["text"] = df.apply(_combine_posting_fields, axis=1)
    df = df[df["text"].str.strip() != ""]
    df["fraudulent"] = pd.to_numeric(df["fraudulent"], errors="coerce")
    return df[df["fraudulent"].isin([0, 1])][["text", "fraudulent"]].copy()


def _combine_posting_fields(row: pd.Series) -> str:
    parts = [f"{column}: {row[column]}" for column in TEXT_COLUMNS if row[column]]
    parts.extend(
        f"{column}: {row[column]}"
        for column in STRUCTURED_COLUMNS
        if row[column]
    )
    return " ".join(parts)
