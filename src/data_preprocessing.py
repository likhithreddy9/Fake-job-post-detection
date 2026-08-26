import pandas as pd
df = pd.read_csv("../data/raw/fake_job_postings.csv")
df = df.drop_duplicates()
text_cols = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]
df[text_cols] = df[text_cols].fillna("")
for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
df["text"] = df[text_cols].agg(" ".join, axis=1)
df = df[df["text"].str.strip() != ""]
df = df[["text", "fraudulent"]]
df = df[df["fraudulent"].isin([0, 1])]
