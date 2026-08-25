# Fake Job Posting Detection — Step-by-Step Execution Guide

---

## Prerequisites

- Python 3.9+
- pip
- Git (optional)
- Jupyter Notebook or JupyterLab
- A code editor (VS Code recommended)

---

## Step 1 — Set Up the Environment

### 1.1 Create and activate a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 1.2 Install dependencies

Populate `requirements.txt` with the following, then install:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
joblib
fastapi
uvicorn
python-multipart
jupyter
```

```bash
pip install -r requirements.txt
```

---

## Step 2 — Get the Dataset

1. Download the dataset from Kaggle:
   👉 https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

2. Place the downloaded file here:
   ```
   data/raw/fake_job_postings.csv
   ```

The dataset contains ~18,000 job postings with a `fraudulent` column (0 = real, 1 = fake).

---

## Step 3 — Exploratory Data Analysis (EDA)

**File:** `notebooks/01_eda.ipynb`

Goals:
- Understand the class distribution (real vs fake)
- Identify missing values per column
- Explore text fields: `title`, `description`, `requirements`, `company_profile`
- Visualize correlations and patterns

Key things to check:
- What % of postings are fraudulent?
- Which columns have the most nulls?
- Are there patterns in fake postings (e.g. missing salary, location)?

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/raw/fake_job_postings.csv')
print(df.shape)
print(df['fraudulent'].value_counts())
df.isnull().sum().plot(kind='bar')
plt.show()
```

---

## Step 4 — Data Preprocessing

**Files:** `notebooks/02_data_preparation.ipynb` → `src/data_preprocessing.py`

Goals:
- Fill or drop missing values
- Combine text columns into a single `text` feature
- Remove duplicates
- Save cleaned data to `data/processed/cleaned_job_postings.csv`

Key steps:

```python
# Combine text fields
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
df['text'] = df[text_cols].fillna('').agg(' '.join, axis=1)

# Drop rows with no text at all
df = df[df['text'].str.strip() != '']

# Keep only needed columns
df = df[['text', 'fraudulent']]

df.to_csv('../data/processed/cleaned_job_postings.csv', index=False)
```

Once validated in the notebook, move the logic into `src/data_preprocessing.py` as a reusable function.

---

## Step 5 — Feature Engineering

**Files:** `notebooks/03_feature_engineering.ipynb` → `src/feature_engineering.py`

Goals:
- Convert raw text into numerical features using TF-IDF
- Optionally add handcrafted features (text length, exclamation count, etc.)

Key steps:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['fraudulent']
```

Handcrafted features to consider:
- `has_salary` — whether salary range is provided
- `has_company_logo` — binary flag
- `text_length` — character count of combined text
- `exclamation_count` — number of `!` in description

Move finalized logic into `src/feature_engineering.py`.

---

## Step 6 — Model Training

**Files:** `notebooks/04_model_training.ipynb` → `src/train_model.py`

Goals:
- Split data into train/test sets
- Train multiple classifiers
- Save the best model to `models/fake_job_pipeline.pkl`

Recommended models to try:
- Logistic Regression (fast baseline)
- Random Forest
- Gradient Boosting (XGBoost or LightGBM)

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['fraudulent'], test_size=0.2, random_state=42, stratify=df['fraudulent']
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, '../models/fake_job_pipeline.pkl')
```

Move finalized logic into `src/train_model.py`.

---

## Step 7 — Model Evaluation

**File:** `notebooks/05_model_evaluation.ipynb`

Goals:
- Evaluate model on the test set
- Generate and save report artifacts to `reports/`

Key metrics to report (use F1 over accuracy due to class imbalance):

```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig('../reports/confusion_matrix.png')

# Feature importance (for tree-based models)
# For Logistic Regression, use coefficients
```

Save all plots to `reports/`:
- `confusion_matrix.png`
- `model_comparison.png` (bar chart of F1 scores across models)
- `feature_importance.png`

---

## Step 8 — Build the Inference Module

**File:** `src/predict.py`

This module loads the saved model and exposes a `predict()` function used by the API.

```python
import joblib

MODEL_PATH = 'models/fake_job_pipeline.pkl'
pipeline = joblib.load(MODEL_PATH)

def predict(text: str) -> dict:
    label = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    return {
        "prediction": "fake" if label == 1 else "real",
        "confidence": round(float(max(proba)), 4)
    }
```

---

## Step 9 — Build the FastAPI App

**File:** `app/app.py`

Goals:
- Expose a `/predict` POST endpoint
- Accept a job posting as JSON input
- Return prediction and confidence score

```python
from fastapi import FastAPI
from pydantic import BaseModel
import sys
sys.path.append('..')
from src.predict import predict

app = FastAPI(title="Fake Job Detection API")

class JobPosting(BaseModel):
    text: str

@app.post("/predict")
def predict_job(posting: JobPosting):
    return predict(posting.text)

@app.get("/health")
def health():
    return {"status": "ok"}
```

### Run the API

```bash
uvicorn app.app:app --reload
```

Visit the interactive docs at: http://127.0.0.1:8000/docs

---

## Step 10 — Test the API

Using the Swagger UI at `/docs`, or via curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Work from home, earn $5000/week, no experience needed!"}'
```

Expected response:
```json
{
  "prediction": "fake",
  "confidence": 0.9312
}
```

---

## Step 11 — Capture Screenshots

Once the app is running:
1. Open http://127.0.0.1:8000/docs in your browser
2. Test a prediction via the Swagger UI
3. Take a screenshot and save it to `screenshots/app_preview.png`

---

## Step 12 — Retraining Feedback Loop (Optional)

To close the loop described in the architecture:

1. Log every prediction (input text + predicted label) to a database or CSV
2. Periodically review flagged predictions
3. Add confirmed labels back into `data/raw/`
4. Re-run Steps 4–7 to retrain and replace `models/fake_job_pipeline.pkl`

This can be automated with a scheduled script or a workflow tool like Apache Airflow.

---

## Execution Order Summary

| Step | Action | File(s) |
|------|--------|---------|
| 1 | Set up environment | `requirements.txt` |
| 2 | Download dataset | `data/raw/` |
| 3 | EDA | `notebooks/01_eda.ipynb` |
| 4 | Preprocessing | `notebooks/02_data_preparation.ipynb` → `src/data_preprocessing.py` |
| 5 | Feature engineering | `notebooks/03_feature_engineering.ipynb` → `src/feature_engineering.py` |
| 6 | Model training | `notebooks/04_model_training.ipynb` → `src/train_model.py` |
| 7 | Evaluation | `notebooks/05_model_evaluation.ipynb` → `reports/` |
| 8 | Inference module | `src/predict.py` |
| 9 | API | `app/app.py` |
| 10 | Test API | curl / Swagger UI |
| 11 | Screenshots | `screenshots/` |
| 12 | Retraining loop | Optional |

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Make sure venv is activated and `pip install -r requirements.txt` was run |
| `FileNotFoundError` for model | Run the training notebook before starting the API |
| Class imbalance in metrics | Use F1-score and `class_weight='balanced'` in your classifier |
| API returns 500 error | Check that `models/fake_job_pipeline.pkl` exists and the path in `predict.py` is correct |
| Jupyter can't find `src/` | Use `sys.path.append('..')` at the top of notebooks |
