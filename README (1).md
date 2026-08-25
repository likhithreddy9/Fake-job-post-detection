# Fake Job Posting Detection

An end-to-end ML system that classifies job postings as **real** or **fake**, structured as an offline training pipeline feeding an online serving system with a model registry and a retraining feedback loop.

## Architecture

**Offline training pipeline**
Data collection → Preprocessing → Feature engineering → Train & evaluate → Model registry

**Online serving system**
Client → Backend API (FastAPI) → Inference service (loads model from registry) → Database (postings + predictions)

Stored predictions feed back into data collection, so the model can be periodically retrained on real-world data.

```
fake-job-detection/
│
├── 📁 data/
│   ├── 📁 raw/
│   │   └── fake_job_postings.csv
│   │
│   └── 📁 processed/
│       └── cleaned_job_postings.csv
│
├── 📁 notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── 📁 src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
│
├── 📁 models/
│   └── fake_job_pipeline.pkl
│
├── 📁 app/
│   └── app.py
│
├── 📁 reports/
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── 📁 screenshots/
│   └── app_preview.png
│
├── README.md
├── requirements.txt
└── .gitignore