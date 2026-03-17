# Student Dropout Explainable AI System

An end-to-end **explainable machine learning system** to predict university student dropout risk, with interpretable predictions powered by **SHAP**, and a modern UI built with **React**.

---

## Project Overview

Student dropout is a major challenge for higher education institutions. This project builds a system that not only predicts **dropout risk** but also explains **why a particular student is at risk**, making it actionable for educators and administrators.

The system uses the [Predict Students' Dropout and Academic Success Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Dropout+and+Academic+Success) from the UCI Machine Learning Repository and includes:

- **Data preprocessing & feature engineering**
- **Machine learning model training**
- **Explainability with SHAP**
- **Prediction API with FastAPI**
- **Interactive React dashboard** for visualizations

---

## Features

- Predicts student dropout risk with high accuracy
- Provides **local and global explanations** for model predictions
- Modular architecture suitable for production deployment
- Interactive dashboard built with React for a polished user experience
- Full pipeline from raw data to deployment

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| API | FastAPI |
| Frontend | React, Chart.js / D3.js |
| Deployment | Docker, GitHub |

---

## Repository Structure

```
student-dropout-xai-system/
├── data/
│   ├── raw/                        # Original dataset
│   └── processed/                  # Cleaned and preprocessed data
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── data/                       # Data loading and preprocessing
│   ├── models/                     # Model training, evaluation, prediction
│   └── explainability/             # SHAP analysis scripts
├── api/                            # FastAPI backend
├── frontend/                       # React UI dashboard
├── models/                         # Saved trained models and explainers
├── config/                         # Configuration files
├── tests/                          # Unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js & npm / yarn
- Docker *(optional, for containerized deployment)*

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Start the Backend API

```bash
uvicorn api.main:app --reload
```

### Start the React Frontend

```bash
cd frontend
npm install
npm start
```

The dashboard will be available at [http://localhost:3000](http://localhost:3000).
