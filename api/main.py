from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import shap
from pathlib import Path
from xgboost import XGBClassifier

app = FastAPI(title="Student Dropout Prediction API")

# Allow React frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_path = Path("models")

with open(models_path / "trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(models_path / "shap_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

# Define the input data model
class StudentFeatures(BaseModel):
    Marital_status: int
    Application_mode: int
    Application_order: int
    Course: int
    Daytime_evening_attendance: int
    Previous_qualification: int
    Previous_qualification_grade: float
    Nacionality: int
    Mothers_qualification: int
    Fathers_qualification: int
    Mothers_occupation: int
    Fathers_occupation: int
    Admission_grade: float
    Displaced: int
    Educational_special_needs: int
    Debtor: int
    Tuition_fees_up_to_date: int
    Gender: int
    Scholarship_holder: int
    Age_at_enrollment: int
    International: int
    Curricular_units_1st_sem_credited: int
    Curricular_units_1st_sem_enrolled: int
    Curricular_units_1st_sem_evaluations: int
    Curricular_units_1st_sem_approved: int
    Curricular_units_1st_sem_grade: float
    Curricular_units_1st_sem_without_evaluations: int
    Curricular_units_2nd_sem_credited: int
    Curricular_units_2nd_sem_enrolled: int
    Curricular_units_2nd_sem_evaluations: int
    Curricular_units_2nd_sem_approved: int
    Curricular_units_2nd_sem_grade: float
    Curricular_units_2nd_sem_without_evaluations: int
    Unemployment_rate: float
    Inflation_rate: float
    GDP: float

# Feature engineering helper function
def engineer_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Rename columns to match training data
    df.columns = [c.replace("_", " ") for c in df.columns]
    df.rename(columns={
        "Daytime evening attendance": "Daytime/evening attendance",
        "Previous qualification grade": "Previous qualification (grade)",
        "Mothers qualification": "Mother's qualification",
        "Fathers qualification": "Father's qualification",
        "Mothers occupation": "Mother's occupation",
        "Fathers occupation": "Father's occupation",
        "Curricular units 1st sem credited": "Curricular units 1st sem (credited)",
        "Curricular units 1st sem enrolled": "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem evaluations": "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem approved": "Curricular units 1st sem (approved)",
        "Curricular units 1st sem grade": "Curricular units 1st sem (grade)",
        "Curricular units 1st sem without evaluations": "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem credited": "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem enrolled": "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem evaluations": "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem approved": "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem grade": "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem without evaluations": "Curricular units 2nd sem (without evaluations)",
    }, inplace=True)

    # Engineer features
    enrolled_1st = df["Curricular units 1st sem (enrolled)"].replace(0, np.nan)
    enrolled_2nd = df["Curricular units 2nd sem (enrolled)"].replace(0, np.nan)
    total_enrolled = (df["Curricular units 1st sem (enrolled)"] + 
                      df["Curricular units 2nd sem (enrolled)"]).replace(0, np.nan)

    df["success_rate_1st_sem"] = df["Curricular units 1st sem (approved)"] / enrolled_1st
    df["success_rate_2nd_sem"] = df["Curricular units 2nd sem (approved)"] / enrolled_2nd
    df["overall_approval_rate"] = (df["Curricular units 1st sem (approved)"] + 
                                    df["Curricular units 2nd sem (approved)"]) / total_enrolled
    df["average_grade"] = (df["Curricular units 1st sem (grade)"] + 
                            df["Curricular units 2nd sem (grade)"]) / 2
    df["financial_risk"] = ((df["Debtor"] == 1) | 
                             (df["Tuition fees up to date"] == 0)).astype(int)
    df.fillna(0, inplace=True)

    return df

@app.get("/")
def root():
    return {"message": "Student Dropout Prediction API is running"}


@app.post("/predict")
def predict(student: StudentFeatures):
    df = engineer_features(student.model_dump())

    probability = model.predict_proba(df)[:, 1][0]
    prediction = int(probability >= 0.5)

    shap_values = explainer.shap_values(df)
    explanation = dict(zip(df.columns.tolist(), shap_values[0].tolist()))
    top_factors = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    return {
        "dropout_probability": round(float(probability), 4),
        "prediction": prediction,
        "prediction_label": "Dropout Risk" if prediction == 1 else "Low Risk",
        "top_factors": [{"feature": k, "shap_value": round(v, 4)} 
                        for k, v in top_factors]
    }