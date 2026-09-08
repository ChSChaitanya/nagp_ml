from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------- Feature engineering (must match notebook definition) ----------
def create_feature_table(df_input: pd.DataFrame) -> pd.DataFrame:
    """Replicate the notebook's feature engineering so joblib can unpickle
    the FunctionTransformer that wraps this function."""
    df_out = df_input.copy()
    df_out["TotalCharges"] = pd.to_numeric(
        df_out["TotalCharges"].replace(" ", np.nan), errors="coerce"
    )
    df_out["TotalCharges"] = df_out["TotalCharges"].fillna(0)
    df_out["SeniorCitizen"] = df_out["SeniorCitizen"].astype(int)

    df_out["tenure_group"] = pd.cut(
        df_out["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    ).astype(str)

    df_out["avg_monthly_charge"] = np.where(
        df_out["tenure"] > 0,
        df_out["TotalCharges"] / df_out["tenure"],
        df_out["MonthlyCharges"],
    )

    df_out["service_count"] = (
        (df_out["PhoneService"] == "Yes").astype(int)
        + (df_out["MultipleLines"] == "Yes").astype(int)
        + (df_out["OnlineSecurity"] == "Yes").astype(int)
        + (df_out["OnlineBackup"] == "Yes").astype(int)
        + (df_out["DeviceProtection"] == "Yes").astype(int)
        + (df_out["TechSupport"] == "Yes").astype(int)
        + (df_out["StreamingTV"] == "Yes").astype(int)
        + (df_out["StreamingMovies"] == "Yes").astype(int)
    )
    return df_out


# Inject create_feature_table into __main__ so joblib can unpickle the
# FunctionTransformer that was originally saved from the notebook's __main__.
import __main__ as _main_module
_main_module.create_feature_table = create_feature_table

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Customer Churn Prediction API")


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: str | float | int


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running."}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        features = pd.DataFrame([customer.model_dump()])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        classes = list(model.named_steps["model"].classes_)
        churn_probability = float(probabilities[classes.index("Yes")])
        return {
            "prediction": prediction,
            "churn_probability": round(churn_probability, 4),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid input or prediction error: {exc}")
