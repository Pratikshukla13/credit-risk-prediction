from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap

from src.predict import predict

app = FastAPI()

# 🔹 Input Schema
from typing import Literal

class InputData(BaseModel):
    # 💰 Financial
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float | None = None
    AMT_GOODS_PRICE: float | None = None

    # 👨‍👩‍👧 Personal
    CNT_CHILDREN: int | None = None
    DAYS_BIRTH: float | None = None

    # 💼 Employment
    DAYS_EMPLOYED: float | None = None
    NAME_INCOME_TYPE: str | None = None
    OCCUPATION_TYPE: str | None = None
    ORGANIZATION_TYPE: str | None = None

    # 🏠 Housing / Assets
    FLAG_OWN_CAR: Literal["Y", "N"]
    FLAG_OWN_REALTY: Literal["Y", "N"]
    NAME_HOUSING_TYPE: str | None = None

    # 🎓 Social / Education
    NAME_EDUCATION_TYPE: str | None = None
    NAME_FAMILY_STATUS: str | None = None

    # 👤 Demographics
    CODE_GENDER: Literal["M", "F"]
    NAME_CONTRACT_TYPE: Literal["Cash loans", "Revolving loans"]

    # 🌍 Region / Stability
    REGION_POPULATION_RELATIVE: float | None = None
    DAYS_REGISTRATION: float | None = None
    DAYS_ID_PUBLISH: float | None = None

    # 📊 External Scores (VERY IMPORTANT)
    EXT_SOURCE_1: float | None = None
    EXT_SOURCE_2: float | None = None
    EXT_SOURCE_3: float | None = None

    # 📅 Process Info
    WEEKDAY_APPR_PROCESS_START: str | None = None
    HOUR_APPR_PROCESS_START: float | None = None

# 🔹 Load pipeline for SHAP
pipeline = joblib.load("pipeline_full.pkl")
preprocessor = pipeline.named_steps["preprocessor"]
model_only = pipeline.named_steps["model"]

explainer = shap.TreeExplainer(model_only)


@app.get("/")
def home():
    return {"message": "Home Credit Default Risk API Running 🚀"}


# 🔥 PREDICTION API
@app.post("/predict")
def make_prediction(data: InputData):
    try:
        df = pd.DataFrame([data.model_dump()])

        prob = predict(df)[0]

        prediction = "Default" if prob >= 0.5 else "No Default"

        if prob >= 0.75:
            risk = "HIGH"
        elif prob >= 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "default_probability": round(float(prob), 4),
            "prediction": prediction,
            "risk_level": risk
        }

    except Exception as e:
        return {"error": str(e)}

# 🔥 SHAP EXPLAIN API
@app.post("/explain")
def explain_prediction(data: InputData):
    try:
        df = pd.DataFrame([data.model_dump()])

        # 🔥 FIX: align columns like predict.py
        expected_cols = list(pipeline.feature_names_in_)

        full_data = pd.DataFrame(columns=expected_cols)

        for col in df.columns:
            full_data[col] = df[col]

        full_data = full_data[expected_cols]

        # Transform
        X_transformed = preprocessor.transform(full_data)

        # Prediction
        prob = model_only.predict_proba(X_transformed)[:, 1][0]

        # SHAP values
        # SHAP values
        shap_values = explainer.shap_values(X_transformed)

        feature_names = preprocessor.get_feature_names_out()

        # 🔥 FIX HERE
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        shap_dict = dict(zip(feature_names, values))

        # Clean names
        clean_dict = {k.split("__")[-1]: v for k, v in shap_dict.items()}

        # Top features
        top_features = {
            k: {
                "impact": round(float(v), 4),
                "effect": "increase risk" if v > 0 else "decrease risk"
            }
            for k, v in sorted(clean_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        }

        # Risk
        if prob >= 0.75:
            risk = "HIGH"
        elif prob >= 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "default_probability": round(float(prob), 4),
            "risk_level": risk,
            "top_features": top_features
        }

    except Exception as e:
        return {"error": str(e)}