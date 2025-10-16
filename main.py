from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import Optional

# -----------------------------
# Load model and features
# -----------------------------
with open("best_model_logistic_regression.pkl", "rb") as f:
    model = pickle.load(f)

with open("best_model_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(
    title="TMKOC Episode Success Predictor API",
    description="Predict success of TMKOC episodes based on features",
    version="1.0"
)

# -----------------------------
# Input schema
# -----------------------------
class EpisodeFeatures(BaseModel):
    lead_cast: int = 3
    supporting_cast: int = 5
    desc_length: int = 300
    desc_word_count: int = 50
    question_count: int = 2
    runtime: int = 22
    release_year: int = 2020
    release_month: int = 6
    is_weekend: int = 0
    has_conflict: int = 1
    has_main_char: int = 1
    has_society: int = 1

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_episode(features: EpisodeFeatures):
    # Convert input to DataFrame
    input_dict = features.dict()
    # Ensure all features from feature_names are present
    for feat in feature_names:
        if feat not in input_dict:
            input_dict[feat] = 0
    df_input = pd.DataFrame([input_dict])[feature_names]  # order features correctly

    # Make prediction
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]  # probability of success

    return {
        "prediction": int(pred),
        "success_probability": float(prob),
        "input_features": input_dict
    }

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "TMKOC Episode Success Predictor API is running!"}
