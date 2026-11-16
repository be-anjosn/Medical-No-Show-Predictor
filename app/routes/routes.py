from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import ollama
from schemas.schema import (
    AppointmentInput,
    PredictionResponse,
    ExplanationResponse,
    AnalysisResponse
)
from services.date_processing import process_dates

# Creating a Router
router = APIRouter()

# Importing ML Model
MODEL_PATH = Path("models/model.pkl")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("../models/model.pkl")

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = artifact["threshold"]
    expected_features = artifact["features"]

except Exception as e:
    model = None
    threshold = 0.5
    expected_features = []
    print(f"WARNING: Model not imported: {e}")


# Health Check Endpoint
@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "medical-no-show-api",
        "message": "API is running"
    }


def predict_appointment(appointment: AppointmentInput) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=500, detail="Model was not imported.")
    
    date_features = process_dates(appointment.scheduled_day, appointment.appointment_day)
    input_data = {
        "Gender": 1 if appointment.gender == "M" else 0,
        "Age": appointment.age,
        "Scholarship": appointment.scholarship,
        "Hipertension": appointment.hipertension,
        "Diabetes": appointment.diabetes,
        "Alcoholism": appointment.alcoholism,
        "Handcap": appointment.handcap,
        **date_features
    }

    df_input = pd.DataFrame([input_data])
    for feature in expected_features:
        if feature not in df_input.columns:
            df_input[feature] = 0
    df_input = df_input[expected_features]

    probabilities = model.predict_proba(df_input)
    prob_no_show = float(probabilities[0][1])

    prediction = 1 if prob_no_show >= threshold else 0
    prediction_label = "possible_no_show" if prediction == 1 else "attended"

    return PredictionResponse(
        prediction=prediction,
        prediction_label=prediction_label,
        probability=prob_no_show,
        threshold=threshold
    )

def generate_explanation(appointment: AppointmentInput, probability: float, threshold: float) -> ExplanationResponse:
    waiting_days = (appointment.appointment_day.date() - appointment.scheduled_day.date()).days
    
    prob_percentage = probability * 100
    
    status_risco = "High No show risk" if probability >= threshold else "Low no show risk"
    
    prompt = f"""
        You are an AI clinical assistant focused on medical clinic management.
        A Machine Learning model analyzed a patient's appointment and determined a {status_risco} with an exact no-show probability of {prob_percentage:.1f}% (the cutoff threshold is {threshold * 100:.1f}%).

        Patient data:
        - Age: {appointment.age} years
        - Gender: {appointment.gender}
        - Waiting days until the appointment: {waiting_days} days
        - Social program beneficiary (Welfare/Scholarship): {'Yes' if appointment.scholarship == 1 else 'No'}
        - Hypertension: {'Yes' if appointment.hipertension == 1 else 'No'}
        - Diabetes: {'Yes' if appointment.diabetes == 1 else 'No'}
        - Alcoholism: {'Yes' if appointment.alcoholism == 1 else 'No'}

        Strict instructions:
        1. Write a short, professional, and objective paragraph in English.
        2. Clearly mention the risk percentage ({prob_percentage:.1f}%).
        3. Justify the risk based on the waiting days or age, without inventing medical information that is not in the data.
        4. End with a practical suggestion for the clinic's reception (e.g., send an SMS reminder or confirm attendance in advance).
        5. Output ONLY the paragraph text. Do not include introductory text, conversational filler, notes, or quotation marks.
        """

    try:
        response = ollama.chat(
            model='llama3.2:1b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        explanation_text = response['message']['content'].strip()

    except Exception as e:
        explanation_text = f"The patient has {status_risco} ({prob_percentage:.1f}% based on history and {waiting_days} days of wait. (WARNING: Ollama not running - {str(e)})"

    return ExplanationResponse(
        explanation=explanation_text,
        model="LightGBM + Ollama (Llama 3)"
    )


# Predict Appointment Endpoint
@router.post("/predict", response_model=PredictionResponse)
def predict(appointment: AppointmentInput):
    try:
        return predict_appointment(appointment)
    
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

# Explanation of the results Endpoint
@router.post("/explain", response_model=AnalysisResponse)
def explain_prediction(appointment: AppointmentInput):
    try:
        prediction_res = predict_appointment(appointment)
        explanation_res = generate_explanation(appointment, prediction_res.probability, prediction_res.threshold)
        
        return AnalysisResponse(
            prediction=prediction_res,
            explanation=explanation_res
        )
    
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))