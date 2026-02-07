import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
from ai_analyzer import analyze_with_groq
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ClaimWatch AI - Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models/fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
label_encoders = joblib.load(os.path.join(BASE_DIR, "models/label_encoders.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "models/feature_columns.pkl"))


class ClaimRequest(BaseModel):
    data: dict


@app.post("/analyze-claim")
def analyze_claim(request: ClaimRequest):
    claim_data = request.data

    # Step 1: Create default full feature input
    full_data = {}

    for col in feature_columns:
        if col in label_encoders:
            # categorical default = first class
            full_data[col] = label_encoders[col].classes_[0]
        else:
            # numeric default = 0
            full_data[col] = 0

    # Step 2: Override with user inputs
    for key, value in claim_data.items():
        if key in full_data:
            full_data[key] = value

    # Step 3: Encode categorical columns safely
    for col, le in label_encoders.items():
        value = str(full_data[col])

        if value in le.classes_:
            full_data[col] = le.transform([value])[0]
        else:
            # fallback to first class if unseen value
            full_data[col] = le.transform([le.classes_[0]])[0]

    # Step 4: Convert into correct ordered array
    X = np.array([full_data[col] for col in feature_columns]).reshape(1, -1)

    # Step 5: Scale
    X_scaled = scaler.transform(X)

    # Step 6: Predict
    prob = model.predict_proba(X_scaled)[0][1]
    pred = model.predict(X_scaled)[0]

    prediction_label = "Fraud Claim" if pred == 1 else "Legal Claim"

    # Step 7: Groq Explanation (safe)
    try:
        explanation = analyze_with_groq(claim_data, prediction_label, prob)
    except Exception as e:
        explanation = f"Groq explanation failed: {str(e)}"

    return {
        "prediction": prediction_label,
        "fraud_probability": float(prob),
        "explanation": explanation
    }
