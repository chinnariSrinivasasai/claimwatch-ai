import requests
import os


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-8b-8192"


def fallback_explanation(claim_data: dict, prediction: str, probability: float):
    """
    Basic rule-based fallback explanation when Groq is unavailable.
    """

    reasons = []

    # Some common fraud suspicious patterns (basic rules)
    if "policy_annual_premium" in claim_data:
        premium = float(claim_data["policy_annual_premium"])
        if premium > 2000:
            reasons.append("High annual premium compared to typical customers.")

    if "total_claim_amount" in claim_data:
        claim_amt = float(claim_data["total_claim_amount"])
        if claim_amt > 50000:
            reasons.append("Unusually high total claim amount.")

    if "police_report_available" in claim_data:
        if str(claim_data["police_report_available"]).upper() in ["NO", "0"]:
            reasons.append("Police report is not available, which may raise suspicion.")

    if "property_damage" in claim_data:
        if str(claim_data["property_damage"]).upper() in ["YES", "1"]:
            reasons.append("Property damage is reported, which can increase claim risk.")

    if "incident_severity" in claim_data:
        sev = str(claim_data["incident_severity"]).lower()
        if "total" in sev or "major" in sev:
            reasons.append("Incident severity is high (major damage / total loss).")

    if "age" in claim_data:
        age = float(claim_data["age"])
        if age < 25:
            reasons.append("Young insured age group is statistically higher risk.")

    # Default reasons if none triggered
    if not reasons:
        reasons.append("Claim pattern is within normal expected range based on dataset.")
        reasons.append("No strong anomaly detected in provided input features.")
        reasons.append("Prediction mainly based on learned dataset patterns.")

    recommendation = "Investigate" if prediction == "Fraud Claim" else "Approve"

    explanation = f"""
⚡ Fallback Fraud Analysis

Prediction: {prediction}
Fraud Probability: {probability:.2f}

Key Reasons:
- {reasons[0]}
"""

    if len(reasons) > 1:
        explanation += f"- {reasons[1]}\n"
    if len(reasons) > 2:
        explanation += f"- {reasons[2]}\n"

    explanation += f"\nFinal Recommendation: {recommendation}\n"

    return explanation.strip()


def analyze_with_groq(claim_data: dict, prediction: str, probability: float):
    # If Groq key missing → fallback
    if not GROQ_API_KEY:
        return fallback_explanation(claim_data, prediction, probability)

    prompt = f"""
You are an insurance fraud investigation assistant.

Claim details:
{claim_data}

Machine Learning Prediction:
Prediction: {prediction}
Fraud Probability: {probability:.2f}

Explain why this claim might be fraudulent or legal.
Provide 3 reasons in bullet points.
Also provide final recommendation: Investigate / Approve.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a fraud detection expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4
    }

    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload)

        if response.status_code != 200:
            # If Groq fails → fallback
            return fallback_explanation(claim_data, prediction, probability)

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception:
        # If request crashes → fallback
        return fallback_explanation(claim_data, prediction, probability)
