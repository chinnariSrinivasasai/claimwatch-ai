import streamlit as st
import requests
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/analyze-claim")


st.set_page_config(page_title="ClaimWatch AI", layout="wide")
st.title("🚨 ClaimWatch AI - Insurance Fraud Detection")

st.write("Enter claim details and click Analyze.")

# Input fields
age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
months_as_customer = st.number_input("Months as Customer", min_value=1, max_value=500, value=120)
policy_annual_premium = st.number_input("Policy Annual Premium", min_value=100.0, max_value=10000.0, value=1200.0)

insured_sex = st.selectbox("Insured Sex", ["MALE", "FEMALE"])
incident_severity = st.selectbox("Incident Severity", ["Minor Damage", "Major Damage", "Total Loss", "Trivial Damage"])
property_damage = st.selectbox("Property Damage", ["YES", "NO"])
police_report_available = st.selectbox("Police Report Available", ["YES", "NO"])

# Button
if st.button("🔍 Analyze Claim"):

    # ✅ claim_data must be inside button block
    claim_data = {
        "age": age,
        "months_as_customer": months_as_customer,
        "policy_annual_premium": policy_annual_premium,
        "insured_sex": insured_sex,
        "incident_severity": incident_severity,
        "property_damage": property_damage,
        "police_report_available": police_report_available
    }

    try:
        response = requests.post(BACKEND_URL, json={"data": claim_data})

        if response.status_code != 200:
            st.error("Backend Error")
            st.write("Status Code:", response.status_code)
            st.write("Response Text:", response.text)
        else:
            result = response.json()

            st.subheader("✅ Prediction Result")
            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Fraud Probability: {result['fraud_probability']:.2f}")

            st.subheader("🤖 AI Explanation")
            st.write(result["explanation"])

    except Exception as e:
        st.error(f"Error connecting backend: {e}")
