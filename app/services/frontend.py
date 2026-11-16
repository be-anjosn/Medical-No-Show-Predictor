import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="Medical No-Show Predictor", page_icon="🏥", layout="centered")

st.title("Medical No-Show Prediction")
st.write("Enter the patient's details below to predict the likelihood of a no-show for the appointment.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["F", "M"])
        age = st.number_input("Age", min_value=0, max_value=115, value=30)
        scholarship = st.selectbox("Bolsa Família", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        hipertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
    with col2:
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        alcoholism = st.selectbox("Alcoholism", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        handcap = st.selectbox("Disability (Degree)", [0, 1, 2, 3, 4])
        
    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:
        scheduled_date = st.date_input("Scheduled Date", value=datetime.today())
    with col4:
        appointment_date = st.date_input("Appointment Date", value=datetime.today())

    submit_button = st.form_submit_button(label="Analyze No-Show Risk")

if submit_button:
    payload = {
        "gender": gender,
        "age": int(age),
        "scholarship": int(scholarship),
        "hipertension": int(hipertension),
        "diabetes": int(diabetes),
        "alcoholism": int(alcoholism),
        "handcap": int(handcap),
        "scheduled_day": f"{scheduled_date}T08:00:00",
        "appointment_day": f"{appointment_date}T09:00:00"
    }

    try:
        response = requests.post("http://127.0.0.1:8000/explain", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            pred = data["prediction"]
            prob = data["prediction"]["probability"]
            label = data["prediction"]["prediction_label"]
            explanation = data["explanation"]["explanation"]
            
            st.subheader("Analysis Results")
            
            if label == "possible_no_show":
                st.error(f"⚠️ **High Risk of No-Show**")
            else:
                st.success(f"✅ **Patient with High Likelihood of Attendance**")
                
            st.metric(label="Probability of Absence", value=f"{prob * 100:.2f}%")
            
            st.info(f"**Model Explanation:** {explanation}")
            
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown Erro')}")
            
    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to the FastAPI API. Make sure uvicorn is running on port 8000!")