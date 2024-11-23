import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.multioutput import MultiOutputClassifier
from datetime import datetime

st.set_page_config(
    page_title="Multi Target Accident Severity Predictor",
    page_icon="🚗",
    layout="wide"
)

# Caching the model and encoders to avoid reloading on every interaction
@st.cache_resource
def load_model():
    try:
        model = joblib.load('multi_estimators_accident_severity_road_accident_spot_model.joblib')
        class_names_severity = joblib.load('label_encoder_severity.joblib')
        class_names_road = joblib.load('label_encoder_road.joblib')
        class_names_section = joblib.load('label_encoder_accident_spot.joblib')
        return model, class_names_severity, class_names_road, class_names_section
    except Exception as e:
        st.error(f"Error loading model or encoders: {e}")
        return None, None, None, None

# Load model and encoders
model, class_names_severity, class_names_road, class_names_section = load_model()

# Load model parameters
@st.cache_data
def load_model_params():
    try:
        with open('model_params.json', 'r') as f:
            params = json.load(f)
        return params
    except Exception as e:
        st.warning(f"Could not load model parameters: {e}")
        return {}

model_params = load_model_params()

# Title and description
st.title("🚗 **Traffic Accident Severity Prediction**")
st.markdown("""
This application predicts the **severity of traffic accidents** based on the **time of occurrence** and **date**.
""")

# Input section for Accident Date and Time
st.header("📅 **Input Accident Date and Time**")
col1, col2, col3 = st.columns(3)

with col1:
    accident_date = st.date_input("📆 **Accident Date**", datetime.today())

with col2:
    hour = st.slider("🕒 **Hour (24-hour format)**", 0, 23, 12)

with col3:
    minute = st.slider("🕓 **Minute**", 0, 59, 30)


def derive_features(date):
    day_of_week = date.weekday() 
    month = date.month
    season = (month % 12) // 3 + 1  # 1: Winter, 2: Spring, etc.
    return day_of_week, month, season

day_of_week, month, season = derive_features(accident_date)

# Display derived features (optional)
st.subheader("🔍 **Derived Features:**")
st.write(f"**Day of Week:** {day_of_week} (0=Monday, 6=Sunday)")
st.write(f"**Month:** {month}")
st.write(f"**Season:** {season} (1: Winter, 2: Spring, 3: Summer, 4: Autumn)")

# Add Predict button
if st.button("🔮 **Predict Severity**"):

    input_data = np.array([[hour, minute, day_of_week, month, season]])

    # Make prediction
    try:
        if model is not None:
            prediction = model.predict(input_data)[0]
            
            # Check if prediction has multiple outputs
            if isinstance(prediction, (list, np.ndarray)):
                if len(prediction) >= 3:
                    severity = class_names_severity.inverse_transform([prediction[0]])[0]
                    road = class_names_road.inverse_transform([prediction[1]])[0]
                    section = class_names_section.inverse_transform([prediction[2]])[0]
                else:
                    st.error("Unexpected number of prediction outputs.")
                    severity = road = section = "N/A"
            else:
                st.error("Unexpected prediction format.")
                severity = road = section = "N/A"

            # Display prediction results
            st.subheader("🎯 **Prediction Results:**")
            st.write(f"**Predicted Severity Level:** {severity}")
            st.write(f"**Most Dangerous Road:** {road}")
            st.write(f"**Most Dangerous Section:** {section}")

        else:
            st.error("Model is not loaded properly.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Sidebar Information
st.sidebar.header("ℹ️ **About the Model**")
st.sidebar.info("""
🛠 **Model Description:**  
This model predicts accident severity based on historical accident data in Kenya.  
**Features Considered:**
- **Hour of Day**
- **Minute of Hour**
- **Day of The Week**
- **Month of the Year**
- **Season**

**Severity Levels:**
- **No Fatality:** Safe
- **Low Severity:** 1-2 deaths
- **Moderate Severity:** 3-5 deaths
- **High Severity:** >5 deaths
""")

# Model Performance Metrics
st.sidebar.header("📊 **Model Performance**")
accuracy = model_params.get('accuracy', 0.1313) 
st.sidebar.metric("✅ **Model Accuracy**", f"{accuracy:.2%}")

