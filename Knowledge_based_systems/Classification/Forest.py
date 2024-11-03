import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# Load the saved model and class names using joblib
model = joblib.load('accident_severity_model.joblib')
class_names = joblib.load('class_names.joblib')

# Set page configurations
st.set_page_config(
    page_title="Accident Severity Predictor Random Forest",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("Traffic Accident Severity Prediction")
st.markdown("""
This app predicts the severity of traffic accidents based on the time of occurrence.
""")

# Create time input
st.header("Input Accident Time")
col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Hour (24-hour format)", 0, 23, 12)
with col2:
    minute = st.slider("Minute", 0, 59, 30)

# Add predict button
if st.button("Predict Severity"):

    input_data = np.array([[hour, minute]])
    
    # Make prediction
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[0]
    st.subheader("Prediction Results:")
    st.write(f"Predicted Severity Level: {class_names[prediction[0]]}")
    
    # Display probability distribution
    st.subheader("Probability Distribution:")
    prob_df = pd.DataFrame({
        'Severity Level': class_names,
        'Probability': probabilities
    })
    
    # Create bar chart
    st.bar_chart(prob_df.set_index('Severity Level'))

st.sidebar.header("About the Model")
st.sidebar.info("""
This model predicts accident severity based on historical accident data in Kenya.
The model considers:
- Hour of day
- Minute of hour

Severity Levels:
- No Fatality thus safe
- Low Severity (1-2 deaths)
- Moderate Severity (3-5 deaths)
- High severity (>5 deaths)
""")

# Add model metrics
accuracy = 0.39 
st.sidebar.header("Model Performance")
st.sidebar.metric("Model Accuracy", f"{accuracy:.2%}")