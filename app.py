# Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import streamlit as st

# Streamlit UI setup
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("Diabetes Prediction App")
st.markdown("Enter the following details to predict diabetes:")

# Try to load the model
try:
    model = tf.keras.models.load_model('Diabetes_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Try to load the scaler
try:
    scaler = joblib.load('Scaler.pkl')
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

# Input fields for user data
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin Level", min_value=0, value=79)
bmi = st.number_input("BMI", min_value=1.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=1, value=30)

# Button to trigger prediction
if st.button("Predict Diabetes"):
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

    # Apply scaling
    try:
        input_scaled = scaler.transform(input_data)
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.stop()

    # Make prediction
    try:
        prediction = model.predict(input_scaled)
        result = "Not Diabetic" if prediction[0][0] < 0.5 else "Diabetic"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
