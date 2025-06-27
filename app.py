import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import joblib
from tensorflow import keras
import streamlit as st
import matplotlib.pyplot as plt



model = tf.keras.models.load_model("diabetes_model.h5")
scaler = joblib.load("Scaler.pkl")



st.set_page_config(page_title="Diabetes Prediction App",layout="Centered")
st.title("Diabetes Prediction App")
st.markdown("enter the following details")


pregnancies = st.number_input("Number of Pregnancies",min_value=0,max_value=20)
glucose = st.number_input("Glucose Level",min_value=0)
blood_pressure = st.number_input("Blood Pressure",min_value=0)
skinthickness = st.number_input("Skin Thickness",min_value=0)
insulin = st.number_input("Insulin Level",min_value=0)
bmi = st.number_input("BMI",min_value=1)
diabetespedigreefunction = st.number_input("Diabetes Pedigree Function",min_value=0.0)
age = st.number_input("Age",min_value=1,max_value=100)


if st.button("Predict"):
    input_data = np.array([[pregnancies,glucose,blood_pressure,skinthickness,insulin,bmi,diabetespedigreefunction,age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = "not diabetic" if prediction[0][0] < 0.5 else "diabetic"

    st.subheader("Prediction Result")
    st.write(f"The person is {result}")