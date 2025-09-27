import streamlit as st
import pickle
import numpy as np

st.title("🩺 Diabetes Progression Predictor")
st.write("Enter the following health measurements to predict the diabetes progression score.")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature sliders (10 features in the diabetes dataset)
feature_names = [
    "Age", "Sex", "BMI", "Average Blood Pressure",
    "S1", "S2", "S3", "S4", "S5", "S6"
]

values = []
for name in feature_names:
    val = st.slider(f"{name}", -0.1, 0.2, 0.0)
    values.append(val)

if st.button("Predict"):
    features = np.array(values).reshape(1, -1)
    prediction = model.predict(features)[0]
    st.success(f"Predicted Diabetes Progression Score: {prediction:.2f}")
