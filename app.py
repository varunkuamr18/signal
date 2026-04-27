import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load model and scaler
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open('signal_model_optimized.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('signal_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# -------------------------------
# UI
# -------------------------------
st.title("Signal Data Quality Prediction")
st.write("Upload sensor data CSV to predict Pass/Fail")

# ✅ CORRECT FUNCTION
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# -------------------------------
# File Handling
# -------------------------------
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(input_df.head())

    if st.button("Predict"):
        try:
            # Convert to numpy (important)
            data = input_df.values

            # Scale
            scaled_data = scaler.transform(data)

            # Predict
            prediction = model.predict(scaled_data)

            # Add result
            input_df['Prediction'] = np.where(prediction == -1, 'Pass', 'Fail')

            st.success("Predictions completed!")
            st.dataframe(input_df)

        except Exception as e:
            st.error(f"Error in prediction: {e}")

else:
    st.info("Please upload a CSV file containing the sensor data features.")
