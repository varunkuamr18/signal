import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model and scaler
@st.cache_resource
def load_artifacts():
    with open('signal_model_optimized.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('signal_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

st.title("Signal Data Quality Prediction")
st.write("Enter the sensor readings to predict if the process resulted in a **Pass** or **Fail**.")

# Note: In a real app with 446 features, you'd likely upload a CSV
uploaded_file = st.file_file("signal-data (1).csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(input_df.head())
    
    if st.button("Predict"):
        # Ensure the columns match the training features (excluding Time/Target)
        # For this demo, we assume the CSV matches the processed feature count
        try:
            scaled_data = scaler.transform(input_df)
            prediction = model.predict(scaled_data)
            
            input_df['Prediction'] = np.where(prediction == -1, 'Pass', 'Fail')
            st.success("Predictions completed!")
            st.dataframe(input_df)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
else:
    st.info("Please upload a CSV file containing the sensor data features.")
