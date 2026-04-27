import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load model, scaler, feature list
# -------------------------------
@st.cache_resource
def load_artifacts():
    with open('signal_model_optimized.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('signal_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # 🔥 IMPORTANT: feature names used during training
    with open('features.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# -------------------------------
# UI
# -------------------------------
st.title("Signal Data Quality Prediction")
st.write("Upload CSV file to predict Pass/Fail")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(input_df.head())

    if st.button("Predict"):
        try:
            # -------------------------------
            # 🔥 STEP 1: Remove unwanted columns
            # -------------------------------
            input_df = input_df.select_dtypes(include=[np.number])

            # -------------------------------
            # 🔥 STEP 2: Align with training features
            # -------------------------------
            missing_cols = set(feature_names) - set(input_df.columns)
            extra_cols = set(input_df.columns) - set(feature_names)

            # Add missing columns as 0
            for col in missing_cols:
                input_df[col] = 0

            # Remove extra columns
            input_df = input_df.drop(columns=extra_cols, errors='ignore')

            # Reorder columns EXACTLY
            input_df = input_df[feature_names]

            st.write(f"Final feature count: {input_df.shape[1]}")

            # -------------------------------
            # 🔥 STEP 3: Scale + Predict
            # -------------------------------
            scaled_data = scaler.transform(input_df.values)
            prediction = model.predict(scaled_data)

            input_df['Prediction'] = np.where(prediction == -1, 'Pass', 'Fail')

            st.success("Predictions completed!")
            st.dataframe(input_df)

        except Exception as e:
            st.error(f"Error in prediction: {e}")

else:
    st.info("Please upload a CSV file.")
