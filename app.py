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

    with open('features.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# -------------------------------
# UI
# -------------------------------
st.title("Signal Data Quality Prediction")
st.write("Upload your processed CSV file (same format as training data)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# -------------------------------
# Prediction Logic
# -------------------------------
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(input_df.head())

    if st.button("Predict"):
        try:
            original_rows = input_df.shape[0]

            # -------------------------------
            # STEP 1: Remove non-numeric columns (like datetime)
            # -------------------------------
            input_df = input_df.select_dtypes(include=[np.number])

            # -------------------------------
            # STEP 2: Align columns with training features
            # -------------------------------
            missing_cols = set(feature_names) - set(input_df.columns)
            extra_cols = set(input_df.columns) - set(feature_names)

            # Warn user
            if len(extra_cols) > 0:
                st.warning(f"Removing {len(extra_cols)} extra columns")

            if len(missing_cols) > 0:
                st.warning(f"Adding {len(missing_cols)} missing columns as 0")

            # Add missing columns
            for col in missing_cols:
                input_df[col] = 0

            # Remove extra columns
            input_df = input_df.drop(columns=extra_cols, errors='ignore')

            # Reorder columns EXACTLY
            input_df = input_df[feature_names]

            # -------------------------------
            # STEP 3: Validate feature count
            # -------------------------------
            if input_df.shape[1] != len(feature_names):
                st.error("Feature mismatch! Please upload correct dataset.")
            else:
                # -------------------------------
                # STEP 4: Scale + Predict
                # -------------------------------
                scaled_data = scaler.transform(input_df.values)
                prediction = model.predict(scaled_data)

                # -------------------------------
                # STEP 5: Clean result output
                # -------------------------------
                result_df = pd.DataFrame({
                    "Row": range(original_rows),
                    "Prediction": np.where(prediction == -1, "Pass", "Fail")
                })

                st.success("Predictions completed!")

                st.write("Final Results:")
                st.dataframe(result_df)

        except Exception as e:
            st.error(f"Error in prediction: {e}")

else:
    st.info("Please upload a CSV file to proceed.")
