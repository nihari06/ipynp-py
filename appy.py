import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load model and columns
model = joblib.load('pollution_model.pkl')
model_cols = joblib.load('model_columns.pkl')

st.title("Water Pollutants Predictor")
st.write("Predict the water pollutants based on Year and Station ID")

year_input = st.number_input("Enter Year", step=1)
station_id = st.text_input("Enter Station ID")

if st.button('Predict'):
    if not station_id:
        st.warning('Please enter the Station ID')
    else:
        input_data = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_data, columns=['id'])

        # Align columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        st.subheader(f"Predicted pollutant levels for the station '{station_id}' in {year_input}:")
        for p, val in zip(pollutants, predicted_pollutants):
            st.write(f"{p}:{val:.2f}")
