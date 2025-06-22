import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("house_price_model_tuned.pkl")

st.title("🏠 House Price Prediction App")

st.write("### Enter the house details below:")

OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 500, 6000, 1500)
GarageCars = st.slider("Garage Cars", 0, 4, 2)
TotalBsmtSF = st.number_input("Basement Area (sq ft)", 0, 3000, 800)
FullBath = st.slider("Full Bathrooms", 0, 4, 2)
YearBuilt = st.number_input("Year Built", 1800, 2024, 2000)

if st.button("Predict Price"):
    input_data = pd.DataFrame({
        'OverallQual': [OverallQual],
        'GrLivArea': [GrLivArea],
        'GarageCars': [GarageCars],
        'TotalBsmtSF': [TotalBsmtSF],
        'FullBath': [FullBath],
        'YearBuilt': [YearBuilt]
    })

    all_features = model.named_steps['preprocessor'].transformers_[0][2].tolist() + model.named_steps['preprocessor'].transformers_[1][2].tolist()

    for col in all_features:
        if col not in input_data.columns:
            input_data[col] = np.nan

    pred = model.predict(input_data)
    st.success(f"Predicted House Price: ${pred[0]:,.0f}")
