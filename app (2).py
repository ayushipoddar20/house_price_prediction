import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip

# Load the compressed model
with gzip.open("house_price_model_compressed.pkl.gz", "rb") as f:
    model = joblib.load(f)

# Streamlit App Title
st.title("🏠 House Price Prediction App")

# User Input Fields
sqft = st.number_input("Enter square footage", min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of bathrooms", min_value=1, max_value=10, value=2)

# Predict Button
if st.button("Predict Price"):
    features = np.array([[sqft, bedrooms, bathrooms]])
    prediction = model.predict(features)[0]
    st.success(f"Estimated Price: ${prediction:,.2f}")
