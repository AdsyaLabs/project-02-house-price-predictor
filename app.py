import streamlit as st
import pickle
import numpy as np

# Load model
with open('model/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# UI
st.title("🏠 House Price Predictor")
st.write("Masukkan detail rumah untuk mendapatkan estimasi harga.")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sqft)", 500, 6000, 1500)
    garage_cars = st.slider("Garage Capacity (cars)", 0, 5, 2)
    total_bsmt_sf = st.number_input("Basement Area (sqft)", 0, 6000, 800)

with col2:
    first_flr_sf = st.number_input("1st Floor Area (sqft)", 300, 5000, 1000)
    year_built = st.slider("Year Built", 1870, 2010, 1990)
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)

# Predict
if st.button("Predict Price"):
    features = np.array([[
        overall_qual, gr_liv_area, garage_cars,
        total_bsmt_sf, first_flr_sf, year_built, full_bath
    ]])
    
    prediction = model.predict(features)[0]
    
    st.success(f"### Estimated Price: ${prediction:,.0f}")
    st.caption("*Estimasi berdasarkan model Ridge Regression — Ames Housing Dataset*")