# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline (includes preprocessing + model)
model = joblib.load("car_price_model_pipeline.pkl")

# Streamlit page configuration
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Title and description
st.title("🚗 Car Price Prediction App")
st.markdown("""
Welcome to the Car Price Predictor App!  
Enter your car details and we'll predict the estimated *resale price* in Indian Rupees.
---
""")

# Input form using columns
col1, col2, col3 = st.columns(3)

with col1:
    car_name = st.selectbox("Select Car Name", [
        "Ritz", "Swift", "Dzire", "Etios", "Corolla Altis", "Verna", "City",
        "Creta", "i20", "i10", "Indica", "Scorpio", "Xcent", "Ciaz", "Ertiga",
        "Wagon R", "Alto", "Fortuner", "EcoSport", "Endeavour", "Duster", "KWID",
        "Figo", "BR-V", "Jazz", "Sail", "Ignis", "Vitara Brezza", "Baleno"
        # Add more based on your dataset
    ])
    year = st.selectbox("Year of Purchase", list(range(2000, 2025)))

with col2:
    present_price = st.number_input("Present Price (in Lakhs)", step=0.1)
    kms_driven = st.number_input("Kilometers Driven", step=1000)

with col3:
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# Prepare input dataframe
input_dict = {
    'Car_Name': car_name,
    'Year': year,
    'Present_Price': present_price,
    'Kms_Driven': kms_driven,
    'Fuel_Type': fuel_type,
    'Seller_Type': seller_type,
    'Transmission': transmission,
    'Owner': owner
}

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict Price 💰"):
    try:
        prediction = model.predict(input_df)[0]
        prediction_lakh = prediction

        st.success(f"Predicted Car Price: ₹ {prediction_lakh:,.2f} Lakhs")

        if prediction_lakh < 3:
            st.info("💡 Budget Category")
        elif prediction_lakh < 7:
            st.info("🚘 Medium Category")
        else:
            st.info("🚗 Premium Category")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("© 2025 | Developed by Rahul | Contact: rpawar1839@gmail.com")