import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# ✅ Load model & data
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

st.title("🚗 Car Price Predictor")
st.markdown("Get an AI-powered estimate for your vehicle")

# ✅ Dropdown Data
companies = sorted(car['company'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()

# ✅ Company → Models Mapping
company_model_dict = {}

for company in companies:
    company_model_dict[company] = sorted(
        car[car['company'] == company]['name'].unique()
    )

# ✅ UI Inputs
selected_company = st.selectbox("Select Company", companies)

selected_model = st.selectbox(
    "Select Model",
    company_model_dict[selected_company]
)

selected_year = st.selectbox("Select Year", years)

selected_fuel = st.selectbox("Fuel Type", fuel_types)

kms_driven = st.number_input("Kilometres Driven", min_value=0)

# ✅ Prediction Button
if st.button("Predict Price 💰"):

    input_df = pd.DataFrame([[
        selected_model,
        selected_company,
        selected_year,
        kms_driven,
        selected_fuel
    ]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    prediction = model.predict(input_df)[0]

    # ✅ Premium Counter Animation 😎
    result = st.empty()

    for i in np.linspace(0, prediction, 50):
        result.markdown(f"### Estimated Price: ₹ {i:,.2f}")
        time.sleep(0.02)

    result.markdown(f"### Estimated Price: ₹ {prediction:,.2f}")
