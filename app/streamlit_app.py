import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from io import StringIO
from utils.utils import load_model

# Load the trained pipeline
# with open('model_pipeline.pkl', 'rb') as f:
#     model_pipeline = pickle.load(f)
model_pipeline = load_model('model_pipeline.pkl')

st.title("Sales Prediction")

# Single Prediction
st.header("Single Prediction")
store = st.number_input("Store ID", min_value=0, max_value=1115, step=1)
day_of_week = st.number_input("Day of Week", min_value=1, max_value=7, step=1)
date = st.date_input("Date")
customers = st.number_input("Customers", min_value=0, step=1)
open_status = st.number_input("Open (1 = open, 0 = closed)", min_value=0, max_value=1, step=1)
promo = st.number_input("Promo (1 = promo, 0 = no promo)", min_value=0, max_value=1, step=1)
state_holiday = st.text_input("State Holiday (0 = none, a = public holiday, b = Easter holiday, c = Christmas)")
school_holiday = st.number_input("School Holiday (1 = holiday, 0 = no holiday)", min_value=0, max_value=1, step=1)
store_type = st.text_input("Store Type (a, b, c, d)")
assortment = st.text_input("Assortment (a, b, c)")
competition_distance = st.number_input("Competition Distance", min_value=0, step=1)
competition_open_since_month = st.number_input("Competition Open Since Month", min_value=1, max_value=12, step=1)
competition_open_since_year = st.number_input("Competition Open Since Year", min_value=1900, step=1)
promo2 = st.number_input("Promo2 (1 = continuing, 0 = none)", min_value=0, max_value=1, step=1)
promo2_since_week = st.number_input("Promo2 Since Week", min_value=1, max_value=52, step=1)
promo2_since_year = st.number_input("Promo2 Since Year", min_value=1900, step=1)
promo_interval = st.text_input("Promo Interval (Jan,Apr,Jul,Oct)")

single_predict_button = st.button("Predict Sales")

if single_predict_button:
    single_data = {
        'store': store,
        'day_of_week': day_of_week,
        'date': pd.to_datetime(date),
        'customers': customers,
        'open': open_status,
        'promo': promo,
        'state_holiday': state_holiday,
        'school_holiday': school_holiday,
        'store_type': store_type,
        'assortment': assortment,
        'competition_distance': competition_distance,
        'competition_open_since_month': competition_open_since_month,
        'competition_open_since_year': competition_open_since_year,
        'promo2': promo2,
        'promo2_since_week': promo2_since_week,
        'promo2_since_year': promo2_since_year,
        'promo_interval': promo_interval
    }
    
    single_df = pd.DataFrame([single_data])
    single_df['date'] = pd.to_datetime(single_df['date'])  # Ensure 'date' is datetime
    single_prediction = model_pipeline.predict(single_df)
    st.write(f"Predicted Sales: {single_prediction[0]}")

# Batch Prediction
st.header("Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    batch_data['date'] = pd.to_datetime(batch_data['date'])  # Convert to datetime
    batch_predictions = model_pipeline.predict(batch_data)
    batch_data['predicted_sales'] = batch_predictions

    st.write(batch_data)
    
    # Download the result as CSV
    csv = batch_data.to_csv(index=False)
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv',
    )

    # Plotting the predictions
    st.header("Sales Predictions Plot")
    plt.figure(figsize=(10, 6))
    plt.plot(batch_data['date'], batch_data['predicted_sales'], marker='o')
    plt.title("Predicted Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Predicted Sales")
    plt.xticks(rotation=45)
    st.pyplot(plt)
