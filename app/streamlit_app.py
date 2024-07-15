# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# from utils.utils import load_model
# import inflection

# # Function to load the model with caching
# @st.cache_resource
# def load_model_pipeline():
#     return load_model('model_pipeline.joblib')

# # Load the trained pipeline
# model_pipeline = load_model_pipeline()

# st.title("Sales Prediction")

# # Single Prediction
# st.header("Single Prediction")
# store = st.number_input("Store ID", min_value=0, max_value=1115, step=1)
# day_of_week = st.number_input("Day of Week", min_value=1, max_value=7, step=1)
# date = st.date_input("Date")
# customers = st.number_input("Customers", min_value=0, step=1)
# open_status = st.number_input("Open (1 = open, 0 = closed)", min_value=0, max_value=1, step=1)
# promo = st.number_input("Promo (1 = promo, 0 = no promo)", min_value=0, max_value=1, step=1)
# state_holiday = st.text_input("State Holiday (0 = none, a = public holiday, b = Easter holiday, c = Christmas)")
# school_holiday = st.number_input("School Holiday (1 = holiday, 0 = no holiday)", min_value=0, max_value=1, step=1)
# store_type = st.text_input("Store Type (a, b, c, d)")
# assortment = st.text_input("Assortment (a, b, c)")
# competition_distance = st.number_input("Competition Distance", min_value=0, step=1)
# competition_open_since_month = st.number_input("Competition Open Since Month", min_value=1, max_value=12, step=1)
# competition_open_since_year = st.number_input("Competition Open Since Year", min_value=1900, step=1)
# promo2 = st.number_input("Promo2 (1 = continuing, 0 = none)", min_value=0, max_value=1, step=1)
# promo2_since_week = st.number_input("Promo2 Since Week (leave blank if not applicable)", value=0)
# promo2_since_year = st.number_input("Promo2 Since Year (leave blank if not applicable)", value=0)
# promo_interval = st.text_input("Promo Interval (Jan,Apr,Jul,Oct) (leave blank if not applicable)")

# single_predict_button = st.button("Predict Sales")

# if single_predict_button:
#     single_data = {
#         'store': store,
#         'day_of_week': day_of_week,
#         'date': pd.to_datetime(date),
#         'customers': customers,
#         'open': open_status,
#         'promo': promo,
#         'state_holiday': state_holiday,
#         'school_holiday': school_holiday,
#         'store_type': store_type,
#         'assortment': assortment,
#         'competition_distance': competition_distance,
#         'competition_open_since_month': competition_open_since_month,
#         'competition_open_since_year': competition_open_since_year,
#         'promo2': promo2,
#         'promo2_since_week': promo2_since_week,
#         'promo2_since_year': promo2_since_year,
#         'promo_interval': promo_interval
#     }
    
#     single_df = pd.DataFrame([single_data])
#     single_df['date'] = pd.to_datetime(single_df['date'])  # Ensure 'date' is datetime

#     # Convert columns to snake_case
#     snakecase = lambda x: inflection.underscore(x)
#     single_df.columns = list(map(snakecase, single_df.columns))

#     if open_status == 0:
#         single_prediction = [0]
#     else:
#         single_prediction = model_pipeline.predict(single_df)
#     single_prediction_adjusted = np.maximum(single_prediction, 0)  # Ensure no negative sales
#     st.write(f"Predicted Sales: {single_prediction_adjusted[0]}")

import streamlit as st
import pandas as pd
import numpy as np
import inflection
from utils.utils import load_model

# Function to load the model with caching
@st.cache_resource
def load_model_pipeline():
    return load_model('model_pipeline.joblib')

# Load the trained pipeline
model_pipeline = load_model_pipeline()

# Load store data
store_data = pd.read_csv('input_data/rossmann-store-sales/store.csv')

st.title("Sales Prediction")

# Single Prediction
st.header("Single Prediction")
store = st.number_input("Store ID", min_value=1, max_value=1115, step=1)
day_of_week = st.number_input("Day of Week", min_value=1, max_value=7, step=1)
date = st.date_input("Date")
open_status = st.number_input("Open (1 = open, 0 = closed)", min_value=0, max_value=1, step=1)
promo = st.number_input("Promo (1 = promo, 0 = no promo)", min_value=0, max_value=1, step=1)
state_holiday = st.text_input("State Holiday (0 = none, a = public holiday, b = Easter holiday, c = Christmas)")
school_holiday = st.number_input("School Holiday (1 = holiday, 0 = no holiday)", min_value=0, max_value=1, step=1)
customers = st.number_input("Customers", min_value=0, step=1)

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
        'school_holiday': school_holiday
    }
    
    single_df = pd.DataFrame([single_data])
    single_df['date'] = pd.to_datetime(single_df['date'])  # Ensure 'date' is datetime

    # Convert columns to snake_case
    snakecase = lambda x: inflection.underscore(x)
    single_df.columns = list(map(snakecase, single_df.columns))

    # Merge with store data
    store_info = store_data[store_data['Store'] == store]
    store_info.columns = list(map(snakecase, store_info.columns))
    single_df = single_df.merge(store_info, how='left', left_on='store', right_on='store')

    # Implement the logic for predicting sales
    if open_status == 0:
        single_prediction_adjusted = [0]
    elif open_status == 1 and customers == 0:
        single_prediction_adjusted = [0]
    else:
        single_prediction = model_pipeline.predict(single_df)
        single_prediction_adjusted = np.maximum(single_prediction, 0)  # Ensure no negative sales

    st.write(f"Predicted Sales: {single_prediction_adjusted[0]}")
    