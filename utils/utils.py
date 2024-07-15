import pandas as pd
import numpy as np
from datetime import datetime
from joblib import dump, load

def get_csv_path(input_dir='input_data', filename='sales_data.csv'):
    import os
    for file_name in os.listdir(input_dir):
        if file_name.endswith(filename):
            return os.path.join(input_dir, file_name)
    raise FileNotFoundError(f"No CSV file found in the input directory: {input_dir}")

def custom_date_parser(x):
    for fmt in ('%d-%m-%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            pass
    raise ValueError(f"No valid date format found for {x}")

def parse_dates(date_series):
    return date_series.apply(custom_date_parser)

# def save_model(pipeline, filename='model_pipeline.pkl'):
#     import pickle
#     with open(filename, 'wb') as f:
#         pickle.dump(pipeline, f)

# def load_model(filename='model_pipeline.pkl'):
#     import pickle
#     with open(filename, 'rb') as f:
#         return pickle.load(f)
def save_model(model, filename='model_pipeline.joblib', compress=3):
    dump(model, filename, compress=compress)

def load_model(filename='model_pipeline.joblib'):
    return load(filename)