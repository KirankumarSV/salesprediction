import os
import pandas as pd
import pickle
from datetime import datetime
from app.pipeline import create_pipeline

# Function to get the CSV file path from the input directory
def get_csv_path(input_dir='input'):
    for file_name in os.listdir(input_dir):
        if file_name.endswith('sales_data.csv'):
            return os.path.join(input_dir, file_name)
    raise FileNotFoundError("No CSV file found in the input directory.")

# # Custom date parser that tries multiple formats
# def custom_date_parser(x):
#     for fmt in ('%d-%m-%Y', '%Y-%m-%d'):
#         try:
#             return datetime.strptime(x, fmt)
#         except ValueError:
#             pass
#     raise ValueError(f"no valid date format found for {x}")

# # Function to apply the custom date parser
# def parse_dates(date_series):
#     return date_series.apply(custom_date_parser)

# Specify dtypes to avoid DtypeWarning
dtype_spec = {
    'store': int,
    'day_of_week': int,
    'sales': float,
    'customers': int,
    'open': int,
    'promo': int,
    'state_holiday': str,
    'school_holiday': int,
    'store_type': str,
    'assortment': str,
    'competition_distance': float,
    'competition_open_since_month': float,
    'competition_open_since_year': float,
    'promo2': int,
    'promo2_since_week': float,
    'promo2_since_year': float,
    'promo_interval': str
}

# Train and save the pipeline
csv_path = get_csv_path()  # Dynamically get the CSV path
df = pd.read_csv(csv_path, dtype=dtype_spec, low_memory=False, parse_dates=['date'])

# # Convert the date column using the custom date parser
# df['date'] = parse_dates(df['date'])

X = df.drop(columns=['sales'])
y = df['sales']

pipeline = create_pipeline()
pipeline.fit(X, y)

# Save the pipeline
with open('model_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
