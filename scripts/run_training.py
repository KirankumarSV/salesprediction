import pandas as pd
from utils.utils import get_csv_path, save_model
from app.pipeline import create_pipeline

def run_training():
    # Load the data
    csv_path = get_csv_path()
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
    df = pd.read_csv(csv_path, dtype=dtype_spec, low_memory=False, parse_dates=['date'])

    X = df.drop(columns=['sales'])
    y = df['sales']

    # Create and train the pipeline
    pipeline = create_pipeline()
    pipeline.fit(X, y)

    # Save the trained pipeline
    save_model(pipeline)
    print("Training completed and model saved.")

if __name__ == '__main__':
    run_training()
