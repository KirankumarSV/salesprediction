import pandas as pd
import numpy as np
import logging
from utils.utils import get_csv_path, load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_adjusted_r2(r2, n, k):
    return 1 - (1 - r2) * ((n - 1) / (n - k - 1))

def evaluate_model():
    # setup_logging()
    logging.info("Starting model evaluation")

    # Load the data
    logging.info("Loading data")
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
    logging.info("Data loaded successfully")

    X = df.drop(columns=['sales'])
    y = df['sales']

    # Load the trained model
    model_pipeline = load_model('model_pipeline.pkl')

    # Make predictions
    logging.info("Making predictions")
    y_pred = model_pipeline.predict(X)

    # Post-prediction adjustment to ensure sales are never less than zero
    y_pred_adjusted = np.maximum(y_pred, 0)

    # Calculate metrics
    logging.info("Calculating performance metrics")
    r2 = r2_score(y, y_pred_adjusted)
    adjusted_r2 = calculate_adjusted_r2(r2, X.shape[0], X.shape[1])
    mae = mean_absolute_error(y, y_pred_adjusted)
    rmse = np.sqrt(mean_squared_error(y, y_pred_adjusted))

    metrics = {
        'R^2': r2,
        'Adjusted R^2': adjusted_r2,
        'MAE': mae,
        'RMSE': rmse
    }

    logging.info("Model performance metrics calculated")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value}")

    # Optionally, save metrics to a file
    logging.info("Saving performance metrics to file")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('model_performance_metrics.csv', index=False)
    logging.info("Performance metrics saved successfully")

if __name__ == '__main__':
    evaluate_model()
