import pandas as pd
from utils.utils import get_csv_path, save_model
from app.pipeline import create_pipeline
from utils.logger import setup_logging
import inflection
import os


def run_training():

    logger = setup_logging()
    logger.info("Starting training process")

    # Load the data
    data_path = os.path.join('input_data', 'rossmann-store-sales')
    # csv_path = get_csv_path()

    dtype_spec_train = {
        'Store': int,
        'DayOfWeek': int,
        'Sales': float,
        'Customers': int,
        'Open': int,
        'Promo': int,
        'StateHoliday': str,
        'SchoolHoliday': int,
    }

    dtype_spec_store = {
        'Store': int,
        'StoreType': str,
        'Assortment': str,
        'CompetitionDistance': float,
        'CompetitionOpenSinceMonth': float,
        'CompetitionOpenSinceYear': float,
        'Promo2': int,
        'Promo2SinceWeek': float,
        'Promo2SinceYear': float,
        'PromoInterval': str
    }
    # df = pd.read_csv(csv_path, dtype=dtype_spec, low_memory=False, parse_dates=['date'])
    # Read the datasets
    train = pd.read_csv(os.path.join(data_path, 'train.csv'), dtype=dtype_spec_train, low_memory=False, parse_dates=['Date'])
    store = pd.read_csv(os.path.join(data_path, 'store.csv'), dtype=dtype_spec_store)

    # Merge the train and store data
    train_data = pd.merge(left=train, right=store, how="left", on="Store")

    # Convert column names to snake_case
    snakecase = lambda x: inflection.underscore(x)
    new_columns = list(map(snakecase, train_data.columns))
    train_data.columns = new_columns

    logger.info("train_data created and their columns renamed to snake_case")

    X = train_data.drop(columns=['sales'])
    y = train_data['sales']

    # Create and train the pipeline
    logger.info("Creating pipeline")
    pipeline = create_pipeline()

    logger.info("Fitting pipeline")
    pipeline.fit(X, y)
    logger.info("Pipeline fitted successfully")

    # Save the trained pipeline
    logger.info("Saving the trained model")
    save_model(pipeline)
    logger.info("Model saved successfully")

if __name__ == '__main__':
    run_training()
