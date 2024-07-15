# Sales Prediction Project

## Overview
This project aims to predict sales using a machine learning pipeline. The pipeline includes data preprocessing, feature engineering, multicollinearity reduction, and model training.


## Data
- **Source**: The dataset used for this project comes from [Kaggle's Sales Dataset](https://www.kaggle.com/c/rossmann-store-sales/data).
- **Description**: The dataset consists of sales data collected from various stores over a period of time. It includes features such as date, store number, sales, and other related attributes.
- **Columns**:
    - `date`: The date of the sales record.
    - `store_number`: The unique identifier for the store.
    - `sales`: The sales amount for that particular date and store.
    - `customers`: The number of customers for that particular date and store.
    - `open`: Indicates whether the store was open on that date (1 = open, 0 = closed).
    - `promo`: Indicates whether a promo was active on that date (1 = active, 0 = not active).
    - `state_holiday`: Indicates whether the date is a state holiday.
    - `school_holiday`: Indicates whether the date is a school holiday.
The rest of the feature names are self explantory. For further details, check the Kaggle website.

## Project Structure

```plaintext
sales_prediction/
├── app/
│ ├── init.py
| ├── custom_pipeline.py
│ ├── pipeline.py
│ └── streamlit_app.py
├── data_preprocessing/
│ ├── init.py
│ └── preprocessing.py
├── feature_engineering/
│ ├── init.py
| ├── drop_columns.py
| ├── transformations.py
│ └── feature_engineering.py
├── input_data/
| └── rossmann-store-sales/  # The dataset are downloaded here
├── model/
│ ├── init.py
│ ├── vif_selector.py
│ └── select_k_best.py
├── scripts/
│ ├── init.py
| ├── download_data.py
| ├── evaluate_model.py
│ └── run_training.py
├── utils/
│ ├── init.py
| ├── logger.py
│ └── utils.py
├── requirements.txt
├── setup.py
└── README.md
```


## Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/sales_prediction.git
    cd sales_prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the download_data script to download the dataset using Kaggle API. Ensure to have kaggle.json file at ~/.kaggle/kaggle.json:
    ```bash
    python scripts/download_data.py
    ```

4. Run the training script:
    ```bash
    python scripts/run_training.py
    ```

5. Start the Streamlit app:
    ```bash
    streamlit run app/streamlit_app.py
    ```

## Usage
- **Single Prediction**: Enter details in the Streamlit app to predict sales for a single data point.

## License
This project is licensed under the MIT License.
