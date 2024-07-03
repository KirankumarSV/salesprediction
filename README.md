# Sales Prediction Project

## Overview
This project aims to predict sales using a machine learning pipeline. The pipeline includes data preprocessing, feature engineering, multicollinearity reduction, and model training.

## Project Structure
# Sales Prediction Project

## Overview
This project aims to predict sales using a machine learning pipeline. The pipeline includes data preprocessing, feature engineering, multicollinearity reduction, and model training.

## Project Structure
sales_prediction/
├── app/
│ ├── init.py
│ ├── pipeline.py
│ └── streamlit_app.py
├── data/
│ ├── init.py
│ └── preprocessing.py
├── feature_engineering/
│ ├── init.py
│ └── feature_engineering.py
├── model/
│ ├── init.py
│ ├── train.py
│ ├── vif_selector.py
│ └── select_k_best.py
├── scripts/
│ ├── init.py
│ └── run_training.py
├── utils/
│ ├── init.py
│ └── utils.py
├── requirements.txt
├── setup.py
└── README.md



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

3. Run the training script:
    ```bash
    python scripts/run_training.py
    ```

4. Start the Streamlit app:
    ```bash
    streamlit run app/streamlit_app.py
    ```

## Usage
- **Single Prediction**: Enter details in the Streamlit app to predict sales for a single data point.
- **Batch Prediction**: Upload a CSV file in the Streamlit app to predict sales for multiple data points.

## License
This project is licensed under the MIT License.

