import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class FillMissingValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.median_competition_distance = X['competition_distance'].median()
        self.mode_competition_open_since_year = X['competition_open_since_year'].mode()[0]
        self.mode_competition_open_since_month = X['competition_open_since_month'].mode()[0]
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['competition_distance'].fillna(self.median_competition_distance, inplace=True)
        X['competition_open_since_year'].fillna(self.mode_competition_open_since_year, inplace=True)
        X['competition_open_since_month'].fillna(self.mode_competition_open_since_month, inplace=True)
        X['promo2_since_week'].fillna(0, inplace=True)
        X['promo2_since_year'].fillna(0, inplace=True)
        X['promo_interval'].fillna(0, inplace=True)
        return X

class EncodeScaleData(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.assortment_dict = {'a': 1, 'b': 2, 'c': 3}
        self.minmax_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.column_transformer = None

    def fit(self, X, y=None):
        X = X.copy()
        # Fit OneHotEncoder and LabelEncoder
        self.ohe.fit(X[['state_holiday']])
        self.label_encoder.fit(X['store_type'])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Apply OneHotEncoder to state_holiday
        state_holiday_encoded = self.ohe.transform(X[['state_holiday']])
        state_holiday_encoded_df = pd.DataFrame(state_holiday_encoded, columns=self.ohe.get_feature_names_out(['state_holiday']))

        # Apply LabelEncoder to store_type
        X['store_type'] = self.label_encoder.transform(X['store_type'])
        X['assortment'] = X['assortment'].map(self.assortment_dict)

        # Scaling numerical features
        X[['year', 'promo2_since']] = self.minmax_scaler.fit_transform(X[['year', 'promo2_since']])
        X[['competition_distance', 'competition_open']] = self.minmax_scaler.fit_transform(X[['competition_distance', 'competition_open']])
        X[['customers']] = self.standard_scaler.fit_transform(X[['customers']])

        # Drop original state_holiday column and concatenate encoded dataframe
        X = X.drop(columns=['state_holiday'])
        X = pd.concat([X.reset_index(drop=True), state_holiday_encoded_df.reset_index(drop=True)], axis=1)

        return X
