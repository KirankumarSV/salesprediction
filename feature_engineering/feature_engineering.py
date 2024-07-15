from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                  7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        X['is_promo'] = X.apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if months[x['date'].month] in x['promo_interval'].split(",") else 0, axis=1)
        X['day'] = X['date'].dt.day
        X['month'] = X['date'].dt.month  # Ensure month column is created
        X['year'] = X['date'].dt.year
        X['week_of_year'] = X['date'].dt.isocalendar().week
        X['competition_open'] = (X['year'] - X['competition_open_since_year']) * 12 - X['competition_open_since_month'] + X['month']
        X['competition_open'] = X['competition_open'].apply(lambda x: 0 if x < 0 else x)
        promo2_conditional_difference = np.where(X['promo2_since_year'] != 0, X['year'] - X['promo2_since_year'], 0)
        X['promo2_since'] = (promo2_conditional_difference) * 52 + X['week_of_year'] - X['promo2_since_week']
        X['promo2_since'] = X['promo2_since'].apply(lambda x: max(x, 0))
        X['state_holiday'] = X['state_holiday'].astype(str)
        X['store_type'] = X['store_type'].astype(str)
        X['assortment'] = X['assortment'].astype(str)

        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns_.tolist()


class CyclicalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns, max_values):
        self.columns = columns
        self.max_values = max_values

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for column, max_value in zip(self.columns, self.max_values):
            X[column + '_sin'] = np.sin(2 * np.pi * X[column] / max_value)
            X[column + '_cos'] = np.cos(2 * np.pi * X[column] / max_value)
        return X.drop(columns=self.columns)
