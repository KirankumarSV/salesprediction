from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np

class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=5):
        self.threshold = threshold
        self.columns_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        
        print("Initial number of columns:", X.shape[1])  # Debug print
        
        high_vif = True
        while high_vif:
            vif_values = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            max_vif = max(vif_values)
            print("Current max VIF:", max_vif)  # Debug print
            if max_vif > self.threshold:
                max_vif_index = vif_values.index(max_vif)
                column_to_drop = X.columns[max_vif_index]
                print("Dropping column:", column_to_drop)  # Debug print
                X = X.drop(columns=[column_to_drop])
            else:
                high_vif = False
        self.columns_ = X.columns
        print("Final number of columns:", X.shape[1])  # Debug print
        return X

    def get_feature_names_out(self, input_features=None):
        return self.columns_.tolist()
