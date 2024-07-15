from sklearn.pipeline import Pipeline
from data_preprocessing.preprocessing import FillMissingValues, EncodeScaleData
from feature_engineering.feature_engineering import FeatureEngineering, CyclicalFeatures
from feature_engineering.drop_columns import DropColumns
from model.vif_selector import VIFSelector
from model.select_k_best import SelectKBestFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from feature_engineering.transformations import ApplyTransformations  # Import the new transformer
from app.custom_pipeline import CustomPipeline  # Import the CustomPipeline class
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
import numpy as np

def sqrt_transform_with_constant(y, constant=10):
    return np.sqrt(y + constant)

def inverse_sqrt_transform_with_constant(y, constant=10):
    return np.square(y) - constant

class ConvertToFloat64(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype('float64')

def create_pipeline():
    preprocessing_pipeline = CustomPipeline(steps=[
        ('fill_missing', FillMissingValues()),
        ('feature_engineering', FeatureEngineering()),
        ('cyclical_features', CyclicalFeatures(columns=['month', 'day_of_week', 'day', 'week_of_year'], max_values=[12, 7, 31, 52])),
        ('apply_transformations', ApplyTransformations()),  # Add the new transformer
        ('encode_scale', EncodeScaleData())
    ])
    
    vif_pipeline = CustomPipeline(steps=[
        ('drop_columns', DropColumns(columns=['date', 'competition_open_since_month', 'competition_open_since_year', 'promo2_since_week', 
                                              'promo2_since_year', 'promo_interval'])),
        ('convert_to_float64', ConvertToFloat64()),
        ('vif', VIFSelector(threshold=5))
    ])
    
    model_pipeline = CustomPipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('vif', vif_pipeline),
        ('select_k_best', SelectKBestFeatures(k='all', threshold=0.005)),
        ('model',  RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    
    final_pipeline = TransformedTargetRegressor(
        regressor=model_pipeline,
        func=sqrt_transform_with_constant,
        inverse_func=inverse_sqrt_transform_with_constant
    )
    
    return final_pipeline
