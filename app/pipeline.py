from sklearn.pipeline import Pipeline
from data.preprocessing import FillMissingValues, EncodeScaleData
from feature_engineering.feature_engineering import FeatureEngineering, CyclicalFeatures
from feature_engineering.drop_columns import DropColumns
from model.vif_selector import VIFSelector
from model.select_k_best import SelectKBestFeatures
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin

class ConvertToFloat64(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype('float64')

def create_pipeline():
    preprocessing_pipeline = Pipeline(steps=[
        ('fill_missing', FillMissingValues()),
        ('feature_engineering', FeatureEngineering()),
        ('cyclical_features', CyclicalFeatures(columns=['month', 'day_of_week', 'day', 'week_of_year'], max_values=[12, 7, 31, 52])),
        ('encode_scale', EncodeScaleData())
    ])
    
    vif_pipeline = Pipeline(steps=[
        ('drop_columns', DropColumns(columns=['date', 'open', 'competition_open_since_month', 'competition_open_since_year', 'promo2_since_week', 
                                              'promo2_since_year', 'promo_interval'])),
        ('convert_to_float64', ConvertToFloat64()),
        ('vif', VIFSelector(threshold=5))
    ])
    
    model_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessing_pipeline),
        ('vif', vif_pipeline),
        ('select_k_best', SelectKBestFeatures(k='all', threshold=0.005)),
        ('model', LinearRegression())
    ])
    return model_pipeline
