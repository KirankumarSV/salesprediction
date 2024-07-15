from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression, SelectKBest
import pandas as pd

class SelectKBestFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, k='all', threshold=0.005):
        self.k = k
        self.threshold = threshold
        self.feature_scores_ = None  # Use trailing underscore to follow sklearn convention

    def fit(self, X, y):
        self.selector = SelectKBest(score_func=mutual_info_regression, k='all')
        self.selector.fit(X, y)
        
        # Get the scores and identify features with low mutual information
        scores = self.selector.scores_
        self.feature_scores_ = pd.DataFrame({'feature': X.columns, 'score': scores})
        self.columns_ = self.feature_scores_[self.feature_scores_['score'] > self.threshold]['feature'].values
        print("Select K best features done")
        return self

    def transform(self, X, y=None):
        return X[self.columns_]

    def get_feature_names_out(self, input_features=None):
        return self.columns_

    def get_feature_scores(self):
        return self.feature_scores_
