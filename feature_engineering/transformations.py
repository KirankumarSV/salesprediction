from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ApplyTransformations(BaseEstimator, TransformerMixin):
    def __init__(self, small_constant=1):
        self.small_constant = small_constant

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        
        #apply sqrt transformation on customers    
        if 'customers' in X.columns:
            X['customers'] = np.sqrt(X['customers'])

        # Apply log1p transformation to 'competition_distance'
        if 'competition_distance' in X.columns:
            X['competition_distance'] = np.log1p(X['competition_distance'])

        # Apply log transformation with a small constant to handle zeros in 'competition_open' and 'promo2_since'
        if 'competition_open' in X.columns:
            X['competition_open'] = np.log1p(X['competition_open'] + self.small_constant)
        if 'promo2_since' in X.columns:
            X['promo2_since'] = np.log1p(X['promo2_since'] + self.small_constant)

        return X

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove any large attributes that are not necessary
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)