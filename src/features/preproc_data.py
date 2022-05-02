import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


logger = logging.getLogger(__name__)


class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features: list, numerical_features: list):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.enc.fit(X[self.categorical_features])
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_ = X.copy()
        X_cat = self.transform_cat_features(X_)
        X_num = self.transform_num_features(X_)
        return np.concatenate((X_cat, X_num), axis=1)

    def transform_cat_features(self, X: pd.DataFrame) -> np.ndarray:
        return self.enc.transform(X[self.categorical_features])

    def transform_num_features(self, X: pd.DataFrame) -> np.ndarray:
        num_features = self.numerical_features
        X[num_features] = (X[num_features] - X[num_features].mean()) / X[
            num_features
        ].std()
        return X[num_features].values
