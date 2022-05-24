import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


logger = logging.getLogger(__name__)


class DataTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for features extraction.
    Transforms categorical features using one-hot-encoding method.
    Standardizes numeric features by removing the mean and
    scaling to unit variance.
    """

    def __init__(self, categorical_features: list, numerical_features: list):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.mean = 0
        self.std = 1

    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.enc.fit(X[self.categorical_features])
        self.mean = X[self.numerical_features].mean()
        self.std = X[self.numerical_features].std()
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
        X[num_features] = (X[num_features] - self.mean) / self.std
        return X[num_features].values
