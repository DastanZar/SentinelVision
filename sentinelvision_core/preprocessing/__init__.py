from typing import Optional, List, Dict, Any
import numpy as np


class DataProcessor:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> None:
        if self.normalize:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.normalize and self.mean is not None and self.std is not None:
            return (X - self.mean) / (self.std + 1e-8)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.normalize and self.mean is not None and self.std is not None:
            return X * self.std + self.mean
        return X
