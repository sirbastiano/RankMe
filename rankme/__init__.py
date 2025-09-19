"""RankMe: A comprehensive PyTorch-based metrics library for ML tasks.

This package provides metrics for:
- Feature learning (RankMe, effective rank, spectral entropy)
- Classification (accuracy, precision, recall, F1, IoU, etc.)
- Regression (MSE, MAE, R², RMSE, etc.)

All metrics are implemented as PyTorch modules for seamless integration
with training pipelines and support for batched operations.
"""

from rankme.base import BaseMetric
from rankme.feature_learning import RankMe
from rankme.classification import (
    Accuracy,
    F1Score,
    IoU,
    Precision,
    Recall,
)
from rankme.regression import (
    MAE,
    MSE,
    R2Score,
    RMSE,
)

__version__ = '0.1.0'
__author__ = 'Roberto Del Prete'
__email__ = 'roberto.delprete@example.com'

__all__ = [
    # Base
    'BaseMetric',
    # Feature Learning
    'RankMe',
    # Classification
    'Accuracy',
    'F1Score',
    'IoU',
    'Precision',
    'Recall',
    # Regression
    'MAE',
    'MSE',
    'R2Score',
    'RMSE',
]