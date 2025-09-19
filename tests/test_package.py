"""Package-level tests."""

import pytest
import torch

import rankme
from rankme import (
    BaseMetric,
    RankMe,
    Accuracy, F1Score, IoU, Precision, Recall,
    MAE, MSE, R2Score, RMSE
)


def test_package_version():
    """Test that package has a version."""
    assert hasattr(rankme, '__version__')
    assert isinstance(rankme.__version__, str)


def test_package_imports():
    """Test that all main classes can be imported."""
    # Base classes
    assert BaseMetric is not None
    
    # Feature learning
    assert RankMe is not None
    
    # Classification
    assert Accuracy is not None
    assert F1Score is not None
    assert IoU is not None
    assert Precision is not None
    assert Recall is not None
    
    # Regression
    assert MAE is not None
    assert MSE is not None
    assert R2Score is not None
    assert RMSE is not None


def test_basic_functionality():
    """Test basic functionality of main metrics."""
    torch.manual_seed(42)
    
    # Test RankMe
    Z = torch.randn(100, 32)
    rankme = RankMe()
    rankme_score = rankme(Z)
    assert 0 <= rankme_score <= 1
    
    # Test classification metrics
    y_true = torch.randint(0, 3, (50,))
    y_pred = torch.randint(0, 3, (50,))
    
    acc = Accuracy(task='multiclass', num_classes=3)
    accuracy = acc(y_pred, y_true)
    assert 0 <= accuracy <= 1
    
    # Test regression metrics
    y_true_reg = torch.randn(50)
    y_pred_reg = y_true_reg + 0.1 * torch.randn(50)
    
    mse = MSE()
    mse_val = mse(y_pred_reg, y_true_reg)
    assert mse_val >= 0


def test_all_metrics_are_torch_modules():
    """Test that all metrics inherit from torch.nn.Module."""
    metrics = [
        RankMe(),
        Accuracy(task='binary'),
        MSE(),
        MAE(),
    ]
    
    for metric in metrics:
        assert isinstance(metric, torch.nn.Module)
        assert hasattr(metric, 'forward')
        assert callable(metric)


def test_package_structure():
    """Test that package has expected structure."""
    # Check that submodules exist
    assert hasattr(rankme, 'base')
    assert hasattr(rankme, 'feature_learning')
    assert hasattr(rankme, 'classification')
    assert hasattr(rankme, 'regression')
    
    # Check that __all__ is defined
    assert hasattr(rankme, '__all__')
    assert isinstance(rankme.__all__, list)
    assert len(rankme.__all__) > 0