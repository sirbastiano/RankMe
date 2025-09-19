"""Test configuration and shared fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Return the device to use for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def small_embedding_2d():
    """Small 2D embedding matrix for testing."""
    torch.manual_seed(42)
    return torch.randn(100, 32)


@pytest.fixture 
def large_embedding_2d():
    """Larger 2D embedding matrix for testing."""
    torch.manual_seed(42)
    return torch.randn(1000, 256)


@pytest.fixture
def batch_embeddings_3d():
    """Batched 3D embedding matrices for testing."""
    torch.manual_seed(42)
    return torch.randn(8, 512, 128)


@pytest.fixture
def binary_classification_data():
    """Binary classification test data."""
    torch.manual_seed(42)
    y_true = torch.randint(0, 2, (100,))
    y_pred_logits = torch.randn(100)
    y_pred_probs = torch.sigmoid(y_pred_logits)
    return y_true, y_pred_logits, y_pred_probs


@pytest.fixture
def multiclass_classification_data():
    """Multiclass classification test data."""
    torch.manual_seed(42)
    num_classes = 5
    y_true = torch.randint(0, num_classes, (100,))
    y_pred_logits = torch.randn(100, num_classes)
    y_pred_classes = y_pred_logits.argmax(dim=1)
    return y_true, y_pred_logits, y_pred_classes, num_classes


@pytest.fixture
def multilabel_classification_data():
    """Multilabel classification test data."""
    torch.manual_seed(42)
    num_classes = 4
    y_true = torch.randint(0, 2, (100, num_classes)).float()
    y_pred_logits = torch.randn(100, num_classes)
    y_pred_probs = torch.sigmoid(y_pred_logits)
    return y_true, y_pred_logits, y_pred_probs, num_classes


@pytest.fixture
def regression_data():
    """Regression test data."""
    torch.manual_seed(42)
    y_true = torch.randn(100)
    y_pred = y_true + 0.1 * torch.randn(100)  # Add some noise
    return y_true, y_pred


@pytest.fixture
def multioutput_regression_data():
    """Multioutput regression test data."""
    torch.manual_seed(42)
    y_true = torch.randn(100, 3)
    y_pred = y_true + 0.1 * torch.randn(100, 3)  # Add some noise
    return y_true, y_pred


def assert_tensor_close(actual, expected, rtol=1e-4, atol=1e-6):
    """Assert that two tensors are close."""
    if isinstance(expected, (int, float)):
        expected = torch.tensor(expected, dtype=actual.dtype, device=actual.device)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def assert_in_range(tensor, min_val, max_val):
    """Assert that tensor values are in the specified range."""
    assert torch.all(tensor >= min_val), f'Some values are below {min_val}'
    assert torch.all(tensor <= max_val), f'Some values are above {max_val}'