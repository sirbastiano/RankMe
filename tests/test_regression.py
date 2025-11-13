"""Tests for regression metrics."""

import numpy as np
import pytest
import torch

from rankme.regression import (
    MAE,
    MAPE,
    MSE,
    RMSE,
    SMAPE,
    HuberLoss,
    LogCoshLoss,
    R2Score,
)
from tests.conftest import assert_in_range, assert_tensor_close


class TestMSE:
    """Test cases for MSE metric."""

    def test_mse_basic(self, regression_data):
        """Test basic MSE computation."""
        y_true, y_pred = regression_data

        mse = MSE()
        mse_value = mse(y_pred, y_true)

        # MSE should be non-negative
        assert mse_value >= 0
        assert torch.isfinite(mse_value)

    def test_mse_perfect_prediction(self):
        """Test MSE with perfect predictions."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.clone()

        mse = MSE()
        mse_value = mse(y_pred, y_true)

        assert_tensor_close(mse_value, 0.0, atol=1e-6)

    def test_mse_reduction_modes(self, regression_data):
        """Test different reduction modes for MSE."""
        y_true, y_pred = regression_data

        mse_mean = MSE(reduction="mean")
        mse_sum = MSE(reduction="sum")
        mse_none = MSE(reduction="none")

        mse_mean_val = mse_mean(y_pred, y_true)
        mse_sum_val = mse_sum(y_pred, y_true)
        mse_none_val = mse_none(y_pred, y_true)

        # Check relationships
        assert_tensor_close(mse_sum_val, mse_mean_val * len(y_true))
        assert mse_none_val.shape == y_true.shape
        assert_tensor_close(mse_none_val.mean(), mse_mean_val)

    def test_mse_vs_rmse(self, regression_data):
        """Test relationship between MSE and RMSE."""
        y_true, y_pred = regression_data

        mse = MSE(squared=True)
        rmse_from_mse = MSE(squared=False)
        rmse = RMSE()

        mse_val = mse(y_pred, y_true)
        rmse_val1 = rmse_from_mse(y_pred, y_true)
        rmse_val2 = rmse(y_pred, y_true)

        # RMSE should be sqrt of MSE
        assert_tensor_close(rmse_val1, torch.sqrt(mse_val))
        assert_tensor_close(rmse_val2, torch.sqrt(mse_val))
        assert_tensor_close(rmse_val1, rmse_val2)


class TestMAE:
    """Test cases for MAE metric."""

    def test_mae_basic(self, regression_data):
        """Test basic MAE computation."""
        y_true, y_pred = regression_data

        mae = MAE()
        mae_value = mae(y_pred, y_true)

        # MAE should be non-negative
        assert mae_value >= 0
        assert torch.isfinite(mae_value)

    def test_mae_perfect_prediction(self):
        """Test MAE with perfect predictions."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.clone()

        mae = MAE()
        mae_value = mae(y_pred, y_true)

        assert_tensor_close(mae_value, 0.0, atol=1e-6)

    def test_mae_known_values(self):
        """Test MAE with known values."""
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_pred = torch.tensor([1.5, 1.5, 3.5])

        mae = MAE()
        mae_value = mae(y_pred, y_true)

        # Expected MAE = (0.5 + 0.5 + 0.5) / 3 = 0.5
        assert_tensor_close(mae_value, 0.5)

    def test_mae_reduction_modes(self, regression_data):
        """Test different reduction modes for MAE."""
        y_true, y_pred = regression_data

        mae_mean = MAE(reduction="mean")
        mae_sum = MAE(reduction="sum")
        mae_none = MAE(reduction="none")

        mae_mean_val = mae_mean(y_pred, y_true)
        mae_sum_val = mae_sum(y_pred, y_true)
        mae_none_val = mae_none(y_pred, y_true)

        # Check relationships
        assert_tensor_close(mae_sum_val, mae_mean_val * len(y_true))
        assert mae_none_val.shape == y_true.shape
        assert_tensor_close(mae_none_val.mean(), mae_mean_val)


class TestR2Score:
    """Test cases for R2Score metric."""

    def test_r2_basic(self, regression_data):
        """Test basic R² computation."""
        y_true, y_pred = regression_data

        r2 = R2Score()
        r2_value = r2(y_pred, y_true)

        # R² can be negative for very poor predictions
        assert torch.isfinite(r2_value)

    def test_r2_perfect_prediction(self):
        """Test R² with perfect predictions."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.clone()

        r2 = R2Score()
        r2_value = r2(y_pred, y_true)

        assert_tensor_close(r2_value, 1.0)

    def test_r2_mean_prediction(self):
        """Test R² when predictions equal the mean."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = torch.full_like(y_true, y_true.mean())

        r2 = R2Score()
        r2_value = r2(y_pred, y_true)

        # R² should be 0 when predicting the mean
        assert_tensor_close(r2_value, 0.0, atol=1e-6)

    def test_r2_multioutput(self, multioutput_regression_data):
        """Test R² with multioutput regression."""
        y_true, y_pred = multioutput_regression_data

        r2_avg = R2Score(multioutput="uniform_average")
        r2_raw = R2Score(multioutput="raw_values")

        r2_avg_val = r2_avg(y_pred, y_true)
        r2_raw_val = r2_raw(y_pred, y_true)

        # Average should be scalar, raw should have shape (num_outputs,)
        assert r2_avg_val.dim() == 0
        assert r2_raw_val.shape == (y_true.size(1),)
        assert_tensor_close(r2_avg_val, r2_raw_val.mean())

    def test_r2_adjusted(self, regression_data):
        """Test adjusted R²."""
        y_true, y_pred = regression_data

        r2_normal = R2Score(adjusted=False)
        r2_adjusted = R2Score(adjusted=True, num_features=5)

        r2_normal_val = r2_normal(y_pred, y_true)
        r2_adjusted_val = r2_adjusted(y_pred, y_true)

        # Adjusted R² should be lower than normal R² (penalizes complexity)
        assert r2_adjusted_val <= r2_normal_val

    def test_r2_constant_target(self):
        """Test R² with constant target values."""
        y_true = torch.ones(10)  # All targets are the same
        y_pred = torch.randn(10)

        r2 = R2Score()
        r2_value = r2(y_pred, y_true)

        # Should return 0 when all targets are the same
        assert_tensor_close(r2_value, 0.0)


class TestMAPE:
    """Test cases for MAPE metric."""

    def test_mape_basic(self):
        """Test basic MAPE computation."""
        y_true = torch.tensor([100.0, 200.0, 300.0])
        y_pred = torch.tensor([90.0, 210.0, 290.0])

        mape = MAPE()
        mape_value = mape(y_pred, y_true)

        # Expected MAPE = (10/100 + 10/200 + 10/300) / 3 * 100 = (0.1 + 0.05 + 0.033) / 3 * 100
        expected = ((10 / 100 + 10 / 200 + 10 / 300) / 3) * 100
        assert_tensor_close(mape_value, expected, rtol=1e-3)

    def test_mape_perfect_prediction(self):
        """Test MAPE with perfect predictions."""
        y_true = torch.tensor([100.0, 200.0, 300.0])
        y_pred = y_true.clone()

        mape = MAPE()
        mape_value = mape(y_pred, y_true)

        assert_tensor_close(mape_value, 0.0, atol=1e-6)

    def test_mape_with_zeros(self):
        """Test MAPE with zero values in target."""
        y_true = torch.tensor([0.0, 100.0, 200.0])
        y_pred = torch.tensor([1.0, 90.0, 210.0])

        mape = MAPE(epsilon=1e-8)
        mape_value = mape(y_pred, y_true)

        # Should handle zeros with epsilon
        assert torch.isfinite(mape_value)


class TestSMAPE:
    """Test cases for SMAPE metric."""

    def test_smape_basic(self):
        """Test basic SMAPE computation."""
        y_true = torch.tensor([100.0, 200.0, 300.0])
        y_pred = torch.tensor([90.0, 210.0, 290.0])

        smape = SMAPE()
        smape_value = smape(y_pred, y_true)

        # SMAPE should be between 0 and 100
        assert_in_range(smape_value, 0.0, 100.0)

    def test_smape_perfect_prediction(self):
        """Test SMAPE with perfect predictions."""
        y_true = torch.tensor([100.0, 200.0, 300.0])
        y_pred = y_true.clone()

        smape = SMAPE()
        smape_value = smape(y_pred, y_true)

        assert_tensor_close(smape_value, 0.0, atol=1e-6)

    def test_smape_symmetry(self):
        """Test SMAPE symmetry property."""
        y_true = torch.tensor([100.0, 200.0])
        y_pred = torch.tensor([80.0, 250.0])

        smape = SMAPE()
        smape_value1 = smape(y_pred, y_true)
        smape_value2 = smape(y_true, y_pred)  # Swap true and pred

        # SMAPE should be symmetric
        assert_tensor_close(smape_value1, smape_value2)


class TestHuberLoss:
    """Test cases for HuberLoss metric."""

    def test_huber_basic(self, regression_data):
        """Test basic Huber loss computation."""
        y_true, y_pred = regression_data

        huber = HuberLoss(delta=1.0)
        huber_value = huber(y_pred, y_true)

        # Huber loss should be non-negative
        assert huber_value >= 0
        assert torch.isfinite(huber_value)

    def test_huber_perfect_prediction(self):
        """Test Huber loss with perfect predictions."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.clone()

        huber = HuberLoss(delta=1.0)
        huber_value = huber(y_pred, y_true)

        assert_tensor_close(huber_value, 0.0, atol=1e-6)

    def test_huber_vs_mse_mae(self):
        """Test Huber loss relationship with MSE and MAE."""
        y_true = torch.tensor([0.0, 0.0, 0.0])

        # Small errors (should behave like MSE)
        y_pred_small = torch.tensor([0.1, 0.2, 0.3])

        # Large errors (should behave like MAE)
        y_pred_large = torch.tensor([2.0, 3.0, 4.0])

        huber = HuberLoss(delta=1.0)
        mse = MSE()
        mae = MAE()

        huber_small = huber(y_pred_small, y_true)
        mse_small = mse(y_pred_small, y_true)

        huber_large = huber(y_pred_large, y_true)
        mae_large = mae(y_pred_large, y_true)

        # For small errors, Huber ≈ 0.5 * MSE
        assert_tensor_close(huber_small, 0.5 * mse_small, rtol=1e-3)

        # For large errors, Huber should be closer to linear behavior


class TestLogCoshLoss:
    """Test cases for LogCoshLoss metric."""

    def test_logcosh_basic(self, regression_data):
        """Test basic Log-Cosh loss computation."""
        y_true, y_pred = regression_data

        logcosh = LogCoshLoss()
        logcosh_value = logcosh(y_pred, y_true)

        # Log-Cosh loss should be non-negative
        assert logcosh_value >= 0
        assert torch.isfinite(logcosh_value)

    def test_logcosh_perfect_prediction(self):
        """Test Log-Cosh loss with perfect predictions."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.clone()

        logcosh = LogCoshLoss()
        logcosh_value = logcosh(y_pred, y_true)

        # log(cosh(0)) = 0
        assert_tensor_close(logcosh_value, 0.0, atol=1e-6)

    def test_logcosh_reduction_modes(self, regression_data):
        """Test different reduction modes for Log-Cosh loss."""
        y_true, y_pred = regression_data

        logcosh_mean = LogCoshLoss(reduction="mean")
        logcosh_sum = LogCoshLoss(reduction="sum")
        logcosh_none = LogCoshLoss(reduction="none")

        logcosh_mean_val = logcosh_mean(y_pred, y_true)
        logcosh_sum_val = logcosh_sum(y_pred, y_true)
        logcosh_none_val = logcosh_none(y_pred, y_true)

        # Check relationships
        assert_tensor_close(logcosh_sum_val, logcosh_mean_val * len(y_true))
        assert logcosh_none_val.shape == y_true.shape
        assert_tensor_close(logcosh_none_val.mean(), logcosh_mean_val)


class TestRegressionIntegration:
    """Integration tests for regression metrics."""

    def test_metric_consistency(self, regression_data):
        """Test basic consistency across regression metrics."""
        y_true, y_pred = regression_data

        mse = MSE()
        mae = MAE()
        r2 = R2Score()

        mse_val = mse(y_pred, y_true)
        mae_val = mae(y_pred, y_true)
        r2_val = r2(y_pred, y_true)

        # All should be finite
        assert torch.isfinite(mse_val)
        assert torch.isfinite(mae_val)
        assert torch.isfinite(r2_val)

        # MSE should be >= 0, MAE should be >= 0
        assert mse_val >= 0
        assert mae_val >= 0

    def test_outlier_robustness(self):
        """Test robustness to outliers."""
        # Data with outlier
        y_true = torch.tensor([1.0, 2.0, 3.0, 100.0])  # 100.0 is outlier
        y_pred = torch.tensor([1.1, 1.9, 3.1, 4.0])  # Poor prediction for outlier

        mse = MSE()
        mae = MAE()
        huber = HuberLoss(delta=1.0)

        mse_val = mse(y_pred, y_true)
        mae_val = mae(y_pred, y_true)
        huber_val = huber(y_pred, y_true)

        # MSE should be heavily affected by outlier
        # MAE should be less affected
        # Huber should be between them
        assert mse_val > mae_val  # MSE more sensitive to outliers
        assert huber_val < mse_val  # Huber more robust than MSE

    def test_device_consistency(self, regression_data, device):
        """Test device consistency for regression metrics."""
        y_true, y_pred = regression_data

        mse = MSE()

        # CPU computation
        mse_cpu = mse(y_pred, y_true)

        # GPU computation (if available)
        if device.type == "cuda":
            y_true_gpu = y_true.to(device)
            y_pred_gpu = y_pred.to(device)
            mse_gpu = mse.to(device)

            mse_gpu_val = mse_gpu(y_pred_gpu, y_true_gpu)

            # Should be identical
            assert_tensor_close(mse_cpu, mse_gpu_val.cpu())

    def test_shape_consistency(self):
        """Test that metrics handle different input shapes correctly."""
        # 1D case
        y_true_1d = torch.randn(10)
        y_pred_1d = torch.randn(10)

        # 2D case (multioutput)
        y_true_2d = torch.randn(10, 3)
        y_pred_2d = torch.randn(10, 3)

        mse = MSE()

        mse_1d = mse(y_pred_1d, y_true_1d)
        mse_2d = mse(y_pred_2d, y_true_2d)

        # Both should be scalars for mean reduction
        assert mse_1d.dim() == 0
        assert mse_2d.dim() == 0

    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        # Very small values
        y_true_small = torch.tensor([1e-8, 2e-8, 3e-8])
        y_pred_small = torch.tensor([1.1e-8, 1.9e-8, 3.1e-8])

        # Very large values
        y_true_large = torch.tensor([1e8, 2e8, 3e8])
        y_pred_large = torch.tensor([1.1e8, 1.9e8, 3.1e8])

        mse = MSE()
        mae = MAE()

        # Should handle both small and large values
        mse_small = mse(y_pred_small, y_true_small)
        mse_large = mse(y_pred_large, y_true_large)
        mae_small = mae(y_pred_small, y_true_small)
        mae_large = mae(y_pred_large, y_true_large)

        assert torch.isfinite(mse_small)
        assert torch.isfinite(mse_large)
        assert torch.isfinite(mae_small)
        assert torch.isfinite(mae_large)
