"""Regression metrics for evaluating model performance on continuous prediction tasks."""

from typing import Any, Literal, Optional

import torch

from rankme.base import StatelessMetric, check_same_shape


class MSE(StatelessMetric):
    """Compute Mean Squared Error for regression tasks.
    
    MSE = mean((y_true - y_pred)^2)
    
    Args:
        reduction: How to reduce the output ('mean', 'sum', 'none').
        squared: If True, returns MSE. If False, returns RMSE.
        
    Example:
        >>> import torch
        >>> from rankme.regression import MSE
        >>> 
        >>> y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = torch.tensor([1.1, 1.9, 3.2, 3.8])
        >>> mse = MSE()
        >>> mse_value = mse(y_pred, y_true)
        >>> float(mse_value)
        0.025
    """
    
    def __init__(
        self,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        squared: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
        self.squared = squared
        
    def forward(
        self, 
        preds: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE/RMSE.
        
        Args:
            preds: Predicted values tensor.
            target: Ground truth values tensor.
            
        Returns:
            torch.Tensor: MSE or RMSE value(s).
        """
        check_same_shape(preds, target)
        
        squared_errors = (preds - target) ** 2
        
        if self.reduction == 'mean':
            mse = squared_errors.mean()
        elif self.reduction == 'sum':
            mse = squared_errors.sum()
        else:  # 'none'
            mse = squared_errors
            
        if self.squared:
            return mse
        else:
            return torch.sqrt(mse)


class RMSE(MSE):
    """Compute Root Mean Squared Error for regression tasks.
    
    RMSE = sqrt(MSE) = sqrt(mean((y_true - y_pred)^2))
    
    This is a convenience class that inherits from MSE with squared=False.
    
    Args:
        reduction: How to reduce the output ('mean', 'sum', 'none').
        
    Example:
        >>> import torch
        >>> from rankme.regression import RMSE
        >>> 
        >>> y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = torch.tensor([1.1, 1.9, 3.2, 3.8])
        >>> rmse = RMSE()
        >>> rmse_value = rmse(y_pred, y_true)
        >>> float(rmse_value)
        0.158
    """
    
    def __init__(
        self,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        **kwargs: Any,
    ) -> None:
        super().__init__(reduction=reduction, squared=False, **kwargs)


class MAE(StatelessMetric):
    """Compute Mean Absolute Error for regression tasks.
    
    MAE = mean(|y_true - y_pred|)
    
    Args:
        reduction: How to reduce the output ('mean', 'sum', 'none').
        
    Example:
        >>> import torch
        >>> from rankme.regression import MAE
        >>> 
        >>> y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = torch.tensor([1.1, 1.9, 3.2, 3.8])
        >>> mae = MAE()
        >>> mae_value = mae(y_pred, y_true)
        >>> float(mae_value)
        0.15
    """
    
    def __init__(
        self,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
        
    def forward(
        self, 
        preds: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute MAE.
        
        Args:
            preds: Predicted values tensor.
            target: Ground truth values tensor.
            
        Returns:
            torch.Tensor: MAE value(s).
        """
        check_same_shape(preds, target)
        
        absolute_errors = torch.abs(preds - target)
        
        if self.reduction == 'mean':
            return absolute_errors.mean()
        elif self.reduction == 'sum':
            return absolute_errors.sum()
        else:  # 'none'
            return absolute_errors


class R2Score(StatelessMetric):
    """Compute R² (coefficient of determination) for regression tasks.
    
    R² = 1 - SS_res / SS_tot
    where:
    - SS_res = sum((y_true - y_pred)^2)  (residual sum of squares)
    - SS_tot = sum((y_true - mean(y_true))^2)  (total sum of squares)
    
    R² represents the proportion of variance in the dependent variable that
    is predictable from the independent variables.
    
    Args:
        multioutput: How to handle multioutput regression ('uniform_average', 'raw_values').
        adjusted: If True, compute adjusted R² accounting for number of features.
        num_features: Number of features (required if adjusted=True).
        
    Example:
        >>> import torch
        >>> from rankme.regression import R2Score
        >>> 
        >>> y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = torch.tensor([1.1, 1.9, 3.2, 3.8])
        >>> r2 = R2Score()
        >>> r2_value = r2(y_pred, y_true)
        >>> float(r2_value)
        0.98
    """
    
    def __init__(
        self,
        multioutput: Literal['uniform_average', 'raw_values'] = 'uniform_average',
        adjusted: bool = False,
        num_features: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.multioutput = multioutput
        self.adjusted = adjusted
        self.num_features = num_features
        
        if adjusted and num_features is None:
            raise ValueError('num_features must be specified when adjusted=True')
        
    def forward(
        self, 
        preds: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute R² score.
        
        Args:
            preds: Predicted values tensor.
            target: Ground truth values tensor.
            
        Returns:
            torch.Tensor: R² score value(s).
        """
        check_same_shape(preds, target)
        
        # Handle multioutput case
        if target.dim() > 1:
            return self._compute_multioutput_r2(preds, target)
        
        # Compute R² for single output
        ss_res = ((target - preds) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        
        # Handle edge case where all targets are the same
        if ss_tot == 0:
            return torch.tensor(0.0, device=target.device, dtype=target.dtype)
            
        r2 = 1 - (ss_res / ss_tot)
        
        if self.adjusted:
            assert self.num_features is not None
            n = target.size(0)
            p = self.num_features
            r2_adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            return r2_adjusted
            
        return r2
        
    def _compute_multioutput_r2(self, preds: torch.Tensor, target: torch.Tensor):
        """Compute R² for multioutput regression."""
        # Compute R² for each output dimension
        ss_res = ((target - preds) ** 2).sum(dim=0)
        ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
        
        # Handle edge cases where all targets in a dimension are the same
        r2_per_output = torch.where(
            ss_tot == 0,
            torch.tensor(0.0, device=target.device, dtype=target.dtype),
            1 - (ss_res / ss_tot)
        )
        
        if self.adjusted:
            assert self.num_features is not None
            n = target.size(0)
            p = self.num_features
            r2_per_output = 1 - (1 - r2_per_output) * (n - 1) / (n - p - 1)
        
        if self.multioutput == 'uniform_average':
            return r2_per_output.mean()
        else:  # 'raw_values'
            return r2_per_output


class MAPE(StatelessMetric):
    """Compute Mean Absolute Percentage Error for regression tasks.
    
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    
    Note: MAPE is undefined when y_true contains zeros.
    
    Args:
        reduction: How to reduce the output ('mean', 'sum', 'none').
        epsilon: Small value to add to denominator to avoid division by zero.
        
    Example:
        >>> import torch
        >>> from rankme.regression import MAPE
        >>> 
        >>> y_true = torch.tensor([100.0, 200.0, 300.0])
        >>> y_pred = torch.tensor([90.0, 210.0, 290.0])
        >>> mape = MAPE()
        >>> mape_value = mape(y_pred, y_true)
        >>> float(mape_value)
        6.67  # percent
    """
    
    def __init__(
        self,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        epsilon: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
        self.epsilon = epsilon
        
    def forward(
        self, 
        preds: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute MAPE.
        
        Args:
            preds: Predicted values tensor.
            target: Ground truth values tensor.
            
        Returns:
            torch.Tensor: MAPE value(s) in percentage.
        """
        check_same_shape(preds, target)
        
        # Compute absolute percentage errors
        ape = torch.abs((target - preds) / (torch.abs(target) + self.epsilon)) * 100
        
        if self.reduction == 'mean':
            return ape.mean()
        elif self.reduction == 'sum':
            return ape.sum()
        else:  # 'none'
            return ape


class SMAPE(StatelessMetric):
    """Compute Symmetric Mean Absolute Percentage Error for regression tasks.
    
    SMAPE = mean(|y_true - y_pred| / (|y_true| + |y_pred|)) * 100
    
    SMAPE is a variation of MAPE that is symmetric and bounded between 0 and 100%.
    
    Args:
        reduction: How to reduce the output ('mean', 'sum', 'none').
        epsilon: Small value to add to denominator to avoid division by zero.
        
    Example:
        >>> import torch
        >>> from rankme.regression import SMAPE
        >>> 
        >>> y_true = torch.tensor([100.0, 200.0, 300.0])
        >>> y_pred = torch.tensor([90.0, 210.0, 290.0])
        >>> smape = SMAPE()
        >>> smape_value = smape(y_pred, y_true)
        >>> float(smape_value)
        3.39  # percent
    """
    
    def __init__(
        self,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        epsilon: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
        self.epsilon = epsilon
        
    def forward(
        self, 
        preds: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute SMAPE.
        
        Args:
            preds: Predicted values tensor.
            target: Ground truth values tensor.
            
        Returns:
            torch.Tensor: SMAPE value(s) in percentage.
        """
        check_same_shape(preds, target)
        
        # Compute symmetric absolute percentage errors
        numerator = torch.abs(target - preds)
        denominator = torch.abs(target) + torch.abs(preds) + self.epsilon
        smape = (numerator / denominator) * 100
        
        if self.reduction == 'mean':
            return smape.mean()
        elif self.reduction == 'sum':
            return smape.sum()
        else:  # 'none'
            return smape


class HuberLoss(StatelessMetric):
    """Compute Huber Loss for regression tasks.
    
    Huber Loss is less sensitive to outliers than MSE. It is quadratic for small
    errors and linear for large errors.
    
    HuberLoss = {
        0.5 * (y_true - y_pred)^2  if |y_true - y_pred| <= delta
        delta * (|y_true - y_pred| - 0.5 * delta)  otherwise
    }
    
    Args:
        delta: Threshold for switching between quadratic and linear loss.
        reduction: How to reduce the output ('mean', 'sum', 'none').
        
    Example:
        >>> import torch
        >>> from rankme.regression import HuberLoss
        >>> 
        >>> y_true = torch.tensor([1.0, 2.0, 3.0, 10.0])  # outlier at index 3
        >>> y_pred = torch.tensor([1.1, 1.9, 3.2, 8.0])
        >>> huber = HuberLoss(delta=1.0)
        >>> huber_value = huber(y_pred, y_true)
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.delta = delta
        self.reduction = reduction
        
    def forward(
        self, 
        preds: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Huber Loss.
        
        Args:
            preds: Predicted values tensor.
            target: Ground truth values tensor.
            
        Returns:
            torch.Tensor: Huber loss value(s).
        """
        check_same_shape(preds, target)
        
        residual = torch.abs(target - preds)
        
        # Quadratic loss for small errors, linear for large errors
        loss = torch.where(
            residual <= self.delta,
            0.5 * residual ** 2,
            self.delta * (residual - 0.5 * self.delta)
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class LogCoshLoss(StatelessMetric):
    """Compute Logarithm of Hyperbolic Cosine Loss for regression tasks.
    
    LogCoshLoss = log(cosh(y_true - y_pred))
    
    This loss function has the benefits of both MSE and MAE:
    - For small errors, it behaves like MSE (smooth gradients)
    - For large errors, it behaves like MAE (robust to outliers)
    
    Args:
        reduction: How to reduce the output ('mean', 'sum', 'none').
        
    Example:
        >>> import torch
        >>> from rankme.regression import LogCoshLoss
        >>> 
        >>> y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = torch.tensor([1.1, 1.9, 3.2, 3.8])
        >>> logcosh = LogCoshLoss()
        >>> logcosh_value = logcosh(y_pred, y_true)
    """
    
    def __init__(
        self,
        reduction: Literal['mean', 'sum', 'none'] = 'mean',
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.reduction = reduction
        
    def forward(
        self, 
        preds: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Log-Cosh Loss.
        
        Args:
            preds: Predicted values tensor.
            target: Ground truth values tensor.
            
        Returns:
            torch.Tensor: Log-cosh loss value(s).
        """
        check_same_shape(preds, target)
        
        errors = target - preds
        loss = torch.log(torch.cosh(errors))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss