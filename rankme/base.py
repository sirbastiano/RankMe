"""Base classes and utilities for all metrics."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn


class BaseMetric(nn.Module, ABC):
    """Base class for all metrics in the RankMe library.
    
    This class provides a common interface for all metrics, ensuring consistency
    across different metric types (classification, regression, feature learning).
    
    All metrics inherit from PyTorch's nn.Module, allowing them to be used
    seamlessly in training pipelines and supporting features like device movement,
    gradient computation control, and state management.
    
    Args:
        compute_on_step: If True, computes and returns metric on every call.
                        If False, accumulates inputs and computes on explicit call.
        dist_sync_on_step: If True, synchronizes metric state across processes
                          on every step (for distributed training).
        process_group: Process group for distributed synchronization.
        dist_sync_fn: Function to use for distributed synchronization.
    """
    
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Any] = None,
    ) -> None:
        super().__init__()
        
        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.process_group = process_group
        self.dist_sync_fn = dist_sync_fn
        
        self._update_signature = self._get_update_signature()
        self._computed: Optional[torch.Tensor] = None
        
    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the metric state with new data.
        
        This method should be implemented by subclasses to accumulate
        metric state across multiple batches.
        """
        pass
    
    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Compute the metric value from accumulated state.
        
        Returns:
            torch.Tensor: The computed metric value.
        """
        pass
    
    def reset(self) -> None:
        """Reset the metric state.
        
        This method should be called at the beginning of each epoch
        for stateful metrics that accumulate across batches.
        """
        self._computed = None
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass of the metric.
        
        Args:
            *args: Positional arguments for metric computation.
            **kwargs: Keyword arguments for metric computation.
            
        Returns:
            torch.Tensor: The computed metric value.
        """
        # Update state if needed
        if not self.compute_on_step:
            self.update(*args, **kwargs)
            
        # Compute metric
        if self.compute_on_step:
            self.update(*args, **kwargs)
            self._computed = self.compute()
        elif self._computed is None:
            self._computed = self.compute()
            
        return self._computed
    
    def _get_update_signature(self) -> Any:
        """Get the signature of the update method."""
        import inspect
        return inspect.signature(self.update)
    
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Make the metric callable."""
        return self.forward(*args, **kwargs)
    
    def __repr__(self) -> str:
        """String representation of the metric."""
        class_name = self.__class__.__name__
        return f'{class_name}()'


class StatefulMetric(BaseMetric):
    """Base class for metrics that accumulate state across multiple updates.
    
    This class is suitable for metrics that need to accumulate predictions
    and targets across multiple batches before computing the final result.
    Examples include metrics that require the full dataset to compute
    properly (like some ranking metrics) or metrics where averaging
    across batches is not equivalent to computing on the full dataset.
    """
    
    def __init__(
        self,
        compute_on_step: bool = False,  # Usually False for stateful metrics
        **kwargs: Any,
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        
    def reset(self) -> None:
        """Reset accumulated state."""
        super().reset()
        # Subclasses should override this to reset their specific state


class StatelessMetric(BaseMetric):
    """Base class for metrics that can be computed directly without state.
    
    This class is suitable for metrics that can be computed directly
    from predictions and targets without needing to accumulate state.
    Examples include MSE, accuracy, F1-score for single batches.
    """
    
    def __init__(
        self,
        compute_on_step: bool = True,  # Usually True for stateless metrics
        **kwargs: Any,
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Stateless metrics don't accumulate state."""
        pass
    
    def compute(self) -> torch.Tensor:
        """Compute metric directly from last inputs."""
        raise NotImplementedError(
            'Stateless metrics should override forward() directly'
        )


def check_same_shape(tensor1: torch.Tensor, tensor2: torch.Tensor) -> None:
    """Check that two tensors have the same shape.
    
    Args:
        tensor1: First tensor.
        tensor2: Second tensor.
        
    Raises:
        ValueError: If tensors have different shapes.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f'Tensors must have the same shape, got {tensor1.shape} '
            f'and {tensor2.shape}'
        )


def check_tensor_dtype(
    tensor: torch.Tensor, 
    expected_dtype: Union[torch.dtype, tuple[torch.dtype, ...]]
) -> None:
    """Check that tensor has expected dtype.
    
    Args:
        tensor: Tensor to check.
        expected_dtype: Expected dtype or tuple of allowed dtypes.
        
    Raises:
        TypeError: If tensor has unexpected dtype.
    """
    if isinstance(expected_dtype, tuple):
        if tensor.dtype not in expected_dtype:
            raise TypeError(
                f'Tensor must have dtype in {expected_dtype}, '
                f'got {tensor.dtype}'
            )
    else:
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f'Tensor must have dtype {expected_dtype}, got {tensor.dtype}'
            )


def to_onehot(
    label_tensor: torch.Tensor, 
    num_classes: int,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Convert label tensor to one-hot encoding.
    
    Args:
        label_tensor: Integer label tensor of shape (...,).
        num_classes: Number of classes.
        dtype: Output dtype. Defaults to float32.
        
    Returns:
        torch.Tensor: One-hot encoded tensor of shape (..., num_classes).
    """
    if dtype is None:
        dtype = torch.float32
        
    return torch.nn.functional.one_hot(
        label_tensor.long(), num_classes=num_classes
    ).to(dtype)


def reduce_tensor(
    tensor: torch.Tensor,
    reduction: str = 'mean',
    dim: Optional[Union[int, tuple[int, ...]]] = None
) -> torch.Tensor:
    """Reduce tensor along specified dimensions.
    
    Args:
        tensor: Input tensor.
        reduction: Reduction method ('mean', 'sum', 'none').
        dim: Dimensions to reduce over. If None, reduces over all dims.
        
    Returns:
        torch.Tensor: Reduced tensor.
        
    Raises:
        ValueError: If reduction method is not supported.
    """
    if reduction == 'none':
        return tensor
    elif reduction == 'mean':
        return tensor.mean(dim=dim)
    elif reduction == 'sum':
        return tensor.sum(dim=dim)
    else:
        raise ValueError(
            f"Reduction '{reduction}' not supported. "
            "Choose from ['mean', 'sum', 'none']"
        )


def apply_to_collection(
    data: Any,
    dtype: type,
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any
) -> Any:
    """Apply function to all elements of given type in a nested structure.
    
    Args:
        data: Input data structure.
        dtype: Type to apply function to.
        function: Function to apply.
        *args: Additional arguments for function.
        **kwargs: Additional keyword arguments for function.
        
    Returns:
        Data structure with function applied to elements of specified type.
    """
    if isinstance(data, dtype):
        return function(data, *args, **kwargs)
    elif isinstance(data, dict):
        return {k: apply_to_collection(v, dtype, function, *args, **kwargs) 
                for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        result = [apply_to_collection(item, dtype, function, *args, **kwargs) 
                 for item in data]
        return type(data)(result)
    else:
        return data