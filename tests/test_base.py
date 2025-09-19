"""Tests for base metric classes and utilities."""

import pytest
import torch

from rankme.base import (
    BaseMetric, 
    StatelessMetric, 
    StatefulMetric,
    check_same_shape, 
    check_tensor_dtype,
    to_onehot,
    reduce_tensor,
    apply_to_collection
)


class DummyStatelessMetric(StatelessMetric):
    """Dummy stateless metric for testing."""
    
    def forward(self, x, y):
        return (x - y).abs().mean()


class DummyStatefulMetric(StatefulMetric):
    """Dummy stateful metric for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('sum_errors', torch.tensor(0.0))
        self.register_buffer('count', torch.tensor(0))
        
    def update(self, x, y):
        self.sum_errors += (x - y).abs().sum()
        self.count += x.numel()
        
    def compute(self):
        return self.sum_errors / self.count if self.count > 0 else torch.tensor(0.0)
        
    def reset(self):
        super().reset()
        self.sum_errors.zero_()
        self.count.zero_()


class TestBaseMetric:
    """Test cases for BaseMetric abstract class."""
    
    def test_stateless_metric_basic(self):
        """Test basic stateless metric functionality."""
        metric = DummyStatelessMetric()
        
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.1, 1.9, 3.1])
        
        result = metric(x, y)
        
        assert torch.isfinite(result)
        assert result >= 0
        
    def test_stateful_metric_basic(self):
        """Test basic stateful metric functionality."""
        metric = DummyStatefulMetric()
        
        x1 = torch.tensor([1.0, 2.0])
        y1 = torch.tensor([1.1, 1.9])
        x2 = torch.tensor([3.0, 4.0])
        y2 = torch.tensor([3.1, 3.9])
        
        # Update with first batch
        metric.update(x1, y1)
        
        # Update with second batch
        metric.update(x2, y2)
        
        # Compute accumulated result
        result = metric.compute()
        
        assert torch.isfinite(result)
        assert result >= 0
        
    def test_stateful_metric_reset(self):
        """Test reset functionality for stateful metrics."""
        metric = DummyStatefulMetric()
        
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([2.0, 3.0])
        
        # Update and compute
        metric.update(x, y)
        result1 = metric.compute()
        
        # Reset and update again
        metric.reset()
        metric.update(x, y)
        result2 = metric.compute()
        
        # Results should be the same
        assert torch.allclose(result1, result2)
        
    def test_metric_compute_on_step(self):
        """Test compute_on_step behavior."""
        metric_on_step = DummyStatefulMetric(compute_on_step=True)
        metric_off_step = DummyStatefulMetric(compute_on_step=False)
        
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([1.1, 1.9])
        
        # With compute_on_step=True, forward should return result immediately
        result_on = metric_on_step(x, y)
        assert torch.isfinite(result_on)
        
        # With compute_on_step=False, need explicit compute call
        metric_off_step(x, y)  # Just updates
        result_off = metric_off_step.compute()
        assert torch.isfinite(result_off)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_check_same_shape_valid(self):
        """Test check_same_shape with valid inputs."""
        t1 = torch.randn(3, 4)
        t2 = torch.randn(3, 4)
        
        # Should not raise an error
        check_same_shape(t1, t2)
        
    def test_check_same_shape_invalid(self):
        """Test check_same_shape with invalid inputs."""
        t1 = torch.randn(3, 4)
        t2 = torch.randn(3, 5)
        
        with pytest.raises(ValueError):
            check_same_shape(t1, t2)
            
    def test_check_tensor_dtype_valid(self):
        """Test check_tensor_dtype with valid inputs."""
        t_float = torch.randn(3, 4).float()
        t_long = torch.randint(0, 10, (3, 4)).long()
        
        # Should not raise errors
        check_tensor_dtype(t_float, torch.float32)
        check_tensor_dtype(t_long, torch.long)
        check_tensor_dtype(t_float, (torch.float32, torch.float64))
        
    def test_check_tensor_dtype_invalid(self):
        """Test check_tensor_dtype with invalid inputs."""
        t_float = torch.randn(3, 4).float()
        
        with pytest.raises(TypeError):
            check_tensor_dtype(t_float, torch.long)
            
        with pytest.raises(TypeError):
            check_tensor_dtype(t_float, (torch.long, torch.int))
            
    def test_to_onehot_basic(self):
        """Test basic one-hot encoding."""
        labels = torch.tensor([0, 1, 2, 1])
        num_classes = 3
        
        onehot = to_onehot(labels, num_classes)
        
        expected = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])
        
        assert torch.allclose(onehot, expected)
        assert onehot.dtype == torch.float32
        
    def test_to_onehot_custom_dtype(self):
        """Test one-hot encoding with custom dtype."""
        labels = torch.tensor([0, 1, 2])
        num_classes = 3
        
        onehot = to_onehot(labels, num_classes, dtype=torch.float64)
        
        assert onehot.dtype == torch.float64
        
    def test_reduce_tensor_mean(self):
        """Test tensor reduction with mean."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # Reduce all
        result_all = reduce_tensor(tensor, 'mean')
        assert torch.allclose(result_all, torch.tensor(2.5))
        
        # Reduce along dimension
        result_dim0 = reduce_tensor(tensor, 'mean', dim=0)
        assert torch.allclose(result_dim0, torch.tensor([2.0, 3.0]))
        
        result_dim1 = reduce_tensor(tensor, 'mean', dim=1)
        assert torch.allclose(result_dim1, torch.tensor([1.5, 3.5]))
        
    def test_reduce_tensor_sum(self):
        """Test tensor reduction with sum."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        result = reduce_tensor(tensor, 'sum')
        assert torch.allclose(result, torch.tensor(10.0))
        
    def test_reduce_tensor_none(self):
        """Test tensor reduction with none."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        result = reduce_tensor(tensor, 'none')
        assert torch.allclose(result, tensor)
        
    def test_reduce_tensor_invalid(self):
        """Test tensor reduction with invalid method."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            reduce_tensor(tensor, 'invalid')
            
    def test_apply_to_collection_basic(self):
        """Test apply_to_collection with basic types."""
        def add_one(x):
            return x + 1
            
        # Test with single tensor
        tensor = torch.tensor([1, 2, 3])
        result = apply_to_collection(tensor, torch.Tensor, add_one)
        expected = torch.tensor([2, 3, 4])
        assert torch.allclose(result, expected)
        
        # Test with list of tensors
        tensor_list = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result_list = apply_to_collection(tensor_list, torch.Tensor, add_one)
        expected_list = [torch.tensor([2, 3]), torch.tensor([4, 5])]
        
        assert len(result_list) == len(expected_list)
        for r, e in zip(result_list, expected_list):
            assert torch.allclose(r, e)
            
    def test_apply_to_collection_nested(self):
        """Test apply_to_collection with nested structures."""
        def multiply_by_two(x):
            return x * 2
            
        nested_dict = {
            'a': torch.tensor([1, 2]),
            'b': {
                'c': torch.tensor([3, 4]),
                'd': [torch.tensor([5, 6]), torch.tensor([7, 8])]
            }
        }
        
        result = apply_to_collection(nested_dict, torch.Tensor, multiply_by_two)
        
        assert torch.allclose(result['a'], torch.tensor([2, 4]))
        assert torch.allclose(result['b']['c'], torch.tensor([6, 8]))
        assert torch.allclose(result['b']['d'][0], torch.tensor([10, 12]))
        assert torch.allclose(result['b']['d'][1], torch.tensor([14, 16]))
        
    def test_apply_to_collection_with_non_target_types(self):
        """Test apply_to_collection ignores non-target types."""
        def add_one(x):
            return x + 1
            
        mixed_data = {
            'tensor': torch.tensor([1, 2]),
            'int': 5,
            'string': 'hello',
            'list_of_tensors': [torch.tensor([3, 4]), 'not_a_tensor']
        }
        
        result = apply_to_collection(mixed_data, torch.Tensor, add_one)
        
        assert torch.allclose(result['tensor'], torch.tensor([2, 3]))
        assert result['int'] == 5  # unchanged
        assert result['string'] == 'hello'  # unchanged
        assert torch.allclose(result['list_of_tensors'][0], torch.tensor([4, 5]))
        assert result['list_of_tensors'][1] == 'not_a_tensor'  # unchanged


class TestBaseMetricIntegration:
    """Integration tests for base metric functionality."""
    
    def test_metric_inheritance_consistency(self):
        """Test that metric inheritance works correctly."""
        stateless = DummyStatelessMetric()
        stateful = DummyStatefulMetric()
        
        # Both should be instances of BaseMetric
        assert isinstance(stateless, BaseMetric)
        assert isinstance(stateful, BaseMetric)
        
        # Both should have required methods
        assert hasattr(stateless, 'forward')
        assert hasattr(stateless, 'reset')
        assert hasattr(stateful, 'update')
        assert hasattr(stateful, 'compute')
        
    def test_metric_device_movement(self):
        """Test that metrics can be moved between devices."""
        metric = DummyStatefulMetric()
        
        # Should start on CPU
        assert next(metric.parameters()).device.type == 'cpu'
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            metric_cuda = metric.cuda()
            assert next(metric_cuda.parameters()).device.type == 'cuda'
            
            # Move back to CPU
            metric_cpu = metric_cuda.cpu()
            assert next(metric_cpu.parameters()).device.type == 'cpu'
            
    def test_metric_state_dict(self):
        """Test that metric state can be saved and loaded."""
        metric = DummyStatefulMetric()
        
        # Update with some data
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([1.1, 1.9])
        metric.update(x, y)
        
        # Save state
        state_dict = metric.state_dict()
        
        # Create new metric and load state
        new_metric = DummyStatefulMetric()
        new_metric.load_state_dict(state_dict)
        
        # Both should compute the same result
        result1 = metric.compute()
        result2 = new_metric.compute()
        
        assert torch.allclose(result1, result2)
        
    def test_metric_train_eval_modes(self):
        """Test that metrics respect train/eval modes."""
        metric = DummyStatefulMetric()
        
        # Should start in training mode
        assert metric.training
        
        # Switch to eval mode
        metric.eval()
        assert not metric.training
        
        # Switch back to train mode
        metric.train()
        assert metric.training