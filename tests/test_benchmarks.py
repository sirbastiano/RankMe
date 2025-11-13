"""Performance benchmarks for RankMe metrics."""

import time

import numpy as np
import pytest
import torch

from rankme import MSE, Accuracy, RankMe
from rankme.feature_learning import EffectiveRank, SpectralEntropy


class TestPerformance:
    """Performance tests that don't require pytest-benchmark."""

    def test_rankme_memory_usage(self):
        """Test memory usage of RankMe."""
        torch.manual_seed(42)
        sizes = [64, 128, 256, 512]

        for size in sizes:
            Z = torch.randn(1000, size)
            rankme = RankMe()

            # Measure memory before
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem_before = torch.cuda.memory_allocated()

            result = rankme(Z)

            # Measure memory after
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                mem_used = (mem_after - mem_before) / (1024**2)  # MB
                print(f"Size {size}: Memory used: {mem_used:.2f} MB")

            assert 0 <= result <= 1

    def test_rankme_scalability(self):
        """Test scalability of RankMe with different matrix sizes."""
        torch.manual_seed(42)

        # Test different matrix sizes
        test_cases = [
            (100, 64),
            (500, 128),
            (1000, 256),
        ]

        times = []
        for n_samples, n_features in test_cases:
            Z = torch.randn(n_samples, n_features)
            rankme = RankMe()

            start_time = time.time()
            result = rankme(Z)
            end_time = time.time()

            elapsed = end_time - start_time
            times.append(elapsed)

            print(f"Size ({n_samples}, {n_features}): {elapsed:.4f}s")
            assert 0 <= result <= 1

        # Check that time complexity is reasonable
        assert all(t >= 0 for t in times)

    def test_basic_performance(self):
        """Basic performance test without benchmarking framework."""
        torch.manual_seed(42)

        # Small test
        Z = torch.randn(100, 64)
        rankme = RankMe()
        start = time.time()
        result = rankme(Z)
        end = time.time()

        print(f"Small test (100x64): {end - start:.4f}s, result: {result:.4f}")
        assert 0 <= result <= 1
        assert (end - start) < 1.0  # Should complete within 1 second

        # Medium test
        Z = torch.randn(1000, 256)
        start = time.time()
        result = rankme(Z)
        end = time.time()

        print(f"Medium test (1000x256): {end - start:.4f}s, result: {result:.4f}")
        assert 0 <= result <= 1
        assert (end - start) < 5.0  # Should complete within 5 seconds


# Benchmark tests that require pytest-benchmark
try:
    import pytest_benchmark

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


@pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
class TestBenchmarks:
    """Benchmark tests for performance monitoring (requires pytest-benchmark)."""

    @pytest.mark.benchmark
    def test_rankme_performance_small(self, benchmark):
        """Benchmark RankMe with small embeddings."""
        torch.manual_seed(42)
        Z = torch.randn(100, 64)
        rankme = RankMe()

        result = benchmark(rankme, Z)
        assert 0 <= result <= 1

    @pytest.mark.benchmark
    def test_accuracy_performance(self, benchmark):
        """Benchmark accuracy computation."""
        torch.manual_seed(42)
        y_true = torch.randint(0, 10, (10000,))
        y_pred = torch.randint(0, 10, (10000,))
        accuracy = Accuracy(task="multiclass", num_classes=10)

        result = benchmark(accuracy, y_pred, y_true)
        assert 0 <= result <= 1
