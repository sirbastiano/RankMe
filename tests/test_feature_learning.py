"""Tests for feature learning metrics."""

import numpy as np
import pytest
import torch

from rankme.feature_learning import EffectiveRank, RankMe, SpectralEntropy
from tests.conftest import assert_in_range, assert_tensor_close


class TestRankMe:
    """Test cases for RankMe metric."""

    def test_rankme_2d_input(self, small_embedding_2d):
        """Test RankMe with 2D input."""
        rankme = RankMe()
        score = rankme(small_embedding_2d)

        # RankMe score should be between 0 and 1
        assert_in_range(score, 0.0, 1.0)
        assert score.dim() == 0  # Should be scalar

    def test_rankme_3d_input(self, batch_embeddings_3d):
        """Test RankMe with 3D (batched) input."""
        rankme = RankMe()
        scores = rankme(batch_embeddings_3d)

        # Should return one score per batch
        assert scores.shape == (8,)
        assert_in_range(scores, 0.0, 1.0)

    def test_rankme_perfect_rank(self):
        """Test RankMe with perfect rank matrix."""
        # Create a matrix with uniform singular values
        D = 10
        U = torch.eye(D)
        S = torch.ones(D)
        V = torch.eye(D)
        Z = U @ torch.diag(S) @ V.T

        rankme = RankMe()
        score = rankme(Z)

        # Should be close to 1 for uniform distribution
        assert score > 0.9

    def test_rankme_rank_deficient(self):
        """Test RankMe with rank-deficient matrix."""
        # Create a rank-1 matrix
        Z = torch.randn(100, 1) @ torch.randn(1, 50)

        rankme = RankMe()
        score = rankme(Z)

        # Should be close to 0 for rank-deficient matrix
        assert score < 0.1

    def test_rankme_centering(self, small_embedding_2d):
        """Test RankMe with and without centering."""
        rankme_no_center = RankMe(center=False)
        rankme_center = RankMe(center=True)

        score_no_center = rankme_no_center(small_embedding_2d)
        score_center = rankme_center(small_embedding_2d)

        # Scores should be different when centering is applied
        assert not torch.allclose(score_no_center, score_center)

    def test_rankme_log_base(self, small_embedding_2d):
        """Test RankMe with different logarithm bases."""
        rankme_natural = RankMe(log_base=None)
        rankme_base2 = RankMe(log_base=2.0)

        score_natural = rankme_natural(small_embedding_2d)
        score_base2 = rankme_base2(small_embedding_2d)

        # Both should be valid scores
        assert_in_range(score_natural, 0.0, 1.0)
        assert_in_range(score_base2, 0.0, 1.0)

    def test_rankme_detach(self, small_embedding_2d):
        """Test RankMe with gradient detachment."""
        small_embedding_2d.requires_grad_(True)

        rankme_detach = RankMe(detach=True)
        rankme_no_detach = RankMe(detach=False)

        score_detach = rankme_detach(small_embedding_2d)
        score_no_detach = rankme_no_detach(small_embedding_2d)

        # Scores should be the same, but gradients different
        assert torch.allclose(score_detach, score_no_detach)
        assert not score_detach.requires_grad
        assert score_no_detach.requires_grad

    def test_rankme_edge_cases(self):
        """Test RankMe edge cases."""
        rankme = RankMe()

        # Zero matrix
        Z_zero = torch.zeros(10, 5)
        score_zero = rankme(Z_zero)
        assert torch.isfinite(score_zero)

        # Single sample
        Z_single = torch.randn(1, 10)
        score_single = rankme(Z_single)
        assert torch.isfinite(score_single)

    def test_rankme_invalid_input(self):
        """Test RankMe with invalid input shapes."""
        rankme = RankMe()

        # 1D input should raise error
        with pytest.raises(ValueError):
            rankme(torch.randn(10))

        # 4D input should raise error
        with pytest.raises(ValueError):
            rankme(torch.randn(2, 3, 4, 5))


class TestEffectiveRank:
    """Test cases for EffectiveRank metric."""

    def test_effective_rank_basic(self, small_embedding_2d):
        """Test basic EffectiveRank functionality."""
        eff_rank = EffectiveRank()
        rank = eff_rank(small_embedding_2d)

        # Effective rank should be positive
        assert rank > 0
        assert torch.isfinite(rank)

    def test_effective_rank_vs_rankme(self, small_embedding_2d):
        """Test relationship between EffectiveRank and RankMe."""
        eff_rank = EffectiveRank()
        rankme = RankMe()

        rank_val = eff_rank(small_embedding_2d)
        rankme_val = rankme(small_embedding_2d)

        # They should be related but different
        assert not torch.allclose(rank_val, rankme_val)

    def test_effective_rank_full_rank(self):
        """Test EffectiveRank with full-rank matrix."""
        # Create a full-rank square matrix
        D = 20
        Z = torch.randn(D, D)
        U, S, V = torch.svd(Z)
        Z_full_rank = U @ torch.diag(torch.ones(D)) @ V.T

        eff_rank = EffectiveRank()
        rank = eff_rank(Z_full_rank)

        # Should be close to the actual rank
        assert rank > D * 0.8  # Allow some tolerance


class TestSpectralEntropy:
    """Test cases for SpectralEntropy metric."""

    def test_spectral_entropy_basic(self, small_embedding_2d):
        """Test basic SpectralEntropy functionality."""
        entropy = SpectralEntropy()
        ent_val = entropy(small_embedding_2d)

        # Entropy should be positive
        assert ent_val >= 0
        assert torch.isfinite(ent_val)

    def test_spectral_entropy_log_bases(self, small_embedding_2d):
        """Test SpectralEntropy with different log bases."""
        entropy_natural = SpectralEntropy(log_base=None)
        entropy_base2 = SpectralEntropy(log_base=2.0)
        entropy_base10 = SpectralEntropy(log_base=10.0)

        ent_natural = entropy_natural(small_embedding_2d)
        ent_base2 = entropy_base2(small_embedding_2d)
        ent_base10 = entropy_base10(small_embedding_2d)

        # All should be valid
        assert torch.isfinite(ent_natural)
        assert torch.isfinite(ent_base2)
        assert torch.isfinite(ent_base10)

        # Different log bases should give different values
        # The relationship depends on implementation, just check they're different
        assert not torch.allclose(ent_base2, ent_natural)
        assert not torch.allclose(ent_base10, ent_natural)

    def test_spectral_entropy_vs_rankme(self, small_embedding_2d):
        """Test relationship between SpectralEntropy and RankMe."""
        entropy = SpectralEntropy()
        rankme = RankMe()

        ent_val = entropy(small_embedding_2d)
        rankme_val = rankme(small_embedding_2d)

        # RankMe is normalized entropy, so entropy should be larger
        D = small_embedding_2d.size(1)
        expected_rankme = ent_val / torch.log(torch.tensor(float(D)))

        # Should be approximately equal (within tolerance)
        assert_tensor_close(rankme_val, expected_rankme, rtol=1e-3)


class TestFeatureLearningIntegration:
    """Integration tests for feature learning metrics."""

    def test_metrics_consistency(self, small_embedding_2d):
        """Test that different metrics give consistent results."""
        rankme = RankMe()
        eff_rank = EffectiveRank()
        entropy = SpectralEntropy()

        rankme_score = rankme(small_embedding_2d)
        eff_rank_score = eff_rank(small_embedding_2d)
        entropy_score = entropy(small_embedding_2d)

        # All should be finite and valid
        assert torch.isfinite(rankme_score)
        assert torch.isfinite(eff_rank_score)
        assert torch.isfinite(entropy_score)

    def test_batch_vs_individual(self, batch_embeddings_3d):
        """Test that batched computation matches individual computation."""
        rankme = RankMe()

        # Compute batched scores
        batched_scores = rankme(batch_embeddings_3d)

        # Compute individual scores
        individual_scores = []
        for i in range(batch_embeddings_3d.size(0)):
            score = rankme(batch_embeddings_3d[i])
            individual_scores.append(score)
        individual_scores = torch.stack(individual_scores)

        # Should be close
        assert_tensor_close(batched_scores, individual_scores, rtol=1e-5)

    def test_device_consistency(self, small_embedding_2d, device):
        """Test that metrics work consistently across devices."""
        rankme = RankMe()

        # CPU computation
        score_cpu = rankme(small_embedding_2d)

        # GPU computation (if available)
        if device.type == "cuda":
            Z_gpu = small_embedding_2d.to(device)
            rankme_gpu = rankme.to(device)
            score_gpu = rankme_gpu(Z_gpu)

            # Should be close
            assert_tensor_close(score_cpu, score_gpu.cpu(), rtol=1e-5)

    def test_gradient_flow(self, small_embedding_2d):
        """Test that gradients flow correctly through metrics."""
        small_embedding_2d.requires_grad_(True)

        rankme = RankMe(detach=False)
        score = rankme(small_embedding_2d)

        # Backward pass should work
        score.backward()

        # Gradients should exist
        assert small_embedding_2d.grad is not None
        assert not torch.allclose(
            small_embedding_2d.grad, torch.zeros_like(small_embedding_2d.grad)
        )
