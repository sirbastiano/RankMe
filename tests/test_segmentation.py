"""Tests for segmentation metrics."""

import pytest
import torch

import rankme


class TestSegmentationMetrics:
    """Test segmentation metrics implementation."""

    def setup_method(self):
        """Setup test data."""
        # Simple 2x3 segmentation masks with 3 classes
        self.y_true = torch.tensor([[0, 1, 2], [1, 2, 0]])
        self.y_pred = torch.tensor([[0, 1, 1], [1, 2, 2]])
        self.num_classes = 3

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        cm = rankme.ConfusionMatrix(num_classes=self.num_classes)
        matrix = cm(self.y_pred, self.y_true)
        
        assert matrix.shape == (self.num_classes, self.num_classes)
        assert torch.allclose(matrix.sum(), torch.tensor(6.0))  # Total pixels
        
        # Test normalized confusion matrix
        cm_norm = rankme.ConfusionMatrix(num_classes=self.num_classes, normalize='true')
        matrix_norm = cm_norm(self.y_pred, self.y_true)
        
        # Each row should sum to 1 (normalized by true labels)
        assert torch.allclose(matrix_norm.sum(dim=1), torch.ones(self.num_classes), atol=1e-6)

    def test_dice_score(self):
        """Test Dice score computation."""
        dice = rankme.DiceScore(num_classes=self.num_classes, average='macro')
        score = dice(self.y_pred, self.y_true)
        
        assert isinstance(score, torch.Tensor)
        assert 0.0 <= score <= 1.0
        
        # Test per-class Dice
        dice_none = rankme.DiceScore(num_classes=self.num_classes, average='none')
        scores = dice_none(self.y_pred, self.y_true)
        assert scores.shape == (self.num_classes,)

    def test_pixel_accuracy(self):
        """Test pixel accuracy computation."""
        pa = rankme.PixelAccuracy()
        accuracy = pa(self.y_pred, self.y_true)
        
        assert isinstance(accuracy, torch.Tensor)
        assert 0.0 <= accuracy <= 1.0
        
        # Perfect prediction should give 1.0
        perfect_pred = self.y_true.clone()
        accuracy_perfect = pa(perfect_pred, self.y_true)
        assert torch.allclose(accuracy_perfect, torch.tensor(1.0))

    def test_specificity(self):
        """Test specificity computation."""
        spec = rankme.Specificity(num_classes=self.num_classes, average='macro')
        specificity = spec(self.y_pred, self.y_true)
        
        assert isinstance(specificity, torch.Tensor)
        assert 0.0 <= specificity <= 1.0

    def test_cohens_kappa(self):
        """Test Cohen's Kappa computation."""
        kappa = rankme.CohensKappa(num_classes=self.num_classes)
        kappa_value = kappa(self.y_pred, self.y_true)
        
        assert isinstance(kappa_value, torch.Tensor)
        # Kappa can be negative, but for reasonable predictions should be positive
        assert kappa_value >= -1.0

    def test_volumetric_similarity(self):
        """Test volumetric similarity computation."""
        vs = rankme.VolumetricSimilarity(class_id=1)
        similarity = vs(self.y_pred, self.y_true)
        
        assert isinstance(similarity, torch.Tensor)
        assert 0.0 <= similarity <= 1.0

    def test_relative_volume_difference(self):
        """Test relative volume difference computation."""
        rvd = rankme.RelativeVolumeDifference(class_id=1)
        difference = rvd(self.y_pred, self.y_true)
        
        assert isinstance(difference, torch.Tensor)
        # Can be positive or negative

    def test_spectral_angle_mapper(self):
        """Test Spectral Angle Mapper computation."""
        # Create hyperspectral-like data
        h, w, c = 4, 4, 10
        preds = torch.randn(h, w, c)
        target = torch.randn(h, w, c)
        
        sam = rankme.SpectralAngleMapper(dim=-1, reduction='mean')
        angle = sam(preds, target)
        
        assert isinstance(angle, torch.Tensor)
        assert angle >= 0.0  # Angles are non-negative
        assert angle <= torch.pi  # Max angle is π radians

    def test_label_consistency(self):
        """Test label consistency computation."""
        lc = rankme.LabelConsistency()
        consistency = lc(self.y_pred, self.y_true)
        
        assert isinstance(consistency, torch.Tensor)
        assert 0.0 <= consistency <= 1.0
        
        # Perfect consistency should give 1.0
        perfect_consistency = lc(self.y_true, self.y_true)
        assert torch.allclose(perfect_consistency, torch.tensor(1.0))

    def test_iou_degradation(self):
        """Test IoU degradation computation."""
        # Create baseline (perfect) and perturbed predictions
        y_base = self.y_true.clone()  # Perfect baseline
        y_pert = self.y_pred.clone()  # Perturbed
        
        degradation = rankme.IoUDegradation(num_classes=self.num_classes)
        result = degradation(self.y_true, y_base, y_pert)
        
        assert isinstance(result, dict)
        assert 'iou_base' in result
        assert 'iou_perturbed' in result
        assert 'delta_iou' in result
        
        # Baseline should be better than perturbed
        assert torch.all(result['iou_base'] >= result['iou_perturbed'])
        assert torch.all(result['delta_iou'] >= 0)

    def test_batched_input(self):
        """Test that metrics work with batched inputs."""
        batch_size = 2
        y_true_batch = torch.stack([self.y_true, self.y_true], dim=0)
        y_pred_batch = torch.stack([self.y_pred, self.y_pred], dim=0)
        
        # Test a few metrics with batched inputs
        pa = rankme.PixelAccuracy()
        accuracy = pa(y_pred_batch, y_true_batch)
        assert isinstance(accuracy, torch.Tensor)
        
        dice = rankme.DiceScore(num_classes=self.num_classes)
        score = dice(y_pred_batch, y_true_batch)
        assert isinstance(score, torch.Tensor)

    def test_device_compatibility(self):
        """Test that metrics work on different devices."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            y_true_cuda = self.y_true.to(device)
            y_pred_cuda = self.y_pred.to(device)
            
            pa = rankme.PixelAccuracy().to(device)
            accuracy = pa(y_pred_cuda, y_true_cuda)
            assert accuracy.device == device

    def test_ignore_index(self):
        """Test ignore_index functionality where applicable."""
        # Create data with ignore index
        y_true_ignore = self.y_true.clone()
        y_true_ignore[0, 0] = -1  # Ignore index
        
        pa = rankme.PixelAccuracy(ignore_index=-1)
        accuracy = pa(self.y_pred, y_true_ignore)
        assert isinstance(accuracy, torch.Tensor)
        
        dice = rankme.DiceScore(num_classes=self.num_classes, ignore_index=2)
        score = dice(self.y_pred, self.y_true)
        assert isinstance(score, torch.Tensor)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Shape mismatch
        with pytest.raises(Exception):
            pa = rankme.PixelAccuracy()
            pa(self.y_pred, self.y_true.unsqueeze(0))
            
        # Invalid normalize parameter
        with pytest.raises(ValueError):
            rankme.ConfusionMatrix(num_classes=3, normalize='invalid')


if __name__ == '__main__':
    pytest.main([__file__])