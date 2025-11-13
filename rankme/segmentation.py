"""Segmentation metrics for semantic segmentation and hyperspectral analysis.

This module implements metrics specifically designed for semantic segmentation tasks,
including traditional computer vision metrics and specialized metrics for hyperspectral
Earth observation data.

All metrics are implemented as PyTorch modules for seamless integration with training
pipelines and support batched operations on segmentation masks with shape (N, H, W)
where labels are integers in [0, num_classes-1].
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from rankme.base import StatelessMetric, check_same_shape


class ConfusionMatrix(StatelessMetric):
    """Compute confusion matrix for segmentation tasks.

    Args:
        num_classes: Number of semantic classes.
        normalize: Normalization mode ('true', 'pred', 'all', None).

    Example:
        >>> import torch
        >>> from rankme.segmentation import ConfusionMatrix
        >>>
        >>> y_true = torch.tensor([[0, 1, 2], [1, 2, 0]])
        >>> y_pred = torch.tensor([[0, 1, 1], [1, 2, 2]]) 
        >>> cm = ConfusionMatrix(num_classes=3)
        >>> matrix = cm(y_pred, y_true)
        >>> matrix.shape
        torch.Size([3, 3])
    """

    def __init__(
        self,
        num_classes: int,
        normalize: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.normalize = normalize
        
        if normalize and normalize not in ['true', 'pred', 'all']:
            raise ValueError("normalize must be one of ['true', 'pred', 'all', None]")

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute confusion matrix.

        Args:
            preds: Predicted labels, shape (N, H, W) or (B, N, H, W).
            target: Ground truth labels, same shape as preds.

        Returns:
            torch.Tensor: Confusion matrix of shape (num_classes, num_classes).
        """
        check_same_shape(preds, target)
        
        # Flatten tensors
        preds = preds.long().view(-1)
        target = target.long().view(-1)
        
        # Filter valid pixels
        mask = (target >= 0) & (target < self.num_classes)
        preds = preds[mask]
        target = target[mask]
        
        # Compute confusion matrix
        idx = target * self.num_classes + preds
        cm = torch.bincount(idx, minlength=self.num_classes * self.num_classes)
        cm = cm.reshape(self.num_classes, self.num_classes).float()
        
        # Apply normalization
        if self.normalize == 'true':
            cm = cm / (cm.sum(dim=1, keepdim=True) + 1e-7)
        elif self.normalize == 'pred':
            cm = cm / (cm.sum(dim=0, keepdim=True) + 1e-7)
        elif self.normalize == 'all':
            cm = cm / (cm.sum() + 1e-7)
            
        return cm


class DiceScore(StatelessMetric):
    """Compute Dice Score (F1-Score for segmentation) per class and mean.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        num_classes: Number of semantic classes.
        average: How to average across classes ('macro', 'micro', 'none').
        ignore_index: Class index to ignore in computation.
        eps: Small constant to avoid division by zero.

    Example:
        >>> import torch
        >>> from rankme.segmentation import DiceScore
        >>>
        >>> y_true = torch.tensor([[0, 1, 2], [1, 2, 0]])
        >>> y_pred = torch.tensor([[0, 1, 1], [1, 2, 2]])
        >>> dice = DiceScore(num_classes=3, average='macro')
        >>> score = dice(y_pred, y_true)
    """

    def __init__(
        self,
        num_classes: int,
        average: str = 'macro',
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.average = average
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice score.

        Args:
            preds: Predicted labels, shape (N, H, W) or (B, N, H, W).
            target: Ground truth labels, same shape as preds.

        Returns:
            torch.Tensor: Dice score(s).
        """
        check_same_shape(preds, target)
        
        # Convert to one-hot encoding
        preds_onehot = F.one_hot(preds.long(), self.num_classes).float()
        target_onehot = F.one_hot(target.long(), self.num_classes).float()
        
        # Move channel dimension to front: (C, N, H, W, ...)
        preds_onehot = preds_onehot.moveaxis(-1, 0)  
        target_onehot = target_onehot.moveaxis(-1, 0)
        
        # Flatten spatial dimensions: (C, -1)
        preds_flat = preds_onehot.flatten(start_dim=1)
        target_flat = target_onehot.flatten(start_dim=1)
        
        # Compute intersection and sums per class
        intersection = (preds_flat * target_flat).sum(dim=1)  # (C,)
        preds_sum = preds_flat.sum(dim=1)  # (C,)
        target_sum = target_flat.sum(dim=1)  # (C,)
        
        # Compute Dice score per class
        dice = (2.0 * intersection + self.eps) / (preds_sum + target_sum + self.eps)
        
        # Handle ignore index
        if self.ignore_index is not None:
            mask = torch.ones(self.num_classes, dtype=torch.bool, device=dice.device)
            mask[self.ignore_index] = False
            dice = dice[mask]
            intersection = intersection[mask]
            preds_sum = preds_sum[mask] 
            target_sum = target_sum[mask]
        
        # Average across classes
        if self.average == 'macro':
            return dice.mean()
        elif self.average == 'micro':
            return (2.0 * intersection.sum() + self.eps) / (preds_sum.sum() + target_sum.sum() + self.eps)
        else:  # 'none'
            return dice


class PixelAccuracy(StatelessMetric):
    """Compute pixel-wise accuracy for segmentation.

    Pixel Accuracy = correct_pixels / total_pixels

    Args:
        ignore_index: Class index to ignore in computation.

    Example:
        >>> import torch
        >>> from rankme.segmentation import PixelAccuracy
        >>>
        >>> y_true = torch.tensor([[0, 1, 2], [1, 2, 0]])
        >>> y_pred = torch.tensor([[0, 1, 1], [1, 2, 2]])
        >>> pa = PixelAccuracy()
        >>> accuracy = pa(y_pred, y_true)
    """

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ignore_index = ignore_index

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute pixel accuracy.

        Args:
            preds: Predicted labels, shape (N, H, W) or (B, N, H, W).
            target: Ground truth labels, same shape as preds.

        Returns:
            torch.Tensor: Pixel accuracy value.
        """
        check_same_shape(preds, target)
        
        # Create mask for valid pixels
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            preds = preds[mask]
            target = target[mask]
        
        correct = (preds == target).float()
        return correct.mean()


class Specificity(StatelessMetric):
    """Compute specificity (true negative rate) per class.

    Specificity = TN / (TN + FP)

    Args:
        num_classes: Number of semantic classes.
        average: How to average across classes ('macro', 'weighted', 'none').
        ignore_index: Class index to ignore in computation.
        eps: Small constant to avoid division by zero.

    Example:
        >>> import torch
        >>> from rankme.segmentation import Specificity
        >>>
        >>> y_true = torch.tensor([[0, 1, 2], [1, 2, 0]])
        >>> y_pred = torch.tensor([[0, 1, 1], [1, 2, 2]])
        >>> spec = Specificity(num_classes=3, average='macro')
        >>> specificity = spec(y_pred, y_true)
    """

    def __init__(
        self,
        num_classes: int,
        average: str = 'macro',
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.average = average
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute specificity.

        Args:
            preds: Predicted labels, shape (N, H, W) or (B, N, H, W).
            target: Ground truth labels, same shape as preds.

        Returns:
            torch.Tensor: Specificity value(s).
        """
        check_same_shape(preds, target)
        
        # Convert to one-hot
        preds_onehot = F.one_hot(preds.long(), self.num_classes).float()
        target_onehot = F.one_hot(target.long(), self.num_classes).float()
        
        # Move channel dimension to front and flatten: (C, -1)
        preds_onehot = preds_onehot.moveaxis(-1, 0).flatten(start_dim=1)
        target_onehot = target_onehot.moveaxis(-1, 0).flatten(start_dim=1)
        
        # Compute TP, FP, TN per class
        tp = (preds_onehot * target_onehot).sum(dim=1)
        fp = (preds_onehot * (1 - target_onehot)).sum(dim=1)
        fn = ((1 - preds_onehot) * target_onehot).sum(dim=1)
        tn = ((1 - preds_onehot) * (1 - target_onehot)).sum(dim=1)
        
        # Compute specificity per class
        specificity = tn / (tn + fp + self.eps)
        
        # Handle ignore index
        if self.ignore_index is not None:
            mask = torch.ones(self.num_classes, dtype=torch.bool, device=specificity.device)
            mask[self.ignore_index] = False
            specificity = specificity[mask]
            target_onehot = target_onehot[mask]
        
        if self.average == 'macro':
            return specificity.mean()
        elif self.average == 'weighted':
            weights = target_onehot.sum(dim=1)
            return (specificity * weights).sum() / (weights.sum() + self.eps)
        else:  # 'none'
            return specificity


class CohensKappa(StatelessMetric):
    """Compute Cohen's Kappa coefficient for segmentation.

    Kappa = (p_o - p_e) / (1 - p_e)
    where p_o is observed accuracy and p_e is expected accuracy by chance.

    Args:
        num_classes: Number of semantic classes.
        eps: Small constant to avoid division by zero.

    Example:
        >>> import torch
        >>> from rankme.segmentation import CohensKappa
        >>>
        >>> y_true = torch.tensor([[0, 1, 2], [1, 2, 0]])
        >>> y_pred = torch.tensor([[0, 1, 1], [1, 2, 2]])
        >>> kappa = CohensKappa(num_classes=3)
        >>> kappa_value = kappa(y_pred, y_true)
    """

    def __init__(
        self,
        num_classes: int,
        eps: float = 1e-7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Cohen's Kappa.

        Args:
            preds: Predicted labels, shape (N, H, W) or (B, N, H, W).
            target: Ground truth labels, same shape as preds.

        Returns:
            torch.Tensor: Cohen's Kappa coefficient.
        """
        check_same_shape(preds, target)
        
        # Get confusion matrix
        cm_metric = ConfusionMatrix(self.num_classes)
        cm = cm_metric(preds, target)
        
        total = cm.sum()
        if total == 0:
            return torch.tensor(0.0, device=cm.device)
        
        # Observed accuracy
        po = torch.trace(cm) / total
        
        # Expected accuracy
        row_marginals = cm.sum(dim=1)
        col_marginals = cm.sum(dim=0) 
        pe = (row_marginals * col_marginals).sum() / (total * total + self.eps)
        
        # Cohen's Kappa
        kappa = (po - pe) / (1.0 - pe + self.eps)
        return kappa


class VolumetricSimilarity(StatelessMetric):
    """Compute Volumetric Similarity for a specific class.

    VS = 1 - |V_pred - V_true| / (V_pred + V_true)

    Args:
        class_id: Class index to evaluate.
        eps: Small constant to avoid division by zero.

    Example:
        >>> import torch
        >>> from rankme.segmentation import VolumetricSimilarity
        >>>
        >>> y_true = torch.tensor([[0, 1, 1], [1, 1, 0]])
        >>> y_pred = torch.tensor([[0, 1, 0], [1, 1, 1]])
        >>> vs = VolumetricSimilarity(class_id=1)
        >>> similarity = vs(y_pred, y_true)
    """

    def __init__(
        self,
        class_id: int,
        eps: float = 1e-7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.class_id = class_id
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute volumetric similarity.

        Args:
            preds: Predicted labels, shape (N, H, W) or (B, N, H, W).
            target: Ground truth labels, same shape as preds.

        Returns:
            torch.Tensor: Volumetric similarity value.
        """
        check_same_shape(preds, target)
        
        v_true = (target == self.class_id).sum().float()
        v_pred = (preds == self.class_id).sum().float()
        
        num = torch.abs(v_pred - v_true)
        den = v_pred + v_true + self.eps
        vs = 1.0 - num / den
        return vs


class RelativeVolumeDifference(StatelessMetric):
    """Compute Relative Volume Difference for a specific class.

    RVD = (V_pred - V_true) / (V_true + eps)

    Args:
        class_id: Class index to evaluate.
        eps: Small constant to avoid division by zero.

    Example:
        >>> import torch
        >>> from rankme.segmentation import RelativeVolumeDifference
        >>>
        >>> y_true = torch.tensor([[0, 1, 1], [1, 1, 0]])
        >>> y_pred = torch.tensor([[0, 1, 0], [1, 1, 1]])
        >>> rvd = RelativeVolumeDifference(class_id=1)
        >>> difference = rvd(y_pred, y_true)
    """

    def __init__(
        self,
        class_id: int,
        eps: float = 1e-7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.class_id = class_id
        self.eps = eps

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute relative volume difference.

        Args:
            preds: Predicted labels, shape (N, H, W) or (B, N, H, W).
            target: Ground truth labels, same shape as preds.

        Returns:
            torch.Tensor: Relative volume difference value.
        """
        check_same_shape(preds, target)
        
        v_true = (target == self.class_id).sum().float()
        v_pred = (preds == self.class_id).sum().float()
        
        return (v_pred - v_true) / (v_true + self.eps)


class SpectralAngleMapper(StatelessMetric):
    """Compute Spectral Angle Mapper (SAM) for hyperspectral data.

    SAM measures the angle between spectral vectors in radians.

    Args:
        dim: Dimension corresponding to spectral channels (default: -1).
        eps: Small constant to avoid division by zero.
        reduction: How to reduce spatial dimensions ('mean', 'sum', 'none').

    Example:
        >>> import torch
        >>> from rankme.segmentation import SpectralAngleMapper
        >>>
        >>> # Hyperspectral data: (H, W, C) 
        >>> s = torch.randn(10, 10, 50)  # predicted
        >>> r = torch.randn(10, 10, 50)  # reference
        >>> sam = SpectralAngleMapper(dim=-1)
        >>> angle = sam(s, r)
    """

    def __init__(
        self,
        dim: int = -1,
        eps: float = 1e-7,
        reduction: str = 'mean',
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SAM angles.

        Args:
            preds: Predicted spectral vectors.
            target: Reference spectral vectors.

        Returns:
            torch.Tensor: SAM angles in radians.
        """
        check_same_shape(preds, target)
        
        # Ensure float precision
        preds = preds.float()
        target = target.float()
        
        # Compute dot product and norms
        dot = (preds * target).sum(dim=self.dim)
        norm_pred = preds.norm(dim=self.dim)
        norm_target = target.norm(dim=self.dim)
        
        # Compute cosine and angle
        denom = norm_pred * norm_target + self.eps
        cos_theta = torch.clamp(dot / denom, -1.0, 1.0)
        sam_angles = torch.acos(cos_theta)
        
        # Apply reduction
        if self.reduction == 'mean':
            return sam_angles.mean()
        elif self.reduction == 'sum':
            return sam_angles.sum()
        else:  # 'none'
            return sam_angles


class LabelConsistency(StatelessMetric):
    """Compute label consistency between baseline and perturbed predictions.

    LC = mean(y_base == y_perturbed)

    This metric is useful for evaluating model robustness to perturbations.

    Example:
        >>> import torch
        >>> from rankme.segmentation import LabelConsistency
        >>>
        >>> y_base = torch.tensor([[0, 1, 2], [1, 2, 0]])
        >>> y_pert = torch.tensor([[0, 1, 1], [1, 2, 2]])
        >>> lc = LabelConsistency()
        >>> consistency = lc(y_pert, y_base)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute label consistency.

        Args:
            preds: Perturbed predictions, shape (N, H, W) or (B, N, H, W).
            target: Baseline predictions, same shape as preds.

        Returns:
            torch.Tensor: Label consistency value in [0, 1].
        """
        check_same_shape(preds, target)
        
        equal = (preds == target).float()
        return equal.mean()


class IoUDegradation(StatelessMetric):
    """Compute IoU degradation between baseline and perturbed predictions.

    This metric evaluates how much the IoU degrades under perturbations,
    useful for robustness analysis.

    Args:
        num_classes: Number of semantic classes.
        eps: Small constant to avoid division by zero.

    Returns a dictionary with:
        - 'iou_base': IoU for baseline predictions
        - 'iou_perturbed': IoU for perturbed predictions  
        - 'delta_iou': IoU loss (iou_base - iou_perturbed)

    Example:
        >>> import torch
        >>> from rankme.segmentation import IoUDegradation
        >>>
        >>> y_true = torch.tensor([[0, 1, 2], [1, 2, 0]])
        >>> y_base = torch.tensor([[0, 1, 2], [1, 2, 0]])  # perfect
        >>> y_pert = torch.tensor([[0, 1, 1], [1, 2, 2]])  # degraded
        >>> degradation = IoUDegradation(num_classes=3)
        >>> result = degradation(y_true, y_base, y_pert)
    """

    def __init__(
        self,
        num_classes: int,
        eps: float = 1e-7,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.eps = eps
        
        # Import IoU metric from classification module
        from rankme.classification import IoU
        self.iou_metric = IoU(
            task='multiclass',
            num_classes=num_classes,
            average='none'  # Get per-class IoU
        )

    def forward(
        self, 
        target: torch.Tensor, 
        baseline: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute IoU degradation.

        Args:
            target: Ground truth labels, shape (N, H, W) or (B, N, H, W).
            baseline: Baseline predictions, same shape as target.
            perturbed: Perturbed predictions, same shape as target.

        Returns:
            Dict with 'iou_base', 'iou_perturbed', and 'delta_iou' tensors.
        """
        check_same_shape(target, baseline)
        check_same_shape(target, perturbed)
        
        # Flatten tensors for IoU computation
        target_flat = target.flatten()
        baseline_flat = baseline.flatten()  
        perturbed_flat = perturbed.flatten()
        
        iou_base = self.iou_metric(baseline_flat, target_flat)
        iou_pert = self.iou_metric(perturbed_flat, target_flat)
        delta_iou = iou_base - iou_pert
        
        return {
            'iou_base': iou_base,
            'iou_perturbed': iou_pert, 
            'delta_iou': delta_iou
        }