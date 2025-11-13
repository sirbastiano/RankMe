"""
Example usage of RankMe segmentation metrics.

This script demonstrates how to use the segmentation metrics
for computer vision and hyperspectral image analysis.
"""

import torch
from rankme import (
    ConfusionMatrix,
    DiceScore,
    PixelAccuracy,
    Specificity,
    CohensKappa,
    VolumetricSimilarity,
    RelativeVolumeDifference,
    SpectralAngleMapper,
    LabelConsistency,
    IoUDegradation,
    # Also available from classification module:
    IoU,
    Precision,
    Recall,
    F1Score,
    Accuracy,
)


def main():
    """Demonstrate segmentation metrics usage."""
    # Create sample segmentation data (batch_size=2, height=4, width=4, num_classes=3)
    batch_size, height, width, num_classes = 2, 4, 4, 3
    
    # Ground truth segmentation masks
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Predicted segmentation masks
    predictions = torch.randint(0, num_classes, (batch_size, height, width))
    
    print("Segmentation Metrics Example")
    print("=" * 40)
    print(f"Input shape: {targets.shape}")
    print(f"Number of classes: {num_classes}")
    print()
    
    # Basic segmentation metrics
    print("Basic Metrics:")
    print("-" * 20)
    
    pixel_acc = PixelAccuracy()
    dice_score = DiceScore(num_classes=num_classes)
    iou = IoU(num_classes=num_classes, task='multiclass')
    
    print(f"Pixel Accuracy: {pixel_acc(predictions, targets):.4f}")
    print(f"Dice Score: {dice_score(predictions, targets):.4f}")
    print(f"IoU: {iou(predictions, targets):.4f}")
    print()
    
    # Advanced segmentation metrics
    print("Advanced Metrics:")
    print("-" * 20)
    
    specificity = Specificity(num_classes=num_classes)
    cohens_kappa = CohensKappa(num_classes=num_classes)
    vol_sim = VolumetricSimilarity(class_id=1)  # Check similarity for class 1
    rel_vol_diff = RelativeVolumeDifference(class_id=1)  # Volume difference for class 1
    
    print(f"Specificity: {specificity(predictions, targets):.4f}")
    print(f"Cohen's Kappa: {cohens_kappa(predictions, targets):.4f}")
    print(f"Volumetric Similarity (class 1): {vol_sim(predictions, targets):.4f}")
    print(f"Relative Volume Difference (class 1): {rel_vol_diff(predictions, targets):.4f}")
    print()
    
    # Hyperspectral-specific metrics
    print("Hyperspectral Metrics:")
    print("-" * 20)
    
    # For spectral metrics, we need hyperspectral data (batch_size, channels, height, width)
    hyperspectral_shape = (batch_size, 10, height, width)  # 10 spectral bands
    hs_targets = torch.randn(hyperspectral_shape)
    hs_predictions = torch.randn(hyperspectral_shape)
    
    sam = SpectralAngleMapper()
    print(f"Spectral Angle Mapper: {sam(hs_predictions, hs_targets):.4f}")
    print()
    
    # Confusion matrix
    print("Confusion Matrix:")
    print("-" * 20)
    
    conf_matrix = ConfusionMatrix(num_classes=num_classes)
    cm = conf_matrix(predictions, targets)
    print(f"Shape: {cm.shape}")
    print(f"Matrix:\n{cm}")
    print()
    
    # Degradation analysis
    print("Degradation Analysis:")
    print("-" * 20)
    
    # Ground truth
    ground_truth = targets
    # Original high-quality segmentation (baseline model)
    baseline_pred = torch.randint(0, num_classes, (batch_size, height, width))
    # Degraded version (e.g., after compression or smaller model)
    degraded_pred = predictions
    
    iou_deg = IoUDegradation(num_classes=num_classes)
    result = iou_deg(ground_truth, baseline_pred, degraded_pred)
    baseline_iou = result['iou_base']
    degraded_iou = result['iou_perturbed'] 
    delta = result['delta_iou']
    print(f"Baseline IoU: {baseline_iou.mean():.4f}")
    print(f"Degraded IoU: {degraded_iou.mean():.4f}")
    print(f"IoU Degradation: {delta.mean():.4f}")
    print("(Positive values indicate degradation)")
    print()
    
    # Label consistency for temporal analysis
    print("Label Consistency:")
    print("-" * 20)
    
    # Two consecutive frames
    frame1 = targets
    frame2 = predictions
    
    label_cons = LabelConsistency()
    consistency = label_cons(frame2, frame1)
    print(f"Label Consistency: {consistency:.4f}")
    print("(Higher values indicate better temporal consistency)")
    print()
    
    print("All segmentation metrics computed successfully!")


if __name__ == '__main__':
    main()