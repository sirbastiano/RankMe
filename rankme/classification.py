"""Classification metrics for evaluating model performance on discrete prediction tasks."""

from typing import Any, Literal, Optional, Union

import torch
import torch.nn.functional as F

from rankme.base import StatelessMetric, check_same_shape, to_onehot


class Accuracy(StatelessMetric):
    """Compute accuracy for classification tasks.

    Supports binary, multiclass, and multilabel classification tasks.

    Args:
        task: Type of classification task ('binary', 'multiclass', 'multilabel').
        num_classes: Number of classes for multiclass/multilabel tasks.
        threshold: Threshold for binary/multilabel classification (default: 0.5).
        top_k: If specified, computes top-k accuracy (predictions are correct
               if true class is among top k predicted classes).

    Example:
        >>> import torch
        >>> from rankme.classification import Accuracy
        >>>
        >>> # Multiclass accuracy
        >>> y_true = torch.tensor([0, 1, 2, 1, 0])
        >>> y_pred = torch.tensor([0, 1, 2, 2, 0])
        >>> acc = Accuracy(task='multiclass', num_classes=3)
        >>> accuracy = acc(y_pred, y_true)
        >>> float(accuracy)
        0.8

        >>> # Top-k accuracy
        >>> logits = torch.randn(5, 3)
        >>> acc_top2 = Accuracy(task='multiclass', num_classes=3, top_k=2)
        >>> top2_acc = acc_top2(logits, y_true)
    """

    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"],
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.task = task
        self.num_classes = num_classes
        self.threshold = threshold
        self.top_k = top_k

        if task in ["multiclass", "multilabel"] and num_classes is None:
            raise ValueError(f"num_classes must be specified for task={task}")

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute accuracy.

        Args:
            preds: Predictions tensor. For binary/multilabel: probabilities or logits.
                   For multiclass: class predictions or logits.
            target: Ground truth tensor. Same shape as preds for multilabel,
                    or class indices for multiclass/binary.

        Returns:
            torch.Tensor: Scalar accuracy value.
        """
        if self.task == "binary":
            if preds.dim() > 1 and preds.size(-1) > 1:
                # Convert logits/probabilities to predictions
                preds = (torch.sigmoid(preds) >= self.threshold).float()
            else:
                preds = (preds >= self.threshold).float()
            target = target.float()

        elif self.task == "multiclass":
            if self.top_k is not None:
                # Top-k accuracy
                assert preds.dim() == 2, "For top-k accuracy, preds must be 2D (logits)"
                _, top_k_preds = preds.topk(self.top_k, dim=1)
                target_expanded = target.unsqueeze(1).expand_as(top_k_preds)
                correct = (top_k_preds == target_expanded).any(dim=1)
                return correct.float().mean()
            else:
                if preds.dim() == 2:
                    # Convert logits to predictions
                    preds = preds.argmax(dim=1)

        elif self.task == "multilabel":
            if preds.dim() == target.dim():
                # Probabilities or logits
                preds = (torch.sigmoid(preds) >= self.threshold).float()
            target = target.float()

        check_same_shape(preds, target)
        correct = (preds == target).float()

        if self.task == "multilabel":
            # For multilabel, compute subset accuracy (all labels must be correct)
            return (correct.sum(dim=1) == target.size(1)).float().mean()
        else:
            return correct.mean()


class Precision(StatelessMetric):
    """Compute precision for classification tasks.

    Precision = TP / (TP + FP)

    Args:
        task: Type of classification task ('binary', 'multiclass', 'multilabel').
        num_classes: Number of classes for multiclass/multilabel tasks.
        average: Averaging method for multiclass ('micro', 'macro', 'weighted', 'none').
        threshold: Threshold for binary/multilabel classification.
        zero_division: Value to return when there are no positive predictions.

    Example:
        >>> import torch
        >>> from rankme.classification import Precision
        >>>
        >>> y_true = torch.tensor([0, 1, 2, 1, 0])
        >>> y_pred = torch.tensor([0, 1, 2, 2, 1])
        >>> precision = Precision(task='multiclass', num_classes=3, average='macro')
        >>> prec_value = precision(y_pred, y_true)
    """

    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"],
        num_classes: Optional[int] = None,
        average: Literal["micro", "macro", "weighted", "none"] = "macro",
        threshold: float = 0.5,
        zero_division: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.task = task
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.zero_division = zero_division

        if task in ["multiclass", "multilabel"] and num_classes is None:
            raise ValueError(f"num_classes must be specified for task={task}")

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute precision.

        Args:
            preds: Predictions tensor.
            target: Ground truth tensor.

        Returns:
            torch.Tensor: Precision value(s).
        """
        preds, target = self._format_inputs(preds, target)

        if self.task == "binary":
            tp = ((preds == 1) & (target == 1)).sum().float()
            fp = ((preds == 1) & (target == 0)).sum().float()
            precision = tp / (tp + fp) if (tp + fp) > 0 else self.zero_division
            if isinstance(precision, torch.Tensor):
                return precision.detach().clone()
            else:
                return torch.tensor(precision, device=preds.device, dtype=torch.float32)

        elif self.task in ["multiclass", "multilabel"]:
            return self._compute_multiclass_precision(preds, target)

    def _format_inputs(self, preds: torch.Tensor, target: torch.Tensor):
        """Format inputs based on task type."""
        if self.task == "binary":
            if preds.dim() > 1 and preds.size(-1) > 1:
                preds = torch.sigmoid(preds)
            preds = (preds >= self.threshold).long()
            target = target.long()

        elif self.task == "multiclass":
            if preds.dim() == 2:
                preds = preds.argmax(dim=1)
            preds = preds.long()
            target = target.long()

        elif self.task == "multilabel":
            if preds.dim() == target.dim():
                preds = torch.sigmoid(preds)
            preds = (preds >= self.threshold).long()
            target = target.long()

        return preds, target

    def _compute_multiclass_precision(self, preds: torch.Tensor, target: torch.Tensor):
        """Compute precision for multiclass/multilabel tasks."""
        if self.task == "multiclass":
            # Convert to one-hot for easier computation
            assert self.num_classes is not None
            preds_onehot = to_onehot(preds, self.num_classes)
            target_onehot = to_onehot(target, self.num_classes)
        else:
            preds_onehot = preds.float()
            target_onehot = target.float()

        tp = (preds_onehot * target_onehot).sum(dim=0)
        fp = (preds_onehot * (1 - target_onehot)).sum(dim=0)

        precision = tp / (tp + fp)
        precision = torch.where(
            (tp + fp) == 0,
            torch.tensor(self.zero_division, device=precision.device),
            precision,
        )

        if self.average == "micro":
            return (
                tp.sum() / (tp.sum() + fp.sum())
                if (tp.sum() + fp.sum()) > 0
                else torch.tensor(self.zero_division)
            )
        elif self.average == "macro":
            return precision.mean()
        elif self.average == "weighted":
            support = target_onehot.sum(dim=0)
            return (
                (precision * support).sum() / support.sum()
                if support.sum() > 0
                else torch.tensor(self.zero_division)
            )
        else:  # 'none'
            return precision


class Recall(StatelessMetric):
    """Compute recall (sensitivity) for classification tasks.

    Recall = TP / (TP + FN)

    Args:
        task: Type of classification task ('binary', 'multiclass', 'multilabel').
        num_classes: Number of classes for multiclass/multilabel tasks.
        average: Averaging method for multiclass ('micro', 'macro', 'weighted', 'none').
        threshold: Threshold for binary/multilabel classification.
        zero_division: Value to return when there are no positive samples.
    """

    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"],
        num_classes: Optional[int] = None,
        average: Literal["micro", "macro", "weighted", "none"] = "macro",
        threshold: float = 0.5,
        zero_division: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.task = task
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.zero_division = zero_division

        if task in ["multiclass", "multilabel"] and num_classes is None:
            raise ValueError(f"num_classes must be specified for task={task}")

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute recall.

        Args:
            preds: Predictions tensor.
            target: Ground truth tensor.

        Returns:
            torch.Tensor: Recall value(s).
        """
        preds, target = self._format_inputs(preds, target)

        if self.task == "binary":
            tp = ((preds == 1) & (target == 1)).sum().float()
            fn = ((preds == 0) & (target == 1)).sum().float()
            recall = tp / (tp + fn) if (tp + fn) > 0 else self.zero_division
            if isinstance(recall, torch.Tensor):
                return recall.detach().clone()
            else:
                return torch.tensor(recall, device=preds.device, dtype=torch.float32)

        elif self.task in ["multiclass", "multilabel"]:
            return self._compute_multiclass_recall(preds, target)

    def _format_inputs(self, preds: torch.Tensor, target: torch.Tensor):
        """Format inputs based on task type."""
        if self.task == "binary":
            if preds.dim() > 1 and preds.size(-1) > 1:
                preds = torch.sigmoid(preds)
            preds = (preds >= self.threshold).long()
            target = target.long()

        elif self.task == "multiclass":
            if preds.dim() == 2:
                preds = preds.argmax(dim=1)
            preds = preds.long()
            target = target.long()

        elif self.task == "multilabel":
            if preds.dim() == target.dim():
                preds = torch.sigmoid(preds)
            preds = (preds >= self.threshold).long()
            target = target.long()

        return preds, target

    def _compute_multiclass_recall(self, preds: torch.Tensor, target: torch.Tensor):
        """Compute recall for multiclass/multilabel tasks."""
        if self.task == "multiclass":
            assert self.num_classes is not None
            preds_onehot = to_onehot(preds, self.num_classes)
            target_onehot = to_onehot(target, self.num_classes)
        else:
            preds_onehot = preds.float()
            target_onehot = target.float()

        tp = (preds_onehot * target_onehot).sum(dim=0)
        fn = ((1 - preds_onehot) * target_onehot).sum(dim=0)

        recall = tp / (tp + fn)
        recall = torch.where(
            (tp + fn) == 0,
            torch.tensor(self.zero_division, device=recall.device),
            recall,
        )

        if self.average == "micro":
            return (
                tp.sum() / (tp.sum() + fn.sum())
                if (tp.sum() + fn.sum()) > 0
                else torch.tensor(self.zero_division)
            )
        elif self.average == "macro":
            return recall.mean()
        elif self.average == "weighted":
            support = target_onehot.sum(dim=0)
            return (
                (recall * support).sum() / support.sum()
                if support.sum() > 0
                else torch.tensor(self.zero_division)
            )
        else:  # 'none'
            return recall


class F1Score(StatelessMetric):
    """Compute F1-score for classification tasks.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        task: Type of classification task ('binary', 'multiclass', 'multilabel').
        num_classes: Number of classes for multiclass/multilabel tasks.
        average: Averaging method for multiclass ('micro', 'macro', 'weighted', 'none').
        threshold: Threshold for binary/multilabel classification.
        zero_division: Value to return when precision + recall = 0.
    """

    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"],
        num_classes: Optional[int] = None,
        average: Literal["micro", "macro", "weighted", "none"] = "macro",
        threshold: float = 0.5,
        zero_division: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.precision = Precision(
            task=task,
            num_classes=num_classes,
            average=average,
            threshold=threshold,
            zero_division=zero_division,
        )
        self.recall = Recall(
            task=task,
            num_classes=num_classes,
            average=average,
            threshold=threshold,
            zero_division=zero_division,
        )
        self.zero_division = zero_division

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute F1-score.

        Args:
            preds: Predictions tensor.
            target: Ground truth tensor.

        Returns:
            torch.Tensor: F1-score value(s).
        """
        precision = self.precision(preds, target)
        recall = self.recall(preds, target)

        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = torch.where(
            (precision + recall) == 0,
            torch.tensor(self.zero_division, device=f1.device),
            f1,
        )

        return f1


class IoU(StatelessMetric):
    """Compute Intersection over Union (Jaccard Index) for classification tasks.

    IoU = TP / (TP + FP + FN)

    Commonly used for semantic segmentation and object detection evaluation.

    Args:
        task: Type of classification task ('binary', 'multiclass', 'multilabel').
        num_classes: Number of classes for multiclass/multilabel tasks.
        average: Averaging method for multiclass ('micro', 'macro', 'weighted', 'none').
        threshold: Threshold for binary/multilabel classification.
        ignore_index: Class index to ignore in computation.
        zero_division: Value to return when union is zero.

    Example:
        >>> import torch
        >>> from rankme.classification import IoU
        >>>
        >>> # Binary IoU (e.g., for segmentation)
        >>> y_true = torch.tensor([[1, 1, 0], [0, 1, 1]])
        >>> y_pred = torch.tensor([[1, 0, 0], [1, 1, 1]])
        >>> iou = IoU(task='binary')
        >>> iou_value = iou(y_pred.flatten(), y_true.flatten())
    """

    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"],
        num_classes: Optional[int] = None,
        average: Literal["micro", "macro", "weighted", "none"] = "macro",
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        zero_division: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.task = task
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.zero_division = zero_division

        if task in ["multiclass", "multilabel"] and num_classes is None:
            raise ValueError(f"num_classes must be specified for task={task}")

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute IoU.

        Args:
            preds: Predictions tensor.
            target: Ground truth tensor.

        Returns:
            torch.Tensor: IoU value(s).
        """
        preds, target = self._format_inputs(preds, target)

        if self.task == "binary":
            intersection = ((preds == 1) & (target == 1)).sum().float()
            union = ((preds == 1) | (target == 1)).sum().float()
            iou = intersection / union if union > 0 else self.zero_division
            if isinstance(iou, torch.Tensor):
                return iou.detach().clone()
            else:
                return torch.tensor(iou, device=preds.device, dtype=torch.float32)

        elif self.task in ["multiclass", "multilabel"]:
            return self._compute_multiclass_iou(preds, target)

    def _format_inputs(self, preds: torch.Tensor, target: torch.Tensor):
        """Format inputs based on task type."""
        if self.task == "binary":
            if preds.dim() > 1 and preds.size(-1) > 1:
                preds = torch.sigmoid(preds)
            preds = (preds >= self.threshold).long()
            target = target.long()

        elif self.task == "multiclass":
            if preds.dim() == 2:
                preds = preds.argmax(dim=1)
            preds = preds.long()
            target = target.long()

        elif self.task == "multilabel":
            if preds.dim() == target.dim():
                preds = torch.sigmoid(preds)
            preds = (preds >= self.threshold).long()
            target = target.long()

        return preds, target

    def _compute_multiclass_iou(self, preds: torch.Tensor, target: torch.Tensor):
        """Compute IoU for multiclass/multilabel tasks."""
        if self.task == "multiclass":
            assert self.num_classes is not None
            preds_onehot = to_onehot(preds, self.num_classes)
            target_onehot = to_onehot(target, self.num_classes)
        else:
            preds_onehot = preds.float()
            target_onehot = target.float()

        intersection = (preds_onehot * target_onehot).sum(dim=0)
        union = (preds_onehot + target_onehot - preds_onehot * target_onehot).sum(dim=0)

        iou = intersection / union
        iou = torch.where(
            union == 0, torch.tensor(self.zero_division, device=iou.device), iou
        )

        # Handle ignore_index
        if self.ignore_index is not None and self.task == "multiclass":
            mask = torch.ones_like(iou, dtype=torch.bool)
            mask[self.ignore_index] = False
            iou = iou[mask]

        if self.average == "micro":
            return (
                intersection.sum() / union.sum()
                if union.sum() > 0
                else torch.tensor(self.zero_division)
            )
        elif self.average == "macro":
            return iou.mean()
        elif self.average == "weighted":
            support = target_onehot.sum(dim=0)
            if self.ignore_index is not None and self.task == "multiclass":
                support = support[mask]
            return (
                (iou * support).sum() / support.sum()
                if support.sum() > 0
                else torch.tensor(self.zero_division)
            )
        else:  # 'none'
            return iou
