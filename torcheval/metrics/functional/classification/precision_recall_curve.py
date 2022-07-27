# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

"""
This file contains binary_precision_recall_curve and multiclass_precision_recall_curve functions.
"""


@torch.inference_mode()
def binary_precision_recall_curve(
    input: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute precision recall curve, which is precision-recall pair with corresponding thresholds,
        for binary classification tasks.
    Its class version is ``torcheval.metrics.BinaryPrecisionRecallCurve``.

    Args:
        input: Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, ).
        target: Tensor of ground truth labels with shape of (n_samples, ).

    Return:
        a tuple of (precision: torch.Tensor, recall: torch.Tensor, thresholds: torch.Tensor)
            precision: Tensor of precision result. Its shape is (n_thresholds + 1, )
            recall: Tensor of recall result. Its shape is (n_thresholds + 1, )
            thresholds: Tensor of threshold. Its shape is (n_thresholds, )

    Example:
        >>> import torch
        >>> from torcheval.metrics.functional import binary_precision_recall_curve
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> binary_precision_recall_curve(input, target)
        (tensor([1., 1., 1.]),
        tensor([1.0000, 0.5000, 0.0000]),
        tensor([0.7000, 0.8000]))
    """
    _binary_precision_recall_curve_update(input, target)
    return _binary_precision_recall_curve_compute(input, target)


def _binary_precision_recall_curve_update(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    _binary_precision_recall_curve_update_input_check(input, target)


def _binary_precision_recall_curve_compute(
    input: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    precision, recall, threshold = _compute_for_each_class(input, target, 1)

    return precision, recall, threshold


def _binary_precision_recall_curve_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if input.ndim != 1:
        raise ValueError(
            "input should be a one-dimensional tensor, " f"got shape {input.shape}."
        )

    if target.ndim != 1:
        raise ValueError(
            "target should be a one-dimensional tensor, " f"got shape {target.shape}."
        )

    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same shape, "
            f"got shapes {input.shape} and {target.shape}."
        )


@torch.inference_mode()
def multiclass_precision_recall_curve(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute precision recall curve, which is precision-recall pair with corresponding thresholds,
        for multi-class classification tasks.
    Its class version is ``torcheval.metrics.MulticlassPrecisionRecallCurve``.

    Args:
        input: Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, n_class).
        target: Tensor of ground truth labels with shape of (n_samples, ).
        num_classes (Optional):
            Number of classes. Set to the second dimension of the input if num_classes is None.

    Return:
        a tuple of (precision: List[torch.Tensor], recall: List[torch.Tensor], thresholds: List[torch.Tensor])
            precision: List of precision result. Each index indicates the result of a class.
            recall: List of recall result. Each index indicates the result of a class.
            thresholds: List of threshold. Each index indicates the result of a class.

    Example:
        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_precision_recall_curve
        >>> input = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> precision_recall_curve(input, target, num_classes=4)
        ([tensor([0.2500, 0.0000, 0.0000, 0.0000, 1.0000]),
        tensor([0.3333, 0.0000, 0.0000, 1.0000]),
        tensor([0.5000, 0.0000, 1.0000]),
        tensor([1., 1.])],
        [tensor([1., 0., 0., 0., 0.]),
        tensor([1., 0., 0., 0.]),
        tensor([1., 0., 0.]),
        tensor([1., 0.])],
        [tensor([0.1000, 0.5000, 0.7000, 0.8000]),
        tensor([0.5000, 0.7000, 0.8000]),
        tensor([0.7000, 0.8000]),
        tensor([0.8000])])
    """
    if num_classes is None and input.ndim == 2:
        num_classes = input.shape[1]
    _multiclass_precision_recall_curve_update(input, target, num_classes)
    return _multiclass_precision_recall_curve_compute(input, target, num_classes)


def _multiclass_precision_recall_curve_update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int],
) -> None:
    _multiclass_precision_recall_curve_update_input_check(input, target, num_classes)


def _multiclass_precision_recall_curve_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    if num_classes is None and input.ndim == 2:
        num_classes = input.shape[1]
    num_unique_classes = torch.unique(target)
    if len(num_unique_classes) != num_classes:
        logging.warning(
            "Warning: Some classes do not exist in the target. Precision Recall Curve for these classes will be straight lines from (1, 0) to (0, 1)."
        )

    precisions, recalls, thresholds = [], [], []
    assert isinstance(num_classes, int)
    for class_idx in range(num_classes):
        precision, recall, threshold = _compute_for_each_class(
            input[:, class_idx], target, class_idx
        )
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(threshold)
    return (
        precisions,
        recalls,
        thresholds,
    )


def _multiclass_precision_recall_curve_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int],
) -> None:
    if input.size(0) != target.size(0):
        raise ValueError(
            "The `input` and `target` should have the same first dimension, "
            f"got shapes {input.shape} and {target.shape}."
        )

    if target.ndim != 1:
        raise ValueError(
            "target should be a one-dimensional tensor, " f"got shape {target.shape}."
        )

    if not (input.ndim == 2 and (num_classes is None or input.shape[1] == num_classes)):
        raise ValueError(
            f"input should have shape of (num_sample, num_classes), "
            f"got {input.shape} and num_classes={num_classes}."
        )


def _compute_for_each_class(
    input: torch.Tensor,
    target: torch.Tensor,
    pos_label: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    threshold, indices = input.sort(descending=True)
    mask = F.pad(threshold.diff(dim=0) != 0, [0, 1], value=True)
    num_tp = (target[indices] == pos_label).cumsum(0)[mask]
    num_fp = (1 - (target[indices] == pos_label).long()).cumsum(0)[mask]
    precision = num_tp / (num_tp + num_fp)
    recall = num_tp / num_tp[-1]

    # Remove redundant thresholds that result in a recall of 1.0.
    last_ind = torch.searchsorted(num_tp, num_tp[-1])
    sl = slice(last_ind + 1)
    precision = precision[sl].flip(0)
    recall = recall[sl].flip(0)
    threshold = threshold[mask][sl].flip(0)

    # The last precision and recall values are 1.0 and 0.0 without a corresponding threshold.
    # This ensures that the graph starts on the y-axis.
    precision = torch.cat([precision, precision.new_ones(1)])
    recall = torch.cat([recall, recall.new_zeros(1)])

    # If all recalls are NaN, the curve will be a straight line from (1, 0) to (0, 1).
    if torch.isnan(recall[0]):
        precision = torch.tensor([0.0, 1.0], device=precision.device)
        recall = torch.tensor([1.0, 0.0], device=recall.device)

    return precision, recall, threshold
