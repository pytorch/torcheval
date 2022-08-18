# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F
from torcheval.metrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_update_input_check,
    _multiclass_precision_recall_curve_update_input_check,
)


@torch.inference_mode()
def binary_binned_precision_recall_curve(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: Union[int, List[float], torch.Tensor] = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute precision recall curve with given thresholds.
    Its class version is ``torcheval.metrics.BinaryBinnedPrecisionRecallCurve``.

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, ).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        threshold:
            a integer representing number of bins, a list of thresholds,
            or a tensor of thresholds.

    Returns:
        Tuple:
            - precision (Tensor): Tensor of precision result. Its shape is (n_thresholds + 1, )
            - recall (Tensor): Tensor of recall result. Its shape is (n_thresholds + 1, )
            - thresholds (Tensor): Tensor of threshold. Its shape is (n_thresholds, )

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_binned_precision_recall_curve
        >>> input = torch.tensor([0.2, 0.8, 0.5, 0.9])
        >>> target = torch.tensor([0, 1, 0, 1])
        >>> threshold = 5
        >>> binary_binned_precision_recall_curve(input, target, threshold)
        (tensor([0.5000, 0.6667, 0.6667, 1.0000, 1.0000, 1.0000]),
        tensor([1., 1., 1., 1., 0., 0.]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

        >>> input = torch.tensor([0.2, 0.3, 0.4, 0.5])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])
        >>> binary_binned_precision_recall_curve(input, target, threshold)
        (tensor([0.5000, 0.6667, 1.0000, 1.0000, 1.0000]),
        tensor([1., 1., 0., 0., 0.]),
        tensor([0.0000, 0.2500, 0.7500, 1.0000]))
    """
    threshold = _create_threshold_tensor(threshold, target.device)
    _binned_precision_recall_curve_param_check(threshold)
    num_tp, num_fp, num_fn = _binary_binned_precision_recall_curve_update(
        input, target, threshold
    )
    return _binary_binned_precision_recall_curve_compute(
        num_tp, num_fp, num_fn, threshold
    )


def _binary_binned_precision_recall_curve_update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _binary_precision_recall_curve_update_input_check(input, target)
    return _update(input, target, threshold)


@torch.jit.script
def _update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_label = input >= threshold[:, None]
    num_tp = (pred_label & target).sum(dim=1)
    num_fp = pred_label.sum(dim=1) - num_tp
    num_fn = target.sum() - num_tp
    return num_tp, num_fp, num_fn


@torch.jit.script
def _binary_binned_precision_recall_curve_compute(
    num_tp: torch.Tensor,
    num_fp: torch.Tensor,
    num_fn: torch.Tensor,
    threshold: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Set precision to 1.0 if all predictions are zeros.
    precision = torch.nan_to_num(num_tp / (num_tp + num_fp), 1.0)
    recall = num_tp / (num_tp + num_fn)

    # The last precision and recall values are 1.0 and 0.0 without a corresponding threshold.
    # This ensures that the graph starts on the y-axis.
    precision = torch.cat([precision, precision.new_ones(1)], dim=0)
    recall = torch.cat([recall, recall.new_zeros(1)], dim=0)

    return precision, recall, threshold


@torch.inference_mode()
def multiclass_binned_precision_recall_curve(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    threshold: Union[int, List[float], torch.Tensor] = 100,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Compute precision recall curve with given thresholds.
    Its class version is ``torcheval.metrics.MulticlassBinnedPrecisionRecallCurve``.

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, n_class).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        num_classes (Optional):
            Number of classes. Set to the second dimension of the input if num_classes is None.
        threshold:
            a integer representing number of bins, a list of thresholds,
            or a tensor of thresholds.

    Return:
        a tuple of (precision: List[torch.Tensor], recall: List[torch.Tensor], thresholds: torch.Tensor)
            precision: List of precision result. Each index indicates the result of a class.
            recall: List of recall result. Each index indicates the result of a class.
            thresholds: Tensor of threshold. The threshold is used for all classes.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_binned_precision_recall_curve
        >>> input = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_binned_precision_recall_curve(input, target, num_classes=4, threshold=5)
        ([tensor([0.2500, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.3333, 0.0000, 1.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.3333, 0.0000, 1.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.3333, 1.0000, 1.0000, 1.0000])],
        [tensor([1., 0., 0., 0., 0., 0.]),
        tensor([1., 1., 1., 0., 0., 0.]),
        tensor([1., 1., 1., 0., 0., 0.]),
        tensor([1., 1., 1., 1., 0., 0.])],
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

        >>> input = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        >>> multiclass_binned_precision_recall_curve(input, target, num_classes=4, threshold=threshold)
        ([tensor([0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.3333, 0.3333, 0.3333, 0.5000, 0.5000, 0.0000, 1.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.3333, 0.3333, 0.3333, 0.5000, 0.5000, 1.0000, 1.0000, 1.0000])],
        [tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        tensor([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]),
        tensor([1., 1., 1., 1., 1., 1., 1., 0., 0., 0.]),
        tensor([1., 1., 1., 1., 1., 1., 1., 1., 0., 0.])],
        tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000]))
    """
    threshold = _create_threshold_tensor(threshold, target.device)
    _binned_precision_recall_curve_param_check(threshold)

    if num_classes is None and input.ndim == 2:
        num_classes = input.shape[1]
    num_tp, num_fp, num_fn = _multiclass_binned_precision_recall_curve_update(
        input, target, num_classes, threshold
    )
    return _multiclass_binned_precision_recall_curve_compute(
        num_tp, num_fp, num_fn, num_classes, threshold
    )


def _multiclass_binned_precision_recall_curve_update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int],
    threshold: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _multiclass_precision_recall_curve_update_input_check(input, target, num_classes)
    labels = input >= threshold[:, None, None]
    target = F.one_hot(target, num_classes)
    num_tp = (labels & target).sum(dim=1)
    num_fp = labels.sum(dim=1) - num_tp
    num_fn = target.sum(dim=0) - num_tp

    return num_tp, num_fp, num_fn


def _multiclass_binned_precision_recall_curve_compute(
    num_tp: torch.Tensor,
    num_fp: torch.Tensor,
    num_fn: torch.Tensor,
    num_classes: Optional[int],
    threshold: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    # Set precision to 1.0 if all predictions are zeros.
    precision = torch.nan_to_num(num_tp / (num_tp + num_fp), 1.0)
    recall = num_tp / (num_tp + num_fn)

    # The last precision and recall values are 1.0 and 0.0 without a corresponding threshold.
    # This ensures that the graph starts on the y-axis.
    assert isinstance(num_classes, int)
    precision = torch.cat([precision, precision.new_ones(1, num_classes)], dim=0)
    recall = torch.cat([recall, recall.new_zeros(1, num_classes)], dim=0)

    return (
        list(precision.T),
        list(recall.T),
        threshold,
    )


def _create_threshold_tensor(
    threshold: Union[int, List[float], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if isinstance(threshold, int):
        threshold = torch.linspace(0, 1.0, threshold, device=device)
    elif isinstance(threshold, list):
        threshold = torch.tensor(threshold, device=device)
    return threshold


def _binned_precision_recall_curve_param_check(
    threshold: torch.Tensor,
) -> None:
    if (torch.diff(threshold) < 0.0).any():
        raise ValueError("The `threshold` should be a sorted array.")

    if (threshold < 0.0).any() or (threshold > 1.0).any():
        raise ValueError("The values in `threshold` should be in the range of [0, 1].")
