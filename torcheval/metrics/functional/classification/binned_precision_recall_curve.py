# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Union

import torch


@torch.inference_mode()
def binary_binned_precision_recall_curve(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: Union[int, List[float], torch.Tensor] = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute precision recall curve with given thresholds.
    Its class version is ``torcheval.metrics.BinnedPrecisionRecallCurve``.

    Args:
        input: Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, ).
        target: Tensor of ground truth labels with shape of (n_samples, ).
        threshold:
            a integer representing number of bins, a list of thresholds,
            or a tensor of thresholds.

    Return:
        a tuple of (precision: torch.Tensor, recall: torch.Tensor, thresholds: torch.Tensor)
            precision: Tensor of precision result. Its shape is (n_thresholds + 1, )
            recall: Tensor of recall result. Its shape is (n_thresholds + 1, )
            thresholds: Tensor of threshold. Its shape is (n_thresholds, )

    Example:
        >>> import torch
        >>> from torcheval.metrics.functional import binned_precision_recall_curve
        >>> input = torch.tensor([0.2, 0.8, 0.5, 0.9])
        >>> target = torch.tensor([0, 1, 0, 1])
        >>> threshold = 5
        >>> binned_precision_recall_curve(input, target, threshold)
        (tensor([0.5000, 0.6667, 0.6667, 1.0000, 1.0000, 1.0000]),
        tensor([1., 1., 1., 1., 0., 0.]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

        >>> input = torch.tensor([0.2, 0.3, 0.4, 0.5])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])
        >>> binned_precision_recall_curve(input, target, threshold)
        (tensor([0.5000, 0.6667, 1.0000, 1.0000, 1.0000]),
        tensor([1., 1., 0., 0., 0.]),
        tensor([0.0000, 0.2500, 0.7500, 1.0000]))
    """
    if isinstance(threshold, int):
        threshold = torch.linspace(0, 1.0, threshold, device=target.device)
    elif isinstance(threshold, list):
        threshold = torch.tensor(threshold, device=target.device)
    num_tp, num_fp, num_fn = _binary_binned_precision_recall_curve_update(
        input, target, threshold
    )
    return _binary_binned_compute_for_each_class(num_tp, num_fp, num_fn, threshold)


def _binary_binned_precision_recall_curve_update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _binary_binned_precision_recall_curve_update_input_check(input, target, threshold)
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


def _binary_binned_precision_recall_curve_compute(
    num_tp: torch.Tensor,
    num_fp: torch.Tensor,
    num_fn: torch.Tensor,
    threshold: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _binary_binned_compute_for_each_class(num_tp, num_fp, num_fn, threshold)


@torch.jit.script
def _binary_binned_compute_for_each_class(
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


def _binary_binned_precision_recall_curve_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: torch.Tensor,
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

    if (torch.diff(threshold) < 0.0).any():
        raise ValueError("The `threshold` should be a sorted array.")

    if (threshold < 0.0).any() or (threshold > 1.0).any():
        raise ValueError("The values in `threshold` should be in the range of [0, 1].")
