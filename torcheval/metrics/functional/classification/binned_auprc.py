# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Union

import torch
from torcheval.metrics.functional.classification.binned_precision_recall_curve import (
    _binary_binned_precision_recall_curve_compute,
    _binary_binned_precision_recall_curve_update,
)
from torcheval.metrics.functional.tensor_utils import (
    _create_threshold_tensor,
    _riemann_integral,
)

DEFAULT_NUM_THRESHOLD = 100


@torch.inference_mode()
def binary_binned_auprc(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_tasks: int = 1,
    threshold: Union[int, List[float], torch.Tensor] = DEFAULT_NUM_THRESHOLD,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Binned Version of AUPRC, which is the area under the AUPRC Curve, for binary classification.
    Its class version is ``torcheval.metrics.BinaryBinnedAUPRC``.

    Computation is done by computing the area under the precision/recall curve; precision and recall
    are computed for the buckets defined by `threshold`.

    Args:
        input (Tensor): Tensor of label predictions
            It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).

        target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).

        num_tasks (int):  Number of tasks that need binary_binned_auroc calculation. Default value
                    is 1. binary_binned_auroc for each task will be calculated independently.

        threshold (Tensor, int, List[float]): Either an integer representing the number of bins, a list of thresholds, or a tensor of thresholds.
                    The same thresholds will be used for all tasks.
                    If `threshold` is a tensor, it must be 1D.
                    If list or tensor is given, the first element must be 0 and the last must be 1.

    Examples:

        >>> import torch
        >>> from torcheval.metrics.functional import binary_binned_auroc
        >>> input = torch.tensor([0.2, 0.3, 0.4, 0.5])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> binary_binned_auroc(input, target, threshold=5)
        (tensor(1.0),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

        >>> input = torch.tensor([0.2, 0.3, 0.4, 0.5])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])
        >>> binary_binned_auroc(input, target, threshold=threshold)
        (tensor(0.6667),
        tensor([0.0000, 0.2500, 0.7500, 1.0000]))

        >>> input = torch.tensor([[0.2, 0.3, 0.4, 0.5], [0, 1, 2, 3]])
        >>> target = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        >>> threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])
        >>> binary_binned_auroc(input, target, num_tasks=2, threshold=threshold)
        (tensor([0.6667, 1.0000],
        tensor([0.0000, 0.2500, 0.7500, 1.0000]))
    """
    threshold = _create_threshold_tensor(threshold, target.device)
    _binary_binned_auprc_param_check(num_tasks, threshold)
    _binary_binned_auprc_update_input_check(input, target, num_tasks, threshold)
    return _binary_binned_auprc_compute(input, target, num_tasks, threshold)


def _binary_binned_auprc_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    num_tasks: int,
    threshold: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_tasks == 1 and input.ndim == 1:
        num_tp, num_fp, num_fn = _binary_binned_precision_recall_curve_update(
            input, target, threshold
        )
        precision, recall, _ = _binary_binned_precision_recall_curve_compute(
            num_tp, num_fp, num_fn, threshold
        )
        auprc_result = _riemann_integral(recall, precision)
    else:
        auprcs = []
        for i in range(num_tasks):
            num_tp, num_fp, num_fn = _binary_binned_precision_recall_curve_update(
                input[i, :], target[i, :], threshold
            )
            precision, recall, _ = _binary_binned_precision_recall_curve_compute(
                num_tp, num_fp, num_fn, threshold
            )
            auprcs.append(_riemann_integral(recall, precision))
        auprc_result = torch.tensor(auprcs, device=input.device)
    auprc_result = torch.nan_to_num(auprc_result, nan=0.0)
    return auprc_result, threshold


def _binary_binned_auprc_param_check(
    num_tasks: int,
    threshold: torch.Tensor,
) -> None:
    if num_tasks < 1:
        raise ValueError("`num_tasks` has to be at least 1.")

    if threshold.ndim != 1:
        raise ValueError(
            f"`threshold` should be 1-dimensional, but got {threshold.ndim}D tensor."
        )

    if (torch.diff(threshold) < 0.0).any():
        raise ValueError("The `threshold` should be a sorted tensor.")

    if (threshold < 0.0).any() or (threshold > 1.0).any():
        raise ValueError("The values in `threshold` should be in the range of [0, 1].")

    if threshold[0] != 0:
        raise ValueError("First value in `threshold` should be 0.")

    if threshold[-1] != 1:
        raise ValueError("Last value in `threshold` should be 1.")


def _binary_binned_auprc_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    num_tasks: int,
    threshold: torch.Tensor,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same shape, "
            f"got shapes {input.shape} and {target.shape}."
        )
    if num_tasks == 1:
        if input.ndim == 2 and input.shape[0] > 1:
            raise ValueError(
                f"`num_tasks = 1`, `input` and `target` are expected to be one-dimensional tensors or 1xN tensors, but got shape input: {input.shape}, target: {target.shape}."
            )
        elif input.ndim > 2:
            raise ValueError(
                f"`num_tasks = 1`, `input` and `target` are expected to be one-dimensional tensors or 1xN tensors, but got shape input: {input.shape}, target: {target.shape}."
            )
    elif input.shape[0] != num_tasks:
        raise ValueError(
            f"`num_tasks = {num_tasks}`, `input` and `target` shape is expected to be ({num_tasks}, num_samples), but got shape input: {input.shape}, target: {target.shape}."
        )
    elif threshold.ndim != 1:
        raise ValueError(
            f"`threshold` should be 1-dimensional, but got {threshold.ndim}D tensor."
        )
