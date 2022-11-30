# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torcheval.metrics.functional.classification.precision_recall_curve import (
    _compute_for_each_class,
)


def _reimann_integral(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reimann integral approximates the area of each cell with a rectangle positioned at the egde.
    It is conventionally used rather than trapezoid approximation, which uses a rectangle positioned in the
    center, for PRAUC"""
    return -torch.sum((x[1:] - x[:-1]) * y[:-1])


@torch.inference_mode()
def binary_auprc(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_tasks: int = 1,
) -> torch.Tensor:
    """
    Compute AUPRC, also called Average Precision, which is the area under the Precision-Recall Curve, for binary classification.
    Its class version is ``torcheval.metrics.BinaryAUPRC``.

    Precision is defined as :math:`\frac{T_p}{T_p+F_p}`, it is the probability that a positive prediction from the model is a true positive.
    Recall is defined as :math:`\frac{T_p}{T_p+F_n}`, it is the probability that a true positive is predicted to be positive by the model.

    The precision-recall curve plots the recall on the x axis and the precision on the y axis, both of which are bounded between 0 and 1.
    This function returns the area under that graph. If the area is near one, the model supports a threshold which correctly identifies
    a high percentage of true positives while also rejecting enough false examples so that most of the true predictions are true positives.

    Args:
        input (Tensor): Tensor of label predictions
            It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).
        target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).
        num_tasks (int):  Number of tasks that need BinaryAUPRC calculation. Default value
                    is 1. Binary AUPRC for each task will be calculated independently. Results are
                    equivelent to calling binary_auprc for each row.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_auprc
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> binary_auprc(input, target)
        tensor(0.9167) # scalar returned with 1D input tensors

        >>> input = torch.tensor([[1, 1, 1, 0]])
        >>> target = torch.tensor([[1, 0, 1, 0]])
        >>> binary_auprc(input, target)
        tensor([0.6667]) # 1D tensor returned with 2D input tensors

        >>> input = torch.tensor([[0.1, 0.5, 0.7, 0.8],
        >>>                       [1, 1, 1, 0]])
        >>> target = torch.tensor([[1, 0, 1, 1],
        >>>                        [1, 0, 1, 0]])
        >>> binary_auprc(input, target, num_tasks=2)
        tensor([0.9167, 0.6667])
    """
    _binary_auprc_update_input_check(input, target, num_tasks)
    return _binary_auprc_compute(input, target, num_tasks)


def _binary_auprc_compute(
    input: torch.Tensor, target: torch.Tensor, num_tasks: int = 1
) -> torch.Tensor:
    # for one task preserve the ndim of the input and target tensor
    if num_tasks == 1 and input.ndim == 1:
        p, r, t = _compute_for_each_class(input, target, 1)
        return _reimann_integral(r, p)
    else:
        auprcs = []
        for i in range(num_tasks):
            p, r, t = _compute_for_each_class(input[i, :], target[i, :], 1)
            auprcs.append(_reimann_integral(r, p))
        return torch.tensor(auprcs, device=input.device)


def _binary_auprc_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    num_tasks: int,
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
