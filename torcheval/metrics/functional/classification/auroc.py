# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F


@torch.inference_mode()
def binary_auroc(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_tasks: int = 1,
) -> torch.Tensor:
    """
    Compute AUROC, which is the area under the ROC Curve, for binary classification.
    Its class version is ``torcheval.metrics.BinaryAUROC``.

    Args:
        input (Tensor): Tensor of label predictions
            It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).
        target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).
        num_tasks (int):  Number of tasks that need BinaryAUROC calculation. Default value
                    is 1. BinaryAUROC for each task will be calculated independently.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_auroc
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> binary_auroc(input, target)
        tensor(0.6667)

        >>> input = torch.tensor([1, 1, 1, 0])
        >>> target = torch.tensor([1, 0, 1, 0])
        >>> binary_auroc(input, target)
        tensor(0.7500)

        >>> input = torch.tensor([[1, 1, 1, 0], [0.1, 0.5, 0.7, 0.8]])
        >>> target = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 1]])
        >>> binary_auroc(input, target, num_tasks=2)
        tensor([0.7500, 0.6667])
    """
    _auroc_update(input, target, num_tasks)
    return _auroc_compute(input, target)


def _auroc_update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_tasks: int,
) -> None:
    _auroc_update_input_check(input, target, num_tasks)


@torch.jit.script
def _auroc_compute(
    input: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    threshold, indices = input.sort(descending=True)
    mask = F.pad(threshold.diff(dim=-1) != 0, [0, 1], value=1.0)
    sorted_target = torch.gather(target, -1, indices)
    cum_tp_before_pad = sorted_target.cumsum(-1) * mask
    cum_fp_before_pad = (1 - sorted_target).cumsum(-1) * mask
    if len(mask.shape) > 1:
        cum_tp = F.pad(cum_tp_before_pad, pad=[1, 0], value=0.0)
        cum_fp = F.pad(cum_fp_before_pad, pad=[1, 0], value=0.0)
        factor = cum_tp[:, -1] * cum_fp[:, -1]
    else:
        cum_tp = F.pad(cum_tp_before_pad[mask], pad=[1, 0], value=0.0)
        cum_fp = F.pad(cum_fp_before_pad[mask], pad=[1, 0], value=0.0)
        factor = cum_tp[-1] * cum_fp[-1]

    # Set AUROC to 0.5 when the target contains all ones or all zeros.
    auroc = torch.where(
        factor == 0,
        0.5,
        torch.trapz(cum_tp, cum_fp).double() / factor,
    )
    return auroc


def _auroc_update_input_check(
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
        if len(input.shape) > 1:
            raise ValueError(
                f"`num_tasks = 1`, `input` is expected to be one-dimensional tensor, but got shape ({input.shape})."
            )
    elif len(input.shape) == 1 or input.shape[0] != num_tasks:
        raise ValueError(
            f"`num_tasks = {num_tasks}`, `input`'s shape is expected to be ({num_tasks}, num_samples), but got shape ({input.shape})."
        )
