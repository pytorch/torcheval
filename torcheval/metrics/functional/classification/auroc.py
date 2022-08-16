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
) -> torch.Tensor:
    """
    Compute AUROC, which is the area under the ROC Curve, for binary classification.
    Its class version is ``torcheval.metrics.BinaryAUROC``.

    Args:
        input (Tensor): Tensor of label predictions
            It should be predicted label, probabilities or logits with shape of (n_sample, ),
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).

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
    """
    _auroc_update(input, target)
    return _auroc_compute(input, target)


def _auroc_update(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    _auroc_update_input_check(input, target)


@torch.jit.script
def _auroc_compute(
    input: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    threshold, indices = input.sort(descending=True)
    mask = F.pad(threshold.diff(dim=0) != 0, [0, 1], value=1.0)
    cum_tp = F.pad(target[indices].cumsum(0)[mask], pad=[1, 0], value=0.0)
    cum_fp = F.pad((1 - target[indices]).cumsum(0)[mask], pad=[1, 0], value=0.0)

    # Set AUROC to 0.5 when the target contains all ones or all zeros.
    factor = cum_tp[-1] * cum_fp[-1]
    auroc = torch.where(
        factor == 0,
        0.5,
        torch.trapz(cum_tp, cum_fp).double() / factor,
    )
    return auroc


def _auroc_update_input_check(
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
