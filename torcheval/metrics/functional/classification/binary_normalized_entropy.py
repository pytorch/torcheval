# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@torch.inference_mode()
def binary_normalized_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    from_logits: bool = False,
) -> torch.Tensor:
    """
    Compute the normalized binary cross entropy between predicted input and
    ground-truth binary target.
    Its class version is ``torcheval.metrics.binary_normalized_entropy``

    Args:
        input (Tensor): Predicted unnormalized scores (often referred to as logits) or
                binary class probabilities (num_samples, ).
        target (Tensor): Ground truth binary class indices (num_samples, ).
        weight (Tensor): Optional. A manual rescaling weight to match input tensor shape (num_samples, ).
        from_logits: bool. A boolean indicator whether the predicted value `y_pred` is
                a floating-point logit value (i.e., value in [-inf, inf] when `from_logits=True`)
                or a probablity value (i.e., value in [0., 1.] when `from_logits=False`)
                Default value is False.
    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_normalized_entropy

        >>> input = torch.tensor([0.2, 0.3])
        >>> target = torch.tensor([1.0, 0.0])
        >>> weight = None
        >>> binary_normalized_entropy(input, target, weight, from_logits=False)
        tensor(1.4183, dtype=torch.float64)

        >>> input = torch.tensor([0.2, 0.3])
        >>> target = torch.tensor([1.0, 0.0])
        >>> weight = torch.tensor([5.0, 1.0])
        >>> binary_normalized_entropy(input, target, weight, from_logits=False)
        tensor(3.1087, dtype=torch.float64)

        >>> input = tensor([-1.3863, -0.8473])
        >>> target = torch.tensor([1.0, 0.0])
        >>> weight = None
        >>> binary_normalized_entropy(input, target, weight, from_logits=True)
        tensor(1.4183, dtype=torch.float64)
    """
    cross_entropy, num_positive, num_examples = _binary_normalized_entropy_update(
        input, target, from_logits, weight
    )
    base_pos_rate = torch.clamp(
        num_positive / num_examples,
        min=torch.finfo(torch.float64).eps,
        max=1 - torch.finfo(torch.float64).eps,
    )
    baseline_entropy = torch.nn.functional.binary_cross_entropy(
        input.new_ones(input.size()) * base_pos_rate, target, weight, reduction="sum"
    )
    return (cross_entropy / baseline_entropy).double()


def _binary_normalized_entropy_update(
    input: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool,
    weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _ne_input_check(input, target, from_logits, weight)
    return _update(input, target, from_logits, weight)


def _update(
    input: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool,
    weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if from_logits:
        cross_entropy = F.binary_cross_entropy_with_logits(
            input, target, weight, reduction="sum"
        )
    else:
        cross_entropy = F.binary_cross_entropy(input, target, weight, reduction="sum")
    weight = target.new_ones(target.size()) * 1.0 if weight is None else weight
    num_examples = torch.sum(weight).double()
    num_positive = torch.sum(weight * target).double()
    return cross_entropy, num_positive, num_examples


def _ne_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool,
    weight: Optional[torch.Tensor] = None,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            f"`input` shape ({input.shape}) is different from `target` shape ({target.shape})"
        )
    if weight is not None and input.shape != weight.shape:
        raise ValueError(
            f"`weight` shape ({weight.shape}) is different from `input` shape ({input.shape})"
        )
    input_max = input.max()
    input_min = input.min()
    if not from_logits and (input_max > 1.0 or input_min < 0.0):
        raise ValueError(
            f"`from_logits`={from_logits}, `input` should be probability in range [0., 1.], but got `input` ranging",
            f"from {input_min} to {input_max}.",
            "Please set `from_logits = True` or convert `input` into valid probability value. ",
        )
