# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[6]: expect int got Optional[int] for num_classes

import logging
from typing import Optional, Tuple

import torch


@torch.inference_mode()
def f1_score(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int] = None,
    average: Optional[str] = "micro",
) -> torch.Tensor:
    """
    Compute f1 score, which is defined as the harmonic mean of precision and recall.
    We convert NaN to zero when f1 score is NaN. This happens when either precision
    or recall is NaN or when both precision and recall are zero.
    Its class version is ``torcheval.metrics.F1Score``.

    Args:
        input: Tensor of label predictions.
            It could be the predicted labels, with shape of (n_sample, ).
            It could also be probabilities or logits with shape of (n_sample, n_class).
            ``torch.argmax`` will be used to convert input into predicted labels.
        target: Tensor of ground truth labels with shape of (n_sample, ).
        num_classes:
            Number of classes.
        average:
            - ``'micro'``[default]:
                Calculate the metrics globally.
            - ``'macro'``:
                Calculate metrics for each class separately, and return their unweighted mean.
                Classes with 0 true and predicted instances are ignored.
            - ``'weighted'``"
                Calculate metrics for each class separately, and return their weighted sum.
                Weights are defined as the proportion of occurrences of each class in "target".
                Classes with 0 true and predicted instances are ignored.
            - ``None``:
                Calculate the metric for each class separately, and return
                the metric for every class.

    Example:
        >>> import torch
        >>> from torcheval.metrics.functional import f1_score
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> f1_score(input, target, num_classes=4)
        tensor(0.5000)

        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> f1_score(input, target, num_classes=4, average=None)
        tensor([1., 0., 0., 1.])

        >>> input = torch.tensor([0, 0, 1, 1, 1])
        >>> target = torch.tensor([0, 0, 0, 0, 1])
        >>> f1_score(input, target, num_classes=2, average="macro")
        tensor(0.5833)

        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0.3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> f1_score(input, target, num_classes=4)
        tensor(0.5)
    """
    _f1_score_param_check(num_classes, average)
    num_tp, num_fp, num_fn, num_label = _f1_score_update(
        input, target, num_classes, average
    )
    return _f1_score_compute(num_tp, num_fp, num_fn, num_label, average)


def _f1_score_update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: Optional[int],
    average: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _f1_score_update_input_check(input, target, num_classes)

    if input.ndim == 2:
        input = torch.argmax(input, dim=1)

    if average == "micro":
        num_tp = (input == target).sum()
        num_fp = (input != target).sum()
        num_fn = num_fp
        return num_tp, num_fp, num_fn, torch.tensor(0.0)

    num_label = target.new_zeros(num_classes).scatter_(0, target, 1, reduce="add")
    num_tp = target.new_zeros(num_classes).scatter_(
        0, target[input == target], 1, reduce="add"
    )
    num_fp = target.new_zeros(num_classes).scatter_(
        0, input[input != target], 1, reduce="add"
    )
    num_fn = num_label - num_tp
    return num_tp, num_fp, num_fn, num_label


def _f1_score_compute(
    num_tp: torch.Tensor,
    num_fp: torch.Tensor,
    num_fn: torch.Tensor,
    num_label: torch.Tensor,
    average: Optional[str],
) -> torch.Tensor:
    # Check if all classes exist in either ``input`` or ``target``
    mask = (num_label != 0) | ((num_tp + num_fp) != 0)
    if False in mask:
        logging.warning(
            "Warning: there are classes that do not exist in both input and target."
        )

    if average in ("macro", "weighted"):
        # Ignore the class that has no samples in both ``input`` and `target`
        num_tp, num_fp, num_fn = num_tp[mask], num_fp[mask], num_fn[mask]

    precision = num_tp / (num_tp + num_fp)
    recall = num_tp / (num_tp + num_fn)
    f1 = 2 * precision * recall / (precision + recall)

    # Convert NaN to zero when f1 score is NaN. This happens when either precision or recall is NaN or when both precision and recall are zero.
    f1 = torch.nan_to_num(f1)

    if average == "micro":
        return f1
    elif average == "macro":
        return f1.mean()
    elif average == "weighted":
        return (f1 * (num_label[mask] / num_label.sum())).sum()
    else:  # average is None
        return f1


def _f1_score_param_check(
    num_classes: Optional[int],
    average: Optional[str],
) -> None:
    average_options = ("micro", "macro", "weighted", None)
    if average not in average_options:
        raise ValueError(
            f"`average` was not in the allowed value of {average_options}, got {average}."
        )
    if average != "micro" and (num_classes is None or num_classes <= 0):
        raise ValueError(
            f"num_classes should be a positive number when average={average}, "
            f"got num_classes={num_classes}."
        )


def _f1_score_update_input_check(
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

    if not input.ndim == 1 and not (
        input.ndim == 2 and (num_classes is None or input.shape[1] == num_classes)
    ):
        raise ValueError(
            f"input should have shape of (num_sample,) or (num_sample, num_classes), "
            f"got {input.shape}."
        )
