# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch


@torch.inference_mode()
def binary_accuracy(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute binary accuracy score, which is the frequency of input matching target.
    Its functional version is ``torcheval.metrics.functional.binary_accuracy``.

    Args:
        input: Tensor of label predictions with shape of (n_sample,).
            ``torch.where(input < threshold, 0, 1)`` will be applied to the input.
        target: Tensor of ground truth labels with shape of (n_sample,).
        threshold [default: 0.5]: Threshold for converting input into predicted labels for each sample.
            ``torch.where(input < threshold, 0, 1)`` will be applied to the ``input``.
    Example:
        >>> import torch
        >>> from torcheval.metrics import BinaryAccuracy
        >>> metric = BinaryAccuracy()
        >>> input = torch.tensor([0, 0, 1, 1])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.75)  # 3 / 4

        >>> metric = BinaryAccuracy(threshold=0.7)
        >>> input = torch.tensor([0, 0.2, 0.6, 0.7])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5)  # 2 / 4
    """

    num_correct, num_total = _binary_accuracy_update(input, target, threshold)
    return _accuracy_compute(num_correct, num_total, "micro")


@torch.inference_mode()
def multiclass_accuracy(
    input: torch.Tensor,
    target: torch.Tensor,
    average: Optional[str] = "micro",
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute accuracy score, which is the frequency of input matching target.
    Its class version is ``torcheval.metrics.MultiClassAccuracy``.

    Args:
        input: Tensor of label predictions
            It could be the predicted labels, with shape of (n_sample, ).
            It could also be probabilities or logits with shape of (n_sample, n_class).
            ``torch.argmax`` will be used to convert input into predicted labels.
        target: Tensor of ground truth labels with shape of (n_sample, ).
        average:
            - ``'micro'``[default]:
                Calculate the metrics globally.
            - ``'macro'``:
                Calculate metrics for each class separately, and return their unweighted
                mean. Classes with 0 true instances are ignored.
            - ``None``:
                Calculate the metric for each class separately, and return
                the metric for every class.
                NaN is returned if a class has no sample in ``target``.
        num_classes:
            Number of classes. Required for ``'macro'`` and ``None`` average methods.

    Example:
        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_accuracy
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_accuracy(input, target)
        tensor(0.5)
        >>> multiclass_accuracy(input, target, average=None, num_classes=4)
        tensor([1., 0., 0., 1.])
        >>> multiclass_accuracy(input, target, average="macro", num_classes=4)
        tensor(0.5)
        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0,3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> multiclass_accuracy(input, target)
        tensor(0.5)
    """

    _accuracy_param_check(average, num_classes)
    num_correct, num_total = _multiclass_accuracy_update(
        input, target, average, num_classes
    )
    return _accuracy_compute(num_correct, num_total, average)


@torch.inference_mode()
def multilabel_accuracy(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    criteria: str = "exact_match",
) -> torch.Tensor:
    """
    Compute multilabel accuracy score, which is the frequency of input matching target.
    Its class version is ``torcheval.metrics.MultilabelAccuracy``.

    Args:
        input: Tensor of label predictions with shape of (n_sample, n_class).
            `torch.where(input < threshold, 0, 1)`` will be applied to the ``input``.
        target: Tensor of ground truth labels with shape of (n_sample, n_class).
        threshold: Threshold for converting input into predicted labels for each sample.
        criteria:
        - ``'exact_match'``[default]:
            The set of labels predicted for a sample must exactly match the corresponding
            set of labels in target. Also known as subset accuracy.
        - ``'hamming'``:
            Fraction of correct labels over total number of labels.

    Example:
        >>> import torch
        >>> from torcheval.metrics.functional import multilabel_accuracy
        >>> input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        >>> target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> multilabel_accuracy(input, target)
        tensor(0.5)  # 2 / 4

        >>> input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        >>> target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> multilabel_accuracy(input, target, criteria="hamming")
        tensor(0.75)  # 6 / 8
    """

    _multilabel_accuracy_param_check(criteria)
    num_correct, num_total = _multilabel_accuracy_update(
        input, target, threshold, criteria
    )
    return _accuracy_compute(num_correct, num_total, "micro")


def _multiclass_accuracy_update(
    input: torch.Tensor,
    target: torch.Tensor,
    average: Optional[str],
    num_classes: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:

    _accuracy_update_input_check(input, target, num_classes)

    if input.ndim == 2:
        input = torch.argmax(input, dim=1)

    if average == "micro":
        num_correct = (input == target).sum()
        num_total = torch.tensor(target.shape[0])
        return num_correct, num_total

    # pyre-ignore[6]: expect int got Optional[int] for num_classes
    num_correct = input.new_zeros((num_classes)).scatter_(
        0, target, (target == input).long(), reduce="add"
    )
    # pyre-ignore[6]: expect int got Optional[int] for num_classes
    num_total = target.new_zeros((num_classes)).scatter_(0, target, 1, reduce="add")
    return num_correct, num_total


@torch.jit.script
def _accuracy_compute(
    num_correct: torch.Tensor,
    num_total: torch.Tensor,
    average: Optional[str],
) -> torch.Tensor:
    if isinstance(average, str) and average == "macro":
        mask = num_total != 0
        return (num_correct[mask] / num_total[mask]).mean()
    else:
        return num_correct / num_total


def _accuracy_param_check(
    average: Optional[str],
    num_classes: Optional[int] = None,
) -> None:
    average_options = ("micro", "macro", "none", None)
    if average not in average_options:
        raise ValueError(
            f"`average` was not in the allowed value of {average_options}, got {average}."
        )
    if average != "micro" and (num_classes is None or num_classes <= 0):
        raise ValueError(
            f"num_classes should be a positive number when average={average}."
            f" Got num_classes={num_classes}."
        )


def _accuracy_update_input_check(
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
            f"target should be a one-dimensional tensor, got shape {target.shape}."
        )

    if not input.ndim == 1 and not (
        input.ndim == 2 and (num_classes is None or input.shape[1] == num_classes)
    ):
        raise ValueError(
            "input should have shape of (num_sample,) or (num_sample, num_classes), "
            f"got {input.shape}."
        )


def _binary_accuracy_update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:

    _binary_accuracy_update_input_check(input, target)

    input = torch.where(input < threshold, 0, 1)

    num_correct = (input == target).sum()
    num_total = target.new_tensor(target.shape[0])
    return num_correct, num_total


def _binary_accuracy_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same dimensions, "
            f"got shapes {input.shape} and {target.shape}."
        )
    if target.ndim != 1:
        raise ValueError(
            f"target should be a one-dimensional tensor, got shape {target.shape}."
        )


def _multilabel_accuracy_update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    criteria: str = "exact_match",
) -> Tuple[torch.Tensor, torch.Tensor]:
    _multilabel_accuracy_update_input_check(input, target)
    return _multilabel_update(input, target, threshold, criteria)


@torch.jit.script
def _multilabel_update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    criteria: str = "exact_match",
) -> Tuple[torch.Tensor, torch.Tensor]:
    input = torch.where(input < threshold, 0, 1)

    if criteria == "exact_match":
        num_correct = torch.all(input == target, dim=1).sum()
        num_total = torch.tensor(target.shape[0])
        return num_correct, num_total

    # Hamming
    num_correct = (input == target).sum()
    num_total = torch.tensor(target.numel(), device=target.device)
    return num_correct, num_total


def _multilabel_accuracy_param_check(
    criteria: str,
) -> None:
    criteria_options = ("exact_match", "hamming")
    if criteria not in criteria_options:
        raise ValueError(
            f"`criteria` was not in the allowed value of {criteria_options}, got {criteria}."
        )


def _multilabel_accuracy_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same dimensions, "
            f"got shapes {input.shape} and {target.shape}."
        )
