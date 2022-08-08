# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from typing import Iterable, Optional, TypeVar

import torch
from torcheval.metrics.functional.classification.recall import (
    _recall_compute,
    _recall_param_check,
    _recall_update,
)
from torcheval.metrics.metric import Metric

TRecall = TypeVar("TRecall")


class MulticlassRecall(Metric[torch.Tensor]):
    """
    Compute the recall score, the ratio of the true positives and the sum of
    true positives and false negatives.
    Its functional version is ``torcheval.metrics.functional.multiclass_recall``.
    We cast NaNs to 0 when classes have zero instances in the ground-truth labels
    (when TP + FN = 0).

    Args:
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
        >>> from torcheval.metrics.classification import MulticlassRecall
        >>> metric = MulticlassRecall(num_classes=4)
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5000)

        >>> metric = MulticlassRecall(average=None, num_classes=4)
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([1., 0., 0., 1.])

        >>> metric = MulticlassRecall(average="macro", num_classes=2)
        >>> input = torch.tensor([0, 0, 1, 1, 1])
        >>> target = torch.tensor([0, 0, 0, 0, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5000)

        >>> metric = MulticlassRecall(num_classes=4)
        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0.3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5000)
    """

    def __init__(
        self: TRecall,
        num_classes: Optional[int] = None,
        average: Optional[str] = "micro",
    ) -> None:
        super().__init__()
        _recall_param_check(num_classes, average)
        self.num_classes = num_classes
        self.average = average

        if average == "micro":
            self._add_state("num_tp", torch.tensor(0.0))
            self._add_state("num_labels", torch.tensor(0.0))
            self._add_state("num_predictions", torch.tensor(0.0))
        else:
            assert isinstance(
                num_classes, int
            ), f"`num_classes` must be an integer, but got {num_classes}."
            self._add_state("num_tp", torch.zeros(num_classes))
            self._add_state("num_labels", torch.zeros(num_classes))
            self._add_state("num_predictions", torch.zeros(num_classes))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(self: TRecall, input: torch.Tensor, target: torch.Tensor) -> TRecall:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input: Tensor of label predictions.
                It could be the predicted labels, with shape of (n_sample, ).
                It could also be probabilities or logits with shape of (n_sample, n_class).
                ``torch.argmax`` will be used to convert input into predicted labels.
            target: Tensor of ground truth labels with shape of (n_sample, ).
        """
        num_tp, num_labels, num_predictions = _recall_update(
            input, target, self.num_classes, self.average
        )
        self.num_tp += num_tp
        self.num_labels += num_labels
        self.num_predictions += num_predictions
        return self

    @torch.inference_mode()
    def compute(self: TRecall) -> torch.Tensor:
        """
        Return the recall score.

        NaN is returned if no calls to ``update()`` are made before ``compute()`` is called.
        """
        return _recall_compute(
            self.num_tp, self.num_labels, self.num_predictions, self.average
        )

    @torch.inference_mode()
    def merge_state(self: TRecall, metrics: Iterable[TRecall]) -> TRecall:
        for metric in metrics:
            self.num_tp += metric.num_tp.to(self.device)
            self.num_labels += metric.num_labels.to(self.device)
            self.num_predictions += metric.num_predictions.to(self.device)
        return self
