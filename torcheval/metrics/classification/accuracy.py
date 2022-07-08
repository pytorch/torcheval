# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from typing import Iterable, Optional, TypeVar

import torch
from torcheval.metrics.functional.classification.accuracy import (
    _accuracy_compute,
    _accuracy_param_check,
    _accuracy_update,
)
from torcheval.metrics.metric import Metric


TAccuracy = TypeVar("TAccuracy")


class Accuracy(Metric[torch.Tensor]):
    """
    Compute accuracy score, which is the frequency of input matching target.
    Its functional version is ``torcheval.metrics.functional.accuracy``.

    Args:
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
        >>> from torcheval.metrics import Accuracy
        >>> metric = Accuracy()
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5)

        >>> metric = Accuracy(average=None, num_classes=4)
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([1., 0., 0., 1.])

        >>> metric = Accuracy(average="macro", num_classes=2)
        >>> input = torch.tensor([0, 0, 1, 1, 1])
        >>> target = torch.tensor([0, 0, 0, 0, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.75)

        >>> metric = Accuracy()
        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0,3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5)
    """

    def __init__(
        self: TAccuracy,
        average: Optional[str] = "micro",
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        _accuracy_param_check(average, num_classes)
        self.average = average
        self.num_classes = num_classes
        if average == "micro":
            self._add_state("num_correct", torch.tensor(0.0))
            self._add_state("num_total", torch.tensor(0.0))
        else:
            # num_classes is verified to be not None when average != "micro"
            self._add_state("num_correct", torch.zeros(num_classes or 0))
            self._add_state("num_total", torch.zeros(num_classes or 0))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(self: TAccuracy, input: torch.Tensor, target: torch.Tensor) -> TAccuracy:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input: Tensor of label predictions
                It could be the predicted labels, with shape of (n_sample, ).
                It could also be probabilities or logits with shape of (n_sample, n_class).
                ``torch.argmax`` will be used to convert input into predicted labels.
            target: Tensor of ground truth labels with shape of (n_sample, ).
        """
        num_correct, num_total = _accuracy_update(
            input, target, self.average, self.num_classes
        )
        self.num_correct += num_correct
        self.num_total += num_total
        return self

    @torch.inference_mode()
    def compute(self: TAccuracy) -> torch.Tensor:
        """
        Return the accuracy score.

        NaN is returned if no calls to ``update()`` are made before ``compute()`` is called.
        """
        return _accuracy_compute(self.num_correct, self.num_total, self.average)

    @torch.inference_mode()
    def merge_state(self: TAccuracy, metrics: Iterable[TAccuracy]) -> TAccuracy:
        for metric in metrics:
            self.num_correct += metric.num_correct.to(self.device)
            self.num_total += metric.num_total.to(self.device)
        return self
