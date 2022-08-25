# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from typing import Iterable, Optional, Tuple, TypeVar, Union

import torch

from torcheval.metrics.functional.classification.binary_normalized_entropy import (
    _baseline_update,
    _binary_normalized_entropy_update,
)
from torcheval.metrics.metric import Metric

TWindowedNormalizedEntropy = TypeVar("TWindowedNormalizedEntropy")


class WindowedBinaryNormalizedEntropy(
    Metric[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
):
    """
    The windowed version of BinaryNormalizedEntropy that provides both windowed and liftime values.
    Compute the normalized binary cross entropy between predicted input and
    ground-truth binary target.

    Args:
        from_logits (bool): A boolean indicator whether the predicted value `y_pred` is
                    a floating-point logit value (i.e., value in [-inf, inf] when `from_logits=True`)
                    or a probablity value (i.e., value in [0., 1.] when `from_logits=False`)
                    Default value is False.
        num_tasks (int): Number of tasks that need BinaryNormalizedEntropy calculation. Default value
                    is 1. BinaryNormalizedEntropy for each task will be calculated independently.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import WindowedBinaryNormalizedEntropy

        >>> metric = WindowedBinaryNormalizedEntropy(window_size=2)
        >>> metric.update(torch.tensor([0.2, 0.3]), torch.tensor([1.0, 0.0]))
        >>> metric.update(torch.tensor([0.5, 0.6]), torch.tensor([1.0, 1.0]))
        >>> metric.update(torch.tensor([0.6, 0.2]), torch.tensor([0.0, 1.0]))
        >>> metric.num_examples, metric.windowed_num_examples
        (tensor([6.], dtype=torch.float64), tensor([[2., 2.]], dtype=torch.float64))
        >>> metric.compute()
        (tensor([1.4914], dtype=torch.float64), tensor([1.6581], dtype=torch.float64))

        >>> metric = WindowedBinaryNormalizedEntropy(window_size=2, enable_lifetime=False)
        >>> metric.update(torch.tensor([0.2, 0.3]), torch.tensor([1.0, 0.0]))
        >>> metric.update(torch.tensor([0.5, 0.6]), torch.tensor([1.0, 1.0]))
        >>> metric.update(torch.tensor([0.6, 0.2]), torch.tensor([0.0, 1.0]))
        >>> metric.windowed_num_examples
        tensor([[2., 2.]], dtype=torch.float64)
        >>> metric.compute()
        tensor([1.6581], dtype=torch.float64)

        >>> metric = WindowedBinaryNormalizedEntropy(window_size=2, num_tasks=2)
        >>> metric.update(torch.tensor([[0.2, 0.3], [0.5, 0.1]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        >>> metric.update(torch.tensor([[0.8, 0.3], [0.6, 0.1]]), torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
        >>> metric.update(torch.tensor([[0.5, 0.1], [0.3, 0.9]]), torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
        >>> metric.num_examples, metric.windowed_num_examples
        (tensor([6., 6.], dtype=torch.float64),
        tensor([[2., 2.],
                [2., 2.]], dtype=torch.float64))
        >>> metric.compute()
        (tensor([1.6729, 1.6421], dtype=torch.float64),
        tensor([1.9663, 1.4562], dtype=torch.float64))
    """

    def __init__(
        self: TWindowedNormalizedEntropy,
        *,
        from_logits: bool = False,
        num_tasks: int = 1,
        window_size: int = 100,
        enable_lifetime: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        self.from_logits = from_logits
        if num_tasks < 1:
            raise ValueError(
                "`num_tasks` value should be greater than and equal to 1, but received {num_tasks}. "
            )
        self.num_tasks = num_tasks
        self.max_num_updates = window_size
        self.next_inserted = 0
        self.enable_lifetime = enable_lifetime

        if self.enable_lifetime:
            self._add_state(
                "total_entropy",
                torch.zeros(self.num_tasks, dtype=torch.float64, device=self.device),
            )
            self._add_state(
                "num_examples",
                torch.zeros(self.num_tasks, dtype=torch.float64, device=self.device),
            )
            self._add_state(
                "num_positive",
                torch.zeros(self.num_tasks, dtype=torch.float64, device=self.device),
            )

        self._add_state(
            "windowed_total_entropy",
            torch.zeros(
                self.num_tasks,
                self.max_num_updates,
                dtype=torch.float64,
                device=self.device,
            ),
        )
        self._add_state(
            "windowed_num_examples",
            torch.zeros(
                self.num_tasks,
                self.max_num_updates,
                dtype=torch.float64,
                device=self.device,
            ),
        )
        self._add_state(
            "windowed_num_positive",
            torch.zeros(
                self.num_tasks,
                self.max_num_updates,
                dtype=torch.float64,
                device=self.device,
            ),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TWindowedNormalizedEntropy,
        input: torch.Tensor,
        target: torch.Tensor,
        *,
        weight: Optional[torch.Tensor] = None,
    ) -> TWindowedNormalizedEntropy:
        """
        Update the metric state with the total entropy, total number of examples and total number of
        positive targets.

        Args:
            input (Tensor): Predicted unnormalized scores (often referred to as logits) or
                binary class probabilities (num_tasks, num_samples).
            target (Tensor): Ground truth binary class indices (num_tasks, num_samples).
            weight (Tensor, optional): A manual rescaling weight to match input tensor shape (num_tasks, num_samples).
        """

        cross_entropy, num_positive, num_examples = _binary_normalized_entropy_update(
            input, target, self.from_logits, self.num_tasks, weight
        )
        if self.enable_lifetime:
            self.total_entropy += cross_entropy
            self.num_examples += num_examples
            self.num_positive += num_positive
        self.windowed_total_entropy[:, self.next_inserted] = cross_entropy
        self.windowed_num_examples[:, self.next_inserted] = num_examples
        self.windowed_num_positive[:, self.next_inserted] = num_positive
        self.next_inserted += 1
        self.next_inserted %= self.max_num_updates
        return self

    @torch.inference_mode()
    def compute(
        self: TWindowedNormalizedEntropy,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Return the normalized binary cross entropy.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            Tensor or Tuple of Tensors: The return value of binary normalized entropy for each task (num_tasks,).
        """

        windowed_baseline_entropy = _baseline_update(
            self.windowed_num_positive.sum(dim=1), self.windowed_num_examples.sum(dim=1)
        )
        windowed_cross_entropy = self.windowed_total_entropy.sum(
            dim=1
        ) / self.windowed_num_examples.sum(dim=1)
        windowed_normalized_entropy = windowed_cross_entropy / windowed_baseline_entropy

        if self.enable_lifetime:
            if torch.any(self.num_examples == 0):
                normalized_entropy = torch.empty(0)
            else:
                baseline_entropy = _baseline_update(
                    self.num_positive, self.num_examples
                )
                cross_entropy = self.total_entropy / self.num_examples
                normalized_entropy = cross_entropy / baseline_entropy
            return (
                normalized_entropy,
                windowed_normalized_entropy,
            )
        else:
            return windowed_normalized_entropy

    @torch.inference_mode()
    def merge_state(
        self: TWindowedNormalizedEntropy, metrics: Iterable[TWindowedNormalizedEntropy]
    ) -> TWindowedNormalizedEntropy:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """

        for metric in metrics:
            if self.enable_lifetime:
                self.total_entropy += metric.total_entropy.to(self.device)
                self.num_examples += metric.num_examples.to(self.device)
                self.num_positive += metric.num_positive.to(self.device)
            self.windowed_total_entropy = torch.cat(
                [
                    self.windowed_total_entropy,
                    metric.windowed_total_entropy.to(self.device),
                ],
                dim=1,
            )
            self.windowed_num_examples = torch.cat(
                [
                    self.windowed_num_examples,
                    metric.windowed_num_examples.to(self.device),
                ],
                dim=1,
            )
            self.windowed_num_positive = torch.cat(
                [
                    self.windowed_num_positive,
                    metric.windowed_num_positive.to(self.device),
                ],
                dim=1,
            )
        return self
