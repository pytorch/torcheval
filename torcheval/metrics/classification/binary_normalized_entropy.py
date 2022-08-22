# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from typing import Iterable, Optional, TypeVar

import torch

from torcheval.metrics.functional.classification.binary_normalized_entropy import (
    _binary_normalized_entropy_update,
)
from torcheval.metrics.metric import Metric

TNormalizedEntropy = TypeVar("TNormalizedEntropy")


class BinaryNormalizedEntropy(Metric[torch.Tensor]):
    """
    Compute the normalized binary cross entropy between predicted input and
    ground-truth binary target.
    Its functional version is :func:`torcheval.metrics.functional.binary_normalized_entropy`

    Args:
        from_logits (bool): A boolean indicator whether the predicted value `y_pred` is
                    a floating-point logit value (i.e., value in [-inf, inf] when `from_logits=True`)
                    or a probablity value (i.e., value in [0., 1.] when `from_logits=False`)
                    Default value is False.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import BinaryNormalizedEntropy

        >>> metric = BinaryNormalizedEntropy()
        >>> metric.update(torch.tensor([0.2, 0.3]), torch.tensor([1.0, 0.0]))
        >>> metric.compute()
        tensor(1.4183, dtype=torch.float64)

        >>> metric = BinaryNormalizedEntropy()
        >>> metric.update(torch.tensor([0.2, 0.3]), torch.tensor([1.0, 0.0]), torch.tensor([5.0, 1.0]))
        >>> metric.compute()
        tensor(3.1087, dtype=torch.float64)

        >>> metric = BinaryNormalizedEntropy(from_logits = True)
        >>> metric.update(tensor([-1.3863, -0.8473]), torch.tensor([1.0, 0.0]))
        >>> metric.compute()
        tensor(1.4183, dtype=torch.float64)
    """

    def __init__(
        self: TNormalizedEntropy,
        *,
        from_logits: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        self.from_logits = from_logits
        self._add_state(
            "total_entropy", torch.tensor(0.0, dtype=torch.float64, device=self.device)
        )
        self._add_state(
            "num_examples", torch.tensor(0.0, dtype=torch.float64, device=self.device)
        )
        self._add_state(
            "num_positive", torch.tensor(0.0, dtype=torch.float64, device=self.device)
        )

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TNormalizedEntropy,
        input: torch.Tensor,
        target: torch.Tensor,
        *,
        weight: Optional[torch.Tensor] = None,
    ) -> TNormalizedEntropy:
        """
        Update the metric state with the total entropy, total number of examples and total number of
        positive targets.

        Args:
            input (Tensor): Predicted unnormalized scores (often referred to as logits) or
                binary class probabilities (num_samples, ).
            target (Tensor): Ground truth binary class indices (num_samples, ).
            weight (Tensor): Optional. A manual rescaling weight to match input tensor shape (num_samples, ).
        """

        cross_entropy, num_positive, num_examples = _binary_normalized_entropy_update(
            input, target, self.from_logits, weight
        )
        self.total_entropy += cross_entropy
        self.num_examples += num_examples
        self.num_positive += num_positive
        return self

    @torch.inference_mode()
    def compute(self: TNormalizedEntropy) -> torch.Tensor:
        """
        Return the normalized binary cross entropy.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.
        """
        if self.num_examples == 0.0:
            return torch.empty(0)

        base_pos_rate = torch.clamp(
            self.num_positive / self.num_examples,
            min=torch.finfo(torch.float64).eps,
            max=1 - torch.finfo(torch.float64).eps,
        )
        baseline_entropy = -base_pos_rate * torch.log(base_pos_rate) - (
            1 - base_pos_rate
        ) * torch.log(1 - base_pos_rate)
        entropy = self.total_entropy / self.num_examples
        return entropy / baseline_entropy

    @torch.inference_mode()
    def merge_state(
        self: TNormalizedEntropy, metrics: Iterable[TNormalizedEntropy]
    ) -> TNormalizedEntropy:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        for metric in metrics:
            self.total_entropy += metric.total_entropy.to(self.device)
            self.num_examples += metric.num_examples.to(self.device)
            self.num_positive += metric.num_positive.to(self.device)
        return self
