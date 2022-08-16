# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from typing import Iterable, Optional, TypeVar

import torch

from torcheval.metrics.functional.classification.auroc import (
    _auroc_compute,
    _auroc_update,
)
from torcheval.metrics.metric import Metric


TAUROC = TypeVar("TAUROC")


class BinaryAUROC(Metric[torch.Tensor]):
    """
    Compute AUROC, which is the area under the ROC Curve, for binary classification.
    Its functional version is :func:`torcheval.metrics.functional.binary_auroc`.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import BinaryAUROC
        >>> metric = BinaryAUROC()
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.6667)
        >>> input = torch.tensor([1, 1, 1, 0])
        >>> target = torch.tensor([1, 1, 1, 0])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.9167)
    """

    def __init__(
        self: TAUROC,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TAUROC,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TAUROC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be predicted label, probabilities or logits with shape of (n_sample, ).
            target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        """
        _auroc_update(input, target)
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TAUROC,
    ) -> torch.Tensor:
        return _auroc_compute(torch.cat(self.inputs), torch.cat(self.targets))

    @torch.inference_mode()
    def merge_state(self: TAUROC, metrics: Iterable[TAUROC]) -> TAUROC:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs).to(self.device)
                metric_targets = torch.cat(metric.targets).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TAUROC) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs)]
            self.targets = [torch.cat(self.targets)]
