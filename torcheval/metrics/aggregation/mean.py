# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

import logging
from typing import Iterable, TypeVar

import torch
from torcheval.metrics.metric import Metric


TMean = TypeVar("TMean")


class Mean(Metric[torch.Tensor]):
    """
    Calculate the mean value of all elements in all the input tensors.
    Its functional version is ``torch.mean(input)``.

    Example:
        >>> import torch
        >>> from torcheval.metrics import Mean
        >>> metric = Mean()
        >>> metric.update(1)
        >>> metric.update(torch.tensor([2, 3]))
        >>> metric.compute()
        tensor(2.)

        >>> metric.update(torch.tensor(-1)).compute()
        tensor(1.2500)

        >>> metric.reset()
        >>> metric.update(torch.tensor(-1)).compute()
        tensor(-1.)
    """

    def __init__(self: TMean) -> None:
        super().__init__()
        self._add_state("value_sum", torch.tensor(0.0))
        self._add_state("weight", torch.tensor(0.0))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(self: TMean, input: torch.Tensor) -> TMean:
        self.value_sum += input.sum()
        self.weight += input.numel()
        return self

    @torch.inference_mode()
    def compute(self: TMean) -> torch.Tensor:
        """
        If no calls to ``update()`` are made before ``compute()`` is called,
        the function throws a warning and returns 0.0.
        """
        if not self.weight:
            logging.warning("No calls to update() have been made - returning 0.0")
            return torch.tensor(0.0)

        return self.value_sum / self.weight

    @torch.inference_mode()
    def merge_state(self: TMean, metrics: Iterable[TMean]) -> TMean:
        for metric in metrics:
            self.value_sum += metric.value_sum.to(self.device)
            self.weight += metric.weight.to(self.device)
        return self
