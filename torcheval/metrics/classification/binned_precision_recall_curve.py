# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from typing import Iterable, List, Optional, Tuple, TypeVar, Union

import torch

from torcheval.metrics.functional.classification.binned_precision_recall_curve import (
    _binary_binned_precision_recall_curve_compute,
    _binary_binned_precision_recall_curve_update,
    _binned_precision_recall_curve_param_check,
)
from torcheval.metrics.metric import Metric


TBinaryBinnedPrecisionRecallCurve = TypeVar("TBinaryBinnedPrecisionRecallCurve")


class BinaryBinnedPrecisionRecallCurve(Metric[torch.Tensor]):
    """
    Compute precision recall curve with given thresholds.
    Its functional version is :func:`torcheval.metrics.functional.binary_binned_precision_recall_curve`.

    Args:
        threshold (int):
            an integer representing number of bins, a list of thresholds,
            or a tensor of thresholds.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import BinaryBinnedPrecisionRecallCurve
        >>> input = torch.tensor([0.2, 0.8, 0.5, 0.9])
        >>> target = torch.tensor([0, 1, 0, 1])
        >>> threshold = 5
        >>> metric = BinaryBinnedPrecisionRecallCurve(threshold)
        >>> metric.update(input, target)
        >>> metric.compute()
        (tensor([0.5000, 0.6667, 0.6667, 1.0000, 1.0000, 1.0000]),
        tensor([1., 1., 1., 1., 0., 0.]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

        >>> input = torch.tensor([0.2, 0.3, 0.4, 0.5])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])
        >>> metric = BinaryBinnedPrecisionRecallCurve(threshold)
        >>> metric.update(input, target)
        >>> metric.compute()
        (tensor([0.5000, 0.6667, 1.0000, 1.0000, 1.0000]),
        tensor([1., 1., 0., 0., 0.]),
        tensor([0.0000, 0.2500, 0.7500, 1.0000]))
    """

    def __init__(
        self: TBinaryBinnedPrecisionRecallCurve,
        *,
        threshold: Union[int, List[float], torch.Tensor] = 100,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        threshold = (
            torch.linspace(0, 1.0, threshold, device=self.device)
            if isinstance(threshold, int)
            else torch.tensor(threshold, device=self.device)
        )
        _binned_precision_recall_curve_param_check(threshold)
        self._add_state("threshold", threshold)
        self._add_state("num_tp", torch.zeros(len(threshold), device=self.device))
        self._add_state("num_fp", torch.zeros(len(threshold), device=self.device))
        self._add_state("num_fn", torch.zeros(len(threshold), device=self.device))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TBinaryBinnedPrecisionRecallCurve,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TBinaryBinnedPrecisionRecallCurve:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be probabilities or logits with shape of (n_sample, ).
            target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        """
        num_tp, num_fp, num_fn = _binary_binned_precision_recall_curve_update(
            input, target, self.threshold
        )
        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_fn += num_fn
        return self

    @torch.inference_mode()
    # pyre-ignore[15]: compute() return tuple of precision, recall and thresholds
    def compute(
        self: TBinaryBinnedPrecisionRecallCurve,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple:
                - precision (Tensor): Tensor of precision result. Its shape is (n_thresholds + 1, )
                - recall (Tensor): Tensor of recall result. Its shape is (n_thresholds + 1, )
                - thresholds (Tensor): Tensor of threshold. Its shape is (n_thresholds, )
        """
        return _binary_binned_precision_recall_curve_compute(
            self.num_tp, self.num_fp, self.num_fn, self.threshold
        )

    @torch.inference_mode()
    def merge_state(
        self: TBinaryBinnedPrecisionRecallCurve,
        metrics: Iterable[TBinaryBinnedPrecisionRecallCurve],
    ) -> TBinaryBinnedPrecisionRecallCurve:
        for metric in metrics:
            self.num_tp += metric.num_tp.to(self.device)
            self.num_fp += metric.num_fp.to(self.device)
            self.num_fn += metric.num_fn.to(self.device)
        return self
