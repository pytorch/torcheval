# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.


from typing import Iterable, List, Optional, Tuple, TypeVar, Union

import torch
from torcheval.metrics.functional.classification.binned_auprc import (
    _binary_binned_auprc_compute,
    _binary_binned_auprc_param_check,
    _binary_binned_auprc_update_input_check,
    _multiclass_binned_auprc_compute,
    _multiclass_binned_auprc_param_check,
    _multiclass_binned_auprc_update_input_check,
    DEFAULT_NUM_THRESHOLD,
)
from torcheval.metrics.functional.classification.binned_precision_recall_curve import (
    _create_threshold_tensor,
)
from torcheval.metrics.metric import Metric


TBinaryBinnedAUPRC = TypeVar("TBinaryBinnedAUPRC")
TMulticlassBinnedAUPRC = TypeVar("TMulticlassBinnedAUPRC")


class BinaryBinnedAUPRC(Metric[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Compute Binned AUPRC, which is the area under the binned version of the Precision Recall Curve, for binary classification.
    Its functional version is :func:`torcheval.metrics.functional.binary_binned_auprc`.

    Args:
        num_tasks (int):  Number of tasks that need binary_binned_auprc calculation. Default value
                    is 1. binary_binned_auprc for each task will be calculated independently.
        threshold: A integeter representing number of bins, a list of thresholds, or a tensor of thresholds.


    Examples::

        >>> import torch
        >>> from torcheval.metrics import BinaryBinnedAUPRC
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> metric = BinaryBinnedAUPRC(threshold=5)
        >>> metric.update(input, target)
        >>> metric.compute()
        (tensor([0.8056]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
        )
        >>> input = torch.tensor([1, 1, 1, 0])
        >>> target = torch.tensor([1, 1, 1, 0])
        >>> metric.update(input, target)
        >>> metric.compute()
        (tensor([0.9306]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
        )
        >>> metric = BinaryBinnedAUPRC(num_tasks=2, threshold=[0, 0.2, 0.4, 0.6, 0.8, 1])
        >>> input = torch.tensor([[1, 1, 1, 0], [0.1, 0.5, 0.7, 0.8]])
        >>> target = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        (tensor([0.6667, 0.8056]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """

    def __init__(
        self: TBinaryBinnedAUPRC,
        *,
        num_tasks: int = 1,
        threshold: Union[int, List[float], torch.Tensor] = DEFAULT_NUM_THRESHOLD,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        threshold = _create_threshold_tensor(
            threshold,
            self.device,
        )
        _binary_binned_auprc_param_check(num_tasks, threshold)
        self.num_tasks = num_tasks
        self.threshold = threshold
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TBinaryBinnedAUPRC,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TBinaryBinnedAUPRC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).
            target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).
        """
        _binary_binned_auprc_update_input_check(
            input, target, self.num_tasks, self.threshold
        )
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TBinaryBinnedAUPRC,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return Binned_AUPRC.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            Tuple:
                - Binned_AUPRC (Tensor): The return value of Binned_AUPRC for each task (num_tasks,).
                - threshold (Tensor): Tensor of threshold. Its shape is (n_thresholds, ).
        """
        return _binary_binned_auprc_compute(
            torch.cat(self.inputs, -1),
            torch.cat(self.targets, -1),
            self.num_tasks,
            self.threshold,
        )

    @torch.inference_mode()
    def merge_state(
        self: TBinaryBinnedAUPRC, metrics: Iterable[TBinaryBinnedAUPRC]
    ) -> TBinaryBinnedAUPRC:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs, -1).to(self.device)
                metric_targets = torch.cat(metric.targets, -1).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TBinaryBinnedAUPRC) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs, -1)]
            self.targets = [torch.cat(self.targets, -1)]


class MulticlassBinnedAUPRC(Metric[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Compute Binned AUPRC, which is the area under the binned version of the Precision Recall Curve, for multiclass classification.
    Its functional version is :func:`torcheval.metrics.functional.multiclass_binned_auprc`.

    Args:
        num_classes (int): Number of classes.
        threshold (Tensor, int, List[float]): Either an integer representing the number of bins, a list of thresholds, or a tensor of thresholds.
                    The same thresholds will be used for all tasks.
                    If `threshold` is a tensor, it must be 1D.
                    If list or tensor is given, the first element must be 0 and the last must be 1.
        average (str, optional):
            - ``'macro'`` [default]:
                Calculate metrics for each class separately, and return their unweighted mean.
            - ``None``:
                Calculate the metric for each class separately, and return
                the metric for every class.


    Examples::

        >>> import torch
        >>> from torcheval.metrics import MulticlassBinnedAUPRC
        >>> input = torch.tensor([[0.1, 0.2, 0.1], [0.4, 0.2, 0.1], [0.6, 0.1, 0.2], [0.4, 0.2, 0.3], [0.6, 0.2, 0.4]])
        >>> target = torch.tensor([0, 1, 2, 1, 0])
        >>> metric = MulticlassBinnedAUPRC(num_classes=3, threshold=5, average='macro')
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.35)
        >>> metric = MulticlassBinnedAUPRC(num_classes=3, threshold=5, average=None)
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.4500, 0.4000, 0.2000])
        >>> input = torch.tensor([[0.1, 0.2, 0.1, 0.4], [0.4, 0.2, 0.1, 0.7], [0.6, 0.1, 0.2, 0.4], [0.4, 0.2, 0.3, 0.2], [0.6, 0.2, 0.4, 0.5]])
        >>> target = torch.tensor([0, 1, 2, 1, 0])
        >>> threshold = torch.tensor([0.0, 0.1, 0.4, 0.7, 0.8, 1.0])
        >>> metric = MulticlassBinnedAUPRC(input, target, num_classes=4, threshold=threshold, average='macro')
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.24375)
        >>> metric = MulticlassBinnedAUPRC(input, target, num_classes=4, threshold=threshold, average='macro')
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.3250, 0.2000, 0.2000, 0.2500])
    """

    def __init__(
        self: TMulticlassBinnedAUPRC,
        *,
        num_classes: int,
        threshold: Union[int, List[float], torch.Tensor] = DEFAULT_NUM_THRESHOLD,
        average: Optional[str] = "macro",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(device=device)
        threshold = _create_threshold_tensor(
            threshold,
            self.device,
        )
        _multiclass_binned_auprc_param_check(num_classes, threshold, average)
        self.num_classes = num_classes
        self.threshold = threshold
        self.average = average
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMulticlassBinnedAUPRC,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TMulticlassBinnedAUPRC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).
            target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).
        """
        _multiclass_binned_auprc_update_input_check(input, target, self.num_classes)
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TMulticlassBinnedAUPRC,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return Binned_AUPRC.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            Tuple:
                - Binned_AUPRC (Tensor): The return value of Binned_AUPRC for each task (num_tasks,).
                - threshold (Tensor): Tensor of threshold. Its shape is (n_thresholds, ).
        """
        return _multiclass_binned_auprc_compute(
            torch.cat(self.inputs),
            torch.cat(self.targets),
            self.num_classes,
            self.threshold,
            self.average,
        )

    @torch.inference_mode()
    def merge_state(
        self: TMulticlassBinnedAUPRC, metrics: Iterable[TMulticlassBinnedAUPRC]
    ) -> TMulticlassBinnedAUPRC:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs).to(self.device)
                metric_targets = torch.cat(metric.targets).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TMulticlassBinnedAUPRC) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs)]
            self.targets = [torch.cat(self.targets)]
