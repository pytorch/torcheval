# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, TypeVar, Union

import torch
import torch.distributed as dist

from torcheval.metrics import Metric
from torcheval.metrics.metric import TComputeReturn

from torchtnt.utils import PGWrapper
from typing_extensions import Literal

log: logging.Logger = logging.getLogger(__name__)

_TMetrics = TypeVar("_TMetrics", bound=Iterable[Metric])


def sync_and_compute(
    metric: Metric[TComputeReturn],
    process_group: Optional[dist.ProcessGroup] = None,
    recipient_rank: Union[int, Literal["all"]] = 0,
) -> Optional[TComputeReturn]:
    """
    Sync metric states and returns the ``metric.compute()`` result of
    synced metric on recipient rank. Return ``None`` on other ranks.

    Args:
        metric: The metric object to be synced and computed.
        process_group: The process group on which the metric states are
            gathered. default: ``None`` (the entire world)
        recipient_rank: The destination rank. If string "all" is passed in,
            then all ranks are the destination ranks.

    Examples::

        >>> # Assumes world_size of 3.
        >>> # Process group initialization omitted on each rank.
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torcheval.metrics import Max
        >>> max = Max()
        >>> max.update(torch.tensor(dist.get_rank())).compute()
        tensor(0.) # Rank 0
        tensor(1.) # Rank 1
        tensor(2.) # Rank 2
        >>> sync_and_compute(max)
        tensor(2.) # Rank 0
        None # Rank 1
        None # Rank 2
        >>> sync_and_compute(max, recipient_rank="all")
        tensor(2.) # Rank 0
        tensor(2.) # Rank 1
        tensor(2.) # Rank 2
    """
    # Sync metric states to rank 0, compute and broadcast the results
    # when recipient_rank is "all".
    # It uses less memory than syncing metric states to all ranks,
    # since results are usually smaller in size than metric states.
    dst_rank = 0 if recipient_rank == "all" else recipient_rank
    synced_metric = get_synced_metric(metric, process_group, dst_rank)
    compute_result = synced_metric.compute() if synced_metric else None

    if recipient_rank == "all":
        obj_list = [compute_result]
        pg = PGWrapper(process_group)
        if pg.get_rank() == dst_rank:
            pg.broadcast_object_list(obj_list, src=int(dst_rank))
        else:
            pg.broadcast_object_list(obj_list, src=int(dst_rank))
            compute_result = obj_list[0]

    return compute_result


def get_synced_state_dict(
    metric: Metric,
    process_group: Optional[dist.ProcessGroup] = None,
    recipient_rank: Union[int, Literal["all"]] = 0,
) -> Dict[str, Any]:
    """
    Return the state dict of a metric after syncing on recipient_rank.
    Return an empty dict on other ranks.

    Args:
        metric: The metric object to sync and get ``state_dict()``
        process_group: The process group on which the metric states are
            gathered. default: ``None`` (the entire world)
        recipient_rank: The destination rank. If string "all" is passed in,
            then all ranks are the destination ranks.
    Returns:
        state dict of synced metric

    Examples::

        >>> # Assumes world_size of 3.
        >>> # Process group initialization omitted on each rank.
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torcheval import Max
        >>> max = Max()
        >>> max.update(torch.tensor(dist.get_rank()))
        >>> get_synced_state_dict(max)
        {"max", tensor(2.)} # Rank 0
        {} # Rank 1
        {} # Rank 2
        >>> get_synced_state_dict(max, recipient_rank="all")
        {"max", tensor(2.)} # Rank 0
        {"max", tensor(2.)} # Rank 1
        {"max", tensor(2.)} # Rank 2
    """
    synced_metric = get_synced_metric(metric, process_group, recipient_rank)
    return synced_metric.state_dict() if synced_metric else {}


def clone_metric(metric: Metric) -> Metric:
    """
    Return a new metric instance which is cloned from the input metric.

    Args:
        metric: The metric object to clone
    Returns:
        A new metric instance from cloning
    """
    return deepcopy(metric)


def clone_metrics(metrics: _TMetrics) -> List[Metric]:
    """
    Return a list of new metric instances which are cloned from the input metrics.

    Args:
        metrics: The metric objects to clone
    Returns:
        A list of metric instances from cloning
    """
    return [clone_metric(metric) for metric in metrics]


def get_synced_metric(
    metric: Metric,
    process_group: Optional[dist.ProcessGroup] = None,
    recipient_rank: Union[int, Literal["all"]] = 0,
) -> Optional[Metric]:
    """
    Returns a metric object on recipient_rank whose internal state
    variables are synced across processes in the process_group.
    Returns ``None`` on non-recipient rank.

    If ``all`` is passed as recipient_rank, all ranks in the
    ``process_group`` are considered as recipient ranks.

    Args:
        metric: The metric object to sync.
        process_group: The process group on which the metric states are
            gathered. default: ``None`` (the entire world)
        recipient_rank: The destination rank. If string "all" is passed in,
            then all ranks are the destination ranks.
    Raises:
        ValueError: when ``recipient_rank`` is not an integer or string
            "all".

    Examples::

        >>> # Assumes world_size of 3.
        >>> # Process group initialization omitted on each rank.
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torcheval import Max
        >>> max = Max()
        >>> max.update(torch.tensor(dist.get_rank())).compute()
        tensor(0.) # Rank 0
        tensor(1.) # Rank 1
        tensor(2.) # Rank 2
        >>>  get_synced_metric(max).compute()
        tensor(2.) # Rank 0
        None # Rank 1
        None # Rank 2
         >>>  get_synced_metric(max, recipient_rank=1).compute()
        None # Rank 0
        tensor(2.) # Rank 1
        None # Rank 2
        >>>  get_synced_metric(max, recipient_rank="all").compute()
        tensor(2.) # Rank 0
        tensor(2.) # Rank 1
        tensor(2.) # Rank 2
    """
    if not (isinstance(recipient_rank, int) or recipient_rank == "all"):
        raise ValueError(
            "``recipient_rank`` should be an integer or 'all', "
            f"got {recipient_rank} instead."
        )

    world_size = PGWrapper(process_group).get_world_size()
    if world_size == 1:
        log.warning(
            "World size is 1, and metric is not synced. "
            "``get_synced_metric()`` returns the input metric."
        )
        return metric
    elif world_size == -1:
        log.warning(
            "World size is -1, and current process might not be "
            "in the process group. ``get_synced_metric()`` returns ``None``."
        )
        return None
    if world_size <= 1:
        raise RuntimeError(
            f"Unexpected world_size {world_size} is seen when syncing metrics!"
        )

    gathered_metric_list = _sync_metric_object(
        metric,
        # pyre-fixme[6]: For 2nd param expected `ProcessGroup` but got `Union[None,
        #  dist.ProcessGroup, _distributed_c10d.ProcessGroup]`.
        process_group if process_group else dist.group.WORLD,
        # pyre-ignore[6]: expect `Union[int, Literal["all"]`, got `Union[int, str]`.
        recipient_rank,
        world_size,
    )

    if gathered_metric_list is None:
        return None
    return (
        clone_metric(gathered_metric_list[0])
        .to(metric.device)
        .merge_state(gathered_metric_list[1:])
    )


def _sync_metric_object(
    metric: Metric,
    process_group: dist.ProcessGroup,
    recipient_rank: Union[int, Literal["all"]],
    world_size: int,
) -> Optional[List[Metric]]:
    metric._prepare_for_merge_state()
    gathered_metric_list = (
        [None] * world_size
        if recipient_rank == "all" or dist.get_rank() == recipient_rank
        else None
    )
    if isinstance(recipient_rank, int):
        dist.gather_object(
            metric,
            gathered_metric_list,
            dst=recipient_rank,
            group=process_group,
        )
    elif recipient_rank == "all":
        dist.all_gather_object(gathered_metric_list, metric, group=process_group)
    # pyre-ignore[7]: Expected `Optional[List[Metric[typing.Any]]]` but got `Optional[List[None]]`
    return gathered_metric_list


def reset_metrics(metrics: _TMetrics) -> _TMetrics:
    """
    Reset input metrics and returns the reset collection back to users.

    Args:
        metrics: The metrics to be reset

    Examples::

        >>> from torcheval.metrics import Max, Min
        >>> max = Max()
        >>> min = Min()
        >>> max.update(torch.tensor(1)).compute()
        >>> min.update(torch.tensor(2)).compute()
        >>> max, min = reset_metrics((max, min))
        >>> max.compute()
        tensor(0.)
        >>> min.compute()
        tensor(0.)
    """

    for metric in metrics:
        metric.reset()
    return metrics


def to_device(
    metrics: _TMetrics, device: torch.device, *args: Any, **kwargs: Any
) -> _TMetrics:
    """
    Moves input metrics to the target device and returns the moved metrics back to users.

    Args:
        metrics: The metrics to be moved to the device
        device: the device to move te metrics to
        *args: Variadic arguments forwarded to ``Metric.to``
        **kwargs: Named arguments forwarded to ``Metric.to``

    Examples::

        >>> from torcheval.metrics import Max, Min
        >>> max = Max()
        >>> min = Min()
        >>> max, min = to_device((max, min), torch.device("cuda"))
        >>> max.device
        torch.device("cuda")
        >>> min.device
        torch.device("cuda")
    """
    for metric in metrics:
        metric.to(device, *args, **kwargs)
    return metrics


def classwise_converter(
    input: torch.Tensor, name: str, labels: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Converts an unaveraged metric result tensor into a dictionary with
    each key being 'metricname_classlabel' and value being the data
    associated with that class.

    Args:
        input (torch.Tensor): The tensor to be split along its first dimension.
        name (str): Name of the metric.
        labels (List[str], Optional): Optional list of strings indicating the different classes.

    Raises:
        ValueError: When the length of `labels` is not equal to the number of classes.
    """
    if labels is None:
        return {f"{name}_{i}": val for i, val in enumerate(input)}

    if input.size(dim=0) != len(labels):
        raise ValueError(
            f"Number of labels {len(labels)} must be equal to the number of classes {input.size(dim=0)}!"
        )
    return {f"{name}_{label}": val for label, val in zip(labels, input)}
