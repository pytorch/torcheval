# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from copy import deepcopy
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    overload,
    TypeVar,
    Union,
)

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


def sync_and_compute_collection(
    metrics: MutableMapping[str, Metric],
    process_group: Optional[dist.ProcessGroup] = None,
    recipient_rank: Union[int, Literal["all"]] = 0,
) -> Optional[Dict[str, Any]]:
    """
    Sync metric states across a dict of metrics and returns the
    ``metric.compute()`` result of synced metrics on recipient rank.
    Returns ``None`` on other ranks.

    Args:
        metrics: The dict of metric objects to be synced and computed.
        process_group: The process group on which the metric states are
            gathered. default: ``None`` (the entire world)
        recipient_rank: The destination rank. If string "all" is passed in,
            then all ranks are the destination ranks.

    Examples::

        >>> # Assumes world_size of 3.
        >>> # Process group initialization omitted on each rank.
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torcheval.metrics import Max, Min
        >>> metrics = {"max" : Max(), "min": Min()}
        >>> metrics["max"].update(torch.tensor(dist.get_rank())).compute()
        tensor(0.) # Rank 0
        tensor(1.) # Rank 1
        tensor(2.) # Rank 2
        >>> metrics["min"].update(torch.tensor(dist.get_rank())).compute()
        tensor(0.) # Rank 0
        tensor(1.) # Rank 1
        tensor(2.) # Rank 2
        >>> sync_and_compute_collection(metrics)
        {"max" : tensor(2.), "min": tensor(0.)} # Rank 0
        None # Rank 1
        None # Rank 2
        >>> sync_and_compute_collection(metrics, recipient_rank="all")
        {"max" : tensor(2.), "min": tensor(0.)} # Rank 0
        {"max" : tensor(2.), "min": tensor(0.)} # Rank 1
        {"max" : tensor(2.), "min": tensor(0.)} # Rank 2
    """
    # Sync metric states to rank 0, compute and broadcast the results
    # when recipient_rank is "all".
    # It uses less memory than syncing metric states to all ranks,
    # since results are usually smaller in size than metric states.
    dst_rank = 0 if recipient_rank == "all" else recipient_rank

    synced_metrics = get_synced_metric_collection(metrics, process_group, dst_rank)
    compute_result = None
    if synced_metrics is not None:
        compute_result = {key: m.compute() for key, m in synced_metrics.items()}

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


def get_synced_state_dict_collection(
    metric_collection: MutableMapping[str, Metric],
    process_group: Optional[dist.ProcessGroup] = None,
    recipient_rank: Union[int, Literal["all"]] = 0,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Return the state dict of a collection of metrics after syncing on recipient_rank.
    Return an None on other ranks.

    Args:
        metric_collection (Dict[str, Metric]): The metric objects to sync and get ``state_dict()``
        process_group: The process group on which the metric states are
            gathered. default: ``None`` (the entire world)
        recipient_rank: The destination rank. If string "all" is passed in,
            then all ranks are the destination ranks.
    Returns:
        Bundle of state dicts of for the synced metrics

    Examples::

        >>> # Assumes world_size of 3.
        >>> # Process group initialization omitted on each rank.
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torcheval import Max, Min
        >>> maximum = Max()
        >>> maximum.update(torch.tensor(dist.get_rank()))
        >>> minimum = Min()
        >>> minimum.update(torch.tensor(dist.get_rank()))
        >>> get_synced_state_dict({"max rank": maximum, "min rank": minimum})
        {"max rank": {"max", tensor(2.)}, "min rank": {"min", tensor(0.)}} # Rank 0
        None # Rank 1
        None # Rank 2
        >>> get_synced_state_dict({"max rank": maximum, "min rank": minimum}, recipient_rank="all")
        {"max rank": {"max", tensor(2.)}, "min rank": {"min", tensor(0.)}} # Rank 0
        {"max rank": {"max", tensor(2.)}, "min rank": {"min", tensor(0.)}} # Rank 1
        {"max rank": {"max", tensor(2.)}, "min rank": {"min", tensor(0.)}} # Rank 2
    """
    synced_metrics = get_synced_metric_collection(
        metric_collection,
        process_group,
        recipient_rank,
    )

    if synced_metrics is None:
        return None
    return {key: metric.state_dict() for key, metric in synced_metrics.items()}


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
        >>> synced_metric = get_synced_metric(max)  # by default sync metric states to Rank 0
        >>> synced_metric.compute() if synced_metric else None
        tensor(2.)     # Rank 0
        None # Rank 1 -- synced_metric is None
        None # Rank 2 -- synced_metric is None
        >>> synced_metric = get_synced_metric(max, recipient_rank=1)
        >>> synced_metric.compute() if synced_metric else None
        None # Rank 0 -- synced_metric is None
        tensor(2.)     # Rank 1
        None # Rank 2 -- synced_metric is None
        >>>  get_synced_metric(max, recipient_rank="all").compute()
        tensor(2.) # Rank 0
        tensor(2.) # Rank 1
        tensor(2.) # Rank 2
    """
    world_size = PGWrapper(process_group).get_world_size()
    _validate_rank_and_world_size(recipient_rank, world_size)

    if world_size == 1:
        return metric
    elif world_size == -1:
        return None

    gathered_metric_list = _sync_metric_object(
        metric,
        # pyre-fixme[6]: For 2nd param expected `ProcessGroup` but got `Union[None,
        #  dist.ProcessGroup, _distributed_c10d.ProcessGroup]`.
        process_group if process_group else dist.group.WORLD,
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


def get_synced_metric_collection(
    metric_collection: MutableMapping[str, Metric],
    process_group: Optional[dist.ProcessGroup] = None,
    recipient_rank: Union[int, Literal["all"]] = 0,
) -> Union[Optional[Dict[str, Metric]], MutableMapping[str, Metric]]:
    """
    Returns a dict of metric objects to the recipient_rank whose
    internal state variables are synced across processes in the process_group.
    Returns ``None`` on non-recipient rank.


    The data transfer is batched to maximize efficiency.

    If ``all`` is passed as recipient_rank, all ranks in the
    ``process_group`` are considered as recipient ranks.

    Args:
        metric_collection (Dict[str, Metric]): The dict of metric objects to sync.
        process_group (int): The process group on which the metric states are
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
        >>> from torcheval.metrics import Max, Min
        >>> metrics = {"max" : Max(), "min": Min()}
        >>> metrics["max"].update(torch.tensor(dist.get_rank()))
        >>> metrics["min"].update(torch.tensor(dist.get_rank()))
        >>> synced_metrics = get_synced_metric_collection(metrics)

        by default metrics sync to Rank 0
        >>> synced_metrics["max"].compute() if synced_metrics else None
        tensor(2.) # Rank 0
        None       # Rank 1 -- synced_metrics is None
        None       # Rank 2 -- synced_metrics is None
        >>> synced_metrics["min"].compute() if synced_metrics else None
        tensor(0.) # Rank 0
        None       # Rank 1 -- synced_metrics is None
        None       # Rank 2 -- synced_metrics is None

        you can also sync to all ranks or choose a specific rank
        >>> synced_metrics = get_synced_metric_collection(metrics, recipient_rank="all")
        >>> synced_metrics["max"].compute()
        tensor(2.) # Rank 0
        tensor(2.) # Rank 1
        tensor(2.) # Rank 2
        >>> synced_metrics["min"].compute()
        tensor(0.) # Rank 0
        tensor(0.) # Rank 1
        tensor(0.) # Rank 2
    """
    world_size = PGWrapper(process_group).get_world_size()
    _validate_rank_and_world_size(recipient_rank, world_size)

    if world_size == 1:
        return metric_collection
    elif world_size == -1:
        return None

    list_of_metric_collections = _sync_metric_object(
        metric_collection,
        # pyre-fixme[6]: For 2nd param expected `ProcessGroup` but got `Union[None,
        #  dist.ProcessGroup, _distributed_c10d.ProcessGroup]`.
        process_group if process_group else dist.group.WORLD,
        recipient_rank,
        world_size,
    )

    if list_of_metric_collections is None:
        return None  # on non-recipient rank

    elif isinstance(
        list_of_metric_collections[0], MutableMapping
    ):  # metric bundles are dicts.
        synced_metric_dict: Dict[str, Metric] = {}
        # we use rank 0 metric to clone regardless of the recipient rank
        rank_0_dict = list_of_metric_collections[0]

        for metric_key in rank_0_dict.keys():
            base_metric = rank_0_dict[metric_key]
            other_rank_metrics: List[Metric] = [
                list_of_metric_collections[rank][metric_key]
                for rank in range(1, world_size)
            ]

            synced_metric = (
                clone_metric(base_metric)
                .to(base_metric.device)
                .merge_state(other_rank_metrics)
            )

            synced_metric_dict[metric_key] = synced_metric

        return synced_metric_dict


def _validate_rank_and_world_size(
    recipient_rank: Union[int, Literal["all"]],
    world_size: int,
) -> None:
    if not (isinstance(recipient_rank, int) or recipient_rank == "all"):
        raise ValueError(
            "``recipient_rank`` should be an integer or 'all', "
            f"got {recipient_rank} instead."
        )

    if world_size == 1:
        log.warning(
            "World size is 1, and metric(s) not synced. "
            "returning the input metric(s)."
        )
    elif world_size == -1:
        log.warning(
            "World size is -1, and current process might not be "
            "in the process group. Returning ``None`` instead of synced metric(s)."
        )
    if world_size < 1:
        raise RuntimeError(
            f"Unexpected world_size {world_size} is seen when syncing metrics!"
        )


@overload
def _sync_metric_object(
    metric_data: Metric,
    process_group: dist.ProcessGroup,
    recipient_rank: Union[int, Literal["all"]],
    world_size: int,
) -> Optional[List[Metric]]:
    ...


@overload
def _sync_metric_object(
    metric_data: MutableMapping[str, Metric],
    process_group: dist.ProcessGroup,
    recipient_rank: Union[int, Literal["all"]],
    world_size: int,
) -> Optional[List[Dict[str, Metric]]]:
    ...


def _sync_metric_object(
    metric_data: Union[Metric, MutableMapping[str, Metric]],
    process_group: dist.ProcessGroup,
    recipient_rank: Union[int, Literal["all"]],
    world_size: int,
) -> Union[Optional[List[Metric]], Optional[List[Dict[str, Metric]]]]:

    # Prepare metrics in data package for merge
    if isinstance(metric_data, Metric):
        metric_data._prepare_for_merge_state()
    else:
        for m in metric_data.values():
            m._prepare_for_merge_state()

    # create buffer for data to land in on recipient ranks
    gathered_data_list = (
        [None] * world_size
        if recipient_rank == "all" or dist.get_rank() == recipient_rank
        else None
    )

    # sync data
    if isinstance(recipient_rank, int):
        dist.gather_object(
            metric_data,
            gathered_data_list,
            dst=recipient_rank,
            group=process_group,
        )
    elif recipient_rank == "all":
        dist.all_gather_object(gathered_data_list, metric_data, group=process_group)
    # c10d does not have type annotations.
    # pyre-ignore: Incompatible return type [7]: Expected `Union[List[Optional[Dict[str, Metric[typing.Any]]]], List[Optional[List[Metric[typing.Any]]]], List[Optional[Metric[typing.Any]]]]` but got `Optional[List[None]]`.
    return gathered_data_list


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
