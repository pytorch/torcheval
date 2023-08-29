# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains functions to synchronize metric states across processes, to be called within toolkit.py

The overall process goes as follows:

1) Metrics are converted to their state dict representations
2) A traversal order is established through these state dicts (by alphabetical order of keys)
3) An empty dictionary to store states received from other processes is created (this is `gathered_states` in the code)
4) The traversal begins, for each metric's state:
    - validates it is one of Tensor, List[Tensor], Dict[str, Tensor], int, or float type
    - calls the appropriate helper sync function to receive and store that state from all other processes
5) Returns these states to toolkit
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import distributed as dist, Tensor
from torch.nn import functional as F
from torcheval.metrics.metric import TState

_logger: logging.Logger = logging.getLogger(__name__)


def _simple_all_gather_tensors(
    result: Tensor, group: Optional[dist.ProcessGroup], world_size: int
) -> List[Tensor]:
    stacked_result_sizes = [world_size] + list(result.size())
    gathered_result = list(
        torch.zeros(stacked_result_sizes, dtype=result.dtype, device=result.device)
    )
    dist.all_gather(gathered_result, result, group)
    return gathered_result


# @jsenthil TODO: remove all_gather for sizes of all tensors:
def all_gather_tensors(
    result: Tensor, group: Optional[dist.ProcessGroup] = None
) -> List[Tensor]:
    """Function to gather tensors from several distributed processes onto a list that is broadcasted to all processes.
    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    # if torch.distributed is not available or not initialized
    # return single-item list containing the result
    if not dist.is_available() or not dist.is_initialized():
        return [result]

    # convert tensors to contiguous format
    result = result.contiguous()
    world_size = dist.get_world_size(group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_all_gather_tensors(result, group, world_size)

    # gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    stacked_local_size = [world_size] + list(local_size.size())
    local_sizes = list(
        torch.zeros(
            stacked_local_size, dtype=local_size.dtype, device=local_size.device
        )
    )
    dist.all_gather(local_sizes, local_size, group=group)

    # if the backend is NCCL, we can gather the differently sized tensors without padding
    if dist.get_backend(group) == "nccl":
        gathered_result = [result.new_empty(size.tolist()) for size in local_sizes]
        dist.all_gather(gathered_result, result, group)
        return gathered_result

    # if shapes are all the same, then do a simple gather:
    stacked_sizes = torch.stack(local_sizes)
    max_size = stacked_sizes.max(dim=0).values
    min_size = stacked_sizes.min(dim=0).values
    all_sizes_equal = torch.equal(max_size, min_size)
    if all_sizes_equal:
        return _simple_all_gather_tensors(result, group, world_size)

    # if not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    stacked_result_padded = [world_size] + list(result_padded.size())
    gathered_result = list(
        torch.zeros(
            stacked_result_padded,
            dtype=result_padded.dtype,
            device=result_padded.device,
        )
    )
    dist.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result


def metrics_traversal_order(
    state_dict: Dict[str, Dict[str, TState]]
) -> List[Tuple[str, str]]:
    """
    Args:
        state_dict: a dictionary of metric states (metric name -> metric's state dict).

    Returns: a list of tuple defining how to traverse a group of metrics deterministically.
        ie ((metric1 name, metric1 state1), (metric1 name, metric1 state2), (metric2 name, metric2 state1), ...)
    """
    dict_items = []
    for outer_key in sorted(state_dict.keys()):
        inner_dict = state_dict[outer_key]
        for inner_key in sorted(inner_dict.keys()):
            dict_items.append((outer_key, inner_key))
    return dict_items


def _get_empty_metric_state_collection(
    metrics_traversal_order: List[Tuple[str, str]],
) -> Dict[str, Dict[str, Any]]:
    metric_state_collection = {}
    for metric_name, state_name in metrics_traversal_order:
        if metric_name not in metric_state_collection.keys():
            metric_state_collection[metric_name] = {}
        metric_state_collection[metric_name][state_name] = {}
    return metric_state_collection


def _sync_tensor_states(
    metric_name: str,
    state_name: str,
    my_state_data: torch.Tensor,
    gathered_states: List[Dict[str, Dict[str, Any]]],
    process_group: Optional[dist.ProcessGroup],
) -> None:
    gathered_state_data = all_gather_tensors(my_state_data, group=process_group)
    for i, state_tensor in enumerate(gathered_state_data):
        gathered_states[i][metric_name][state_name] = state_tensor


def _sync_dtype_and_shape(
    tensor: Optional[torch.Tensor],
    process_group: Optional[dist.ProcessGroup],
) -> Optional[Tuple[torch.dtype, torch.Size]]:
    my_rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)

    rank_with_dtype = -1
    if tensor is not None:
        rank_with_dtype = my_rank  # send its rank

    object_list = [None for _ in range(world_size)]
    dist.all_gather_object(object_list, rank_with_dtype, group=process_group)
    # pyre-ignore: Incompatible parameter type [6]
    rank_with_dtype = max(object_list)  # choose highest rank to broadcast dtype

    if rank_with_dtype == -1:
        # all ranks have passed None for tensor
        return None

    # Broadcast the dtype from the rank that knows it to all other processes
    if my_rank == rank_with_dtype:
        # pyre-ignore: Undefined attribute [16]
        object_list = [(tensor.dtype, tensor.shape)]
    else:
        object_list = [None]

    dist.broadcast_object_list(object_list, src=rank_with_dtype, group=process_group)
    dtype, shape = object_list[0]
    return dtype, shape


def _sync_list_length(
    state_data: List[torch.Tensor], process_group: Optional[dist.ProcessGroup]
) -> List[int]:
    my_length = len(state_data)
    world_size = dist.get_world_size(group=process_group)

    lengths = [None for _ in range(world_size)]
    dist.all_gather_object(lengths, my_length, group=process_group)

    # pyre-ignore: Incompatible parameter type [7]
    return lengths


def _generate_dummy_tensor(
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def _sync_list_tensor_states(
    metric_name: str,
    state_name: str,
    my_state_data: List[torch.Tensor],
    device: torch.device,
    gathered_states: List[Dict[str, Dict[str, Any]]],
    process_group: Optional[dist.ProcessGroup],
) -> None:
    # get length of lists across all ranks
    gathered_list_lengths = _sync_list_length(
        my_state_data, process_group=process_group
    )
    if any((length == 0 for length in gathered_list_lengths)):
        # one or more ranks has empty list, need to sync dtype and shape
        # so it can send appropriate dummy tensor for allgather

        if len(my_state_data) > 0:
            # if at least one element, send your tensor dtype and shape
            result = _sync_dtype_and_shape(
                my_state_data[0], process_group=process_group
            )
        else:
            # otherwise send None to indicate your list is empty
            result = _sync_dtype_and_shape(None, process_group=process_group)
        if result is None:
            # all ranks state_data is empty, no need to sync
            return
        dtype, shape = result  # unpack results
    else:
        # pull dtype and ndim from local data
        shape = my_state_data[0].shape
        dtype = my_state_data[0].dtype

    max_length = max(gathered_list_lengths)

    # go through each element in list and sync the tensors
    for i in range(max_length):
        if i < len(my_state_data):
            gathered_state_data = all_gather_tensors(
                my_state_data[i], group=process_group
            )
        else:
            # local rank exceeded length of its list, send dummy tensor instead
            dummy_tensor = _generate_dummy_tensor(shape, dtype, device)
            gathered_state_data = all_gather_tensors(dummy_tensor, group=process_group)

        for rank, state_tensor in enumerate(gathered_state_data):
            if len(gathered_states[rank][metric_name][state_name]) == 0:
                # initialize to list
                gathered_states[rank][metric_name][state_name] = []
            # append the received state tensor to rank unless its list length is reached
            if i < gathered_list_lengths[rank]:
                gathered_states[rank][metric_name][state_name].append(state_tensor)


def _sync_dict_tensor_states(
    metric_name: str,
    state_name: str,
    my_state_data: Dict[str, torch.Tensor],
    device: torch.device,
    gathered_states: List[Dict[str, Dict[str, Any]]],
    process_group: Optional[dist.ProcessGroup],
) -> None:
    sorted_keys = sorted(my_state_data.keys())
    tensor_list = [my_state_data[key] for key in sorted_keys]
    _sync_list_tensor_states(
        metric_name, state_name, tensor_list, device, gathered_states, process_group
    )
    for rank in range(len(gathered_states)):
        tensor_list = gathered_states[rank][metric_name][state_name]
        gathered_states[rank][metric_name][state_name] = dict(
            zip(sorted_keys, tensor_list)
        )


def _sync_obj_states(
    metric_name: str,
    state_name: str,
    # pyre-ignore: Missing parameter annotation [2]
    my_state_data: Any,
    gathered_states: List[Dict[str, Dict[str, Any]]],
    process_group: Optional[dist.ProcessGroup],
) -> None:
    world_size = dist.get_world_size(group=process_group)
    gathered_obj_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_obj_data, my_state_data, group=process_group)
    for i, obj in enumerate(gathered_obj_data):
        gathered_states[i][metric_name][state_name] = obj


def sync_states(
    states: Dict[str, Dict[str, Any]],
    devices: Dict[str, torch.device],
    metrics_traversal_order: List[Tuple[str, str]],
    process_group: Optional[dist.ProcessGroup] = None,
) -> List[Dict[str, Dict[str, Any]]]:
    """
    Retrieves metric states across all ranks.

    Args:
        states: local metric state dict
        devices: mapping from metric name to device on which metric resides
        metrics_traversal_order: list of tuple (metric_name, state_name) defining the order of traversal through the metric state dicts

    Returns:
        Metric state dict data from all ranks.
    """
    gathered_states = [
        _get_empty_metric_state_collection(
            metrics_traversal_order=metrics_traversal_order
        )
        for _ in range(dist.get_world_size())
    ]

    for metric_name, state_name in metrics_traversal_order:
        my_state_data = states[metric_name][
            state_name
        ]  # tensor, list, dict, int, or float
        if isinstance(my_state_data, torch.Tensor):
            _sync_tensor_states(
                metric_name,
                state_name,
                my_state_data,
                gathered_states,
                process_group=process_group,
            )
        elif isinstance(my_state_data, list):
            _sync_list_tensor_states(
                metric_name,
                state_name,
                my_state_data,
                devices[metric_name],
                gathered_states,
                process_group=process_group,
            )
        elif isinstance(my_state_data, dict):
            _sync_dict_tensor_states(
                metric_name,
                state_name,
                my_state_data,
                devices[metric_name],
                gathered_states,
                process_group=process_group,
            )
        elif isinstance(
            my_state_data,
            (
                int,
                float,
            ),
        ):
            _sync_obj_states(
                metric_name,
                state_name,
                my_state_data,
                gathered_states,
                process_group=process_group,
            )
        else:
            raise RuntimeError(
                f"Do not know how to sync state of type: {type(my_state_data)} for state {metric_name} {state_name}"
            )

    return gathered_states
